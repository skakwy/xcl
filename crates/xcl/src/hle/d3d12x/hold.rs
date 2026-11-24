//! Submissions held back on the CPU until what they wait for is submitted.
//!
//! A console title submits a list that waits (Wait64BitValueX) for a value a list it submits
//! later writes: on the console the GPU waits and the submission returns at once. Waited for on
//! the CPU, the submitting thread blocked itself; waited for on the GPU (the Wait64 bridge), a
//! queue held a wait on work nobody had submitted, and under memory pressure Windows' video
//! memory manager waited for that queue – every graphics call of the process hung (GTA VI's
//! world loading). Such a submission is kept here instead, with everything the title gives the
//! same queue after it (lists, fence signals and waits), in order; it goes to the GPU when a
//! list writing the value was submitted, the memory holds it, or – a last resort – after
//! `HOLD_LIMIT`. Then nothing on the GPU waits for work the CPU has not submitted.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How long a submission is held at most (then it goes as it would have without holding).
const HOLD_LIMIT: Duration = Duration::from_secs(5);

/// An operation on a queue kept in order behind a held submission.
enum Op {
    /// ExecuteCommandLists with the title's lists (as given, and the PC lists they run as) and the
    /// wait it is held for
    Execute { lists: Vec<usize>, pc: Vec<usize>, wait: Option<(u64, u64)>, since: Instant },
    /// Signal(fence, value) – `x`: SignalX
    Signal { fence: usize, value: u64, x: bool },
    /// Wait(fence, value)
    Wait { fence: usize, value: u64 },
}

/// queue → what is held on it, oldest first
static HELD: Mutex<Option<HashMap<usize, VecDeque<Op>>>> = Mutex::new(None);
/// queues with something held (a lock-free first check)
static HOLDING: AtomicUsize = AtomicUsize::new(0);
/// one thread releases at a time (a release submits, which releases again)
static PUMP: Mutex<()> = Mutex::new(());

thread_local! {
    /// set while this thread submits a released operation: it goes straight through
    static RELEASING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_HOLD_SUBMITS").map_or(true, |v| v.trim() != "0"))
}

fn releasing() -> bool {
    RELEASING.with(|r| r.get())
}

/// Whether `queue` has operations held (then later ones on it are held too, in order).
fn queue_held(queue: usize) -> bool {
    HOLDING.load(Ordering::Acquire) > 0 && HELD.lock().unwrap().as_ref().is_some_and(|m| m.get(&queue).is_some_and(|q| !q.is_empty()))
}

fn push(queue: usize, op: Op) {
    let mut g = HELD.lock().unwrap();
    let q = g.get_or_insert_with(HashMap::new).entry(queue).or_default();
    if q.is_empty() {
        HOLDING.fetch_add(1, Ordering::AcqRel);
    }
    q.push_back(op);
    drop(g);
    start_thread();
}

/// The PC lists the title's lists run as (what `q_execute_command_lists` submits).
unsafe fn mapped(lists: &[usize]) -> Vec<*mut c_void> {
    let mut out = Vec::with_capacity(lists.len());
    for &l in lists {
        let l = l as *mut c_void;
        if let Some(pc) = super::console_list::pc_list(l) {
            out.push(pc);
        } else if !super::title_list::push_segments(l, &mut out) {
            out.push(l);
        }
    }
    out
}

/// ExecuteCommandLists on `queue`: true when it is held (the caller returns), false when it goes
/// to the GPU now.
pub unsafe fn hold_execute(queue: *mut c_void, n: u32, lists: *const *mut c_void) -> bool {
    if !enabled() || releasing() {
        return false;
    }
    let given: Vec<usize> = std::slice::from_raw_parts(lists, n as usize).iter().map(|&l| l as usize).collect();
    let behind = queue_held(queue as usize);
    let pc = mapped(&given);
    let at = super::list_x::unsubmitted_wait_at(&pc);
    if !behind && at.is_none() {
        return false;
    }
    let wait = at.map(|(_, va, v)| (va, v));
    // the segments before the one that waits run now (their work, and the writes the title's
    // other threads wait for, as on the console); the rest is held
    let first = if behind { 0 } else { at.map_or(0, |(i, _, _)| i) };
    if first > 0 {
        RELEASING.with(|r| r.set(true));
        super::real::execute_held(queue, &pc[..first]);
        RELEASING.with(|r| r.set(false));
    }
    let pc: Vec<usize> = pc[first..].iter().map(|&l| l as usize).collect();
    static N: AtomicUsize = AtomicUsize::new(0);
    let k = N.fetch_add(1, Ordering::Relaxed);
    if k < 16 || k.is_power_of_two() {
        let q = super::list_x::queue_label(queue as usize);
        match wait {
            Some((va, v)) => crate::xlog_info!("D3D12.X: submission on q{q} held until a list writing {va:#x} >= {v:#x} is submitted ({} so far)", k + 1),
            None => crate::xlog_info!("D3D12.X: submission on q{q} held behind an earlier one ({} so far)", k + 1),
        }
    }
    push(queue as usize, Op::Execute { lists: given, pc, wait, since: Instant::now() });
    true
}

/// Signal / SignalX on `queue`: true when it is held behind a held submission.
pub fn hold_signal(queue: *mut c_void, fence: usize, value: u64, x: bool) -> bool {
    if !enabled() || releasing() || !queue_held(queue as usize) {
        return false;
    }
    push(queue as usize, Op::Signal { fence, value, x });
    true
}

/// Wait on `queue`: true when it is held behind a held submission.
pub fn hold_wait(queue: *mut c_void, fence: usize, value: u64) -> bool {
    if !enabled() || releasing() || !queue_held(queue as usize) {
        return false;
    }
    push(queue as usize, Op::Wait { fence, value });
    true
}

/// Whether the operation at the front of a queue can go now.
unsafe fn ready(op: &Op) -> bool {
    match op {
        Op::Execute { pc, wait, since, .. } => {
            if since.elapsed() >= HOLD_LIMIT {
                static N: AtomicUsize = AtomicUsize::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 16 {
                    crate::xlog_warn!("D3D12.X: a held submission goes after {} s without what it waits for ({wait:x?})", HOLD_LIMIT.as_secs());
                }
                return true;
            }
            // what it was held for, and whatever else of it waits on an unsubmitted value
            let pc: Vec<*mut c_void> = pc.iter().map(|&l| l as *mut c_void).collect();
            wait.is_none_or(|(va, v)| super::list_x::wait_released(va, v)) && super::list_x::unsubmitted_wait(&pc).is_none()
        }
        _ => true,
    }
}

/// Submit what can go now, queue by queue and in order. Called after every submission and by
/// a thread of its own.
pub unsafe fn release() {
    if HOLDING.load(Ordering::Acquire) == 0 || releasing() {
        return;
    }
    let Ok(_pump) = PUMP.try_lock() else { return };
    loop {
        // the next operation that can go, taken out under the lock and run without it
        let next = {
            let mut g = HELD.lock().unwrap();
            let Some(m) = g.as_mut() else { return };
            let mut found = None;
            for (&queue, q) in m.iter_mut() {
                if q.front().is_some_and(|op| ready(op)) {
                    let op = q.pop_front().unwrap();
                    if q.is_empty() {
                        HOLDING.fetch_sub(1, Ordering::AcqRel);
                    }
                    found = Some((queue, op));
                    break;
                }
            }
            found
        };
        let Some((queue, op)) = next else { return };
        run(queue, op);
    }
}

/// The title resets `list` (a console, title or PC list): if it is held, it and what is held before
/// it on its queue go to the GPU now – as it would have without holding (a GPU wait). On the
/// console a submitted list is the GPU's at once and its object can be recorded again; submitted
/// later, GTA VI's re-recorded list removed the device (0x887a0001, half a second after the first
/// hold).
pub unsafe fn flush_list(list: usize) {
    if HOLDING.load(Ordering::Acquire) == 0 || releasing() {
        return;
    }
    // only a held list waits for the release lock: XCL's own lists are reset with other locks held
    // (title_textures' upload list, under the refresh lock a release takes in turn – the two
    // threads waited on each other)
    let is_held = |m: &HashMap<usize, VecDeque<Op>>| m.values().any(|ops| ops.iter().any(|op| matches!(op, Op::Execute { lists, pc, .. } if lists.contains(&list) || pc.contains(&list))));
    if !HELD.lock().unwrap().as_ref().is_some_and(is_held) {
        return;
    }
    let _pump = PUMP.lock().unwrap();
    let ops: Vec<(usize, Op)> = {
        let mut g = HELD.lock().unwrap();
        let Some(m) = g.as_mut() else { return };
        let Some((&queue, at)) = m.iter().find_map(|(q, ops)| {
            ops.iter().position(|op| matches!(op, Op::Execute { lists, pc, .. } if lists.contains(&list) || pc.contains(&list))).map(|i| (q, i))
        }) else {
            return;
        };
        let q = m.get_mut(&queue).unwrap();
        let taken: Vec<(usize, Op)> = q.drain(..=at).map(|op| (queue, op)).collect();
        if q.is_empty() {
            HOLDING.fetch_sub(1, Ordering::AcqRel);
        }
        taken
    };
    static N: AtomicUsize = AtomicUsize::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 16 {
        crate::xlog_info!("D3D12.X: a held list is reset by the title: {} held operations go now", ops.len());
    }
    for (queue, op) in ops {
        run(queue, op);
    }
}

/// Submit one released operation (straight through: not held again).
unsafe fn run(queue: usize, op: Op) {
    RELEASING.with(|r| r.set(true));
    let q = queue as *mut c_void;
    match op {
        // the PC lists (segments) themselves: the part of a list that already went is not again
        Op::Execute { pc, .. } => {
            let raw: Vec<*mut c_void> = pc.iter().map(|&l| l as *mut c_void).collect();
            super::real::execute_held(q, &raw);
        }
        Op::Signal { fence, value, x } => super::real::signal_held(q, fence, value, x),
        Op::Wait { fence, value } => super::real::wait_held(q, fence, value),
    }
    RELEASING.with(|r| r.set(false));
}

/// The thread that releases what CPU writes or the time limit let go (no submission follows).
fn start_thread() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-held-submits".into()).spawn(|| loop {
            std::thread::sleep(Duration::from_millis(2));
            unsafe { release() };
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Operations on a queue that holds something are kept in order; another queue's are not.
    #[test]
    fn a_queue_with_something_held_keeps_its_later_operations_in_order() {
        let (a, b) = (0x5100_0001usize, 0x5100_0002usize);
        push(a, Op::Execute { lists: vec![1], pc: vec![2], wait: Some((0x1000, 5)), since: Instant::now() });
        assert!(queue_held(a));
        assert!(!queue_held(b));
        assert!(hold_signal(a as *mut c_void, 0x77, 3, false));
        assert!(hold_wait(a as *mut c_void, 0x78, 4));
        assert!(!hold_signal(b as *mut c_void, 0x77, 3, false), "another queue's signal goes at once");
        let kinds: Vec<&'static str> = HELD.lock().unwrap().as_ref().unwrap()[&a]
            .iter()
            .map(|op| match op {
                Op::Execute { .. } => "execute",
                Op::Signal { .. } => "signal",
                Op::Wait { .. } => "wait",
            })
            .collect();
        assert_eq!(kinds, ["execute", "signal", "wait"]);
        // (cleanup: other tests share the map)
        HELD.lock().unwrap().as_mut().unwrap().remove(&a);
        HOLDING.fetch_sub(1, Ordering::AcqRel);
    }
}
