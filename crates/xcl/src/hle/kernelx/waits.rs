//! `XCL_TRACE_WAITS=1`: what the title's threads wait on. Its events, semaphores and mutexes are
//! noted with where they were made and when they were last signalled; its waits with the handles,
//! the call site and since when. When the title has not presented for 3 s, the threads that have
//! waited a second or more are logged with what each waits on – a title that stops presenting
//! while every thread sleeps waits for a signal nobody gives, and this names the object.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;
use windows_sys::Win32::System::Threading as th;

type HANDLE = *mut c_void;
type BOOL = i32;

/// `XCL_TRACE_WAITS=1` is set
pub fn tracing() -> bool {
    on()
}

/// The title's return addresses above the caller of this function (`module+offset`).
pub fn caller() -> String {
    call_site()
}

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_TRACE_WAITS").is_ok_and(|v| v.trim() == "1"))
}

/// an object the title made
struct Obj {
    kind: &'static str,
    made_at: String,
    signals: u64,
    last_signal: Option<(Instant, String)>,
}

/// a wait in progress
struct Waiting {
    handles: Vec<usize>,
    since: Instant,
    from: String,
    timeout: u32,
    all: bool,
}

static OBJS: Mutex<Option<HashMap<usize, Obj>>> = Mutex::new(None);
static WAITS: Mutex<Option<HashMap<u32, Waiting>>> = Mutex::new(None);

/// The title's return addresses above this wrapper (`module+offset`).
fn call_site() -> String {
    use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
    let mut frames = [std::ptr::null_mut::<c_void>(); 4];
    let n = unsafe { RtlCaptureStackBackTrace(2, frames.len() as u32, frames.as_mut_ptr(), std::ptr::null_mut()) };
    frames[..n as usize].iter().map(|&f| crate::modules::describe_address(f as usize)).collect::<Vec<_>>().join(" < ")
}

fn made(h: HANDLE, kind: &'static str) -> HANDLE {
    if on() && !h.is_null() {
        let at = call_site();
        OBJS.lock().unwrap().get_or_insert_with(HashMap::new).insert(h as usize, Obj { kind, made_at: at, signals: 0, last_signal: None });
    }
    h
}

fn signalled(h: HANDLE) {
    if !on() {
        return;
    }
    let at = call_site();
    let tid = unsafe { th::GetCurrentThreadId() };
    if let Some(o) = OBJS.lock().unwrap().get_or_insert_with(HashMap::new).get_mut(&(h as usize)) {
        o.signals += 1;
        o.last_signal = Some((Instant::now(), format!("thread {tid} from {at}")));
    }
}

/// A wait begins on this thread (`None` when tracing is off).
fn begin(handles: &[HANDLE], timeout: u32, all: bool) -> Option<u32> {
    if !on() {
        return None;
    }
    monitor();
    let tid = unsafe { th::GetCurrentThreadId() };
    let w = Waiting { handles: handles.iter().map(|&h| h as usize).collect(), since: Instant::now(), from: call_site(), timeout, all };
    WAITS.lock().unwrap().get_or_insert_with(HashMap::new).insert(tid, w);
    Some(tid)
}

fn end(tid: Option<u32>) {
    if let Some(tid) = tid {
        if let Some(m) = WAITS.lock().unwrap().as_mut() {
            m.remove(&tid);
        }
    }
}

/// Started with the first traced wait: once the title has not presented for 3 s, the threads
/// waiting a second or more and the objects they wait on (once per stall).
fn monitor() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        let _ = std::thread::Builder::new().name("xcl-wait-trace".into()).spawn(|| {
            let (mut last, mut since, mut reported) = (u64::MAX, Instant::now(), false);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let presents = crate::hle::d3d12x::present::counters().0;
                if presents != last {
                    (last, since, reported) = (presents, Instant::now(), false);
                    continue;
                }
                if reported || presents == 0 || since.elapsed().as_secs() < 3 {
                    continue;
                }
                reported = true;
                report(since.elapsed().as_secs());
            }
        });
    });
}

fn report(stalled_s: u64) {
    let waits = WAITS.lock().unwrap();
    let objs = OBJS.lock().unwrap();
    let mut long: Vec<(&u32, &Waiting)> = waits.iter().flatten().filter(|(_, w)| w.since.elapsed().as_millis() >= 1000).collect();
    long.sort_by_key(|(_, w)| std::cmp::Reverse(w.since.elapsed()));
    crate::xlog_warn!("wait trace: no present for {stalled_s} s – {} of the title's threads have waited 1 s or more", long.len());
    for (f, signalled, completed, cell) in crate::hle::d3d12x::fence_x::cells() {
        crate::xlog_warn!("wait trace: fence {f:#x} the title reads: queued to signal up to {signalled:#x}, GPU completed {completed:#x}, cell {cell:#x}");
    }
    for (tid, w) in long {
        let what: Vec<String> = w
            .handles
            .iter()
            .map(|h| match objs.as_ref().and_then(|m| m.get(h)) {
                Some(o) => format!(
                    "{h:#x} ({}, made at {}, signalled {}×, last {})",
                    o.kind,
                    o.made_at,
                    o.signals,
                    o.last_signal.as_ref().map_or("never".to_string(), |(t, by)| format!("{} ms ago by {by}", t.elapsed().as_millis()))
                ),
                None => format!("{h:#x} (not made by the title's CreateEvent/Semaphore/Mutex)"),
            })
            .collect();
        crate::xlog_warn!(
            "wait trace: thread {tid} waits {} ms (timeout {}{}) from {} on {}",
            w.since.elapsed().as_millis(),
            if w.timeout == u32::MAX { "infinite".to_string() } else { format!("{} ms", w.timeout) },
            if w.all { ", all" } else { "" },
            w.from,
            what.join("; ")
        );
    }
}

/// A WaitFrameEventX in progress (`d3d12x::frame`): its handles (the first the frame event's own)
pub fn frame_wait_begin(ty: u32, timeout: u32, flags: u32, handles: &[HANDLE]) -> Option<u32> {
    if !on() {
        return None;
    }
    monitor();
    let tid = unsafe { th::GetCurrentThreadId() };
    let w = Waiting { handles: handles.iter().map(|&h| h as usize).collect(), since: Instant::now(), from: format!("WaitFrameEventX(type {ty}, flags {flags:#x}; first handle the frame event)"), timeout, all: flags & 1 != 0 };
    WAITS.lock().unwrap().get_or_insert_with(HashMap::new).insert(tid, w);
    Some(tid)
}

pub fn frame_wait_end(tid: Option<u32>) {
    end(tid);
}

pub unsafe extern "system" fn CreateEventW(attr: *const c_void, manual: BOOL, initial: BOOL, name: *const u16) -> HANDLE {
    made(th::CreateEventW(attr as _, manual, initial, name), if manual != 0 { "manual-reset event" } else { "auto-reset event" })
}

pub unsafe extern "system" fn CreateEventA(attr: *const c_void, manual: BOOL, initial: BOOL, name: *const u8) -> HANDLE {
    made(th::CreateEventA(attr as _, manual, initial, name), if manual != 0 { "manual-reset event" } else { "auto-reset event" })
}

pub unsafe extern "system" fn CreateSemaphoreW(attr: *const c_void, initial: i32, max: i32, name: *const u16) -> HANDLE {
    made(th::CreateSemaphoreW(attr as _, initial, max, name), "semaphore")
}

pub unsafe extern "system" fn CreateSemaphoreExW(attr: *const c_void, initial: i32, max: i32, name: *const u16, flags: u32, access: u32) -> HANDLE {
    made(th::CreateSemaphoreExW(attr as _, initial, max, name, flags, access), "semaphore")
}

pub unsafe extern "system" fn CreateMutexA(attr: *const c_void, owner: BOOL, name: *const u8) -> HANDLE {
    made(th::CreateMutexA(attr as _, owner, name), "mutex")
}

pub unsafe extern "system" fn SetEvent(h: HANDLE) -> BOOL {
    signalled(h);
    th::SetEvent(h)
}

pub unsafe extern "system" fn ReleaseSemaphore(h: HANDLE, count: i32, prev: *mut i32) -> BOOL {
    signalled(h);
    th::ReleaseSemaphore(h, count, prev)
}

pub unsafe extern "system" fn WaitForSingleObject(h: HANDLE, ms: u32) -> u32 {
    let t = begin(&[h], ms, false);
    let r = th::WaitForSingleObject(h, ms);
    end(t);
    r
}

pub unsafe extern "system" fn WaitForSingleObjectEx(h: HANDLE, ms: u32, alertable: BOOL) -> u32 {
    let t = begin(&[h], ms, false);
    let r = th::WaitForSingleObjectEx(h, ms, alertable);
    end(t);
    r
}

pub unsafe extern "system" fn WaitForMultipleObjects(n: u32, hs: *const HANDLE, all: BOOL, ms: u32) -> u32 {
    let t = if hs.is_null() { None } else { begin(std::slice::from_raw_parts(hs, n as usize), ms, all != 0) };
    let r = th::WaitForMultipleObjects(n, hs, all, ms);
    end(t);
    r
}
