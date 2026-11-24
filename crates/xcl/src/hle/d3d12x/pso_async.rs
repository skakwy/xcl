//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};

struct Pending {
    /// the build, until a worker – or a bind that cannot wait for the queue – takes it
    job: Mutex<Option<Job>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// a bounded compute wait already timed out; subsequent binds skip waiting until `real` lands
    timed_out: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
}

impl Pending {
    fn note_use(&self) {
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
        }
    }
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(p: &Pending, real: usize) {
    if real == 0 {
        FAILED.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILT.fetch_add(1, Ordering::Relaxed);
    }
    let orphan = {
        let mut done = p.done.lock().unwrap();
        let orphan = p.orphan.load(Ordering::Acquire);
        if !orphan {
            p.real.store(real, Ordering::Release);
        }
        *done = true;
        p.cv.notify_all();
        orphan
    };
    // outside `done`: the Release override takes PENDING (`counts` takes PENDING, then `done`)
    if orphan {
        release(real);
    }
}
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnOnce() -> usize + Send>;

fn spawn_pool(prefix: &str, threads: usize) -> Mutex<mpsc::Sender<Arc<Pending>>> {
    let (tx, rx) = mpsc::channel::<Arc<Pending>>();
    let rx = Arc::new(Mutex::new(rx));
    // translation recurses deeply on big programs; DXC itself runs on its own pool
    for k in 0..threads {
        let rx = rx.clone();
        let _ = std::thread::Builder::new().name(format!("{prefix}-{k}")).stack_size(64 << 20).spawn(move || loop {
            let next = rx.lock().unwrap().recv();
            let Ok(p) = next else { return };
            // (None: a bind took it and builds it itself)
            let job = p.job.lock().unwrap().take();
            if let Some(job) = job {
                complete(&p, job());
            }
        });
    }
    Mutex::new(tx)
}

fn queue() -> &'static Mutex<mpsc::Sender<Arc<Pending>>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Arc<Pending>>>> = OnceLock::new();
    Q.get_or_init(|| spawn_pool("xcl-pso", crate::gpu::recomp::dxc::compile_threads()))
}

/// Dedicated worker pool for compute pipelines so a backlog (or driver stall, #156) of graphics
/// builds cannot starve compute dispatches indefinitely.
fn compute_queue() -> &'static Mutex<mpsc::Sender<Arc<Pending>>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Arc<Pending>>>> = OnceLock::new();
    Q.get_or_init(|| spawn_pool("xcl-pso-compute", (crate::gpu::recomp::dxc::compile_threads() / 2).clamp(1, 4)))
}

/// Send `p` to its worker pool if not already sent; clears `sent` on failure so a later bind
/// can retry instead of stalling forever.
fn enqueue(p: &Arc<Pending>) -> bool {
    if !p.sent.swap(true, Ordering::AcqRel) {
        let q = if p.compute { compute_queue() } else { queue() };
        if q.lock().unwrap().send(p.clone()).is_err() {
            p.sent.store(false, Ordering::Release);
            return false;
        }
    }
    true
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true, 0);
}

/// Lazy pipeline creation (`XCL_LAZY_PSO=0`: off): a title's pipelines with console shaders get
/// their stand-in when made and are built when first used – the first bind of a graphics one
/// sends it to the workers, a compute one is built at its first bind. A retail game makes
/// ~51 000 pipelines in its first 30 s and uses few; each one built stays in the driver
/// (~400 KB), and the process ran out of commit.
pub fn lazy() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// `submit`, sent to the workers only on its first `resolve` (`lazy`) – or now, at a low
/// priority, when an earlier run used a pipeline of these shaders (`key`, 0 = none; `used`).
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    submit_as(standin, compute, job, false, key);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool, key: u64) {
    let p = Arc::new(Pending {
        job: Mutex::new(Some(job)),
        sent: std::sync::atomic::AtomicBool::new(false),
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        key,
        used: Default::default(),
        timed_out: Default::default(),
        orphan: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        let _ = enqueue(&p);
    } else if used::known(key) {
        if let Some(q) = preload() {
            q.list.lock().unwrap().push_back(p);
            q.cv.notify_one();
        }
    }
}

/// The pipelines a title used, by a hash of their shaders, kept between runs
/// (`used-pipelines.txt` beside the DXIL cache, a key a line).
///
/// A lazy pipeline is built at its first use: until then a graphics one draws nothing, and a
/// compute one is waited for on the title's thread. A retail title's first frames of its world
/// use hundreds of pipelines for the first time at once. The pipelines of an earlier run are
/// built while the title loads instead (`preload`): measured on a build that skips a dispatch
/// instead of waiting for its pipeline, ~2 170 dispatches were skipped in those frames without
/// this and 94 – 150 with it.
mod used {
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock, RwLock};

    struct Used {
        earlier: RwLock<HashSet<u64>>,
        now: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
    }

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).filter(|&k| k != 0).collect()
    }

    fn get() -> &'static Used {
        static U: OnceLock<Used> = OnceLock::new();
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used { earlier: RwLock::new(HashSet::new()), now: Mutex::new((HashSet::new(), None)) };
            }
            let path = crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt");
            let earlier = parse(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
            Used { now: Mutex::new((earlier.clone(), file)), earlier: RwLock::new(earlier) }
        })
    }

    /// An earlier run used a pipeline of these shaders.
    pub fn known(key: u64) -> bool {
        key != 0 && get().earlier.read().unwrap().contains(&key)
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        if key == 0 {
            return;
        }
        let mut g = get().now.lock().unwrap();
        if g.0.insert(key) {
            if let Some(f) = g.1.as_mut() {
                let _ = writeln!(f, "{key:016x}");
            }
        }
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.lock().unwrap().0.contains(&key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        get().earlier.write().unwrap().insert(key);
    }
}

/// The workers that build the pipelines of an earlier run ahead of their first use
/// (`XCL_PSO_PRELOAD=0`: off). Apart from the ones a bind sends to (`queue`) and at the lowest
/// priority: a pipeline that is bound is not held up by the ones made ahead (whoever takes
/// `job` first builds it), and the title's own threads are not either.
struct Preload {
    list: Mutex<std::collections::VecDeque<Arc<Pending>>>,
    cv: Condvar,
}

fn preload() -> Option<&'static Preload> {
    static Q: OnceLock<Option<Preload>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let q = Q
        .get_or_init(|| {
            if std::env::var("XCL_PSO_PRELOAD").is_ok_and(|v| v.trim() == "0") {
                return None;
            }
            Some(Preload { list: Mutex::new(Default::default()), cv: Condvar::new() })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        for k in 0..n {
            let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
                static DONE: AtomicUsize = AtomicUsize::new(0);
                loop {
                    let p = {
                        let mut g = q.list.lock().unwrap();
                        loop {
                            if let Some(p) = g.pop_front() {
                                break p;
                            }
                            g = q.cv.wait(g).unwrap();
                        }
                    };
                    // (released by the title meanwhile)
                    if p.orphan.load(Ordering::Acquire) {
                        continue;
                    }
                    // (None: a bind took it and builds it itself)
                    let job = p.job.lock().unwrap().take();
                    if let Some(job) = job {
                        let real = job();
                        complete(&p, real);
                        let n = DONE.fetch_add(1, Ordering::Relaxed) + 1;
                        if n.is_power_of_two() {
                            crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
                        }
                    }
                }
            });
        }
    });
    Some(q)
}

#[cfg(test)]
mod used_tests {
    use super::*;

    #[test]
    fn the_keys_of_a_file_are_read_and_other_lines_passed_over() {
        let keys = used::parse("00000000000000ff\n\nnot a key\n  1a2b \n0\n");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0xff) && keys.contains(&0x1a2b));
    }

    /// A lazy pipeline of shaders an earlier run used is built without a bind; another one is
    /// built only at its first use, and that use is kept for the next run.
    #[test]
    fn a_pipeline_an_earlier_run_used_is_built_ahead_of_its_first_use() {
        let (known, unknown) = (0x7E57_7001usize, 0x7E57_7002usize);
        let (known_key, unknown_key) = (0x5EED_0001u64, 0x5EED_0002u64);
        used::earlier_run_used(known_key);
        submit_lazy(known, false, Box::new(|| 0x4321), known_key);
        submit_lazy(unknown, false, Box::new(|| 0x4322), unknown_key);
        let start = std::time::Instant::now();
        while is_pending(known) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(known), "built by the workers that build ahead, with no bind");
        assert_eq!(resolve(known), 0x4321);
        assert!(is_pending(unknown), "nobody used it and no earlier run did");
        assert!(!used::noted(unknown_key));
        assert_eq!(resolve_wait(unknown), 0x4322);
        assert!(used::noted(unknown_key) && used::noted(known_key), "a first use is kept for the next run");
        released(known);
        released(unknown);
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// The pipeline to use for `pso` now: the real one once built, else `pso` itself.
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers
            let _ = enqueue(&p);
            pso
        }
        r => r,
    }
}

/// `resolve`, waiting for a pipeline still being built – built here if no worker has started
/// it yet (a queue of a title's hundreds of pipelines must not hold up the one it binds now).
pub fn resolve_wait(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        complete(&p, job());
    }
    let mut done = p.done.lock().unwrap();
    while !*done {
        done = p.cv.wait(done).unwrap();
    }
    match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    }
}

/// Maximum per-frame time title threads wait for uncached compute pipelines on first
/// bind/dispatch (`XCL_COMPUTE_PSO_WAIT_MS`, default 4000 ms; `0` = unbounded). Huge compute
/// shaders can take 20+ s to compile in DXC / the driver, and a loading frame can bind many
/// uncached compute pipelines in a row; sharing one budget per frame keeps the total bounded.
fn compute_wait_timeout() -> Option<std::time::Duration> {
    static LIMIT: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        let ms = std::env::var("XCL_COMPUTE_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(4000);
        if ms == 0 { None } else { Some(std::time::Duration::from_millis(ms)) }
    })
}

/// Shared per-frame compute wait budget, reset at each `PresentX` boundary (`reset_frame_budget`).
struct FrameBudget {
    spent_ms: std::sync::atomic::AtomicU64,
}

impl FrameBudget {
    const fn new() -> Self {
        Self { spent_ms: std::sync::atomic::AtomicU64::new(0) }
    }

    fn remaining(&self, limit: std::time::Duration) -> std::time::Duration {
        let limit_ms = limit.as_millis().min(u64::MAX as u128) as u64;
        let spent = self.spent_ms.load(Ordering::Relaxed);
        std::time::Duration::from_millis(limit_ms.saturating_sub(spent))
    }

    fn consume(&self, elapsed: std::time::Duration) {
        let ms = elapsed.as_millis().min(u64::MAX as u128) as u64;
        if ms > 0 {
            self.spent_ms.fetch_add(ms, Ordering::Relaxed);
        }
    }

    fn reset(&self) {
        self.spent_ms.store(0, Ordering::Relaxed);
    }
}

static FRAME_BUDGET: FrameBudget = FrameBudget::new();

/// Reset the per-frame compute pipeline wait budget (called at `PresentX`).
pub fn reset_frame_budget() {
    FRAME_BUDGET.reset();
}

/// Bounded wait for a compute pipeline: always builds on `xcl-pso-compute-*` workers (never
/// stalling the title's thread inside DXC/driver), waits at most the remaining per-frame `budget`
/// up to `limit`, and returns the stand-in immediately once timed out or once the frame budget is
/// exhausted until the worker finishes.
fn resolve_compute_within(pso: usize, limit: Option<std::time::Duration>, budget: &FrameBudget) -> usize {
    let Some(limit) = limit else {
        return resolve_wait(pso);
    };
    let Some(p) = entry(pso) else { return pso };
    if let r @ 1.. = p.real.load(Ordering::Acquire) {
        return r;
    }
    if !enqueue(&p) {
        return pso;
    }
    if p.timed_out.load(Ordering::Acquire) {
        return match p.real.load(Ordering::Acquire) {
            0 => pso,
            r => r,
        };
    }
    let rem_budget = budget.remaining(limit);
    if rem_budget.is_zero() {
        return match p.real.load(Ordering::Acquire) {
            0 => pso,
            r => r,
        };
    }
    let started = std::time::Instant::now();
    let deadline = started + rem_budget;
    let mut done = p.done.lock().unwrap();
    while !*done {
        let rem = deadline.saturating_duration_since(std::time::Instant::now());
        if rem.is_zero() {
            break;
        }
        let (g, res) = p.cv.wait_timeout(done, rem).unwrap();
        done = g;
        if res.timed_out() {
            break;
        }
    }
    let elapsed = started.elapsed();
    if !*done {
        budget.consume(elapsed.max(rem_budget));
        if rem_budget >= limit / 2 && !p.timed_out.swap(true, Ordering::AcqRel) {
            static LOGGED: AtomicUsize = AtomicUsize::new(0);
            if LOGGED.fetch_add(1, Ordering::Relaxed) < 16 {
                crate::xlog_warn!(
                    "D3D12.X: compute pipeline {pso:#x} still building after {} ms – proceeding with stand-in (continues in background)",
                    elapsed.as_millis()
                );
            }
        }
        return pso;
    }
    budget.consume(elapsed);
    match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    }
}

/// Resolve a compute pipeline with `XCL_COMPUTE_PSO_WAIT_MS` per-frame bounding.
pub fn resolve_compute(pso: usize) -> usize {
    resolve_compute_within(pso, compute_wait_timeout(), &FRAME_BUDGET)
}

/// For binding: compute pipelines wait up to `XCL_COMPUTE_PSO_WAIT_MS` on background workers;
/// graphics pipelines are queued without blocking.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_compute(pso),
        Some(_) => resolve(pso),
        None => pso,
    }
}

/// For `ExecuteIndirect`: compute pipelines use bounded waits, graphics pipelines keep `resolve_wait`.
pub fn resolve_for_indirect(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_compute(pso),
        Some(_) => resolve_wait(pso),
        None => pso,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    let done = p.done.lock().unwrap();
    if *done {
        p.real.swap(0, Ordering::AcqRel)
    } else {
        // the worker stores its result into a Pending nobody will read: mark it orphaned
        p.orphan.store(true, Ordering::Release);
        0
    }
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// `pso` is a compute stand-in whose real pipeline is still being built.
pub fn is_pending_compute(pso: usize) -> bool {
    entry(pso).is_some_and(|p| p.compute && !*p.done.lock().unwrap())
}

#[cfg(test)]
pub(crate) fn resolve_compute_with_limit_for_test(pso: usize, limit: std::time::Duration) -> usize {
    resolve_compute_within(pso, Some(limit), &FrameBudget::new())
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn bounded_compute_wait_times_out_then_resolves_once_worker_finishes() {
        let standin = 0x7300_0001usize;
        let real_pso = 0x7300_0002usize;
        let budget = FrameBudget::new();
        let (tx, rx) = mpsc::channel::<()>();
        submit_lazy(
            standin,
            true,
            Box::new(move || {
                let _ = rx.recv_timeout(Duration::from_secs(5));
                real_pso
            }),
        );

        // First call sends the lazy build to the compute worker pool and returns the stand-in after ~limit.
        let t0 = Instant::now();
        assert_eq!(resolve_compute_within(standin, Some(Duration::from_millis(60)), &budget), standin);
        assert!(t0.elapsed() < Duration::from_secs(1), "bounded wait must not block indefinitely");
        assert!(is_pending_compute(standin));

        // Subsequent calls while still building return immediately (even on a fresh frame budget).
        budget.reset();
        let t1 = Instant::now();
        assert_eq!(resolve_compute_within(standin, Some(Duration::from_secs(2)), &budget), standin);
        assert!(t1.elapsed() < Duration::from_millis(200), "post-timeout calls must return immediately");

        // Unblock the background worker; once complete, resolution immediately returns the real PSO.
        tx.send(()).unwrap();
        let wait = Instant::now();
        while resolve_compute_within(standin, Some(Duration::from_millis(50)), &budget) != real_pso {
            assert!(wait.elapsed() < Duration::from_secs(5), "background compute PSO never completed");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!is_pending_compute(standin));
        assert_eq!(released(standin), real_pso);
    }

    #[test]
    fn bounded_compute_wait_returns_real_when_build_finishes_in_time() {
        let standin = 0x7300_0003usize;
        let real_pso = 0x7300_0004usize;
        let budget = FrameBudget::new();
        let (name_tx, name_rx) = mpsc::channel::<String>();
        submit_lazy(
            standin,
            true,
            Box::new(move || {
                let _ = name_tx.send(std::thread::current().name().unwrap_or("").to_string());
                real_pso
            }),
        );
        assert_eq!(resolve_compute_within(standin, Some(Duration::from_secs(4)), &budget), real_pso);
        let worker_name = name_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(
            worker_name.starts_with("xcl-pso-compute-"),
            "compute pipelines must run on dedicated compute workers, got {worker_name:?}"
        );
        assert_eq!(released(standin), real_pso);
    }

    #[test]
    fn bounded_compute_wait_shares_per_frame_budget_across_pipelines() {
        let standin_a = 0x7300_0005usize;
        let real_a = 0x7300_0006usize;
        let standin_b = 0x7300_0007usize;
        let real_b = 0x7300_0008usize;
        let budget = FrameBudget::new();
        let (tx_a, rx_a) = mpsc::channel::<()>();
        let (tx_b, rx_b) = mpsc::channel::<()>();
        submit_lazy(
            standin_a,
            true,
            Box::new(move || {
                let _ = rx_a.recv_timeout(Duration::from_secs(5));
                real_a
            }),
        );
        submit_lazy(
            standin_b,
            true,
            Box::new(move || {
                let _ = rx_b.recv_timeout(Duration::from_secs(5));
                real_b
            }),
        );

        // First slow pipeline consumes the entire 60 ms frame budget.
        assert_eq!(resolve_compute_within(standin_a, Some(Duration::from_millis(60)), &budget), standin_a);

        // Second uncached pipeline in the same frame is queued but does NOT stall for another 60 ms.
        let t_b = Instant::now();
        assert_eq!(resolve_compute_within(standin_b, Some(Duration::from_millis(60)), &budget), standin_b);
        assert!(
            t_b.elapsed() < Duration::from_millis(20),
            "second pipeline in exhausted frame must not multiply wait time"
        );

        // Next frame refreshes the budget; once `b` finishes, waiting on it resolves `real_b`.
        budget.reset();
        tx_b.send(()).unwrap();
        assert_eq!(resolve_compute_within(standin_b, Some(Duration::from_secs(4)), &budget), real_b);

        tx_a.send(()).unwrap();
        let wait = Instant::now();
        while resolve_compute_within(standin_a, Some(Duration::from_millis(50)), &budget) != real_a {
            assert!(wait.elapsed() < Duration::from_secs(5), "background compute PSO A never completed");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(released(standin_a), real_a);
        assert_eq!(released(standin_b), real_b);
    }
}
