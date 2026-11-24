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
    /// a bounded compute wait already timed out; subsequent binds skip waiting until `real` lands
    timed_out: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
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
    submit_as(standin, compute, job, true);
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

/// `submit`, sent to the workers only on its first `resolve` (`lazy`).
pub fn submit_lazy(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, false);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool) {
    let p = Arc::new(Pending {
        job: Mutex::new(Some(job)),
        sent: std::sync::atomic::AtomicBool::new(false),
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
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
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// The pipeline to use for `pso` now: the real one once built, else `pso` itself.
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
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
