//! Physical pages mapped into the title's own reservations, committed on first use.
//!
//! A title maps its physical pages wherever it reserved address space and places resources
//! there – render targets and depth buffers too, which on the PC live in video memory
//! (`d3d12x::vram`); the CPU never touches those pages. Committed at map time they doubled
//! that memory (a retail game: 5.4 GB of mapped pages next to its video-memory targets, past
//! the system's commit). So a page whose contents are still zero is mapped as reserved address
//! space only: the CPU's first access faults, the page is committed (zeros, as the page holds)
//! and the access goes on. Kernel calls do not fault in the title – they fail – so the buffers
//! of the file calls XCL routes, and DirectStorage's destinations, are committed first
//! (`ensure`).

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Once, RwLock};

use windows_sys::Win32::Foundation::{
    GetLastError, ERROR_COMMITMENT_LIMIT, ERROR_NOT_ENOUGH_MEMORY, ERROR_OUTOFMEMORY, EXCEPTION_ACCESS_VIOLATION, NTSTATUS,
};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
};

const PAGE: usize = 0x1_0000;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

/// pages (64 KiB) mapped but not committed yet
static LAZY: RwLock<BTreeSet<usize>> = RwLock::new(BTreeSet::new());

static FAILED_COMMITS: AtomicUsize = AtomicUsize::new(0);
static LAST_FAILED_PAGE: AtomicUsize = AtomicUsize::new(0);
static LAST_FAILED_ERR: AtomicU32 = AtomicU32::new(0);

fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
    });
}

/// Commit `page` if it is a lazy one (true: it was and is now committed).
unsafe fn commit(page: usize) -> bool {
    let mut g = LAZY.write().unwrap_or_else(|e| e.into_inner());
    if !g.contains(&page) {
        return false;
    }
    if VirtualAlloc(page as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null() {
        LAST_FAILED_PAGE.store(page, Ordering::Relaxed);
        LAST_FAILED_ERR.store(GetLastError(), Ordering::Relaxed);
        FAILED_COMMITS.fetch_add(1, Ordering::Release);
        return false;
    }
    g.remove(&page);
    true
}

/// Summary of failed lazy commits for crash reporting, or `None` if none failed.
pub(crate) fn failure_summary() -> Option<String> {
    let n = FAILED_COMMITS.load(Ordering::Acquire);
    if n == 0 {
        return None;
    }
    let page = LAST_FAILED_PAGE.load(Ordering::Relaxed);
    let err = LAST_FAILED_ERR.load(Ordering::Relaxed);
    let name = match err {
        ERROR_COMMITMENT_LIMIT => " ERROR_COMMITMENT_LIMIT",
        ERROR_NOT_ENOUGH_MEMORY => " ERROR_NOT_ENOUGH_MEMORY",
        ERROR_OUTOFMEMORY => " ERROR_OUTOFMEMORY",
        _ => "",
    };
    Some(format!("lazy commit failed {n} times, last {page:#x} ({err}{name})"))
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let r = &*(*info).ExceptionRecord;
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let page = r.ExceptionInformation[1] & !(PAGE - 1);
    let lazy = LAZY.read().unwrap_or_else(|e| e.into_inner()).contains(&page);
    if lazy && commit(page) || allows(r.ExceptionInformation[0], r.ExceptionInformation[1]) || wild_commit(page) {
        EXCEPTION_CONTINUE_EXECUTION
    } else {
        EXCEPTION_CONTINUE_SEARCH
    }
}

unsafe fn allows(access: usize, at: usize) -> bool {
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT || m.Protect & PAGE_GUARD != 0 {
        return false;
    }
    let writes = PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    let reads = writes | PAGE_READONLY | PAGE_EXECUTE_READ;
    match access {
        0 => m.Protect & reads != 0,
        1 => m.Protect & writes != 0,
        _ => false,
    }
}

/// `XCL_SPARSE_WILD_COMMIT=1`: the title touched a page of a sparse window that no reservation
/// holds (GTA VI's streaming memcpy, 0x6ac..0x6ff above its last block): the page is committed
/// there and the access goes on (an experiment – where does the title get such addresses?).
unsafe fn wild_commit(page: usize) -> bool {
    static N: AtomicUsize = AtomicUsize::new(0);
    if !wild_enabled() {
        return false;
    }
    let Some(u) = crate::gpu::uma::get() else { return false };
    if !u.is_sparse(page) || u.sparse_reservation(page).is_some() {
        return false;
    }
    if VirtualAlloc(page as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null() {
        return false;
    }
    crate::hle::d3d12x::sparse::committed(page, PAGE);
    let k = N.fetch_add(1, Ordering::Relaxed) + 1;
    if k <= 32 || k.is_power_of_two() {
        crate::xlog_warn!("sparse window: page {page:#x} outside every reservation committed on its first access (#{k})");
    }
    true
}

fn wild_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SPARSE_WILD_COMMIT").is_ok_and(|v| v.trim() == "1"))
}

/// Under `XCL_SPARSE_WILD_COMMIT=1` the handler is there from the start.
pub(crate) fn install_if_wild() {
    if wild_enabled() {
        install();
    }
}

/// `page` (reserved, 64 KiB aligned) is mapped: committed on first use.
pub(crate) fn mark(page: usize) {
    install();
    LAZY.write().unwrap_or_else(|e| e.into_inner()).insert(page);
}

/// Whether `page` is mapped and was never used.
pub(super) fn marked(page: usize) -> bool {
    LAZY.read().unwrap_or_else(|e| e.into_inner()).contains(&page)
}

/// `page` is unmapped: true if it was never used (nothing to keep of it).
pub(super) fn unmark(page: usize) -> bool {
    LAZY.write().unwrap_or_else(|e| e.into_inner()).remove(&page)
}

/// The lazy pages of `[p, p + len)` committed – memory a kernel call or XCL writes.
pub fn ensure(p: usize, len: usize) {
    if len == 0 {
        return;
    }
    let pages: Vec<usize> = {
        let g = LAZY.read().unwrap_or_else(|e| e.into_inner());
        if g.is_empty() {
            return;
        }
        g.range(p & !(PAGE - 1)..p.saturating_add(len)).copied().collect()
    };
    for page in pages {
        unsafe { commit(page) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE, MEM_RESERVE};

    /// A lazy page reads as zeros on first touch, keeps what is written; `ensure` commits it
    /// without a fault.
    #[test]
    fn a_lazy_page_is_committed_on_first_use() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            mark(r);
            mark(r + PAGE);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 0);
            std::ptr::write_volatile((r + 8) as *mut u32, 7);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 7);
            assert!(!unmark(r), "used: no longer lazy");
            ensure(r + PAGE + 4, 4);
            assert!(!LAZY.read().unwrap().contains(&(r + PAGE)));
            assert!(crate::gpu::shader::readable(r + PAGE, 4));
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn threads_first_touching_one_lazy_page_together_all_go_on() {
        const THREADS: usize = 8;
        const PAGES: usize = 256;
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            for i in 0..PAGES {
                mark(r + i * PAGE);
            }
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
            let handles: Vec<_> = (0..THREADS)
                .map(|t| {
                    let barrier = barrier.clone();
                    std::thread::spawn(move || {
                        for i in 0..PAGES {
                            barrier.wait();
                            std::ptr::write_volatile((r + i * PAGE + t * 64) as *mut u64, (i * THREADS + t) as u64);
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            for i in 0..PAGES {
                for t in 0..THREADS {
                    assert_eq!(std::ptr::read_volatile((r + i * PAGE + t * 64) as *const u64), (i * THREADS + t) as u64);
                }
            }
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn failed_lazy_commit_stays_lazy_and_reports_in_crash_line() {
        if std::env::var_os("XCL_LAZY_LIMIT_CHILD").is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::kernelx::lazy::tests::failed_lazy_commit_stays_lazy_and_reports_in_crash_line", "--nocapture"])
                .env("XCL_LAZY_LIMIT_CHILD", "1")
                .output()
                .unwrap();
            assert!(!out.status.success(), "expected child to crash on commit limit");
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(stderr.contains("0xc0000005"), "missing 0xc0000005 in stderr: {stderr}");
            assert!(stderr.contains("lazy commit failed"), "missing lazy commit summary in stderr: {stderr}");
            assert!(stderr.contains("1455 ERROR_COMMITMENT_LIMIT"), "missing ERROR_COMMITMENT_LIMIT in stderr: {stderr}");
            return;
        }
        unsafe {
            use windows_sys::Win32::System::Diagnostics::Debug::{SetErrorMode, SEM_FAILCRITICALERRORS, SEM_NOGPFAULTERRORBOX};
            use windows_sys::Win32::System::JobObjects::{
                AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
            };
            use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
            use windows_sys::Win32::System::Threading::GetCurrentProcess;

            SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
            crate::eh::install_crash_handler();

            // Pre-commit stack pages before applying the job commit limit so the crash handler
            // itself does not hit a stack-guard commit failure.
            let mut stack_warm = [0u8; 64 * 1024];
            std::hint::black_box(&mut stack_warm);

            let mut pmc: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
            assert_ne!(GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, std::mem::size_of_val(&pmc) as u32), 0);

            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            assert!(!job.is_null());
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            info.ProcessMemoryLimit = pmc.PagefileUsage + 4 * PAGE;
            assert_ne!(SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const _, std::mem::size_of_val(&info) as u32), 0);
            assert_ne!(AssignProcessToJobObject(job, GetCurrentProcess()), 0);

            const PAGES: usize = 64;
            let r = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(r, 0);
            for i in 0..PAGES {
                mark(r + i * PAGE);
            }

            // A page under the limit still commits on first touch.
            std::ptr::write_volatile(r as *mut u32, 42);
            assert_eq!(std::ptr::read_volatile(r as *const u32), 42);
            assert!(!LAZY.read().unwrap().contains(&r));

            // Push past the job's commit limit: the failing page must stay in LAZY and record the error.
            for i in 1..PAGES {
                let page = r + i * PAGE;
                ensure(page, PAGE);
                if LAZY.read().unwrap().contains(&page) {
                    assert_eq!(FAILED_COMMITS.load(Ordering::Acquire), 1);
                    assert_eq!(LAST_FAILED_ERR.load(Ordering::Relaxed), ERROR_COMMITMENT_LIMIT);
                    std::ptr::write_volatile(page as *mut u32, 99);
                    unreachable!("touching an uncommitted page past the limit must fault");
                }
            }
            panic!("expected lazy commit to hit process memory limit");
        }
    }
}
