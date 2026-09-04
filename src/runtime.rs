use std::sync::{Arc, Mutex, PoisonError};

use pyo3::PyResult;
use pyo3::exceptions::PyRuntimeError;
use tokio::runtime::Runtime;
use zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncBlockOn;

/// The process-wide tokio runtime, and the process that built it.
///
/// KEYED ON THE PROCESS ID, and behind a `Mutex` rather than a `OnceLock`, because `fork()`
/// copies the parent's memory but only the calling thread. A child inherits a runtime whose
/// worker and driver threads do not exist, and its first `block_on` waits for an I/O readiness
/// that nothing will ever signal -- a permanent hang, not a slowdown.
///
/// It only bites when the parent used the runtime BEFORE forking, so it is data-dependent and
/// invisible to any test that does not fork. The workload where that happens is the ordinary
/// one: read a few batches from an object-store or HTTP backed array, then hand the array to
/// `torch.utils.data.DataLoader(num_workers > 0)`, which forks by default on Linux.
///
/// A `OnceLock` cannot express the reset -- it has no way back from a `static`.
///
/// `Arc`, not the `Runtime` itself: a caller mid-`block_on` holds a clone, so emptying this
/// slot cannot pull a runtime out from under a read that is using it. Dropping the last clone
/// is what shuts it down, and that is the caller's own reference going away, not this one.
static RUNTIME: Mutex<Option<(u32, Arc<Runtime>)>> = Mutex::new(None);

/// Resolves the runtime PER CALL rather than capturing a handle.
///
/// This is the whole mechanism. `AsyncToSyncStorageAdapter` stores its `AsyncToSyncBlockOn` by
/// value and a store is built once, so anything captured here would outlive a fork and the pid
/// check would never run again -- the child would go straight to the dead runtime it inherited.
/// Holding nothing forces every `block_on` back through [`runtime`].
pub struct TokioBlockOn;

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        // The trait cannot report failure, so this panics where `tokio_block_on` returns an
        // error. Reachable only when a REBUILD fails -- the first build already succeeded at
        // store construction, and said so properly.
        runtime()
            .expect("the tokio runtime could not be rebuilt")
            .block_on(future)
    }
}

/// This process's runtime, building it on first use and rebuilding it in a forked child.
///
/// FALLIBLE, because `Runtime::new` fails whenever a thread cannot be spawned, and a modest
/// `pids.max` under Slurm or Kubernetes is enough to do it. `expect` here would be a
/// `PanicException` naming a Rust internal, where the caller could be told which store they
/// cannot open.
pub fn runtime() -> PyResult<Arc<Runtime>> {
    let mut guard = RUNTIME.lock().unwrap_or_else(PoisonError::into_inner);
    let pid = std::process::id();
    if guard.as_ref().is_none_or(|(built, _)| *built != pid) {
        let runtime = Runtime::new().map_err(|e| {
            PyRuntimeError::new_err(format!("could not create a tokio runtime: {e}"))
        })?;
        if let Some(stale) = guard.take() {
            // FORGOTTEN, never dropped: `Runtime::drop` joins its workers, and in a child those
            // workers were never created, so dropping is the same hang by another route. Only
            // reachable for a fork that bypasses Python's handlers -- `release_for_fork` empties
            // the slot first for every fork that does not.
            std::mem::forget(stale);
        }
        *guard = Some((pid, Arc::new(runtime)));
    }
    let (_, runtime) = guard.as_ref().expect("just built");
    Ok(runtime.clone())
}

/// Check at store construction that a runtime can be built, so the failure is reportable.
pub fn tokio_block_on() -> PyResult<TokioBlockOn> {
    runtime()?;
    Ok(TokioBlockOn)
}

/// Empty the slot so a `fork()` cannot inherit threads the child will not have.
///
/// The pid check in [`runtime`] rebuilds in a child, but it runs INSIDE this lock, and `fork`
/// copies a held mutex as held with an owner thread the child does not have. Emptying before
/// the fork means the child takes a free lock and rebuilds.
///
/// NARROWS THE WINDOW, does not close it. Reads run with the GIL released, so another thread
/// can re-enter [`runtime`] and re-take this lock between the handler returning and the syscall.
/// Only a single-threaded forker is airtight; this covers the case that actually happens, which
/// is a loader forking from the thread that has been reading.
pub fn release_for_fork() {
    *RUNTIME.lock().unwrap_or_else(PoisonError::into_inner) = None;
}
