use std::sync::{Mutex, PoisonError};

use pyo3::PyResult;
use pyo3::exceptions::PyRuntimeError;
use tokio::runtime::Runtime;
use zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncBlockOn;

/// The process-wide tokio runtime, and the process that built it.
///
/// KEYED ON THE PROCESS ID, and behind a `Mutex` rather than a `OnceLock`, for the same reason
/// the two rayon pools are: `fork()` copies the parent's memory but only the calling thread, so
/// a child inherits a runtime whose worker and driver threads do not exist. The first
/// `block_on` in that child waits for an I/O readiness that nothing will ever signal -- a
/// permanent hang, not a slowdown, and data-dependent on the parent having read first.
///
/// A `OnceLock` cannot express this: it has no reset reachable from a `static`. That is exactly
/// why `POOLS` in `read_decode.rs` is shaped this way, and this one was left behind -- so
/// `zarrs.release_pools_for_fork` covered an object-store read's pools and not the runtime
/// underneath it, which reads as "forking is handled" while half of it is not.
///
/// The stale runtime is FORGOTTEN, never dropped: `Runtime::drop` joins its workers, and in a
/// child those workers were never created, so dropping is the same hang by another route.
static RUNTIME: Mutex<Option<(u32, Runtime)>> = Mutex::new(None);

pub struct TokioBlockOn(tokio::runtime::Handle);

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

/// A handle to this process's runtime, building it on first use.
///
/// FALLIBLE, because `Runtime::new` fails whenever a thread cannot be spawned -- a modest
/// `pids.max` under Slurm or Kubernetes is enough. It used to `expect`, which turned that into
/// a panic across the pyo3 boundary while opening an array: a `PanicException` naming a Rust
/// internal, where the caller could have been told which store they cannot open.
pub fn tokio_block_on() -> PyResult<TokioBlockOn> {
    let mut guard = RUNTIME.lock().unwrap_or_else(PoisonError::into_inner);
    let pid = std::process::id();
    if guard.as_ref().is_none_or(|(built, _)| *built != pid) {
        let runtime = Runtime::new().map_err(|e| {
            PyRuntimeError::new_err(format!("could not create a tokio runtime: {e}"))
        })?;
        if let Some(stale) = guard.take() {
            // See `RUNTIME`: forget, never drop.
            std::mem::forget(stale);
        }
        *guard = Some((pid, runtime));
    }
    let (_, runtime) = guard.as_ref().expect("just built");
    Ok(TokioBlockOn(runtime.handle().clone()))
}

/// Drop the runtime so a `fork()` cannot inherit a held lock, or threads that do not exist.
///
/// The mirror of `read_decode::release_pools_for_fork`, and registered from the same place. The
/// pid check above rebuilds in a child, but it runs INSIDE this lock, and `fork` copies a held
/// mutex as held with an owner that does not exist -- so a child forked while any thread was
/// inside `tokio_block_on` would block there for ever and never reach the check.
pub fn release_runtime_for_fork() {
    let mut guard = RUNTIME.lock().unwrap_or_else(PoisonError::into_inner);
    // This is the PARENT: its threads exist, so dropping them is correct and is what frees them.
    *guard = None;
}
