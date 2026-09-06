use std::sync::Arc;

use pyo3::PyResult;
use pyo3::exceptions::PyRuntimeError;
use tokio::runtime::Runtime;
use zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncBlockOn;

use crate::per_process::PerProcess;

/// The process-wide tokio runtime, rebuilt in a forked child.
///
/// Only object-store and HTTP arrays reach tokio -- a filesystem store is synchronous -- so this
/// is the remote half of surviving a `fork()`; the rayon half is [`crate::pool`]. A child
/// inherits a runtime whose worker and driver threads do not exist, and a `block_on` on it waits
/// for a readiness nothing will signal.
///
/// THIS DOES NOT MAKE A FORKED REMOTE READ WORK. A store built before the fork carries an HTTP
/// connection pool too, and a pooled socket belongs to the process that dialled it: the child
/// writes a request and the reply is delivered to the parent. Measured: a child reading an array
/// the parent had already read hangs for the full deadline with this in place. A store the child
/// opens ITSELF is the case this buys; an inherited one is refused by name in `lib.rs`.
static RUNTIME: PerProcess<Runtime> = PerProcess::new();

/// Resolves the runtime PER CALL rather than capturing a handle.
///
/// `AsyncToSyncStorageAdapter` stores this by value and a store is built once, so a captured
/// handle would outlive a fork and the pid check would never run again. Holding nothing is what
/// forces every `block_on` back through [`runtime`].
pub struct TokioBlockOn;

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        // The trait cannot report failure, so this panics where `tokio_block_on` returns an error.
        // Reachable only when a REBUILD fails; the first build already succeeded and said so.
        runtime()
            .expect("the tokio runtime could not be rebuilt")
            .block_on(future)
    }
}

/// This process's runtime, built on first use and rebuilt in a forked child.
///
/// Fallible because `Runtime::new` fails whenever a thread cannot spawn, and a modest `pids.max`
/// under Slurm or Kubernetes is enough to do it.
pub fn runtime() -> PyResult<Arc<Runtime>> {
    RUNTIME.get_or_try_init(|| {
        Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("could not create a tokio runtime: {e}")))
    })
}

/// Check at store construction that a runtime can be built, so the failure is reportable.
pub fn tokio_block_on() -> PyResult<TokioBlockOn> {
    runtime()?;
    Ok(TokioBlockOn)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One runtime per process, resolved fresh each time rather than captured. The rebuild half
    /// needs a fork; `tests/test_fork_remote_store.py` covers that.
    #[test]
    fn the_runtime_is_cached_within_one_process() {
        let first = runtime().expect("a runtime must be buildable");
        assert!(
            Arc::ptr_eq(&first, &runtime().expect("buildable")),
            "the runtime must be reused within a process, not rebuilt per call"
        );
    }
}
