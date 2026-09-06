use std::sync::Arc;

use pyo3::PyResult;
use pyo3::exceptions::PyRuntimeError;
use tokio::runtime::Runtime;
use zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncBlockOn;

use crate::per_process::PerProcess;

/// The process-wide tokio runtime, rebuilt in a forked child.
///
/// Only object-store and HTTP arrays reach tokio at all -- a filesystem store is synchronous --
/// so this is the remote half of surviving a `fork()`. The rayon half is `crate::pool`.
///
/// A child inherits a runtime whose worker and driver threads do not exist, and a `block_on` on
/// it waits for a readiness nothing will ever signal. [`PerProcess`] keys it on the process that
/// built it, so the child builds its own.
///
/// THIS DOES NOT MAKE A FORKED REMOTE READ WORK, and cannot. A store built before the fork also
/// carries an HTTP connection pool, and a pooled socket belongs to the process that dialled it:
/// the child writes a request and the reply is delivered to the parent. Rebuilding the runtime
/// does not re-dial. Measured: a child reading an HTTP-backed array the parent had already read
/// hangs for the full deadline with this in place. A store the child opens ITSELF is fine, which
/// is the case this buys; an inherited one is refused by name in `lib.rs` rather than blocked on.
///
/// `Arc`, not the `Runtime` itself, so a caller mid-`block_on` holds a clone and a rebuild
/// cannot pull the runtime out from under it.
static RUNTIME: PerProcess<Runtime> = PerProcess::new();

/// Resolves the runtime PER CALL rather than capturing a handle.
///
/// This is the whole mechanism. `AsyncToSyncStorageAdapter` stores its `AsyncToSyncBlockOn` by
/// value and a store is built once, so a handle captured here would outlive a fork and the pid
/// check would never run again. Holding nothing forces every `block_on` back through
/// [`runtime`].
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

    /// One runtime per process, resolved fresh each time rather than captured.
    ///
    /// The rebuild half cannot be reached from a unit test: it needs the pid to change, which
    /// needs a fork, and a forked child cannot report a failure back to the harness. What is
    /// testable is that the steady state does not build a runtime per call.
    #[test]
    fn the_runtime_is_cached_within_one_process() {
        let first = runtime().expect("a runtime must be buildable");
        let second = runtime().expect("a runtime must be buildable");
        assert!(
            Arc::ptr_eq(&first, &second),
            "the runtime must be reused within a process, not rebuilt per call"
        );
    }
}
