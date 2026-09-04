use std::sync::{Arc, Mutex, PoisonError};

use pyo3::PyResult;
use pyo3::exceptions::PyRuntimeError;
use tokio::runtime::Runtime;
use zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncBlockOn;

/// The process-wide tokio runtime, and the process that built it.
///
/// KEYED ON THE PROCESS ID, and behind a `Mutex` rather than a `OnceLock`, because `fork()`
/// copies the parent's memory but only the calling thread. A child inherits a runtime whose
/// worker and driver threads do not exist, and a `block_on` on it waits for a readiness that
/// nothing will ever signal. Keying on the pid means the child builds its own instead. A
/// `OnceLock` cannot express that: it has no way back from a `static`.
///
/// WHAT THIS DOES NOT DO IS MAKE A FORK SAFE, and the reason is upstream of tokio. Opening any
/// array calls `rayon::current_num_threads` (`lib.rs`), which starts rayon's GLOBAL registry,
/// and the read path runs on it through `iter_concurrent_limit!`. That registry is behind a
/// `Once` -- no reset, the same objection this module raises against `OnceLock` -- so a child
/// inherits a registry reporting N workers and having none, and blocks in `in_worker_cold`
/// before it ever reaches a `block_on`. A live store's connection pool is inherited too.
///
/// Measured: a child reading an HTTP-backed array after the parent read it hangs, with this
/// change in place. So the claim here is only the narrow one -- a child gets a WORKING runtime
/// rather than a dead one. Making a forked child able to READ needs pools this library owns and
/// can release, which the global registry is not.
///
/// `Arc`, not the `Runtime` itself, so a caller mid-`block_on` holds a clone and a rebuild
/// cannot pull the runtime out from under it.
static RUNTIME: Mutex<Option<(u32, Arc<Runtime>)>> = Mutex::new(None);

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
    let mut guard = RUNTIME.lock().unwrap_or_else(PoisonError::into_inner);
    let pid = std::process::id();
    if guard.as_ref().is_none_or(|(built, _)| *built != pid) {
        let runtime = Runtime::new().map_err(|e| {
            PyRuntimeError::new_err(format!("could not create a tokio runtime: {e}"))
        })?;
        if let Some(stale) = guard.take() {
            // FORGOTTEN, never dropped. `Runtime::drop` blocks until every spawned task and
            // every `spawn_blocking` has finished -- tokio documents it as waiting forever --
            // and in a child those threads were never created, so it would never return.
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
