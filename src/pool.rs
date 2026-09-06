//! The rayon pool this crate's parallel work runs on.
//!
//! Everything here exists for one reason: `fork()`. Rayon's GLOBAL pool is built behind a
//! `Once` with no reset reachable from a `static`, so a forked child inherits a registry that
//! reports N workers and has none, and the first task submitted to it parks in `in_worker_cold`
//! on a latch nothing will ever signal. That is zarrs-python issue #171 -- reported against
//! `torch.utils.data.DataLoader(num_workers>0)`, which forks by default on Python 3.13 and
//! older, on Linux.
//!
//! The fix is not to make the global pool forkable, which cannot be done from here. It is to
//! stop SUBMITTING to it: this crate owns one pool, keyed on the process that built it, and
//! every `iter_concurrent_limit!` in `lib.rs` runs inside it.
//!
//! THE GLOBAL REGISTRY IS STILL BUILT, and this module cannot prevent it: `CodecOptions`
//! and `zarrs`'s own `Config` both call `rayon::current_num_threads()` in their `Default`
//! impls, and neither has a constructor that does not. So a process that opens an array still
//! spawns a global pool it never uses, and a child still inherits it dead. What matters is
//! that nothing here ever puts work into it -- an inherited idle registry is only memory. The
//! invariant that keeps it that way is "every rayon entry point in this crate is inside
//! `pool().install(..)`", and it is not enforced by anything but review.

use std::sync::{Arc, Mutex, PoisonError};

use pyo3::exceptions::PyRuntimeError;
use pyo3::{PyResult, Python};

/// The pool, and the process id it was built in.
///
/// A `Mutex<Option<_>>` and not a `OnceLock`, because the whole point is being able to throw
/// the contents away, and a `OnceLock` has no reset reachable from a `static`.
static POOL: Mutex<Option<(u32, Arc<rayon::ThreadPool>)>> = Mutex::new(None);

/// This process's pool, built on first use and rebuilt in a forked child.
///
/// TAKES A `Python` TOKEN so that the GIL invariant is the compiler's business and not the
/// reader's: this locks a mutex, and the GIL is what stops that lock from being held at the
/// instant another thread calls `fork()`. A child inherits a held mutex as held, owned by a
/// thread it does not have, and blocks on it -- the same deadlock by a shorter route. Hold the
/// returned `Arc` across `Python::detach`; do not call this from inside one.
///
/// (The token is necessary and not sufficient: on a future free-threaded build there is no GIL
/// to serialise against, and this argument would have to be replaced rather than adjusted.)
pub(crate) fn pool(_py: Python<'_>) -> PyResult<Arc<rayon::ThreadPool>> {
    pool_for(std::process::id())
}

/// [`pool`], with the process id given rather than asked for, so a test can reach the rebuild
/// branch without forking.
fn pool_for(pid: u32) -> PyResult<Arc<rayon::ThreadPool>> {
    let mut guard = POOL.lock().unwrap_or_else(PoisonError::into_inner);
    if guard.as_ref().is_none_or(|(built, _)| *built != pid) {
        // NO `num_threads`, deliberately. Left unset, rayon resolves the width itself --
        // `RAYON_NUM_THREADS`, then `RAYON_RS_NUM_CPUS`, then the machine's parallelism -- which
        // is what sized the global pool this replaces. Setting it here would give the same
        // number on a bare machine and silently ignore the environment on a shared one.
        let built = rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("zarrs-{i}"))
            .build()
            .map_err(|e| {
                PyRuntimeError::new_err(format!("could not create a thread pool: {e}"))
            })?;
        // FORGOTTEN, not dropped. `ThreadPool::drop` terminates the registry, and terminating
        // takes a per-worker mutex that a child inherits locked by a worker that does not
        // exist. What leaks is a copy of memory this process never owned.
        if let Some(stale) = guard.take() {
            std::mem::forget(stale);
        }
        *guard = Some((pid, Arc::new(built)));
    }
    let (_, pool) = guard.as_ref().expect("just built");
    Ok(pool.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Caching AND the pid key, in ONE test on purpose: both drive the same `static`, and cargo
    /// runs tests in parallel threads, so two tests touching it would race each other.
    #[test]
    fn the_pool_is_kept_per_process_and_rebuilt_when_the_process_changes() {
        let first = pool_for(1).expect("a pool must be buildable");
        assert!(
            Arc::ptr_eq(&first, &pool_for(1).expect("buildable")),
            "the same process must be served the same pool, not a new one per call"
        );

        let other = pool_for(2).expect("a pool must be buildable");
        assert!(
            !Arc::ptr_eq(&first, &other),
            "a pid change must rebuild -- this is the whole fix, and without it a forked \
             child keeps a pool whose threads do not exist"
        );

        assert!(
            !Arc::ptr_eq(&other, &pool_for(1).expect("buildable")),
            "the slot holds one process's pool, so going back rebuilds again"
        );
    }

    /// The width is rayon's to decide, and deliberately not ours: unset `num_threads` is what
    /// lets `RAYON_NUM_THREADS` still reach the pool.
    #[test]
    fn the_pool_has_workers() {
        assert!(pool_for(7).expect("buildable").current_num_threads() >= 1);
    }
}
