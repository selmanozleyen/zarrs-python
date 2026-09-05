//! The rayon pool this crate's parallel work runs on.
//!
//! Everything here exists for one reason: `fork()`. Rayon's GLOBAL pool is built behind a
//! `Once` with no reset reachable from a `static`, so a forked child inherits a registry that
//! reports N workers and has none, and the first task submitted to it parks in `in_worker_cold`
//! on a latch nothing will ever signal. That is zarrs-python issue #171 -- reported against
//! `torch.utils.data.DataLoader(num_workers>0)`, which forks by default on Python 3.13 and
//! older.
//!
//! The fix is not to make the global pool forkable, which cannot be done from here. It is to
//! stop using it: this crate owns one pool, keyed on the process that built it, and every
//! `iter_concurrent_limit!` in `lib.rs` runs inside it. The global registry is then reached
//! from nowhere, so it is never built, so a child has nothing to inherit.

use std::sync::{Arc, Mutex, PoisonError};

/// The pool, and the process id it was built in.
///
/// A `Mutex<Option<_>>` and not a `OnceLock`, because the whole point is being able to throw
/// the contents away, and a `OnceLock` has no reset reachable from a `static`.
static POOL: Mutex<Option<(u32, Arc<rayon::ThreadPool>)>> = Mutex::new(None);

/// The width of the pool, and the default for the two concurrency knobs.
///
/// `available_parallelism` and not `rayon::current_num_threads`, because READING that number
/// builds the global pool -- which put a set of threads in every process that opened an array,
/// and armed for a child the very registry this module exists to avoid.
pub(crate) fn parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

/// This process's pool, built on first use and rebuilt in a forked child.
///
/// CALL THIS WITH THE GIL HELD, and hold the result across `Python::detach`. It takes a lock,
/// and the GIL is what stops that lock from being held at the moment another thread forks: a
/// child inherits a held mutex as held, owned by a thread it does not have, and then blocks on
/// it -- trading one deadlock for another.
pub(crate) fn pool() -> Arc<rayon::ThreadPool> {
    let mut guard = POOL.lock().unwrap_or_else(PoisonError::into_inner);
    let pid = std::process::id();
    if guard.as_ref().is_none_or(|(built, _)| *built != pid) {
        // FORGOTTEN, not dropped. `ThreadPool::drop` joins its workers, and in a child those
        // workers were never created, so dropping is the same hang by another route. What
        // leaks is a copy of memory this process never owned.
        if let Some(stale) = guard.take() {
            std::mem::forget(stale);
        }
        let built = rayon::ThreadPoolBuilder::new()
            .num_threads(parallelism())
            .thread_name(|i| format!("zarrs-{i}"))
            .build()
            .expect("a thread pool of a positive size");
        *guard = Some((pid, Arc::new(built)));
    }
    let (_, pool) = guard.as_ref().expect("just built");
    pool.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One pool per process, not one per call.
    ///
    /// The rebuild half needs the pid to change, which needs a fork, and a forked child cannot
    /// report back to the harness. `tests/test_fork_deadlock.py` covers that half.
    #[test]
    fn the_pool_is_reused_within_one_process() {
        assert!(Arc::ptr_eq(&pool(), &pool()), "the pool must not be rebuilt per call");
    }

    #[test]
    fn the_pool_is_as_wide_as_the_machine() {
        assert_eq!(pool().current_num_threads(), parallelism());
    }
}
