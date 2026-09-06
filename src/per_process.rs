//! A resource this process owns and a `fork()` child must not inherit.

use std::sync::{Arc, Mutex, PoisonError};

/// A value built once per process, and built again in a forked child.
///
/// `fork()` copies the parent's memory but only the calling thread, so anything that owns
/// threads -- a rayon pool, a tokio runtime -- reaches a child as a description of workers that
/// do not exist. Keying on the process id is what makes the child build its own instead. A
/// `OnceLock` is the obvious shape and cannot express this: it has no reset reachable from a
/// `static`.
///
/// CALL THIS WITH THE GIL HELD, and carry the returned `Arc` across `Python::detach`. It takes
/// a lock, and the GIL is what stops that lock from being held at the instant another thread
/// forks: a child inherits a held mutex as held, owned by a thread it does not have, and blocks
/// on it -- trading one deadlock for another. Callers take a `Python` token to say so.
pub(crate) struct PerProcess<T> {
    slot: Mutex<Option<(u32, Arc<T>)>>,
}

impl<T> PerProcess<T> {
    pub(crate) const fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    /// The value for this process, calling `build` on first use and after a fork.
    ///
    /// A failed build leaves the previous value in place: `build` runs BEFORE the stale value is
    /// taken, so a child that cannot spawn threads reports that rather than emptying the slot
    /// and reporting something different on the next call.
    pub(crate) fn get_or_try_init<E>(
        &self,
        build: impl FnOnce() -> Result<T, E>,
    ) -> Result<Arc<T>, E> {
        self.get_or_try_init_for(std::process::id(), build)
    }

    /// [`Self::get_or_try_init`] with the process id given rather than asked for, so the rebuild
    /// branch -- the whole of the fix -- is reachable from a test without forking.
    fn get_or_try_init_for<E>(
        &self,
        pid: u32,
        build: impl FnOnce() -> Result<T, E>,
    ) -> Result<Arc<T>, E> {
        let mut guard = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        if guard.as_ref().is_none_or(|(built, _)| *built != pid) {
            let fresh = Arc::new(build()?);
            // FORGOTTEN, not dropped. Dropping is what the owner does, and in a child this
            // process is not the owner: rayon's `ThreadPool::drop` terminates a registry whose
            // per-worker mutexes may be inherited held, and tokio's `Runtime::drop` waits for
            // tasks that never ran. What leaks is a copy of memory this process never owned.
            if let Some(stale) = guard.take() {
                std::mem::forget(stale);
            }
            *guard = Some((pid, fresh));
        }
        let (_, value) = guard.as_ref().expect("just built");
        Ok(value.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Caching AND the pid key, in ONE test on purpose: both drive the same slot, and cargo runs
    /// tests in parallel threads, so two tests sharing one would race.
    #[test]
    fn a_value_is_kept_per_process_and_rebuilt_when_the_process_changes() {
        let slot: PerProcess<u32> = PerProcess::new();
        let take = |pid: u32| {
            slot.get_or_try_init_for(pid, || Ok::<u32, std::convert::Infallible>(pid))
                .expect("infallible")
        };

        let first = take(1);
        assert!(
            Arc::ptr_eq(&first, &take(1)),
            "the same process must be served the same value, not a new one per call"
        );
        assert!(
            !Arc::ptr_eq(&first, &take(2)),
            "a pid change must rebuild -- without it a forked child keeps threads that do not \
             exist, which is the deadlock this type exists to prevent"
        );
    }

    /// A failed build must not cost the caller the value it already had.
    #[test]
    fn a_failed_build_keeps_what_was_there() {
        let slot: PerProcess<u32> = PerProcess::new();
        let good = slot
            .get_or_try_init_for(1, || Ok::<u32, &str>(7))
            .expect("buildable");
        assert_eq!(
            slot.get_or_try_init_for(2, || Err::<u32, &str>("no")),
            Err("no")
        );
        assert!(
            Arc::ptr_eq(
                &good,
                &slot.get_or_try_init_for(1, || Ok::<u32, &str>(9)).unwrap()
            ),
            "the slot still holds process 1's value, so it is returned rather than rebuilt"
        );
    }
}
