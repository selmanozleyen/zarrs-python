//! A resource this process owns and a `fork()` child must not inherit.

use std::sync::{Arc, Mutex, PoisonError};

/// A value built once per process, and built again in a forked child.
///
/// `fork()` copies memory but only the calling thread, so anything owning threads reaches a child
/// as workers that do not exist; a `OnceLock` has no reset to express the rebuild. Call with the
/// GIL held: a lock held when another thread forks is inherited locked by a vanished thread.
pub(crate) struct PerProcess<T> {
    slot: Mutex<Option<(u32, Arc<T>)>>,
}

impl<T> PerProcess<T> {
    pub(crate) const fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    /// The value for this process, built on first use and after a fork.
    pub(crate) fn get_or_try_init<E>(
        &self,
        build: impl FnOnce() -> Result<T, E>,
    ) -> Result<Arc<T>, E> {
        self.get_or_try_init_for(std::process::id(), build)
    }

    /// As above, with the pid given so a test reaches the rebuild without forking.
    fn get_or_try_init_for<E>(
        &self,
        pid: u32,
        build: impl FnOnce() -> Result<T, E>,
    ) -> Result<Arc<T>, E> {
        let mut guard = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        if guard.as_ref().is_none_or(|(built, _)| *built != pid) {
            // Built before the stale one is taken, so a failed build keeps what was there.
            let fresh = Arc::new(build()?);
            // Forgotten, not dropped: dropping joins or waits on threads a child never had.
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

    /// Caching and the pid key in one test: both drive the same slot, and cargo runs tests
    /// in parallel threads.
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
            "one value per process, not one per call"
        );
        assert!(
            !Arc::ptr_eq(&first, &take(2)),
            "a pid change must rebuild, or a forked child keeps threads that do not exist"
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
            "process 1's value is still there, so it is returned rather than rebuilt"
        );
    }
}
