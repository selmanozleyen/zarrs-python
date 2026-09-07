//! The rayon pool this crate runs on, keyed on the process so a `fork()` child rebuilds it.
//!
//! Rayon's global pool has no reset, so a forked child inherits a registry with no live workers
//! and parks for ever (#171). Nothing here submits to it. It is still built, by
//! `CodecOptions::default` and zarrs' `Config`, which this module cannot prevent.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once, PoisonError};

use pyo3::exceptions::PyRuntimeError;
use pyo3::{PyResult, Python};

/// Bumped by a `pthread_atfork` child handler, which is how POSIX says to notice a fork.
///
/// A pid would be cheaper to read, but pids are not unique: they are recycled, so a fork chain
/// can hand a grandchild the pid its grandparent used and a stale value would look current.
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// Runs in the child, immediately after `fork`. An atomic increment is async-signal-safe, which
/// is the only thing a child of a threaded parent may portably do.
extern "C" fn note_fork() {
    GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// The current fork generation, registering the handler on first use.
///
/// Registration always precedes any stored value, because storing one calls this first, so a
/// fork can never slip past unnoticed.
fn generation() -> u64 {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        #[cfg(unix)]
        // SAFETY: `note_fork` is `extern "C"`, takes nothing, and only touches an atomic.
        unsafe {
            libc::pthread_atfork(None, None, Some(note_fork));
        }
    });
    GENERATION.load(Ordering::Relaxed)
}

/// A value built once per process, and built again in a forked child.
///
/// `fork()` copies memory but only the calling thread, so anything owning threads reaches a child
/// as workers that do not exist; a `OnceLock` has no reset to express the rebuild. Call with the
/// GIL held: a lock held when another thread forks is inherited locked by a vanished thread.
pub(crate) struct PerProcess<T> {
    slot: Mutex<Option<(u64, Arc<T>)>>,
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
        self.get_or_try_init_for(generation(), build)
    }

    /// As above, with the generation given so a test reaches the rebuild without forking.
    fn get_or_try_init_for<E>(
        &self,
        generation: u64,
        build: impl FnOnce() -> Result<T, E>,
    ) -> Result<Arc<T>, E> {
        let mut guard = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        if guard.as_ref().is_none_or(|(built, _)| *built != generation) {
            // Built before the stale one is taken, so a failed build keeps what was there.
            let fresh = Arc::new(build()?);
            // Forgotten, not dropped: dropping joins or waits on threads a child never had.
            if let Some(stale) = guard.take() {
                std::mem::forget(stale);
            }
            *guard = Some((generation, fresh));
        }
        let (_, value) = guard.as_ref().expect("just built");
        Ok(value.clone())
    }
}

static POOL: PerProcess<rayon::ThreadPool> = PerProcess::new();

/// This process's pool. Takes a `Python` token because it locks, and the GIL is what keeps that
/// lock unheld when another thread forks; hold the `Arc` across `Python::detach`.
pub(crate) fn pool(_py: Python<'_>) -> PyResult<Arc<rayon::ThreadPool>> {
    POOL.get_or_try_init(|| {
        // `num_threads` unset, so rayon resolves it as it sized the global pool this replaces.
        rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("zarrs-{i}"))
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("could not create a thread pool: {e}")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The keying is tested in [`PerProcess`]; this pins that a pool is built at all.
    #[test]
    fn the_pool_has_workers() {
        Python::initialize();
        Python::attach(|py| {
            assert!(pool(py).expect("buildable").current_num_threads() >= 1);
        });
    }

    /// Caching and the generation key in one test: both drive the same slot, and cargo runs
    /// tests in parallel threads.
    #[test]
    fn a_value_is_kept_per_process_and_rebuilt_when_the_process_changes() {
        let slot: PerProcess<u32> = PerProcess::new();
        let take = |era: u64| {
            slot.get_or_try_init_for(era, || Ok::<u32, std::convert::Infallible>(era as u32))
                .expect("infallible")
        };

        let first = take(1);
        assert!(
            Arc::ptr_eq(&first, &take(1)),
            "one value per process, not one per call"
        );
        assert!(
            !Arc::ptr_eq(&first, &take(2)),
            "a generation change must rebuild, or a forked child keeps threads that do not exist"
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
