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
//! stop SUBMITTING to it: this crate owns one pool, keyed on the process that built it by
//! [`PerProcess`], and every `iter_concurrent_limit!` in `lib.rs` runs inside it.
//!
//! THE GLOBAL REGISTRY IS STILL BUILT, and this module cannot prevent it: `CodecOptions` and
//! `zarrs`'s own `Config` both call `rayon::current_num_threads()` in their `Default` impls,
//! and neither has a constructor that does not. So a process that opens an array still spawns a
//! global pool it never uses, and a child still inherits it dead. What matters is that nothing
//! here ever puts work into it -- an inherited idle registry is only memory. The invariant that
//! keeps it that way is "every rayon entry point in this crate is inside `pool().install(..)`",
//! and it is not enforced by anything but review.

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::{PyResult, Python};

use crate::per_process::PerProcess;

static POOL: PerProcess<rayon::ThreadPool> = PerProcess::new();

/// This process's pool, built on first use and rebuilt in a forked child.
///
/// TAKES A `Python` TOKEN so that the GIL invariant is the compiler's business and not the
/// reader's: this locks a mutex, and the GIL is what stops that lock from being held at the
/// instant another thread calls `fork()`. Hold the returned `Arc` across `Python::detach`; do
/// not call this from inside one.
///
/// (The token is necessary and not sufficient: on a future free-threaded build there is no GIL
/// to serialise against, and this argument would have to be replaced rather than adjusted.)
pub(crate) fn pool(_py: Python<'_>) -> PyResult<Arc<rayon::ThreadPool>> {
    POOL.get_or_try_init(|| {
        // NO `num_threads`, deliberately. Left unset, rayon resolves the width itself --
        // `RAYON_NUM_THREADS`, then `RAYON_RS_NUM_CPUS`, then the machine's parallelism -- which
        // is what sized the global pool this replaces. Setting it here would give the same
        // number on a bare machine and silently ignore the environment on a shared one.
        rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("zarrs-{i}"))
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("could not create a thread pool: {e}")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The keying itself lives in [`PerProcess`] and is tested there. What is worth pinning
    /// here is that a pool actually gets built, and that its width is rayon's to decide.
    #[test]
    fn the_pool_has_workers() {
        Python::initialize();
        Python::attach(|py| {
            assert!(pool(py).expect("buildable").current_num_threads() >= 1);
        });
    }
}
