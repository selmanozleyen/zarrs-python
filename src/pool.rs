//! The rayon pool this crate's parallel work runs on.
//!
//! Rayon's GLOBAL pool sits behind a `Once` with no reset, so a `fork()`ed child inherits a
//! registry reporting workers it does not have and parks for ever -- issue #171, against
//! `DataLoader(num_workers>0)`. Nothing here submits to it. It is still BUILT and this module
//! cannot prevent it -- `CodecOptions` and zarrs' `Config` read `current_num_threads()` in their
//! `Default` impls -- but an inherited IDLE registry is only memory. The invariant that matters
//! is that every rayon entry point in this crate sits inside `pool().install(..)`.

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::{PyResult, Python};

use crate::per_process::PerProcess;

static POOL: PerProcess<rayon::ThreadPool> = PerProcess::new();

/// This process's pool, built on first use and rebuilt in a forked child.
///
/// Takes a `Python` token so the GIL rule is the compiler's business: this locks, and the GIL is
/// what stops that lock being held when another thread forks. Hold the `Arc` across
/// `Python::detach`, never call this inside one. (Necessary, not sufficient: a free-threaded
/// build has no GIL to serialise against and would need a different argument entirely.)
pub(crate) fn pool(_py: Python<'_>, width: Option<usize>) -> PyResult<Arc<rayon::ThreadPool>> {
    POOL.get_or_try_init(|| {
        let mut builder = rayon::ThreadPoolBuilder::new().thread_name(|i| format!("zarrs-{i}"));
        // Unset means rayon resolves it: `RAYON_NUM_THREADS`, then `RAYON_RS_NUM_CPUS`, then the
        // machine -- the ladder that sized the pool this replaces. A caller that knows the width
        // says so instead, and the FIRST such caller fixes it for the process.
        if let Some(width) = width {
            builder = builder.num_threads(width.max(1));
        }
        builder
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("could not create a thread pool: {e}")))
    })
}

/// What this process already built, or `None` if it has not built it yet.
///
/// Reporting only, so it must not build: the thing it reports on is whether building happened.
pub(crate) fn peek() -> Option<Arc<rayon::ThreadPool>> {
    POOL.peek()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The keying is tested in [`PerProcess`]; what is worth pinning here is that a pool is built
    /// and that its width is rayon's to decide.
    #[test]
    fn the_pool_has_workers() {
        Python::initialize();
        Python::attach(|py| {
            assert!(pool(py, None).expect("buildable").current_num_threads() >= 1);
        });
    }
}
