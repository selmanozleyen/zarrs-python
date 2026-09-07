//! The rayon pool this crate's parallel work runs on.
//!
//! Rayon's global pool sits behind a `Once` with no reset, so a `fork()`ed child inherits a
//! registry reporting workers it does not have and parks for ever (issue #171, against
//! `DataLoader(num_workers>0)`). Nothing here submits to it.
//!
//! That registry is still built and this module cannot prevent it: `CodecOptions` and zarrs'
//! `Config` read `current_num_threads()` in their `Default` impls. An inherited idle registry is
//! only memory, so what matters is that every rayon entry point in this crate sits inside
//! `pool().install(..)`.

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::{PyResult, Python};

use crate::per_process::PerProcess;

static POOL: PerProcess<rayon::ThreadPool> = PerProcess::new();

/// This process's pool, built on first use and rebuilt in a forked child.
///
/// Takes a `Python` token so the GIL rule is the compiler's business: this locks, and the GIL is
/// what stops that lock being held when another thread forks. Hold the `Arc` across
/// `Python::detach`, never call this inside one.
pub(crate) fn pool(_py: Python<'_>) -> PyResult<Arc<rayon::ThreadPool>> {
    POOL.get_or_try_init(|| {
        // `num_threads` unset on purpose: rayon then resolves `RAYON_NUM_THREADS`, then
        // `RAYON_RS_NUM_CPUS`, then the machine -- the ladder that sized the pool this replaces.
        rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("zarrs-{i}"))
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("could not create a thread pool: {e}")))
    })
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
            assert!(pool(py).expect("buildable").current_num_threads() >= 1);
        });
    }
}
