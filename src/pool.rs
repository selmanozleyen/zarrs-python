//! The rayon pool this crate runs on, keyed on the process so a `fork()` child rebuilds it.
//!
//! Rayon's global pool has no reset, so a forked child inherits a registry with no live workers
//! and parks for ever (#171). Nothing here submits to it. It is still built, by
//! `CodecOptions::default` and zarrs' `Config`, which this module cannot prevent.

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::{PyResult, Python};

use crate::per_process::PerProcess;

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
}
