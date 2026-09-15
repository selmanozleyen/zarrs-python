use std::sync::{Arc, Mutex, PoisonError};

use pyo3::exceptions::PyRuntimeError;
use pyo3::{PyResult, Python};

use crate::fork::generation;

pub(crate) struct PerProcess<T> {
    slot: Mutex<Option<(u64, Arc<T>)>>,
}

impl<T> PerProcess<T> {
    pub(crate) const fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    pub(crate) fn get_or_try_init<E>(
        &self,
        build: impl FnOnce() -> Result<T, E>,
    ) -> Result<Arc<T>, E> {
        let generation = generation();
        let mut guard = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        if guard.as_ref().is_none_or(|(built, _)| *built != generation) {
            let fresh = Arc::new(build()?);
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

pub(crate) fn pool(_py: Python<'_>) -> PyResult<Arc<rayon::ThreadPool>> {
    POOL.get_or_try_init(|| {
        rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("zarrs-{i}"))
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("could not create a thread pool: {e}")))
    })
}
