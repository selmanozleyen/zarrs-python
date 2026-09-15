use std::sync::Once;
use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::PyResult;
use pyo3::exceptions::PyRuntimeError;

static GENERATION: AtomicU64 = AtomicU64::new(0);

#[cfg(unix)]
extern "C" fn note_fork() {
    GENERATION.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn generation() -> u64 {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        #[cfg(unix)]
        unsafe {
            libc::pthread_atfork(None, None, Some(note_fork));
        }
    });
    GENERATION.load(Ordering::Relaxed)
}

pub(crate) fn check_era(era: u64) -> PyResult<()> {
    if era == generation() {
        Ok(())
    } else {
        Err(PyRuntimeError::new_err(
            "this store's tokio runtime was created before the process forked and does not \
             survive a fork. Start worker processes with the 'spawn' or 'forkserver' method \
             instead.",
        ))
    }
}
