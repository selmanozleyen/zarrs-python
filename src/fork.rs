use std::sync::Once;
use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::PyResult;
use pyo3::exceptions::PyRuntimeError;

static GENERATION: AtomicU64 = AtomicU64::new(0);
static ARMED: AtomicU64 = AtomicU64::new(u64::MAX);

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

pub(crate) fn check() -> PyResult<()> {
    let now = generation();
    match ARMED.compare_exchange(u64::MAX, now, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => Ok(()),
        Err(armed) if armed == now => Ok(()),
        Err(_) => Err(PyRuntimeError::new_err(
            "zarrs was used in this process before it was forked, and its worker threads -- the \
             read and decode pools, rayon's pool for writes, the tokio runtime for remote stores \
             -- do not survive a fork. Start worker processes with the 'spawn' or 'forkserver' \
             method instead.",
        )),
    }
}
