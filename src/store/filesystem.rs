use std::sync::Arc;

use pyo3::{PyErr, exceptions::PyRuntimeError};
use zarrs::{
    filesystem::{FilesystemStore, FilesystemStoreOptions},
    storage::ReadableWritableListableStorage,
};

use crate::utils::PyErrExt;

#[derive(Debug, Clone)]
pub struct FilesystemStoreConfig {
    pub root: String,
    opts: FilesystemStoreOptions,
}

/// Size of the store's open-file-handle cache. Upstream defaults this to 0,
/// meaning every read is open + pread + close. That is three syscalls per
/// planned range, which dominates once reads are issued concurrently -- the
/// pool then measures syscall overhead rather than read latency.
const FILE_HANDLE_CACHE: &str = "ZARRS_PYTHON_FILE_HANDLE_CACHE";

impl FilesystemStoreConfig {
    pub fn new(root: String) -> Self {
        let mut opts = FilesystemStoreOptions::default();
        if let Some(size) = std::env::var(FILE_HANDLE_CACHE)
            .ok()
            .and_then(|size| size.parse::<usize>().ok())
        {
            opts.file_handle_cache_size(size);
        }
        Self { root, opts }
    }

    pub fn direct_io(&mut self, flag: bool) -> () {
        self.opts.direct_io(flag);
    }
}

impl TryInto<ReadableWritableListableStorage> for &FilesystemStoreConfig {
    type Error = PyErr;

    fn try_into(self) -> Result<ReadableWritableListableStorage, Self::Error> {
        let store = Arc::new(
            FilesystemStore::new_with_options(self.root.clone(), self.opts.clone())
                .map_py_err::<PyRuntimeError>()?,
        );
        Ok(store)
    }
}
