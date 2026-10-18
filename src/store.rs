

use pyo3::{
    Borrowed, FromPyObject, PyAny, PyErr, PyResult,
    exceptions::PyNotImplementedError,
    types::{PyAnyMethods, PyStringMethods, PyTypeMethods},
};
use zarrs::storage::ReadableWritableListableStorage;


mod filesystem;

pub use self::filesystem::FilesystemStoreConfig;

#[derive(Debug, Clone)]
pub struct StoreConfig {
    pub kind: StoreKind,
    /// Whether zarr-python opened the store read-only, i.e. `mode="r"`.
    ///
    /// zarr-python resolves the mode into the store itself (`Store.read_only`), so it comes
    /// off the same object every other field here does.
    pub read_only: bool,
}

#[derive(Debug, Clone)]
pub enum StoreKind {
    Filesystem(FilesystemStoreConfig),
    // TODO: Add support for more stores
}

impl<'py> FromPyObject<'_, 'py> for StoreConfig {
    type Error = PyErr;

    fn extract(store: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let name = store.get_type().name()?;
        let name = name.to_str()?;
        let kind = match name {
            "LocalStore" => {
                let root: String = store.getattr("root")?.call_method0("__str__")?.extract()?;
                StoreKind::Filesystem(FilesystemStoreConfig::new(root))
            }
            "FsspecStore" | "ObjectStore" => {
                return Err(PyErr::new::<PyNotImplementedError, _>(format!(
                    "{name} stores are not in this benchmark build"
                )));
            }
            _ => {
                return Err(PyErr::new::<PyNotImplementedError, _>(format!(
                    "zarrs-python does not support {name} stores"
                )));
            }
        };
        Ok(StoreConfig {
            kind,
            read_only: store.getattr("read_only")?.extract()?,
        })
    }
}

impl StoreConfig {
    pub fn direct_io(&mut self, flag: bool) {
        match &mut self.kind {
            StoreKind::Filesystem(config) => config.direct_io(flag),
        }
    }

    pub fn file_handle_cache_size(&mut self, size: usize) {
        match &mut self.kind {
            StoreKind::Filesystem(config) => config.file_handle_cache_size(size),
        }
    }
}

impl pyo3_stub_gen::PyStubType for StoreConfig {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("zarr.abc.store.Store", "zarr.abc.store".into())
    }
}

impl TryFrom<&StoreConfig> for ReadableWritableListableStorage {
    type Error = PyErr;

    fn try_from(value: &StoreConfig) -> Result<Self, Self::Error> {
        match &value.kind {
            StoreKind::Filesystem(config) => config.try_into(),
        }
    }
}

