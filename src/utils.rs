use std::fmt::Display;
use std::sync::Arc;

use pyo3::{PyErr, PyResult, PyTypeInfo};
use zarrs::array::BytesPartialDecoderTraits;
use zarrs::array::CodecError;
use zarrs::storage::{ReadableWritableListableStorage, StorageHandle, StoreKey};

use crate::ChunkItem;

pub(crate) trait PyErrExt<T> {
    fn map_py_err<PE: PyTypeInfo>(self) -> PyResult<T>;
}

impl<T, E: Display> PyErrExt<T> for Result<T, E> {
    fn map_py_err<PE: PyTypeInfo>(self) -> PyResult<T> {
        self.map_err(|e| PyErr::new::<PE, _>(format!("{e}")))
    }
}

pub(crate) trait PyCodecErrExt<T> {
    fn map_codec_err(self) -> PyResult<T>;
}

impl<T> PyCodecErrExt<T> for Result<T, CodecError> {
    fn map_codec_err(self) -> PyResult<T> {
        // see https://docs.python.org/3/library/exceptions.html#exception-hierarchy
        self.map_err(|e| match e {
            // requested indexing operation doesn’t match shape
            CodecError::IncompatibleIndexer(_)
            | CodecError::IncompatibleDimensionalityError(_)
            | CodecError::InvalidByteRangeError(_) => {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!("{e}"))
            }
            // some pipe, file, or subprocess failed
            CodecError::IOError(_) => PyErr::new::<pyo3::exceptions::PyOSError, _>(format!("{e}")),
            // all the rest: some unknown runtime problem
            e => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")),
        })
    }
}

pub fn is_whole_chunk(item: &ChunkItem) -> bool {
    item.chunk_subset.start().iter().all(|&o| o == 0)
        && item.chunk_subset.shape() == bytemuck::must_cast_slice::<_, u64>(&item.shape)
}

/// Copy `coords.len()` elements out of `scratch` into `out`, in coordinate order.
///
/// `out` must be exactly `coords.len() * size` bytes.
pub(crate) fn gather(
    scratch: &[u8],
    coords: &[u64],
    out: &mut [u8],
    size: usize,
) -> Result<(), String> {
    if out.len() != coords.len() * size {
        return Err("output region does not match the coordinate count".to_string());
    }
    for (n, &c) in coords.iter().enumerate() {
        let src = usize::try_from(c).map_err(|e| e.to_string())? * size;
        let Some(element) = scratch.get(src..src + size) else {
            return Err(format!(
                "coordinate {c} is outside the {} elements decoded",
                scratch.len() / size
            ));
        };
        out[n * size..(n + 1) * size].copy_from_slice(element);
    }
    Ok(())
}

/// A partial decoder that reads one store key.
///
/// The `(storage, key)` tuple is zarrs 0.24's store-backed `BytesPartialDecoderTraits`
/// implementation.
pub(crate) fn key_partial_decoder(
    store: &ReadableWritableListableStorage,
    key: &StoreKey,
) -> Arc<dyn BytesPartialDecoderTraits> {
    Arc::new((StorageHandle::new(store.clone()), key.clone()))
}
