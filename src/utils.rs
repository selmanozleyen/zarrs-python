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

/// Copy one RUN of `run_len` elements per coordinate out of `scratch` into `out`, in
/// coordinate order.
///
/// `run_len` is 1 on the 1-D path, where a coordinate is one element. On a rank-N selection
/// taken whole after axis 0 it is the row length, and then this moves a row per coordinate
/// -- one `copy_from_slice` of `run_len * size` bytes instead of `run_len` of `size`, which
/// is the whole reason grouping a wide read pays.
///
/// `out` must be exactly `coords.len() * run_len * size` bytes.
pub(crate) fn gather(
    scratch: &[u8],
    coords: &[u64],
    run_len: u64,
    out: &mut [u8],
    size: usize,
) -> Result<(), String> {
    let Some(run) = usize::try_from(run_len).ok().and_then(|r| r.checked_mul(size)) else {
        return Err(format!("run length {run_len} is too large to address"));
    };
    if run == 0 {
        return Err("run length must be greater than zero".to_string());
    }
    if coords.len().checked_mul(run) != Some(out.len()) {
        return Err("output region does not match the coordinate count".to_string());
    }
    for (n, &c) in coords.iter().enumerate() {
        // Checked, and not because a coordinate can be that large today -- they are all
        // below the inner chunk extent. Unchecked, a large one wraps in release and can land
        // back INSIDE scratch, so `get` succeeds and the wrong element is copied: exactly
        // the silent-wrong-data mode this function's bounds check exists to refuse.
        let Some(src) = usize::try_from(c).ok().and_then(|c| c.checked_mul(size)) else {
            return Err(format!("coordinate {c} is too large to address"));
        };
        // The END of the run is what has to be in bounds, not its start: a coordinate inside
        // `scratch` whose run walks off the end would otherwise read past the decode.
        let Some(element) = src.checked_add(run).and_then(|end| scratch.get(src..end)) else {
            return Err(format!(
                "coordinate {c} plus {run_len} elements is outside the {} decoded",
                scratch.len() / size
            ));
        };
        out[n * run..(n + 1) * run].copy_from_slice(element);
    }
    Ok(())
}

/// Gather `cols.len()` SCATTERED elements per coordinate, rather than a contiguous run.
///
/// `oindex[rows, cols]` takes the same columns from every selected row, so one shared list
/// serves the whole item and each coordinate is the start of its own row. The output stays
/// contiguous -- a row of the result is those columns in order -- which is what lets an item
/// still be vended as one range.
///
/// Element at a time, where the contiguous gather is one memcpy per coordinate. That is the
/// price of the case: the alternative is the ordinary route, which pays a partial-decode call
/// per element instead.
pub(crate) fn gather_columns(
    scratch: &[u8],
    coords: &[u64],
    cols: &[u64],
    out: &mut [u8],
    size: usize,
) -> Result<(), String> {
    if cols.is_empty() {
        return Err("a column list must select at least one element".to_string());
    }
    let Some(row) = cols.len().checked_mul(size) else {
        return Err("the column list is too long to address".to_string());
    };
    if coords.len().checked_mul(row) != Some(out.len()) {
        return Err("output region does not match the coordinate count".to_string());
    }
    for (n, &c) in coords.iter().enumerate() {
        for (j, &col) in cols.iter().enumerate() {
            // Checked for the same reason the contiguous gather checks: unchecked, a large
            // value wraps in release and can land back INSIDE scratch, so the read succeeds
            // and returns the wrong element rather than failing.
            let Some(src) = c
                .checked_add(col)
                .and_then(|e| usize::try_from(e).ok())
                .and_then(|e| e.checked_mul(size))
            else {
                return Err(format!("coordinate {c} plus column {col} is too large to address"));
            };
            let Some(element) = src.checked_add(size).and_then(|end| scratch.get(src..end)) else {
                return Err(format!(
                    "coordinate {c} column {col} is outside the {} elements decoded",
                    scratch.len() / size
                ));
            };
            let at = n * row + j * size;
            out[at..at + size].copy_from_slice(element);
        }
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
