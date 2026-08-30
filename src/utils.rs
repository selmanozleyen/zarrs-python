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

/// Gather `starts.len()` RUNS of `run` elements per coordinate, rather than one contiguous
/// span.
///
/// A grid takes the same sub-box out of every selected index, and in row-major order that box
/// is a set of runs, not a set of elements: `X[rows, 2:5, 4:12]` of a (8,16) row is three runs
/// of eight, at 36, 52 and 68. Representing it as twenty-four separate offsets and copying
/// them one at a time throws away structure that is actually there -- which is why a box of
/// pure slices used to be left to the ordinary route.
///
/// A fully scattered selection degenerates to `run == 1`, one element per start, so this is
/// never worse than the element-at-a-time version it replaces.
///
/// The output stays contiguous -- the runs land back to back, which is what the output row is
/// -- so an item is still vended as a single range.
pub(crate) fn gather_runs(
    scratch: &[u8],
    coords: &[u64],
    starts: &[u64],
    run: u64,
    out: &mut [u8],
    size: usize,
) -> Result<(), String> {
    if starts.is_empty() || run == 0 {
        return Err("a grid must select at least one element".to_string());
    }
    let Some(span) = usize::try_from(run).ok().and_then(|r| r.checked_mul(size)) else {
        return Err(format!("a run of {run} elements is too large to address"));
    };
    let Some(row) = starts.len().checked_mul(span) else {
        return Err("the gathered rows are too large to address".to_string());
    };
    if coords.len().checked_mul(row) != Some(out.len()) {
        return Err("output region does not match the coordinate count".to_string());
    }
    for (n, &c) in coords.iter().enumerate() {
        for (j, &start) in starts.iter().enumerate() {
            // Checked for the reason the contiguous gather checks: unchecked, a large value
            // wraps in release and can land back INSIDE scratch, so the read succeeds and
            // returns the wrong elements rather than failing.
            let Some(src) = c
                .checked_add(start)
                .and_then(|e| usize::try_from(e).ok())
                .and_then(|e| e.checked_mul(size))
            else {
                return Err(format!("coordinate {c} plus {start} is too large to address"));
            };
            let Some(piece) = src.checked_add(span).and_then(|end| scratch.get(src..end)) else {
                return Err(format!(
                    "coordinate {c} run at {start} leaves the {} elements decoded",
                    scratch.len() / size
                ));
            };
            let at = n * row + j * span;
            out[at..at + span].copy_from_slice(piece);
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
