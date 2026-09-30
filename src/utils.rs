use std::fmt::Display;
use std::ops::Range;
use std::sync::Arc;

use pyo3::{PyErr, PyResult, PyTypeInfo};
use zarrs::array::BytesPartialDecoderTraits;
use zarrs::array::CodecError;
use zarrs::storage::{ReadableStorage, StorageHandle, StoreKey};

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

/// Maximal runs of coordinates stepping by exactly `run_len`, as index ranges into `coords`.
/// A duplicate breaks the run.
pub(crate) fn coord_runs(coords: &[u64], run_len: u64) -> impl Iterator<Item = Range<usize>> + '_ {
    let mut start = 0usize;
    std::iter::from_fn(move || {
        if start >= coords.len() {
            return None;
        }
        let mut end = start + 1;
        while end < coords.len() && coords[end - 1].checked_add(run_len) == Some(coords[end]) {
            end += 1;
        }
        let run = start..end;
        start = end;
        Some(run)
    })
}

/// Writes a sequence of runs across output pieces that need not align with them.
pub(crate) struct PieceWriter<'a, 'b> {
    pieces: &'b mut [&'a mut [u8]],
    piece: usize,
    at: usize,
}

impl<'a, 'b> PieceWriter<'a, 'b> {
    pub(crate) fn new(pieces: &'b mut [&'a mut [u8]]) -> Self {
        Self {
            pieces,
            piece: 0,
            at: 0,
        }
    }

    /// Append `src`, spilling into later pieces as needed.
    pub(crate) fn write(&mut self, mut src: &[u8]) -> Result<(), String> {
        while !src.is_empty() {
            while self.piece < self.pieces.len() && self.at == self.pieces[self.piece].len() {
                self.piece += 1;
                self.at = 0;
            }
            let Some(piece) = self.pieces.get_mut(self.piece) else {
                return Err(format!(
                    "{} bytes left to write with no output piece to take them",
                    src.len()
                ));
            };
            let room = piece.len() - self.at;
            let take = room.min(src.len());
            piece[self.at..self.at + take].copy_from_slice(&src[..take]);
            self.at += take;
            src = &src[take..];
        }
        Ok(())
    }

    /// Every byte of every piece was written.
    pub(crate) fn finished(&self) -> bool {
        self.pieces
            .get(self.piece)
            .is_none_or(|piece| self.at == piece.len())
            && self.pieces[(self.piece + 1).min(self.pieces.len())..]
                .iter()
                .all(|piece| piece.is_empty())
    }
}

fn run_bytes(run_len: u64, size: usize) -> Result<usize, String> {
    let Some(bytes) = usize::try_from(run_len)
        .ok()
        .and_then(|r| r.checked_mul(size))
    else {
        return Err(format!("run length {run_len} is too large to address"));
    };
    if bytes == 0 {
        return Err("run length must be greater than zero".to_string());
    }
    Ok(bytes)
}

pub(crate) fn gather(
    scratch: &[u8],
    coords: &[u64],
    run_len: u64,
    out: &mut [u8],
    size: usize,
) -> Result<(), String> {
    let run = run_bytes(run_len, size)?;
    if coords.len().checked_mul(run) != Some(out.len()) {
        return Err("output region does not match the coordinate count".to_string());
    }
    for (n, &c) in coords.iter().enumerate() {
        let Some(src) = usize::try_from(c).ok().and_then(|c| c.checked_mul(size)) else {
            return Err(format!("coordinate {c} is too large to address"));
        };
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

/// `gather`, writing across several output pieces instead of one slice.
pub(crate) fn gather_pieces(
    scratch: &[u8],
    coords: &[u64],
    run_len: u64,
    pieces: &mut [&mut [u8]],
    size: usize,
) -> Result<(), String> {
    let run = run_bytes(run_len, size)?;
    let total: usize = pieces.iter().map(|p| p.len()).sum();
    if coords.len().checked_mul(run) != Some(total) {
        return Err("output pieces do not match the coordinate count".to_string());
    }
    let mut writer = PieceWriter::new(pieces);
    // A merged span may straddle two pieces; the writer spills it over.
    for r in coord_runs(coords, run_len) {
        let c = coords[r.start];
        let Some(src) = usize::try_from(c).ok().and_then(|c| c.checked_mul(size)) else {
            return Err(format!("coordinate {c} is too large to address"));
        };
        let Some(span) = r.len().checked_mul(run) else {
            return Err("the gathered span is too large to address".to_string());
        };
        let Some(region) = src.checked_add(span).and_then(|end| scratch.get(src..end)) else {
            return Err(format!(
                "coordinate {c} plus {} elements is outside the {} decoded",
                r.len() as u64 * run_len,
                scratch.len() / size
            ));
        };
        writer.write(region)?;
    }
    if !writer.finished() {
        return Err("the gather left part of the output unwritten".to_string());
    }
    Ok(())
}

/// A partial decoder that reads one store key.
pub(crate) fn key_partial_decoder(
    store: &ReadableStorage,
    key: &StoreKey,
) -> Arc<dyn BytesPartialDecoderTraits> {
    Arc::new((StorageHandle::new(store.clone()), key.clone()))
}
