use std::fmt::Display;
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
            // Skip pieces already filled, and any that are empty to begin with.
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

    /// Every byte of every piece was written. The caller's buffer is `np.empty`, so a piece
    /// left short returns whatever was already in memory, as data.
    pub(crate) fn finished(&self) -> bool {
        // Everything before `piece` was filled to its end by construction, so only the
        // current piece and whatever follows it can be short.
        self.pieces
            .get(self.piece)
            .is_none_or(|piece| self.at == piece.len())
            && self.pieces[(self.piece + 1).min(self.pieces.len())..]
                .iter()
                .all(|piece| piece.is_empty())
    }
}

/// Copy one RUN of `run_len` elements per coordinate out of `scratch` into `out`, in
/// coordinate order.
///
/// `out` must be exactly `coords.len() * run_len * size` bytes.
pub(crate) fn gather(
    scratch: &[u8],
    coords: &[u64],
    run_len: u64,
    out: &mut [u8],
    size: usize,
) -> Result<(), String> {
    let Some(run) = usize::try_from(run_len)
        .ok()
        .and_then(|r| r.checked_mul(size))
    else {
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

/// `gather`, writing across several output pieces instead of one slice.
pub(crate) fn gather_pieces(
    scratch: &[u8],
    coords: &[u64],
    run_len: u64,
    pieces: &mut [&mut [u8]],
    size: usize,
) -> Result<(), String> {
    let Some(run) = usize::try_from(run_len)
        .ok()
        .and_then(|r| r.checked_mul(size))
    else {
        return Err(format!("run length {run_len} is too large to address"));
    };
    if run == 0 {
        return Err("run length must be greater than zero".to_string());
    }
    let total: usize = pieces.iter().map(|p| p.len()).sum();
    if coords.len().checked_mul(run) != Some(total) {
        return Err("output pieces do not match the coordinate count".to_string());
    }
    let mut writer = PieceWriter::new(pieces);
    // Consecutive coordinates name ONE contiguous span of the decode and are copied as one.
    // The pieces are written in order, so a merged span still lands correctly when it
    // straddles two of them. (`gather`, the single-piece path, does NOT merge: it writes into
    // one slice at a fixed stride, where a copy per coordinate costs nothing extra.)
    let mut n = 0usize;
    while n < coords.len() {
        let mut m = n + 1;
        while m < coords.len() && coords[m - 1].checked_add(run_len) == Some(coords[m]) {
            m += 1;
        }
        let c = coords[n];
        let Some(src) = usize::try_from(c).ok().and_then(|c| c.checked_mul(size)) else {
            return Err(format!("coordinate {c} is too large to address"));
        };
        let Some(span) = (m - n).checked_mul(run) else {
            return Err("the gathered span is too large to address".to_string());
        };
        let Some(region) = src.checked_add(span).and_then(|end| scratch.get(src..end)) else {
            return Err(format!(
                "coordinate {c} plus {} elements is outside the {} decoded",
                (m - n) as u64 * run_len,
                scratch.len() / size
            ));
        };
        writer.write(region)?;
        n = m;
    }
    if !writer.finished() {
        return Err("the gather left part of the output unwritten".to_string());
    }
    Ok(())
}

/// Gather `starts.len()` RUNS of `run` elements per coordinate, rather than one contiguous
/// span.
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
                return Err(format!(
                    "coordinate {c} plus {start} is too large to address"
                ));
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
pub(crate) fn key_partial_decoder(
    store: &ReadableStorage,
    key: &StoreKey,
) -> Arc<dyn BytesPartialDecoderTraits> {
    Arc::new((StorageHandle::new(store.clone()), key.clone()))
}
