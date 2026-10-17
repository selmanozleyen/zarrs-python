use std::num::NonZeroU64;
use std::sync::Arc;

use numpy::PyReadonlyArray1;
use pyo3::{
    Bound, PyErr, PyResult,
    exceptions::{PyIndexError, PyValueError},
    pyclass, pymethods,
    types::{PySlice, PySliceMethods as _},
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use zarrs::{
    array::ArraySubset,
    storage::StoreKey,
};

use crate::utils::PyErrExt;

fn to_nonzero_u64_vec(v: Vec<u64>) -> PyResult<Vec<NonZeroU64>> {
    v.into_iter()
        .map(|dim| {
            NonZeroU64::new(dim).ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "subset dimensions must be greater than zero".to_string(),
                )
            })
        })
        .collect::<PyResult<Vec<NonZeroU64>>>()
}

#[derive(Clone)]
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct ChunkItem {
    pub key: StoreKey,
    pub chunk_subset: ArraySubset,
    pub subset: ArraySubset,
    pub shape: Vec<NonZeroU64>,
    pub num_elements: u64,
    pub array_shape: Vec<NonZeroU64>,
    /// Element offsets into the decoded inner chunk, each the start of a run of `run_len`.
    /// `None` for a write item.
    pub coords: Option<Arc<[u64]>>,
    pub run_len: u64,
    /// `(axis-0 extent, row stride)` of the inner chunk `coords` were built against.
    pub claimed_inner: (u64, u64),
    /// The flat output element offset of each coordinate's run; `subset` is then unused.
    pub out_offsets: Option<Arc<[u64]>>,
    /// When the item is one whole inner chunk landing as one box of the output, that box:
    /// the chunk can be decoded straight into it.
    pub out_box: Option<ArraySubset>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ChunkItem {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        key: String,
        chunk_subset: Vec<Bound<'_, PySlice>>,
        chunk_shape: Vec<u64>,
        subset: Vec<Bound<'_, PySlice>>,
        shape: Vec<u64>,
    ) -> PyResult<Self> {
        let num_elements = chunk_shape.iter().product();
        let shape_nonzero_u64 = to_nonzero_u64_vec(shape)?;
        let chunk_shape_nonzero_u64 = to_nonzero_u64_vec(chunk_shape)?;
        let chunk_subset = selection_to_array_subset(&chunk_subset, &chunk_shape_nonzero_u64)?;
        let subset = selection_to_array_subset(&subset, &shape_nonzero_u64)?;
        // Check that subset and chunk_subset have the same number of elements.
        // This permits broadcasting of a constant input.
        if subset.num_elements() != chunk_subset.num_elements() && subset.num_elements() > 1 {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "the size of the chunk subset {chunk_subset} and input/output subset {subset} are incompatible",
            )));
        }

        Ok(Self {
            key: StoreKey::new(key).map_py_err::<PyValueError>()?,
            chunk_subset,
            subset,
            shape: chunk_shape_nonzero_u64,
            num_elements,
            array_shape: shape_nonzero_u64,
            coords: None,
            run_len: 1,
            claimed_inner: (0, 0),
            out_offsets: None,
            out_box: None,
        })
    }
}

fn slice_to_range(slice: &Bound<'_, PySlice>, length: isize) -> PyResult<std::ops::Range<u64>> {
    let indices = slice.indices(length)?;
    if indices.start < 0 {
        Err(PyErr::new::<PyValueError, _>(
            "slice start must be greater than or equal to 0".to_string(),
        ))
    } else if indices.stop < 0 {
        Err(PyErr::new::<PyValueError, _>(
            "slice stop must be greater than or equal to 0".to_string(),
        ))
    } else if indices.step != 1 {
        Err(PyErr::new::<PyValueError, _>(
            "slice step must be equal to 1".to_string(),
        ))
    } else {
        Ok(u64::try_from(indices.start)?..u64::try_from(indices.stop)?)
    }
}

fn selection_to_array_subset(
    selection: &[Bound<'_, PySlice>],
    shape: &[NonZeroU64],
) -> PyResult<ArraySubset> {
    if selection.is_empty() {
        Ok(ArraySubset::new_with_shape(vec![1; shape.len()]))
    } else {
        let chunk_ranges = selection
            .iter()
            .zip(shape)
            .map(|(selection, &shape)| slice_to_range(selection, isize::try_from(shape.get())?))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(ArraySubset::new_with_ranges(&chunk_ranges))
    }
}

/// `(shard-local lo, hi, output start)` of a run within one inner chunk.
type Piece = (u64, u64, u64);
/// One axis of a box: each inner chunk index touched, and its pieces.
type ChunkPieces = Vec<(u64, Vec<Piece>)>;

/// A batch of read items, built and held in Rust.
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct ChunkItems {
    items: Vec<ChunkItem>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ChunkItems {
    #[new]
    pub(crate) fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// The number of read items pushed so far.
    fn __len__(&self) -> usize {
        self.items.len()
    }

    /// Push one shard's orthogonal selection: per axis, runs `lo[r]..lo[r] + len[r]` of the
    /// shard, each written from output offset `out[r]` on. The runs need not be sorted or
    /// adjacent, so an index array in any order is a run per index (or per consecutive group).
    #[pyo3(signature = (key, shard_shape, inner, out_shape, lo, len, out))]
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    pub(crate) fn push_box(
        &mut self,
        key: &str,
        shard_shape: Vec<u64>,
        inner: Vec<u64>,
        out_shape: Vec<u64>,
        lo: Vec<PyReadonlyArray1<'_, i64>>,
        len: Vec<PyReadonlyArray1<'_, i64>>,
        out: Vec<PyReadonlyArray1<'_, i64>>,
    ) -> PyResult<()> {
        let rank = shard_shape.len();
        if rank == 0 || [inner.len(), out_shape.len(), lo.len(), len.len(), out.len()] != [rank; 5] {
            return Err(PyErr::new::<PyValueError, _>(
                "one inner extent, output extent and run triple per axis",
            ));
        }
        if shard_shape.iter().chain(&inner).any(|d| *d == 0) {
            return Err(PyErr::new::<PyValueError, _>(
                "shard and inner extents must be non-zero",
            ));
        }
        let mut axes: Vec<ChunkPieces> = Vec::with_capacity(rank);
        for k in 0..rank {
            let pieces = split_axis_runs(
                k,
                (&lo[k], &len[k], &out[k]),
                shard_shape[k],
                inner[k],
                out_shape[k],
            )?;
            if pieces.is_empty() {
                return Ok(());
            }
            axes.push(pieces);
        }
        let layout = BoxLayout {
            key: StoreKey::new(key.to_string()).map_py_err::<PyValueError>()?,
            in_strides: c_strides(&inner),
            out_strides: c_strides(&out_shape),
            num_elements: shard_shape.iter().product(),
            claimed_inner: (inner[0], inner[1..].iter().product()),
            array_shape: to_nonzero_u64_vec(out_shape)?,
            shape: to_nonzero_u64_vec(shard_shape.clone())?,
            shard_shape,
            inner,
        };
        // An odometer over the inner chunks touched, last axis fastest.
        let mut at = vec![0usize; rank];
        loop {
            let chunk: Vec<&(u64, Vec<Piece>)> = at.iter().zip(&axes).map(|(i, a)| &a[*i]).collect();
            self.push_inner_chunk(&layout, &chunk);
            let mut k = rank;
            loop {
                if k == 0 {
                    return Ok(());
                }
                k -= 1;
                at[k] += 1;
                if at[k] < axes[k].len() {
                    break;
                }
                at[k] = 0;
            }
        }
    }
}

/// What every item of one `push_box` call shares.
struct BoxLayout {
    key: StoreKey,
    shard_shape: Vec<u64>,
    inner: Vec<u64>,
    in_strides: Vec<u64>,
    out_strides: Vec<u64>,
    num_elements: u64,
    claimed_inner: (u64, u64),
    array_shape: Vec<NonZeroU64>,
    shape: Vec<NonZeroU64>,
}

/// C-order element strides of `shape`.
fn c_strides(shape: &[u64]) -> Vec<u64> {
    let mut strides = vec![1u64; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    strides
}

/// One axis's runs `(lo, len, out)`, checked against the shard and the output, and split at
/// inner chunk boundaries: each inner chunk touched, in order, with its pieces.
fn split_axis_runs(
    axis: usize,
    (lo, len, out): (
        &PyReadonlyArray1<'_, i64>,
        &PyReadonlyArray1<'_, i64>,
        &PyReadonlyArray1<'_, i64>,
    ),
    shard: u64,
    split: u64,
    out_extent: u64,
) -> PyResult<ChunkPieces> {
    let (los, lens, outs) = (lo.as_array(), len.as_array(), out.as_array());
    if los.len() != lens.len() || los.len() != outs.len() {
        return Err(PyErr::new::<PyValueError, _>("one length and output per run"));
    }
    let u = |v: i64| {
        u64::try_from(v)
            .map_err(|_| PyErr::new::<PyValueError, _>(format!("negative run field {v}")))
    };
    let mut by_chunk: std::collections::BTreeMap<u64, Vec<Piece>> =
        std::collections::BTreeMap::new();
    for ((run_lo, run_len), run_out) in los.iter().zip(lens.iter()).zip(outs.iter()) {
        let (mut at, count, to) = (u(*run_lo)?, u(*run_len)?, u(*run_out)?);
        let end = at.checked_add(count).filter(|end| *end <= shard).ok_or_else(|| {
            PyErr::new::<PyIndexError, _>(format!("a run on axis {axis} is past the shard"))
        })?;
        if to.checked_add(count).is_none_or(|end| end > out_extent) {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "a run on axis {axis} is past the output extent {out_extent}"
            )));
        }
        let first = at;
        while at < end {
            let hi = ((at / split + 1) * split).min(end);
            by_chunk.entry(at / split).or_default().push((at, hi, to + (at - first)));
            at = hi;
        }
    }
    Ok(by_chunk.into_iter().collect())
}

impl ChunkItems {
    /// The items of one inner chunk of a box: one per last-axis run length, whose coordinates
    /// are every run of that length in the chunk and whose `out_offsets` say where each lands.
    fn push_inner_chunk(&mut self, layout: &BoxLayout, chunk: &[&(u64, Vec<Piece>)]) {
        let last = chunk.len() - 1;
        let origin: Vec<u64> = chunk.iter().zip(&layout.inner).map(|((c, _), e)| c * e).collect();
        let inner_box: Vec<_> = origin
            .iter()
            .zip(&layout.inner)
            .zip(&layout.shard_shape)
            .map(|((o, e), s)| *o..(*o + *e).min(*s))
            .collect();
        // One piece spanning the whole inner chunk on every axis: the chunk lands as one box,
        // so it is one run of the whole chunk, written over that box in C order.
        let whole = chunk.iter().zip(&origin).zip(&layout.inner).all(|(((_, pieces), o), e)| {
            matches!(pieces.as_slice(), [(lo, hi, _)] if *lo == *o && *hi == *o + *e)
        });
        if whole {
            let out_box = ArraySubset::new_with_start_shape(
                chunk.iter().map(|(_, pieces)| pieces[0].2).collect(),
                layout.inner.clone(),
            )
            .expect("one start and one extent per axis");
            self.items.push(ChunkItem {
                key: layout.key.clone(),
                chunk_subset: ArraySubset::new_with_ranges(&inner_box),
                subset: out_box.clone(),
                shape: layout.shape.clone(),
                num_elements: layout.num_elements,
                array_shape: layout.array_shape.clone(),
                coords: Some(vec![0].into()),
                run_len: layout.inner.iter().product(),
                claimed_inner: layout.claimed_inner,
                out_offsets: None,
                out_box: Some(out_box),
            });
            return;
        }
        // Trailing axes the chunk covers whole, and the output holds whole, are contiguous in
        // both: they fold into the run, so the run axis is the last one that does not.
        let mut run_axis = last;
        while run_axis > 0 && {
            let extent = layout.inner[run_axis];
            matches!(chunk[run_axis].1.as_slice(),
                [(lo, hi, 0)] if *lo == origin[run_axis] && *hi == *lo + extent)
                && layout.array_shape[run_axis].get() == extent
        } {
            run_axis -= 1;
        }
        let folded: u64 = layout.inner[run_axis + 1..].iter().product();
        // `(inner offset, output offset)` of each row: every element of the axes before it.
        let mut rows = vec![(0u64, 0u64)];
        for k in 0..run_axis {
            let mut next = Vec::new();
            for &(io, oo) in &rows {
                for &(lo, hi, out) in &chunk[k].1 {
                    for e in 0..hi - lo {
                        next.push((
                            io + (lo - origin[k] + e) * layout.in_strides[k],
                            oo + (out + e) * layout.out_strides[k],
                        ));
                    }
                }
            }
            rows = next;
        }
        let mut by_len: std::collections::BTreeMap<u64, Vec<(u64, u64)>> =
            std::collections::BTreeMap::new();
        for &(lo, hi, out) in &chunk[run_axis].1 {
            by_len.entry((hi - lo) * folded).or_default().push((
                (lo - origin[run_axis]) * layout.in_strides[run_axis],
                out * layout.out_strides[run_axis],
            ));
        }
        for (run_len, runs) in by_len {
            let mut coords = Vec::with_capacity(rows.len() * runs.len());
            let mut offsets = Vec::with_capacity(rows.len() * runs.len());
            let mut pairs: Vec<(u64, u64)> = Vec::with_capacity(rows.len() * runs.len());
            for &(io, oo) in &rows {
                pairs.extend(runs.iter().map(|&(lo, out)| (io + lo, oo + out)));
            }
            // In chunk order, so neighbouring runs merge into one read or one copy.
            pairs.sort_unstable_by_key(|&(coord, _)| coord);
            coords.extend(pairs.iter().map(|p| p.0));
            offsets.extend(pairs.iter().map(|p| p.1));
            self.items.push(ChunkItem {
                key: layout.key.clone(),
                chunk_subset: ArraySubset::new_with_ranges(&inner_box),
                subset: ArraySubset::new_with_shape(vec![0; chunk.len()]),
                shape: layout.shape.clone(),
                num_elements: layout.num_elements,
                array_shape: layout.array_shape.clone(),
                coords: Some(coords.into()),
                run_len,
                claimed_inner: layout.claimed_inner,
                out_offsets: Some(offsets.into()),
                out_box: None,
            });
        }
    }
}

impl ChunkItems {
    pub(crate) fn as_slice(&self) -> &[ChunkItem] {
        &self.items
    }

}
