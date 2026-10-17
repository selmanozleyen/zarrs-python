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
        let layout = BoxLayout::new(key, shard_shape, inner, out_shape)?;
        self.push_shard(&layout, &axes.iter().collect::<Vec<_>>());
        Ok(())
    }

    /// Push the product of runs on every axis, each axis's runs back to back in the output,
    /// across every shard they touch, in one call. `shard_ids[k]` ascends and names the shards
    /// touched on axis `k`; `keys` are their product in C order.
    #[pyo3(signature = (keys, shard_ids, starts, lengths, shard_shape, inner))]
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn push_runs(
        &mut self,
        keys: Vec<String>,
        shard_ids: Vec<PyReadonlyArray1<'_, i64>>,
        starts: Vec<PyReadonlyArray1<'_, i64>>,
        lengths: Vec<PyReadonlyArray1<'_, i64>>,
        shard_shape: Vec<u64>,
        inner: Vec<u64>,
    ) -> PyResult<()> {
        let rank = shard_shape.len();
        if rank == 0 || [shard_ids.len(), starts.len(), lengths.len(), inner.len()] != [rank; 4] {
            return Err(PyErr::new::<PyValueError, _>(
                "one shard id array, run pair and inner extent per axis",
            ));
        }
        if shard_shape.iter().chain(&inner).any(|d| *d == 0) {
            return Err(PyErr::new::<PyValueError, _>(
                "shard and inner extents must be non-zero",
            ));
        }
        // Per axis: each shard touched (its position among `shard_ids`) and its inner pieces.
        let mut axes: Vec<Vec<(usize, ChunkPieces)>> = Vec::with_capacity(rank);
        let mut out_shape = Vec::with_capacity(rank);
        for k in 0..rank {
            let ids = shard_ids[k].as_slice().map_err(|_| {
                PyErr::new::<PyValueError, _>("the shard id arrays must be contiguous")
            })?;
            let (s_arr, n_arr) = (starts[k].as_array(), lengths[k].as_array());
            if s_arr.len() != n_arr.len() {
                return Err(PyErr::new::<PyValueError, _>("one length per start"));
            }
            let shard = shard_shape[k];
            let mut by_shard: std::collections::BTreeMap<u64, Vec<(u64, u64, u64)>> =
                std::collections::BTreeMap::new();
            let mut out = 0u64;
            for (s, n) in s_arr.iter().zip(n_arr.iter()) {
                let (mut at, n) = (non_negative(*s)?, non_negative(*n)?);
                let end = at.checked_add(n).ok_or_else(|| {
                    PyErr::new::<PyValueError, _>("a run is too long to address")
                })?;
                // Split at shard boundaries; each piece keeps its place in the output.
                while at < end {
                    let id = at / shard;
                    let hi = ((id + 1) * shard).min(end);
                    by_shard.entry(id).or_default().push((at - id * shard, hi - at, out));
                    (out, at) = (out + hi - at, hi);
                }
            }
            out_shape.push(out);
            let mut per_shard = Vec::with_capacity(by_shard.len());
            for (id, runs) in by_shard {
                let pos = i64::try_from(id)
                    .ok()
                    .and_then(|id| ids.binary_search(&id).ok())
                    .ok_or_else(|| {
                        PyErr::new::<PyIndexError, _>(format!(
                            "no key given for shard {id} of axis {k}"
                        ))
                    })?;
                per_shard.push((pos, split_runs(k, runs, shard, inner[k], u64::MAX)?));
            }
            axes.push(per_shard);
        }
        let id_counts: Vec<usize> = shard_ids.iter().map(|ids| ids.as_array().len()).collect();
        if keys.len() != id_counts.iter().product::<usize>() {
            return Err(PyErr::new::<PyValueError, _>(
                "one key per shard in the product of the shard ids",
            ));
        }
        if axes.iter().any(Vec::is_empty) {
            return Ok(());
        }
        // An odometer over the shards touched; each is one box of the selection.
        let mut at = vec![0usize; rank];
        loop {
            let shard: Vec<&(usize, ChunkPieces)> =
                at.iter().zip(&axes).map(|(i, a)| &a[*i]).collect();
            let key = shard.iter().zip(&id_counts).fold(0, |acc, ((p, _), c)| acc * c + p);
            let layout =
                BoxLayout::new(&keys[key], shard_shape.clone(), inner.clone(), out_shape.clone())?;
            let pieces: Vec<&ChunkPieces> = shard.iter().map(|(_, p)| p).collect();
            self.push_shard(&layout, &pieces);
            if !advance(&mut at, &axes.iter().map(Vec::len).collect::<Vec<_>>()) {
                return Ok(());
            }
        }
    }
}

/// Step an odometer over `lens` (last axis fastest); false once it has wrapped.
fn advance(at: &mut [usize], lens: &[usize]) -> bool {
    for k in (0..at.len()).rev() {
        at[k] += 1;
        if at[k] < lens[k] {
            return true;
        }
        at[k] = 0;
    }
    false
}

fn non_negative(v: i64) -> PyResult<u64> {
    u64::try_from(v).map_err(|_| PyErr::new::<PyValueError, _>(format!("negative run field {v}")))
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

impl BoxLayout {
    fn new(key: &str, shard_shape: Vec<u64>, inner: Vec<u64>, out_shape: Vec<u64>) -> PyResult<Self> {
        Ok(Self {
            key: StoreKey::new(key.to_string()).map_py_err::<PyValueError>()?,
            in_strides: c_strides(&inner),
            out_strides: c_strides(&out_shape),
            num_elements: shard_shape.iter().product(),
            claimed_inner: (inner[0], inner[1..].iter().product()),
            array_shape: to_nonzero_u64_vec(out_shape)?,
            shape: to_nonzero_u64_vec(shard_shape.clone())?,
            shard_shape,
            inner,
        })
    }
}

/// C-order element strides of `shape`.
fn c_strides(shape: &[u64]) -> Vec<u64> {
    let mut strides = vec![1u64; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    strides
}

/// One axis's runs from numpy `(lo, len, out)` arrays; see `split_runs`.
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
    let runs = los
        .iter()
        .zip(lens.iter())
        .zip(outs.iter())
        .map(|((l, n), o)| Ok((non_negative(*l)?, non_negative(*n)?, non_negative(*o)?)))
        .collect::<PyResult<Vec<_>>>()?;
    split_runs(axis, runs, shard, split, out_extent)
}

/// One axis's shard-local runs `(lo, len, out)`, checked against the shard and the output, and
/// split at inner chunk boundaries: each inner chunk touched, in order, with its pieces.
fn split_runs(
    axis: usize,
    runs: Vec<(u64, u64, u64)>,
    shard: u64,
    split: u64,
    out_extent: u64,
) -> PyResult<ChunkPieces> {
    let mut by_chunk: std::collections::BTreeMap<u64, Vec<Piece>> =
        std::collections::BTreeMap::new();
    for (mut at, count, to) in runs {
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
    /// Every inner chunk of one shard's box: the product of each axis's chunks touched.
    fn push_shard(&mut self, layout: &BoxLayout, axes: &[&ChunkPieces]) {
        let lens: Vec<usize> = axes.iter().map(|a| a.len()).collect();
        let mut at = vec![0usize; axes.len()];
        loop {
            let chunk: Vec<&(u64, Vec<Piece>)> =
                at.iter().zip(axes).map(|(i, a)| &a[*i]).collect();
            self.push_inner_chunk(layout, &chunk);
            if !advance(&mut at, &lens) {
                return;
            }
        }
    }

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
