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
    array::{ArraySubset, ravel_indices},
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

/// `(row_stride, run_len, elem_offset)` of a trailing sub-box, which must be one run per row.
fn trailing_layout(inner: &[u64], shape: &[u64], starts: &[u64]) -> PyResult<(u64, u64, u64)> {
    if inner.is_empty() || inner.len() != shape.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "chunk_unit_items splits axis 0 and needs matching arity: the inner chunk has \
             {} axes, the output shape has {}",
            inner.len(),
            shape.len()
        )));
    }
    if starts.len() + 1 != inner.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "chunk_unit_items needs one start per axis after the split: {} starts against a \
             rank-{} inner chunk",
            starts.len(),
            inner.len()
        )));
    }
    let extents = &inner[1..];
    let widths = &shape[1..];
    for (axis, ((start, width), extent)) in starts.iter().zip(widths).zip(extents).enumerate() {
        if *width == 0 || start.checked_add(*width).is_none_or(|end| end > *extent) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "axis {} takes {width} elements from {start}, which leaves its extent {extent}",
                axis + 1
            )));
        }
    }
    let box_ = ArraySubset::new_with_start_shape(starts.to_vec(), widths.to_vec())
        .map_py_err::<PyValueError>()?;
    let runs = box_
        .contiguous_indices(extents)
        .map_py_err::<PyValueError>()?;
    if runs.len() != 1 {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "selecting {widths:?} of {extents:?} is strided within one index, and an \
             item's output is a single contiguous range"
        )));
    }
    let row_stride: u64 = extents.iter().product();
    let run_len = runs.contiguous_elements();
    if run_len == 0 || row_stride == 0 {
        return Err(PyErr::new::<PyValueError, _>(
            "a trailing axis of extent zero selects nothing",
        ));
    }
    let elem_offset = ravel_indices(starts, extents).ok_or_else(|| {
        PyErr::new::<PyValueError, _>(format!(
            "a start of {starts:?} is outside the {extents:?} one index holds"
        ))
    })?;
    if elem_offset
        .checked_add(run_len)
        .is_none_or(|end| end > row_stride)
    {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "a run of {run_len} elements at offset {elem_offset} leaves the {row_stride} \
             elements one index holds"
        )));
    }
    Ok((row_stride, run_len, elem_offset))
}

/// One item per inner chunk for an entry whose `indices` select along axis 0.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_chunk_unit_items(
    key: &str,
    chunk_shape: Vec<u64>,
    shape: Vec<u64>,
    indices: PyReadonlyArray1<'_, i64>,
    out_starts: &[u64],
    out_widths: &[u64],
    inner: &[u64],
    // Shard-relative start of the band on each trailing axis.
    elem_starts: &[u64],
) -> PyResult<Vec<ChunkItem>> {
    if chunk_shape.is_empty() || inner.len() != chunk_shape.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "one inner extent per axis is needed, on a chunk of rank at least one: {} \
             against a rank-{} chunk",
            inner.len(),
            chunk_shape.len()
        )));
    }
    if out_starts.len() != shape.len() || out_widths.len() != shape.len() || shape.is_empty() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "one output start and width per axis is needed: {} and {} against a rank-{} \
             output",
            out_starts.len(),
            out_widths.len(),
            shape.len()
        )));
    }
    if inner.iter().zip(&chunk_shape).any(|(i, c)| i > c) {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "an inner chunk {inner:?} cannot be larger than the shard {chunk_shape:?} it divides"
        )));
    }
    if let Some(axis) = inner.iter().position(|e| *e == 0) {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "the inner chunk has extent zero on axis {axis}: {inner:?}"
        )));
    }
    let split = inner[0];
    let indices = indices.as_array();
    let n = indices.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let within: Vec<u64> = elem_starts
        .iter()
        .zip(&inner[1..])
        .map(|(start, extent)| start % extent)
        .collect();
    let (row_stride, run_len, offset) = trailing_layout(inner, out_widths, &within)?;
    let num_elements: u64 = chunk_shape.iter().product();
    let chunk_shape = to_nonzero_u64_vec(chunk_shape)?;
    let shape = to_nonzero_u64_vec(shape)?;
    let extent = chunk_shape[0].get();
    let out_extent = shape[0].get();
    let key = StoreKey::new(key.to_string()).map_py_err::<PyValueError>()?;

    let at = |i: usize| -> PyResult<u64> {
        u64::try_from(indices[i])
            .map_err(|_| PyErr::new::<PyValueError, _>(format!("index {} is negative", indices[i])))
    };

    let mut items = Vec::new();
    let mut a = 0usize;
    let mut previous = 0u64;
    while a < n {
        let first = at(a)?;
        if a > 0 && first < previous {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "indices must be non-decreasing: {first} follows {previous}"
            )));
        }
        previous = first;
        let chunk_id = first / split;
        let mut b = a + 1;
        while b < n {
            let value = at(b)?;
            if value < previous {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "indices must be non-decreasing: {value} follows {previous}"
                )));
            }
            previous = value;
            if value / split != chunk_id {
                break;
            }
            b += 1;
        }
        let lo = chunk_id * split;
        let hi = (lo + split).min(extent);
        if lo >= extent {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "index {} is past the chunk extent {extent}",
                at(a)?
            )));
        }
        // The last of a non-decreasing group is its largest.
        if at(b - 1)? >= extent {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "index {} is past the chunk extent {extent}",
                at(b - 1)?
            )));
        }
        let out_lo = out_starts[0] + a as u64;
        let out_hi = out_starts[0] + b as u64;
        if out_hi > out_extent {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "output subset {out_lo}..{out_hi} is past the output extent {out_extent}",
            )));
        }
        let mut chunk_ranges = Vec::with_capacity(chunk_shape.len());
        chunk_ranges.push(lo..hi);
        chunk_ranges.extend(
            elem_starts
                .iter()
                .zip(&out_widths[1..])
                .map(|(start, width)| *start..*start + *width),
        );
        let mut out_ranges = Vec::with_capacity(shape.len());
        out_ranges.push(out_lo..out_hi);
        out_ranges.extend(
            out_widths[1..]
                .iter()
                .zip(&out_starts[1..])
                .map(|(width, at)| *at..at + width),
        );
        items.push(ChunkItem {
            key: key.clone(),
            chunk_subset: ArraySubset::new_with_ranges(&chunk_ranges),
            subset: ArraySubset::new_with_ranges(&out_ranges),
            shape: chunk_shape.clone(),
            num_elements,
            array_shape: shape.clone(),
            coords: Some(
                (a..b)
                    .map(|i| at(i).map(|v| (v - lo) * row_stride + offset))
                    .collect::<PyResult<Vec<u64>>>()?
                    .into(),
            ),
            run_len,
            claimed_inner: (inner[0], row_stride),
        });
        a = b;
    }
    Ok(items)
}

/// A batch of read items, built and held in Rust.
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct ChunkItems {
    items: Vec<ChunkItem>,
    /// Where the last pushed output ended on axis 0.
    out_end: u64,
}

#[gen_stub_pymethods]
#[pymethods]
impl ChunkItems {
    #[new]
    pub(crate) fn new() -> Self {
        Self {
            items: Vec::new(),
            out_end: 0,
        }
    }

    /// The number of read items pushed so far.
    fn __len__(&self) -> usize {
        self.items.len()
    }

    /// Push one batch entry: sorted `indices` on axis 0 and a contiguous box on the others.
    ///
    /// `shape` must be the output buffer's real extent; it is not checked.
    #[pyo3(signature = (key, chunk_shape, shape, indices, out_starts, out_widths, inner, elem_starts=Vec::new()))]
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    pub(crate) fn push_entry(
        &mut self,
        key: &str,
        chunk_shape: Vec<u64>,
        shape: Vec<u64>,
        indices: PyReadonlyArray1<'_, i64>,
        out_starts: Vec<u64>,
        out_widths: Vec<u64>,
        inner: Vec<u64>,
        elem_starts: Vec<u64>,
    ) -> PyResult<()> {
        // Overlap is refused when the output is carved: bands of one entry share an axis-0 start.
        let items = build_chunk_unit_items(
            key,
            chunk_shape,
            shape,
            indices,
            &out_starts,
            &out_widths,
            &inner,
            &elem_starts,
        )?;
        self.extend_items(items);
        Ok(())
    }

    /// Push a contiguous span of the split axis, without naming its elements.
    #[pyo3(signature = (key, chunk_shape, shape, first, count, out_start, inner))]
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn push_span(
        &mut self,
        key: &str,
        chunk_shape: Vec<u64>,
        shape: Vec<u64>,
        first: u64,
        count: u64,
        out_start: u64,
        inner: u64,
    ) -> PyResult<()> {
        self.refuse_backwards(out_start)?;
        if count == 0 {
            return Ok(());
        }
        let inner = NonZeroU64::new(inner)
            .ok_or_else(|| PyErr::new::<PyValueError, _>("inner chunk shape must be non-zero"))?
            .get();
        if chunk_shape.is_empty() || chunk_shape.len() != shape.len() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "push_span splits axis 0 and needs matching arity: chunk_shape has {} axes, \
                 the output shape has {}",
                chunk_shape.len(),
                shape.len()
            )));
        }
        if chunk_shape[1..] != shape[1..] {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "push_span takes the trailing axes whole: chunk {:?} against output {:?}",
                &chunk_shape[1..],
                &shape[1..]
            )));
        }
        let row_stride: u64 = chunk_shape[1..].iter().product();
        if row_stride == 0 {
            return Err(PyErr::new::<PyValueError, _>(
                "a trailing axis of extent zero selects nothing",
            ));
        }
        let num_elements: u64 = chunk_shape.iter().product();
        let extent = chunk_shape[0];
        let out_extent = shape[0];
        let last = first
            .checked_add(count - 1)
            .ok_or_else(|| PyErr::new::<PyValueError, _>("the span is too large to address"))?;
        if last >= extent {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "index {last} is past the chunk extent {extent}"
            )));
        }
        let chunk_shape_nz = to_nonzero_u64_vec(chunk_shape.clone())?;
        let shape_nz = to_nonzero_u64_vec(shape.clone())?;
        let key = StoreKey::new(key.to_string()).map_py_err::<PyValueError>()?;

        for chunk_id in (first / inner)..=(last / inner) {
            let lo = chunk_id * inner;
            let hi = (lo + inner).min(extent);
            let span_lo = first.max(lo);
            let span_hi = (first + count).min(hi);
            let rows = span_hi - span_lo;
            let out_lo = out_start + (span_lo - first);
            let out_hi = out_lo + rows;
            if out_hi > out_extent {
                return Err(PyErr::new::<PyIndexError, _>(format!(
                    "output subset {out_lo}..{out_hi} is past the output extent {out_extent}",
                )));
            }
            let mut chunk_ranges = Vec::with_capacity(chunk_shape.len());
            chunk_ranges.push(lo..hi);
            chunk_ranges.extend(chunk_shape[1..].iter().map(|d| 0..*d));
            let mut out_ranges = Vec::with_capacity(shape.len());
            out_ranges.push(out_lo..out_hi);
            out_ranges.extend(shape[1..].iter().map(|d| 0..*d));
            self.items.push(ChunkItem {
                key: key.clone(),
                chunk_subset: ArraySubset::new_with_ranges(&chunk_ranges),
                subset: ArraySubset::new_with_ranges(&out_ranges),
                shape: chunk_shape_nz.clone(),
                num_elements,
                array_shape: shape_nz.clone(),
                coords: Some(vec![(span_lo - lo) * row_stride].into()),
                run_len: rows * row_stride,
                claimed_inner: (inner, row_stride),
            });
            self.out_end = out_hi;
        }
        Ok(())
    }

    /// Push the product of runs on every axis, each axis's runs back to back in the output.
    /// `shard_ids[k]` ascends and names the shards touched on axis `k`; `keys` are their
    /// product in C order.
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
        if rank == 0
            || [shard_ids.len(), starts.len(), lengths.len(), inner.len()] != [rank; 4]
        {
            return Err(PyErr::new::<PyValueError, _>(
                "one shard id array, run pair and inner extent per axis",
            ));
        }
        if shard_shape.iter().chain(&inner).any(|d| *d == 0) {
            return Err(PyErr::new::<PyValueError, _>(
                "shard and inner extents must be non-zero",
            ));
        }
        let u = |v: i64| {
            u64::try_from(v)
                .map_err(|_| PyErr::new::<PyValueError, _>(format!("negative run field {v}")))
        };
        // Per axis, `(shard position, shard-local lo..hi, output start)` of each piece that
        // stays within one inner chunk.
        let mut axes: Vec<Vec<(usize, u64, u64, u64)>> = Vec::with_capacity(rank);
        let mut out_shape = Vec::with_capacity(rank);
        let mut id_counts = Vec::with_capacity(rank);
        for k in 0..rank {
            let ids = shard_ids[k].as_slice().map_err(|_| {
                PyErr::new::<PyValueError, _>("the shard id arrays must be contiguous")
            })?;
            let (s_arr, n_arr) = (starts[k].as_array(), lengths[k].as_array());
            if s_arr.len() != n_arr.len() {
                return Err(PyErr::new::<PyValueError, _>("one length per start"));
            }
            let (shard, split) = (shard_shape[k], inner[k]);
            let mut pieces = Vec::new();
            let mut out = 0u64;
            for (s, n) in s_arr.iter().zip(n_arr.iter()) {
                let mut s = u(*s)?;
                let end = s.checked_add(u(*n)?).ok_or_else(|| {
                    PyErr::new::<PyValueError, _>("a run is too long to address")
                })?;
                while s < end {
                    let (id, lo) = (s / shard, s % shard);
                    let hi = ((lo / split + 1) * split).min(shard).min(lo + (end - s));
                    let pos = i64::try_from(id)
                        .ok()
                        .and_then(|id| ids.binary_search(&id).ok())
                        .ok_or_else(|| {
                            PyErr::new::<PyIndexError, _>(format!(
                                "no key given for shard {id} of axis {k}"
                            ))
                        })?;
                    pieces.push((pos, lo, hi, out));
                    (out, s) = (out + hi - lo, s + hi - lo);
                }
            }
            axes.push(pieces);
            out_shape.push(out);
            id_counts.push(ids.len());
        }
        if keys.len() != id_counts.iter().product::<usize>() {
            return Err(PyErr::new::<PyValueError, _>(
                "one key per shard in the product of the shard ids",
            ));
        }
        if axes.iter().any(Vec::is_empty) {
            return Ok(());
        }
        let keys = keys
            .into_iter()
            .map(|k| StoreKey::new(k).map_py_err::<PyValueError>())
            .collect::<PyResult<Vec<_>>>()?;
        let num_elements = shard_shape.iter().product();
        let claimed_inner = (inner[0], inner[1..].iter().product());
        let array_shape = to_nonzero_u64_vec(out_shape)?;
        let shape = to_nonzero_u64_vec(shard_shape)?;
        // An odometer over the product, last axis fastest, so items come in output order.
        let mut at = vec![0usize; rank];
        loop {
            let pieces: Vec<_> = at.iter().zip(&axes).map(|(i, a)| a[*i]).collect();
            let key = pieces
                .iter()
                .zip(&id_counts)
                .fold(0, |acc, (p, count)| acc * count + p.0);
            let within = ArraySubset::new_with_start_shape(
                pieces.iter().zip(&inner).map(|p| p.0.1 % p.1).collect(),
                pieces.iter().map(|p| p.2 - p.1).collect(),
            )
            .map_py_err::<PyValueError>()?;
            let runs = within
                .contiguous_linearised_indices(&inner)
                .map_py_err::<PyValueError>()?;
            let chunk_ranges: Vec<_> = pieces.iter().map(|p| p.1..p.2).collect();
            let out_ranges: Vec<_> = pieces.iter().map(|p| p.3..p.3 + p.2 - p.1).collect();
            self.items.push(ChunkItem {
                key: keys[key].clone(),
                chunk_subset: ArraySubset::new_with_ranges(&chunk_ranges),
                subset: ArraySubset::new_with_ranges(&out_ranges),
                shape: shape.clone(),
                num_elements,
                array_shape: array_shape.clone(),
                coords: Some(runs.iter().map(|(i, _)| i).collect()),
                run_len: runs.contiguous_elements(),
                claimed_inner,
            });
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

impl ChunkItems {
    pub(crate) fn as_slice(&self) -> &[ChunkItem] {
        &self.items
    }

    fn refuse_backwards(&self, out_start: u64) -> PyResult<()> {
        if out_start < self.out_end {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "output starting at {out_start} overlaps an entry already pushed, which ends \
                 at {}",
                self.out_end
            )));
        }
        Ok(())
    }

    fn extend_items(&mut self, items: Vec<ChunkItem>) {
        if let Some(last) = items.last() {
            self.out_end = last.subset.end_exc()[0];
        }
        self.items.extend(items);
    }
}
