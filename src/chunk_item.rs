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
use zarrs::{array::ArraySubset, storage::StoreKey};

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
    /// Indices within `chunk_subset`, when this item is a whole inner chunk plus the
    /// elements wanted from it. The chunk is decoded once and these are gathered out.
    pub coords: Option<Arc<[u64]>>,
    /// How many CONSECUTIVE elements each coordinate stands for. 1 on the 1-D path, where a
    /// coordinate is one element. On a rank-N selection whose split axis is 0 and whose
    /// trailing axes are taken whole, one coordinate is the start of a whole row, and this
    /// is the row's length -- so `gather` moves a row per coordinate instead of an element,
    /// which is the only reason grouping a wide 2-D read is worth doing.
    pub run_len: u64,
}

#[gen_stub_pymethods]
#[pymethods]
impl ChunkItem {
    /// `coords` is always `None` here -- it is not a parameter, so this constructor cannot
    /// build a chunk-unit item. `ChunkItems::push_entry` builds those.
    #[new]
    #[pyo3(signature = (key, chunk_subset, chunk_shape, subset, shape))]
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

/// Where each selected index's run begins inside that index's own elements.
#[derive(Clone, Copy)]
pub(crate) enum Offsets<'a> {
    /// Per-axis STARTS of a sub-box shared by every index: `X[rows, 8:24]`. The offset is
    /// derived from these by `trailing_layout`, which is also what checks the box is one run.
    Uniform(&'a [u64]),
    /// One per index: a POINT selection, `X[rows, cols]`, where each point names its own
    /// element. The run is then a single element and the output is flat, so the grouping by
    /// inner chunk is the whole win -- the ordinary route costs a partial-decode call PER
    /// POINT.
    PerIndex(&'a [u64]),
}

impl Offsets<'_> {
    /// How many indices this describes, when it describes a fixed number.
    fn len(self) -> Option<usize> {
        match self {
            Self::Uniform(_) => None,
            Self::PerIndex(offsets) => Some(offsets.len()),
        }
    }
}

/// `(row_stride, run_len, elem_offset)` for a selection whose trailing axes may be PARTIAL.
///
/// `row_stride` is one index's worth of the decoded chunk -- the product of every axis after
/// the split -- and is how far apart two selected indices sit in it. `run_len` is how many
/// elements are actually copied out, the product of the output's trailing axes. They are
/// equal when the trailing axes are taken whole, which was once the only case this served.
///
/// The offset is DERIVED here from the per-axis starts rather than accepted as one fused
/// number, because a fused offset cannot be checked. Given only `offset + run_len <=
/// row_stride`, a rank-3 box of 2-of-4 rows by 5-of-10 columns passes -- and `gather` then
/// copies 10 CONSECUTIVE elements which are read back as a 2x5 tile. Wrong data, no error.
/// With the starts the shape is checkable, and so is the wrap: an offset of 8 with width 4 on
/// an axis of extent 10 runs off the end of its own sub-row into the next one.
fn trailing_layout(chunk_shape: &[u64], shape: &[u64], starts: &[u64]) -> PyResult<(u64, u64, u64)> {
    if chunk_shape.is_empty() || chunk_shape.len() != shape.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "chunk_unit_items splits axis 0 and needs matching arity: chunk_shape has {} \
             axes, the output shape has {}",
            chunk_shape.len(),
            shape.len()
        )));
    }
    if starts.len() + 1 != chunk_shape.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "chunk_unit_items needs one start per axis AFTER the split: {} starts against a \
             rank-{} chunk",
            starts.len(),
            chunk_shape.len()
        )));
    }
    let extents = &chunk_shape[1..];
    let widths = &shape[1..];
    for (axis, ((start, width), extent)) in starts.iter().zip(widths).zip(extents).enumerate() {
        if *width == 0 || start.checked_add(*width).is_none_or(|end| end > *extent) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "axis {} takes {width} elements from {start}, which leaves its extent {extent}",
                axis + 1
            )));
        }
    }
    // Row-major, a sub-box is ONE run exactly when every axis before the last PARTIAL one
    // takes a single element. Anything else repeats a short run at a stride, and an item's
    // output is vended as a single range, which cannot express that.
    if let Some(last) = widths
        .iter()
        .zip(extents)
        .rposition(|(width, extent)| width != extent)
    {
        if widths[..last].iter().any(|width| *width != 1) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "selecting {widths:?} of {extents:?} is strided within one index, and an \
                 item's output is a single contiguous range"
            )));
        }
    }
    let row_stride: u64 = extents.iter().product();
    let run_len: u64 = widths.iter().product();
    if run_len == 0 || row_stride == 0 {
        return Err(PyErr::new::<PyValueError, _>(
            "a trailing axis of extent zero selects nothing",
        ));
    }
    let mut elem_offset = 0u64;
    let mut stride = 1u64;
    for axis in (0..starts.len()).rev() {
        elem_offset += starts[axis] * stride;
        stride *= extents[axis];
    }
    // The per-axis checks above already imply this. Kept because `gather` only knows the
    // whole decoded buffer's length, so a run walking into the NEXT index's elements would
    // return them under this index's name rather than fail a bounds check.
    if elem_offset.checked_add(run_len).is_none_or(|end| end > row_stride) {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "a run of {run_len} elements at offset {elem_offset} leaves the {row_stride} \
             elements one index holds"
        )));
    }
    Ok((row_stride, run_len, elem_offset))
}

/// Build one item per inner chunk for a whole entry.
///
/// `indices` selects along AXIS 0 and must be non-negative and non-decreasing. Both are
/// rechecked here: a negative index becomes a wild chunk id, and `inner == 0` divides by
/// zero.
///
/// Axes after the first may be taken whole or as a CONTIGUOUS sub-box, described by
/// `elem_starts` against the output's trailing extents -- so one selected index is still one
/// contiguous run, of `run_len` elements starting `elem_offset` into that index. That restriction is what
/// lets `locate` keep descending on axis 0 alone: with a single subchunk on every other
/// axis, the raveled chunk-grid index IS the axis-0 index. `locate` rechecks it per level.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_chunk_unit_items(
    key: &str,
    chunk_shape: Vec<u64>,
    shape: Vec<u64>,
    indices: PyReadonlyArray1<'_, i64>,
    out_start: u64,
    inner: u64,
    offsets: Offsets<'_>,
) -> PyResult<Vec<ChunkItem>> {
    let inner = NonZeroU64::new(inner)
        .ok_or_else(|| PyErr::new::<PyValueError, _>("inner chunk shape must be non-zero"))?
        .get();
    // Strided views are legal here: an index array can be a slice of a larger one.
    let indices = indices.as_array();
    let n = indices.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let (row_stride, run_len, uniform_offset) = match offsets {
        Offsets::Uniform(starts) => trailing_layout(&chunk_shape, &shape, starts)?,
        // A point names ONE element, so the run is one element and the output is flat. The
        // trailing extents are not a shared sub-box here -- each point carries its own offset
        // -- so only the stride comes from the chunk.
        Offsets::PerIndex(_) => (chunk_shape[1..].iter().product::<u64>(), 1, 0),
    };
    if row_stride == 0 {
        return Err(PyErr::new::<PyValueError, _>(
            "a trailing axis of extent zero selects nothing",
        ));
    }
    if let Some(given) = offsets.len() {
        if given != n {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "a point selection needs one offset per index: {given} against {n}"
            )));
        }
    }
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

    // NON-DECREASING is assumed below: the grouping walks a RUN of equal chunk ids, so out of
    // order the same chunk is grouped twice, and the extent check trusts the last of a group
    // to be its largest. `push_entry` is `#[pymethods]` and takes an arbitrary array, so it is
    // enforced here as well as by Python's `_is_sorted_integer_axis`. A violation that slipped
    // through would be caught downstream by `gather` rather than returning wrong data -- but
    // as an out-of-range coordinate, which says nothing about which index was bad.
    // Checked inside the walk, not in a pass of its own -- a separate pass cost 12%.
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
        let chunk_id = first / inner;
        let mut b = a + 1;
        while b < n {
            let value = at(b)?;
            if value < previous {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "indices must be non-decreasing: {value} follows {previous}"
                )));
            }
            previous = value;
            if value / inner != chunk_id {
                break;
            }
            b += 1;
        }
        let lo = chunk_id * inner;
        // Exactly one inner chunk, clamped to the extent.
        let hi = (lo + inner).min(extent);
        if lo >= extent {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "index {} is past the chunk extent {extent}",
                at(a)?
            )));
        }
        // `lo >= extent` only catches a chunk that STARTS past the end. An index inside the
        // last chunk but past the extent would gather fill bytes, so check it too. Indices
        // are non-decreasing, so the last of the group is the largest.
        if at(b - 1)? >= extent {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "index {} is past the chunk extent {extent}",
                at(b - 1)?
            )));
        }
        let out_lo = out_start + a as u64;
        let out_hi = out_start + b as u64;
        if out_hi > out_extent {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "output subset {out_lo}..{out_hi} is past the output extent {out_extent}",
            )));
        }
        // Axis 0 is the split; every axis after it is taken whole on both sides.
        let mut chunk_ranges = Vec::with_capacity(chunk_shape.len());
        chunk_ranges.push(lo..hi);
        chunk_ranges.extend(chunk_shape[1..].iter().map(|d| 0..d.get()));
        let mut out_ranges = Vec::with_capacity(shape.len());
        out_ranges.push(out_lo..out_hi);
        out_ranges.extend(shape[1..].iter().map(|d| 0..d.get()));
        items.push(ChunkItem {
            key: key.clone(),
            chunk_subset: ArraySubset::new_with_ranges(&chunk_ranges),
            subset: ArraySubset::new_with_ranges(&out_ranges),
            shape: chunk_shape.clone(),
            num_elements,
            array_shape: shape.clone(),
            // Relative to the chunk subset, because that is the buffer gathered from.
            // Scaled by `row_stride`, which is one index's worth of THAT buffer, then
            // stepped by `elem_offset` to where this selection starts inside the row. With
            // the trailing axes whole the offset is 0 and the stride is the run.
            coords: Some(
                (a..b)
                    .map(|i| {
                        let offset = match offsets {
                            Offsets::Uniform(_) => uniform_offset,
                            Offsets::PerIndex(per) => per[i],
                        };
                        // Every offset, not just a shared one, must leave the index's own
                        // elements: `gather` only knows the whole decoded buffer's length, so
                        // a point past its row would return the NEXT row's element under this
                        // point's name.
                        if offset.saturating_add(run_len) > row_stride {
                            return Err(PyErr::new::<PyValueError, _>(format!(
                                "offset {offset} leaves the {row_stride} elements one index \
                                 holds"
                            )));
                        }
                        at(i).map(|v| (v - lo) * row_stride + offset)
                    })
                    .collect::<PyResult<Vec<u64>>>()?
                    .into(),
            ),
            run_len,
        });
        a = b;
    }
    Ok(items)
}

/// A batch of chunk items, built and held in Rust.
///
/// Push one entry at a time, then pass the handle to
/// `retrieve_chunk_items_and_apply_index`.
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct ChunkItems {
    items: Vec<ChunkItem>,
    /// Where the last entry's output ended, so a later one cannot overlap it. Python drives
    /// `push_entry` directly, and two entries sharing an `out_start` would give two items
    /// overlapping output ranges -- which the read path writes CONCURRENTLY through views
    /// whose safety contract is that they are disjoint. A wrong answer would be recoverable;
    /// this would be a data race.
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

    fn __len__(&self) -> usize {
        self.items.len()
    }

    /// Build one batch entry's items and append them.
    ///
    /// `indices` select along AXIS 0 and are checked here: non-negative, non-decreasing, and
    /// inside the chunk extent. So is `out_start` -- entries must be pushed in increasing
    /// order, and one that would reuse output another entry already owns is refused.
    ///
    /// Axes after the first are taken WHOLE and must be the same extent in `chunk_shape` and
    /// in `shape`; that is checked too. It is what makes one index one contiguous run, and
    /// the rank-N case the 1-D case with a run length.
    ///
    /// One obligation this CANNOT check: `shape` must be the real extent of the output buffer,
    /// since the output subset is bounded against it. A larger one describes bytes the buffer
    /// does not have, and that produces wrong data rather than an error.
    #[pyo3(signature = (key, chunk_shape, shape, indices, out_start, inner, elem_starts=Vec::new()))]
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn push_entry(
        &mut self,
        key: &str,
        chunk_shape: Vec<u64>,
        shape: Vec<u64>,
        indices: PyReadonlyArray1<'_, i64>,
        out_start: u64,
        inner: u64,
        elem_starts: Vec<u64>,
    ) -> PyResult<()> {
        if out_start < self.out_end {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "output starting at {out_start} overlaps an entry already pushed, which ends \
                 at {}",
                self.out_end
            )));
        }
        let items = build_chunk_unit_items(key, chunk_shape, shape, indices, out_start, inner, Offsets::Uniform(&elem_starts))?;
        if let Some(last) = items.last() {
            self.out_end = last.subset.end_exc()[0];
        }
        self.items.extend(items);
        Ok(())
    }

    /// Push a POINT selection: one element per index, each naming its own offset inside that
    /// index's elements.
    ///
    /// `X[rows, cols]` and `X[rows, 5]` both arrive as this -- zarr builds a
    /// `CoordinateIndexer` rather than dropping an axis -- and the ordinary route spends two
    /// allocations and a partial-decode call PER POINT. Grouping them by the chunk that gets
    /// decoded is the whole win, and it is the same grouping the row case uses: the output is
    /// flat, so `shape` is 1-D here while `chunk_shape` is not.
    #[pyo3(signature = (key, chunk_shape, shape, indices, offsets, out_start, inner))]
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn push_points(
        &mut self,
        key: &str,
        chunk_shape: Vec<u64>,
        shape: Vec<u64>,
        indices: PyReadonlyArray1<'_, i64>,
        offsets: PyReadonlyArray1<'_, u64>,
        out_start: u64,
        inner: u64,
    ) -> PyResult<()> {
        if out_start < self.out_end {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "output starting at {out_start} overlaps an entry already pushed, which ends \
                 at {}",
                self.out_end
            )));
        }
        // Contiguous so the per-point offsets can be read as a slice; a strided view would be
        // indexed as if it were dense.
        let offsets = offsets.as_slice().map_err(|_| {
            PyErr::new::<PyValueError, _>("the offsets array must be contiguous")
        })?;
        let items = build_chunk_unit_items(
            key,
            chunk_shape,
            shape,
            indices,
            out_start,
            inner,
            Offsets::PerIndex(offsets),
        )?;
        if let Some(last) = items.last() {
            self.out_end = last.subset.end_exc()[0];
        }
        self.items.extend(items);
        Ok(())
    }
}

impl ChunkItems {
    pub(crate) fn as_slice(&self) -> &[ChunkItem] {
        &self.items
    }
}
