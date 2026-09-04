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
    /// Indices within `chunk_subset`, when this item is a whole inner chunk plus the
    /// elements wanted from it. The chunk is decoded once and these are gathered out.
    pub coords: Option<Arc<[u64]>>,
    /// How many CONSECUTIVE elements each coordinate stands for. 1 on the 1-D path, where a
    /// coordinate is one element. On a rank-N selection whose split axis is 0 and whose
    /// trailing axes are taken whole, one coordinate is the start of a whole row, and this
    /// is the row's length -- so `gather` moves a row per coordinate instead of an element,
    /// which is the only reason grouping a wide 2-D read is worth doing.
    pub run_len: u64,
    /// Where each RUN starts inside a coordinate's own elements, and how long a run is, when
    /// the wanted elements are not one consecutive span: `oindex[rows, cols]` takes the same
    /// sub-box out of every selected row, so one shared description serves the whole item.
    /// `None` means a single contiguous run, which is every other case.
    pub grid: Option<(Arc<[u64]>, u64)>,
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
            grid: None,
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
    /// The same sub-box taken out of EVERY index: `oindex[rows, cols]`, and any rank-N grid.
    Grid { starts: &'a [u64], run: u64 },
}

impl Offsets<'_> {
    /// How many indices this describes, when it describes a fixed number.
    fn len(self) -> Option<usize> {
        match self {
            Self::Uniform(_) | Self::Grid { .. } => None,
            Self::PerIndex(offsets) => Some(offsets.len()),
        }
    }
}

/// `(row_stride, run_len, elem_offset)` for a selection whose trailing axes may be PARTIAL.
///
/// The offset is DERIVED here from the per-axis starts rather than accepted as one fused
/// number, because a fused offset cannot be checked. Given only `offset + run_len <=
/// row_stride`, a rank-3 box of 2-of-4 rows by 5-of-10 columns passes -- and `gather` then
/// copies 10 CONSECUTIVE elements which are read back as a 2x5 tile. Wrong data, no error.
/// With the starts the shape is checkable, and so is the wrap: an offset of 8 with width 4 on
/// an axis of extent 10 runs off the end of its own sub-row into the next one.
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
            "chunk_unit_items needs one start per axis AFTER the split: {} starts against a \
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
    // Row-major, a sub-box is ONE run exactly when every axis before the last PARTIAL one
    // takes a single element -- and that rule is zarrs', not ours. `ContiguousIndices` walks
    // the axes in reverse absorbing whole ones, so `len() == 1` IS that condition and
    // `contiguous_elements()` is the run it absorbed. Asked here rather than restated, so the
    // rule has ONE definition: `output_pieces` asks the linearised form of the same thing, and
    // this was a fourth copy of it.
    let box_ = ArraySubset::new_with_start_shape(starts.to_vec(), widths.to_vec())
        .map_py_err::<PyValueError>()?;
    let runs = box_.contiguous_indices(extents).map_py_err::<PyValueError>()?;
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
    // The same fold, and zarrs bounds-checks each index against its extent on the way.
    let elem_offset = ravel_indices(starts, extents).ok_or_else(|| {
        PyErr::new::<PyValueError, _>(format!(
            "a start of {starts:?} is outside the {extents:?} one index holds"
        ))
    })?;
    // The per-axis checks above already imply this. Kept because `gather` only knows the
    // whole decoded buffer's length, so a run walking into the NEXT index's elements would
    // return them under this index's name rather than fail a bounds check.
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

/// `out_start` on axis 0 and zero after it.
fn trailing_zeros(out_start: u64, rank: usize) -> Vec<u64> {
    let mut starts = vec![0u64; rank];
    if let Some(first) = starts.first_mut() {
        *first = out_start;
    }
    starts
}

/// Build one item per inner chunk for a whole entry.
///
/// `indices` selects along AXIS 0 and must be non-negative and non-decreasing. Both are
/// rechecked here: a negative index becomes a wild chunk id, and `inner == 0` divides by
/// zero.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_chunk_unit_items(
    key: &str,
    chunk_shape: Vec<u64>,
    shape: Vec<u64>,
    indices: PyReadonlyArray1<'_, i64>,
    // Where this entry's output begins on EVERY axis. Axis 0 is the split; the rest place the
    // entry's band within the output. They were implicitly zero, which is only right when an
    // entry spans the whole trailing extent -- false as soon as the shard grid divides it.
    out_starts: &[u64],
    // The item's own extent on the trailing axes. Separate from `shape`, which stays the FULL
    // output shape: `shape` gives the row stride an output offset is computed against, and
    // these give how much of a row this entry actually fills. They are equal only while an
    // entry spans the whole trailing extent, which stops being true once the shard grid
    // divides it.
    out_widths: &[u64],
    // The INNER chunk -- the unit of compression, therefore of decoding, therefore the buffer
    // every coordinate below addresses. One extent per axis, not just the split: a shard may
    // hold several inner chunks on a trailing axis, and then the shard's extent is not the row
    // stride of anything that ever gets decoded.
    inner: &[u64],
    offsets: Offsets<'_>,
) -> PyResult<Vec<ChunkItem>> {
    // Arity FIRST, and rank at least one, because everything below indexes: `inner_nz[0]`
    // for the split extent and `out_widths[1..]` for the bands. `push_entry` is a pymethod
    // taking arbitrary vectors, so a short one has to be an error rather than a panic across
    // the FFI. The out_starts/out_widths check that used to sit further down is folded in
    // here for the same reason.
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
    // Rejects a zero extent on any axis, which is what the scalar check used to do for one.
    // Checked in place rather than through `to_nonzero_u64_vec`: that allocates a Vec, and
    // `inner.to_vec()` allocates another, both per ENTRY. A scattered batch pushes thousands
    // of entries per call, and the only thing wanted out of them is `inner[0]`.
    if let Some(axis) = inner.iter().position(|e| *e == 0) {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "the inner chunk has extent zero on axis {axis}: {inner:?}"
        )));
    }
    // What axis 0 groups by. Named apart from `inner` so the two cannot be confused: one is a
    // scalar extent on the split axis, the other the whole decode unit.
    let split = inner[0];
    // Strided views are legal here: an index array can be a slice of a larger one.
    let indices = indices.as_array();
    let n = indices.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let (row_stride, run_len, uniform_offset) = match offsets {
        // The band's position INSIDE its own inner chunk. `trailing_layout` then refuses any
        // band that leaves it -- the same check that already refuses a run walking off its
        // sub-row, reused rather than re-invented.
        Offsets::Uniform(starts) => {
            let within: Vec<u64> = starts
                .iter()
                .zip(&inner[1..])
                .map(|(start, extent)| start % extent)
                .collect();
            trailing_layout(inner, out_widths, &within)?
        }
        // A point names ONE element, so the run is one element and the output is flat. The
        // trailing extents are not a shared sub-box here -- each point carries its own offset
        // -- so only the stride comes from the chunk.
        Offsets::PerIndex(_) => (inner[1..].iter().product::<u64>(), 1, 0),
        // A grid takes the same `cols` from every row, so the run is the list's length and
        // the coordinate itself is the start of the row: the offsets are applied per element
        // by the gather, not folded into the coordinate.
        Offsets::Grid { starts, run } => {
            let stride: u64 = inner[1..].iter().product();
            for &c in starts {
                // The END of the run has to fit, not just its start.
                if c.checked_add(run).is_none_or(|end| end > stride) {
                    return Err(PyErr::new::<PyValueError, _>(format!(
                        "a run of {run} at {c} leaves the {stride} elements one index holds"
                    )));
                }
            }
            let Some(total) = (starts.len() as u64).checked_mul(run) else {
                return Err(PyErr::new::<PyValueError, _>(
                    "the grid is too large to address",
                ));
            };
            (stride, total, 0)
        }
    };
    if row_stride == 0 {
        return Err(PyErr::new::<PyValueError, _>(
            "a trailing axis of extent zero selects nothing",
        ));
    }
    // (start, width) per trailing axis, SHARD-relative -- what the descent divides.
    let trailing: Vec<(u64, u64)> = match offsets {
        Offsets::Uniform(starts) => starts
            .iter()
            .zip(&out_widths[1..])
            .map(|(start, width)| (*start, *width))
            .collect(),
        // Points and grids take the trailing axes whole, and their Python gates require one
        // inner chunk there. If that ever stops being true `locate` refuses the item rather
        // than returning wrong data.
        _ => chunk_shape[1..].iter().map(|d| (0, *d)).collect(),
    };
    // Constant for every index in the two shared cases; unused in the varying one.
    let shared_offset = match offsets {
        Offsets::Uniform(_) => uniform_offset,
        // Applied per element by the gather, not folded into the coordinate.
        Offsets::Grid { .. } | Offsets::PerIndex(_) => 0,
    };
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
        // Exactly one inner chunk, clamped to the extent.
        let hi = (lo + split).min(extent);
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
        let out_lo = out_starts[0] + a as u64;
        let out_hi = out_starts[0] + b as u64;
        if out_hi > out_extent {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "output subset {out_lo}..{out_hi} is past the output extent {out_extent}",
            )));
        }
        // Axis 0 is the split. Every axis after it takes the band this entry describes --
        // `locate` divides `chunk_subset.start()` on EVERY axis, so this is what steers the
        // descent to the right inner chunk on a trailing one. Pinned to `0..extent` it always
        // landed on inner chunk 0, which is correct only while a shard holds exactly one.
        let mut chunk_ranges = Vec::with_capacity(chunk_shape.len());
        chunk_ranges.push(lo..hi);
        chunk_ranges.extend(
            trailing
                .iter()
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
            // Relative to the chunk subset, because that is the buffer gathered from,
            // scaled by `row_stride` -- one index's worth of THAT buffer -- and stepped by
            // the offset to where this selection starts inside the row. With the trailing
            // axes whole the offset is 0 and the stride is the run.
            //
            // The shared cases step by a CONSTANT, so the offset lookup and its bounds check
            // are hoisted out: this closure runs once per selected index, thousands of times
            // a call, and doing loop-invariant work inside it cost 6% on the loader.
            coords: Some(match offsets {
                Offsets::PerIndex(per) => (a..b)
                    .map(|i| {
                        // The only case where the offset genuinely varies, so the only one
                        // that has to be checked per element. `gather` knows the whole
                        // decoded buffer's length and nothing narrower, so a point past its
                        // own row would return the NEXT row's element under this point's name.
                        let offset = per[i];
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
                // `uniform_offset` was checked once by `trailing_layout`, and `Grid`'s columns
                // once by the arm that built it -- neither varies with the index.
                _ => (a..b)
                    .map(|i| at(i).map(|v| (v - lo) * row_stride + shared_offset))
                    .collect::<PyResult<Vec<u64>>>()?
                    .into(),
            }),
            run_len,
            grid: match offsets {
                Offsets::Grid { starts, run } => Some((Arc::from(starts), run)),
                _ => None,
            },
        });
        a = b;
    }
    Ok(items)
}

/// A batch of chunk items, built and held in Rust.
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
    /// Output must not step backwards: the pieces are vended forward-only.
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

    #[new]
    pub(crate) fn new() -> Self {
        Self {
            items: Vec::new(),
            out_end: 0,
        }
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
        // There WAS a monotonicity check here: an entry's output start against the last
        // entry's end. It cannot judge a banded entry -- two bands of one read share an
        // axis-0 start and overlap nothing -- and applying it only to the entries it CAN
        // judge is worse than removing it: a guard that silently does not cover the newest
        // shape reads as protection it no longer gives.
        let items = build_chunk_unit_items(
            key,
            chunk_shape,
            shape,
            indices,
            &out_starts,
            &out_widths,
            &inner,
            Offsets::Uniform(&elem_starts),
        )?;
        self.extend_items(items);
        Ok(())
    }

    /// Push a contiguous SPAN of the split axis, without naming its elements.
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
        // The trailing axes are taken WHOLE -- that is what makes a span of indices one
        // contiguous block. A sub-box there would make each index its own run, which is
        // `push_grid`'s shape, not this one.
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

        // One item per inner chunk the span crosses -- arithmetic, not a walk over elements.
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
                // ONE coordinate: where this chunk's slice of the span begins inside the
                // decoded chunk. `run_len` carries the rest, so `gather` moves the whole
                // block in a single copy.
                coords: Some(vec![(span_lo - lo) * row_stride].into()),
                run_len: rows * row_stride,
                grid: None,
            });
            self.out_end = out_hi;
        }
        Ok(())
    }

    /// Push a GRID selection: the same columns taken from every selected index.
    #[pyo3(signature = (key, chunk_shape, shape, indices, starts, run, out_start, inner))]
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    pub(crate) fn push_grid(
        &mut self,
        key: &str,
        chunk_shape: Vec<u64>,
        shape: Vec<u64>,
        indices: PyReadonlyArray1<'_, i64>,
        starts: PyReadonlyArray1<'_, u64>,
        run: u64,
        out_start: u64,
        inner: u64,
    ) -> PyResult<()> {
        let starts = starts
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("the run-start array must be contiguous"))?;
        self.push_widened(
            key,
            chunk_shape,
            shape,
            indices,
            out_start,
            inner,
            Offsets::Grid { starts, run },
        )
    }

    /// Push a POINT selection: one element per index, each naming its own offset inside that
    /// index's elements.
    #[pyo3(signature = (key, chunk_shape, shape, indices, offsets, out_start, inner))]
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
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
        // Contiguous so the per-point offsets can be read as a slice; a strided view would be
        // indexed as if it were dense.
        let offsets = offsets
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("the offsets array must be contiguous"))?;
        self.push_widened(
            key,
            chunk_shape,
            shape,
            indices,
            out_start,
            inner,
            Offsets::PerIndex(offsets),
        )
    }
}

impl ChunkItems {
    pub(crate) fn as_slice(&self) -> &[ChunkItem] {
        &self.items
    }

    /// Append built items and move the output cursor past them.
    fn extend_items(&mut self, items: Vec<ChunkItem>) {
        if let Some(last) = items.last() {
            self.out_end = last.subset.end_exc()[0];
        }
        self.items.extend(items);
    }

    /// The body `push_grid` and `push_points` share: guard the cursor, widen the trailing-axis
    /// descriptions, build, append.
    #[allow(clippy::too_many_arguments)]
    fn push_widened(
        &mut self,
        key: &str,
        chunk_shape: Vec<u64>,
        shape: Vec<u64>,
        indices: PyReadonlyArray1<'_, i64>,
        out_start: u64,
        inner: u64,
        offsets: Offsets<'_>,
    ) -> PyResult<()> {
        self.refuse_backwards(out_start)?;
        // Before the call: `shape` is moved into it, and argument evaluation is left to right.
        let out_starts = trailing_zeros(out_start, shape.len());
        // These take the trailing axes whole, so the item's extent IS the output shape.
        let out_widths = shape.clone();
        // These paths take the trailing axes whole, and their Python gates require the shard
        // to hold ONE inner chunk on each -- so the shard extent is the inner extent there.
        // Widened here rather than in the signature, so the tautology is written down in one
        // place instead of assumed at every use.
        //
        // `skip(1)`, not `[1..]`: these are pymethods taking arbitrary vectors, and a rank-0
        // chunk must reach `build_chunk_unit_items` to be refused by name rather than panic
        // across the FFI here.
        let inner: Vec<u64> = std::iter::once(inner)
            .chain(chunk_shape.iter().skip(1).copied())
            .collect();
        let items = build_chunk_unit_items(
            key,
            chunk_shape,
            shape,
            indices,
            &out_starts,
            &out_widths,
            &inner,
            offsets,
        )?;
        self.extend_items(items);
        Ok(())
    }
}
