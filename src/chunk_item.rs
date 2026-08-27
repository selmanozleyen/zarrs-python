use std::num::NonZeroU64;

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
    pub coords: Option<Vec<u64>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ChunkItem {
    #[new]
    #[pyo3(signature = (key, chunk_subset, chunk_shape, subset, shape, coords=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        key: String,
        chunk_subset: Vec<Bound<'_, PySlice>>,
        chunk_shape: Vec<u64>,
        subset: Vec<Bound<'_, PySlice>>,
        shape: Vec<u64>,
        coords: Option<Vec<u64>>,
    ) -> PyResult<Self> {
        let num_elements = chunk_shape.iter().product();
        let shape_nonzero_u64 = to_nonzero_u64_vec(shape)?;
        let chunk_shape_nonzero_u64 = to_nonzero_u64_vec(chunk_shape)?;
        let chunk_subset = selection_to_array_subset(&chunk_subset, &chunk_shape_nonzero_u64)?;
        let subset = selection_to_array_subset(&subset, &shape_nonzero_u64)?;
        // Check that subset and chunk_subset have the same number of elements.
        // This permits broadcasting of a constant input.
        // With coordinates the chunk subset is a whole inner chunk and the output holds
        // only the elements wanted from it, so the two counts are not meant to agree.
        if coords.is_none()
            && subset.num_elements() != chunk_subset.num_elements()
            && subset.num_elements() > 1
        {
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
            coords,
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

/// Build one item per inner chunk for a whole entry, without crossing pyo3 per item.
///
/// The Python loop this replaces paid, per item, a pyo3 call with keyword parsing, two
/// `slice` objects, two `PySlice::indices` calls, a fresh `StoreKey` allocation for a key
/// that never changes, and -- per ELEMENT -- a boxed `PyInt` for `coords` that pyo3 then
/// unboxed again -- the dominant cost of such a read, and serial under the GIL.
///
/// Everything an item needs is identical across the entry except two index pairs, so the
/// batch crosses once: the key is validated once, `indices` arrives as a numpy view, and
/// the grouping runs here in one pass.
///
/// The caller's guards still hold the semantics (1-D, non-negative, non-decreasing,
/// contiguous output slice) because they are vectorised numpy and cost nothing. The
/// checks repeated here are the ones whose failure would be silent: a negative index
/// cast to `u64` becomes a wild chunk id, and `inner == 0` divides by zero.
#[allow(clippy::needless_pass_by_value)]
// One range per call is the point: this is the 1-D path.
#[allow(clippy::single_range_in_vec_init)]
pub(crate) fn build_chunk_unit_items(
    key: &str,
    chunk_shape: Vec<u64>,
    shape: Vec<u64>,
    indices: PyReadonlyArray1<'_, i64>,
    out_start: u64,
    inner: u64,
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
    if chunk_shape.len() != 1 || shape.is_empty() {
        return Err(PyErr::new::<PyValueError, _>(
            "chunk_unit_items is the 1-D path: chunk_shape must have one axis",
        ));
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

    let mut items = Vec::new();
    let mut a = 0usize;
    while a < n {
        let chunk_id = at(a)? / inner;
        let mut b = a + 1;
        while b < n && at(b)? / inner == chunk_id {
            b += 1;
        }
        let lo = chunk_id * inner;
        // Exactly one inner chunk, as a subset: zarrs decodes it once.
        let hi = (lo + inner).min(extent);
        if lo >= extent {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "index {} is past the chunk extent {extent}",
                at(a)?
            )));
        }
        let out_lo = out_start + a as u64;
        let out_hi = out_start + b as u64;
        // Python built these as slices, and `slice.indices()` would have CLAMPED an
        // overrun to the shape instead of saying so.
        if out_hi > out_extent {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "output subset {out_lo}..{out_hi} is past the output extent {out_extent}",
            )));
        }
        items.push(ChunkItem {
            key: key.clone(),
            chunk_subset: ArraySubset::new_with_ranges(&[lo..hi]),
            subset: ArraySubset::new_with_ranges(&[out_lo..out_hi]),
            shape: chunk_shape.clone(),
            num_elements,
            array_shape: shape.clone(),
            // Relative to the chunk subset, because that is the buffer gathered from.
            coords: Some(
                (a..b)
                    .map(|i| at(i).map(|v| v - lo))
                    .collect::<PyResult<Vec<u64>>>()?,
            ),
        });
        a = b;
    }
    Ok(items)
}

/// A whole batch of chunk items, held on the Rust side.
///
/// Returning `Vec<ChunkItem>` to Python costs one pyclass object per item, which the read
/// entry point then extracts straight back into a `Vec` -- a round trip through Python for
/// a batch that both ends want as Rust values. A selection over a sharded array can reach
/// a thousand items per call, so that cost is proportional to the selection.
///
/// Behind this handle the items never become Python objects. The caller drives the
/// eligibility checks and calls `push_entry` once per batch ENTRY, then passes the handle
/// to `CodecPipelineImpl.retrieve_chunk_items_and_apply_index`.
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

    fn __len__(&self) -> usize {
        self.items.len()
    }

    /// Build one batch entry's items and append them.
    ///
    /// `indices` must be non-negative and non-decreasing, and `out_start` is where this
    /// entry's elements begin in the output. The caller checks eligibility.
    #[pyo3(signature = (key, chunk_shape, shape, indices, out_start, inner))]
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn push_entry(
        &mut self,
        key: &str,
        chunk_shape: Vec<u64>,
        shape: Vec<u64>,
        indices: PyReadonlyArray1<'_, i64>,
        out_start: u64,
        inner: u64,
    ) -> PyResult<()> {
        self.items.extend(build_chunk_unit_items(
            key,
            chunk_shape,
            shape,
            indices,
            out_start,
            inner,
        )?);
        Ok(())
    }
}

impl ChunkItems {
    pub(crate) fn as_slice(&self) -> &[ChunkItem] {
        &self.items
    }
}
