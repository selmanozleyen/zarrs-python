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
    // 1-D scattered fast path (vindex / `arr[idx]`): when set, decode N
    // length-1 positions via partial_decode and scatter into the output.
    // `out_indices` carries permuted output positions; when None the
    // output is the contiguous range described by `subset`.
    pub chunk_indices: Option<Vec<u64>>,
    pub out_indices: Option<Vec<u64>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ChunkItem {
    #[new]
    #[pyo3(signature = (key, chunk_subset, chunk_shape, subset, shape, *, chunk_indices=None, out_indices=None))]
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn new(
        key: String,
        chunk_subset: Vec<Bound<'_, PySlice>>,
        chunk_shape: Vec<u64>,
        subset: Vec<Bound<'_, PySlice>>,
        shape: Vec<u64>,
        chunk_indices: Option<PyReadonlyArray1<i64>>,
        out_indices: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<Self> {
        let num_elements = chunk_shape.iter().product();
        let shape_nonzero_u64 = to_nonzero_u64_vec(shape)?;
        let chunk_shape_nonzero_u64 = to_nonzero_u64_vec(chunk_shape)?;
        let chunk_subset = selection_to_array_subset(&chunk_subset, &chunk_shape_nonzero_u64)?;
        let subset = selection_to_array_subset(&subset, &shape_nonzero_u64)?;
        // Numpy ndarrays are taken directly here (no python list -> PyInt
        // -> u64 round-trip on the hot path).
        let chunk_indices = chunk_indices.map(ndarray_to_u64_vec).transpose()?;
        let out_indices = out_indices.map(ndarray_to_u64_vec).transpose()?;
        // Check that subset and chunk_subset have the same number of elements.
        // This permits broadcasting of a constant input.
        if chunk_indices.is_none()
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
            chunk_indices,
            out_indices,
        })
    }
}

fn ndarray_to_u64_vec(arr: PyReadonlyArray1<i64>) -> PyResult<Vec<u64>> {
    let s = arr.as_slice().map_py_err::<PyValueError>()?;
    s.iter()
        .map(|&i| {
            u64::try_from(i).map_err(|_| {
                PyErr::new::<PyValueError, _>(format!(
                    "negative index in scattered selector: {i}"
                ))
            })
        })
        .collect()
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
