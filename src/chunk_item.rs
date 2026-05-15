use std::num::NonZeroU64;
use std::ops::Range;

use numpy::PyReadonlyArray1;
use pyo3::{
    Bound, PyErr, PyResult,
    exceptions::{PyIndexError, PyValueError},
    pyclass, pymethods,
    types::{PyAny, PyAnyMethods, PySlice, PySliceMethods as _},
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

/// Per-dimension selector for `ChunkItem::chunk_subset` and
/// `ChunkItem::subset`.
///
/// `Slice` is the legacy contiguous range and is what the slice-splitting
/// fast path produces. `Indices` carries an integer ndarray of within-dim
/// positions and is what the Phase 2 native-indices path passes through
/// from python without exploding it into per-element ChunkItems.
#[derive(Clone, Debug)]
pub(crate) enum DimSelector {
    Slice(Range<u64>),
    Indices(Vec<u64>),
}

impl DimSelector {
    pub fn len(&self) -> u64 {
        match self {
            DimSelector::Slice(r) => r.end - r.start,
            DimSelector::Indices(v) => v.len() as u64,
        }
    }

    pub fn is_slice(&self) -> bool {
        matches!(self, DimSelector::Slice(_))
    }
}

#[derive(Clone)]
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct ChunkItem {
    pub key: StoreKey,
    pub chunk_subset: Vec<DimSelector>,
    pub subset: Vec<DimSelector>,
    pub shape: Vec<NonZeroU64>,
    pub num_elements: u64,
    pub array_shape: Vec<NonZeroU64>,
}

impl ChunkItem {
    /// True iff every per-dim selector on both sides is a `Slice`.
    pub fn is_slice_only(&self) -> bool {
        self.chunk_subset.iter().all(DimSelector::is_slice)
            && self.subset.iter().all(DimSelector::is_slice)
    }

    /// Convert `chunk_subset` to a single `ArraySubset` if it is slice-only.
    pub fn chunk_subset_as_array_subset(&self) -> Option<ArraySubset> {
        let ranges: Option<Vec<Range<u64>>> = self
            .chunk_subset
            .iter()
            .map(|s| match s {
                DimSelector::Slice(r) => Some(r.clone()),
                DimSelector::Indices(_) => None,
            })
            .collect();
        ranges.map(|r| ArraySubset::new_with_ranges(&r))
    }

    /// Convert `subset` to a single `ArraySubset` if it is slice-only.
    pub fn subset_as_array_subset(&self) -> Option<ArraySubset> {
        let ranges: Option<Vec<Range<u64>>> = self
            .subset
            .iter()
            .map(|s| match s {
                DimSelector::Slice(r) => Some(r.clone()),
                DimSelector::Indices(_) => None,
            })
            .collect();
        ranges.map(|r| ArraySubset::new_with_ranges(&r))
    }

    /// Number of elements selected from the chunk by `chunk_subset`.
    pub fn chunk_subset_num_elements(&self) -> u64 {
        self.chunk_subset.iter().map(DimSelector::len).product()
    }

    /// Expand `chunk_subset` and `subset` into the cartesian product of
    /// per-dim runs / per-dim indices, returning paired
    /// `(chunk_subset, output_subset)` `ArraySubset`s.
    ///
    /// For a (rows: Indices([3, 7, 12]), cols: Slice(0..50)) chunk_subset
    /// paired with (rows: Indices([100, 200, 300]), cols: Slice(0..50))
    /// subset, this yields three pairs:
    ///   (ArraySubset([3..4, 0..50]), ArraySubset([100..101, 0..50])),
    ///   (ArraySubset([7..8, 0..50]), ArraySubset([200..201, 0..50])),
    ///   (ArraySubset([12..13, 0..50]), ArraySubset([300..301, 0..50])).
    ///
    /// The order of pairs follows nested cartesian iteration over dims
    /// from last to first, which matches how `partial_decode` lays out
    /// the returned bytes: concat(pair_0 elements in C-order, pair_1, ...).
    /// Empty `Indices` selectors short-circuit the whole expansion to
    /// an empty Vec.
    pub fn expand_to_subset_pairs(&self) -> Vec<(ArraySubset, ArraySubset)> {
        let n = self.chunk_subset.len();
        debug_assert_eq!(n, self.subset.len());

        // Build a per-dim list of (chunk_range, output_range) cells.
        let mut per_dim: Vec<Vec<(Range<u64>, Range<u64>)>> = Vec::with_capacity(n);
        for d in 0..n {
            let cells = match (&self.chunk_subset[d], &self.subset[d]) {
                (DimSelector::Slice(c), DimSelector::Slice(s)) => {
                    vec![(c.clone(), s.clone())]
                }
                (DimSelector::Indices(ci), DimSelector::Indices(si)) => {
                    debug_assert_eq!(ci.len(), si.len());
                    ci.iter()
                        .zip(si.iter())
                        .map(|(c, s)| (*c..*c + 1, *s..*s + 1))
                        .collect()
                }
                (DimSelector::Indices(ci), DimSelector::Slice(s)) => {
                    // Chunk side is a list of N indices; output side is a
                    // contiguous slice of length N. Pair by position.
                    debug_assert_eq!(ci.len() as u64, s.end - s.start);
                    ci.iter()
                        .enumerate()
                        .map(|(i, c)| {
                            let i = i as u64;
                            (*c..*c + 1, s.start + i..s.start + i + 1)
                        })
                        .collect()
                }
                (DimSelector::Slice(c), DimSelector::Indices(si)) => {
                    // Chunk side is a contiguous slice of length N; output
                    // side is a list of N indices. Pair by position.
                    let n = c.end - c.start;
                    debug_assert_eq!(n as usize, si.len());
                    si.iter()
                        .enumerate()
                        .map(|(i, s)| {
                            let i = i as u64;
                            (c.start + i..c.start + i + 1, *s..*s + 1)
                        })
                        .collect()
                }
            };
            if cells.is_empty() {
                return Vec::new();
            }
            per_dim.push(cells);
        }

        let total: usize = per_dim.iter().map(Vec::len).product();
        let mut result = Vec::with_capacity(total);
        let mut idx = vec![0usize; n];
        loop {
            let chunk_ranges: Vec<Range<u64>> =
                (0..n).map(|d| per_dim[d][idx[d]].0.clone()).collect();
            let output_ranges: Vec<Range<u64>> =
                (0..n).map(|d| per_dim[d][idx[d]].1.clone()).collect();
            result.push((
                ArraySubset::new_with_ranges(&chunk_ranges),
                ArraySubset::new_with_ranges(&output_ranges),
            ));
            // Advance the n-D iterator (last dim varies fastest).
            let mut d = n;
            loop {
                if d == 0 {
                    return result;
                }
                d -= 1;
                idx[d] += 1;
                if idx[d] < per_dim[d].len() {
                    break;
                }
                idx[d] = 0;
            }
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl ChunkItem {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        key: String,
        chunk_subset: Vec<Bound<'_, PyAny>>,
        chunk_shape: Vec<u64>,
        subset: Vec<Bound<'_, PyAny>>,
        shape: Vec<u64>,
    ) -> PyResult<Self> {
        let num_elements = chunk_shape.iter().product();
        let shape_nonzero_u64 = to_nonzero_u64_vec(shape.clone())?;
        let chunk_shape_nonzero_u64 = to_nonzero_u64_vec(chunk_shape.clone())?;

        let chunk_subset = parse_dim_selectors(&chunk_subset, &chunk_shape)?;
        let subset = parse_dim_selectors(&subset, &shape)?;

        // Check that subset and chunk_subset have the same number of elements.
        // This permits broadcasting of a constant input.
        let chunk_subset_n: u64 = chunk_subset.iter().map(DimSelector::len).product();
        let subset_n: u64 = subset.iter().map(DimSelector::len).product();
        if subset_n != chunk_subset_n && subset_n > 1 {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "the size of the chunk subset ({chunk_subset_n} elements) \
                 and input/output subset ({subset_n} elements) are incompatible",
            )));
        }

        Ok(Self {
            key: StoreKey::new(key).map_py_err::<PyValueError>()?,
            chunk_subset,
            subset,
            shape: chunk_shape_nonzero_u64,
            num_elements,
            array_shape: shape_nonzero_u64,
        })
    }
}

fn parse_dim_selectors(
    selectors: &[Bound<'_, PyAny>],
    shape: &[u64],
) -> PyResult<Vec<DimSelector>> {
    if shape.is_empty() {
        // Rank-0 shape signals a constant broadcast (the python legacy
        // path passes shape=[] for `arr[:] = constant`). Any selectors
        // the caller still happens to pass alongside become irrelevant
        // because subset.num_elements() collapses to 1; this matches
        // the silent zip-truncation in the pre-Phase-2 Rust constructor.
        return Ok(Vec::new());
    }
    if selectors.is_empty() {
        // Empty selection with non-empty shape is the constant-write
        // path emitted by python: a length-1 Slice per dim of `shape`,
        // mirroring the legacy slice-only selection_to_array_subset.
        return Ok(shape.iter().map(|_| DimSelector::Slice(0..1)).collect());
    }
    if selectors.len() != shape.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "selector count ({}) does not match shape rank ({})",
            selectors.len(),
            shape.len(),
        )));
    }
    selectors
        .iter()
        .zip(shape.iter())
        .map(|(obj, &length)| extract_dim_selector(obj, length))
        .collect()
}

fn extract_dim_selector(obj: &Bound<'_, PyAny>, length: u64) -> PyResult<DimSelector> {
    if let Ok(slice) = obj.cast::<PySlice>() {
        let r = slice_to_range(slice, isize::try_from(length)?)?;
        return Ok(DimSelector::Slice(r));
    }
    let arr: PyReadonlyArray1<i64> = obj.extract().map_err(|_| {
        PyErr::new::<PyValueError, _>(
            "dim selector must be a slice or a 1-D numpy.int64 array".to_string(),
        )
    })?;
    let s = arr.as_slice().map_py_err::<PyValueError>()?;
    let v: Vec<u64> = s
        .iter()
        .map(|&i| {
            u64::try_from(i).map_err(|_| {
                PyErr::new::<PyValueError, _>(format!(
                    "negative or out-of-range index in selector: {i}"
                ))
            })
        })
        .collect::<PyResult<Vec<u64>>>()?;
    Ok(DimSelector::Indices(v))
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
