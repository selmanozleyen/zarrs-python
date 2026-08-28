use pyo3::ffi::c_str;

use numpy::{PyArrayDescrMethods as _, PyUntypedArray, PyUntypedArrayMethods as _};
use pyo3::prelude::*;

use crate::CodecPipelineImpl;

#[test]
fn test_nparray_to_unsafe_cell_slice_empty() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let arr: Bound<'_, PyUntypedArray> = PyModule::from_code(
            py,
            c_str!(
                "def empty_array():
                import numpy as np
                return np.empty(0, dtype=np.uint8)"
            ),
            c_str!(""),
            c_str!(""),
        )?
        .getattr("empty_array")?
        .call0()?
        .extract()?;

        // The size the array actually holds, so the new mismatch check passes rather than
        // being the thing under test here.
        let element_size = arr.dtype().itemsize();
        let slice = CodecPipelineImpl::nparray_to_unsafe_cell_slice(&arr, element_size)?;
        assert!(slice.is_empty());
        Ok(())
    })
}

/// One item per inner chunk: coords relative to it, output runs contiguous and in order.
#[test]
fn test_chunk_unit_items_groups_by_inner_chunk() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let inner = 10u64;
        // Chunk 0: 3, 3, 9 (a duplicate, which the path accepts). Chunk 2: 20, 27.
        // Chunk 9: 94, the last index the extent allows.
        let indices = PyArray1::from_slice(py, &[3i64, 3, 9, 20, 27, 94]);
        let items = crate::chunk_item::build_chunk_unit_items(
            "c/0",
            vec![95],
            vec![100],
            indices.readonly(),
            7,
            inner,
        )?;

        let got: Vec<_> = items
            .iter()
            .map(|i| {
                (
                    i.chunk_subset.start()[0],
                    i.chunk_subset.end_exc()[0],
                    i.subset.start()[0],
                    i.subset.end_exc()[0],
                    i.coords.as_ref().unwrap().to_vec(),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (0, 10, 7, 10, vec![3, 3, 9]),
                (20, 30, 10, 12, vec![0, 7]),
                // hi is min(lo + inner, extent): the last chunk is short.
                (90, 95, 12, 13, vec![4]),
            ]
        );

        // A negative index would cast to a wild chunk id, so it must be refused.
        let bad = PyArray1::from_slice(py, &[-1i64]);
        assert!(
            crate::chunk_item::build_chunk_unit_items(
                "c/0",
                vec![95],
                vec![100],
                bad.readonly(),
                0,
                inner
            )
            .is_err()
        );
        // An output subset past the output extent must be refused.
        let over = PyArray1::from_slice(py, &[0i64, 1]);
        assert!(
            crate::chunk_item::build_chunk_unit_items(
                "c/0",
                vec![95],
                vec![1],
                over.readonly(),
                0,
                inner
            )
            .is_err()
        );
        Ok(())
    })
}

/// `push_entry` must ACCUMULATE, not replace: a selection spanning two shards is two entries,
/// and the second entry's output starts where the first left off.
#[test]
fn test_chunk_items_handle_accumulates_across_entries() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let mut handle = crate::chunk_item::ChunkItems::new();
        let a = PyArray1::from_slice(py, &[3i64, 20]);
        let b = PyArray1::from_slice(py, &[41i64]);
        handle.push_entry("c/0", vec![95], vec![100], a.readonly(), 0, 10)?;
        handle.push_entry("c/1", vec![95], vec![100], b.readonly(), 2, 10)?;

        let got: Vec<_> = handle
            .as_slice()
            .iter()
            .map(|i| (i.key.as_str(), i.subset.start()[0], i.subset.end_exc()[0]))
            .collect();
        assert_eq!(got, vec![("c/0", 0, 1), ("c/0", 1, 2), ("c/1", 2, 3)]);
        Ok(())
    })
}

/// Two entries may not claim the same output bytes.
///
/// `push_entry` is `#[pymethods]` with a caller-chosen `out_start`, and the read path writes
/// items CONCURRENTLY through views whose safety contract is that their subsets are disjoint.
/// Overlap there is a data race, so it has to be refused where the entries are accumulated --
/// nothing downstream sees both.
#[test]
fn test_push_entry_refuses_output_another_entry_owns() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let mut handle = crate::chunk_item::ChunkItems::new();
        let a = PyArray1::from_slice(py, &[3i64, 20]);
        let b = PyArray1::from_slice(py, &[41i64]);
        handle.push_entry("c/0", vec![95], vec![100], a.readonly(), 0, 10)?;

        // `a` produced two items covering output 0..2, so an entry starting at 1 would give
        // two items the same byte.
        assert!(
            handle
                .push_entry("c/1", vec![95], vec![100], b.readonly(), 1, 10)
                .is_err(),
            "an out_start inside an entry already pushed"
        );
        // Starting where the last one ended is exactly what zarr produces, and is allowed.
        handle.push_entry("c/1", vec![95], vec![100], b.readonly(), 2, 10)?;
        assert_eq!(handle.as_slice().len(), 3);
        Ok(())
    })
}

/// `gather` copies by coordinate, and refuses an out-of-range coord or a mismatched output.
#[test]
fn test_gather_copies_by_coordinate_and_refuses_the_rest() {
    let scratch: Vec<u8> = (0..12u8).collect(); // 6 elements of 2 bytes
    let mut out = vec![0u8; 6];

    crate::utils::gather(&scratch, &[0, 2, 5], &mut out, 2).expect("in bounds");
    assert_eq!(out, vec![0, 1, 4, 5, 10, 11]);

    // A coordinate past the decoded buffer must not read adjacent elements.
    let mut out = vec![0u8; 2];
    assert!(crate::utils::gather(&scratch, &[6], &mut out, 2).is_err());

    // An output region that does not match the coordinate count would write short or over.
    let mut out = vec![0u8; 4];
    assert!(crate::utils::gather(&scratch, &[0, 1, 2], &mut out, 2).is_err());
}
