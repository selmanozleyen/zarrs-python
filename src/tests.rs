use pyo3::ffi::c_str;

use numpy::PyUntypedArray;
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

        let slice = CodecPipelineImpl::nparray_to_unsafe_cell_slice(&arr)?;
        assert!(slice.is_empty());
        Ok(())
    })
}

/// The grouping the Python loop used to do: one item per inner chunk, coords relative to
/// it, output runs contiguous and in order. A regression here is silent -- wrong coords
/// gather the wrong elements rather than failing -- so it is asserted, not eyeballed.
#[test]
fn test_chunk_unit_items_groups_by_inner_chunk() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let inner = 10u64;
        // Chunk 0: 3, 3, 9 (a duplicate, which the path accepts). Chunk 2: 20, 27.
        // Chunk 9: 95, capped by the extent below.
        let indices = PyArray1::from_slice(py, &[3i64, 3, 9, 20, 27, 95]);
        let items = crate::chunk_item::chunk_unit_items(
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
                    i.coords.clone().unwrap(),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (0, 10, 7, 10, vec![3, 3, 9]),
                (20, 30, 10, 12, vec![0, 7]),
                // hi is min(lo + inner, extent): the last chunk is short.
                (90, 95, 12, 13, vec![5]),
            ]
        );

        // A negative index would cast to a wild chunk id, so it must be refused.
        let bad = PyArray1::from_slice(py, &[-1i64]);
        assert!(
            crate::chunk_item::chunk_unit_items("c/0", vec![95], vec![100], bad.readonly(), 0, inner)
                .is_err()
        );
        // An output subset past the output extent was silently clamped by `slice.indices`.
        let over = PyArray1::from_slice(py, &[0i64, 1]);
        assert!(
            crate::chunk_item::chunk_unit_items("c/0", vec![95], vec![1], over.readonly(), 0, inner)
                .is_err()
        );
        Ok(())
    })
}

/// The handle must accumulate across entries exactly as the list path concatenates them,
/// or a mixed-looking batch would silently lose items.
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

        let list_path: Vec<_> = crate::chunk_item::chunk_unit_items(
            "c/0",
            vec![95],
            vec![100],
            a.readonly(),
            0,
            10,
        )?
        .into_iter()
        .chain(crate::chunk_item::chunk_unit_items(
            "c/1",
            vec![95],
            vec![100],
            b.readonly(),
            2,
            10,
        )?)
        .collect();

        assert_eq!(handle.as_slice().len(), 3);
        assert_eq!(handle.as_slice().len(), list_path.len());
        for (got, want) in handle.as_slice().iter().zip(&list_path) {
            assert_eq!(got.key, want.key);
            assert_eq!(got.chunk_subset, want.chunk_subset);
            assert_eq!(got.subset, want.subset);
            assert_eq!(got.coords, want.coords);
        }
        Ok(())
    })
}
