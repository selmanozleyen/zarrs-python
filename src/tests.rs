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
        // Chunk 0: 3, 3, 9. Chunk 2: 20, 27. Chunk 9: 94, the last index the extent allows.
        let indices = PyArray1::from_slice(py, &[3i64, 3, 9, 20, 27, 94]);
        let items = crate::chunk_item::build_chunk_unit_items(
            "c/0",
            vec![95],
            vec![100],
            indices.readonly(),
            &[7],
            &[100],
            &[inner],
            &[],
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
                // The last chunk is short.
                (90, 95, 12, 13, vec![4]),
            ]
        );

        let bad = PyArray1::from_slice(py, &[-1i64]);
        assert!(
            crate::chunk_item::build_chunk_unit_items(
                "c/0",
                vec![95],
                vec![100],
                bad.readonly(),
                &[0],
                &[100],
                &[inner],
                &[]
            )
            .is_err()
        );
        let over = PyArray1::from_slice(py, &[0i64, 1]);
        assert!(
            crate::chunk_item::build_chunk_unit_items(
                "c/0",
                vec![95],
                vec![1],
                over.readonly(),
                &[0],
                &[1],
                &[inner],
                &[]
            )
            .is_err()
        );
        Ok(())
    })
}

/// A selection spanning two shards is two entries, the second starting where the first ended.
#[test]
fn test_chunk_items_handle_accumulates_across_entries() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let mut handle = crate::chunk_item::ChunkItems::new();
        let a = PyArray1::from_slice(py, &[3i64, 20]);
        let b = PyArray1::from_slice(py, &[41i64]);
        handle.push_entry(
            "c/0",
            vec![95],
            vec![100],
            a.readonly(),
            vec![0],
            vec![100],
            vec![10],
            vec![],
        )?;
        handle.push_entry(
            "c/1",
            vec![95],
            vec![100],
            b.readonly(),
            vec![2],
            vec![100],
            vec![10],
            vec![],
        )?;

        let got: Vec<_> = handle
            .as_slice()
            .iter()
            .map(|i| (i.key.as_str(), i.subset.start()[0], i.subset.end_exc()[0]))
            .collect();
        assert_eq!(got, vec![("c/0", 0, 1), ("c/0", 1, 2), ("c/1", 2, 3)]);
        Ok(())
    })
}

/// `push_entry` accepts overlapping output; `DisjointBytes` refuses it when the read carves.
#[test]
fn test_push_entry_leaves_overlap_to_the_read() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let mut handle = crate::chunk_item::ChunkItems::new();
        let a = PyArray1::from_slice(py, &[3i64, 20]);
        let b = PyArray1::from_slice(py, &[41i64]);
        handle.push_entry(
            "c/0",
            vec![95],
            vec![100],
            a.readonly(),
            vec![0],
            vec![100],
            vec![10],
            vec![],
        )?;

        handle.push_entry(
            "c/1",
            vec![95],
            vec![100],
            b.readonly(),
            vec![1],
            vec![100],
            vec![10],
            vec![],
        )?;
        handle.push_entry(
            "c/1",
            vec![95],
            vec![100],
            b.readonly(),
            vec![2],
            vec![100],
            vec![10],
            vec![],
        )?;
        assert_eq!(handle.as_slice().len(), 4);
        Ok(())
    })
}

/// The rank-2 case: the same grouping, with every column taken whole.
#[test]
fn test_chunk_unit_items_rank_two_takes_columns_whole() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let inner = 4u64;
        let cols = 3u64;
        // Chunk 0: rows 1, 1 and 3. Chunk 2: row 9.
        let indices = PyArray1::from_slice(py, &[1i64, 1, 3, 9]);
        let items = crate::chunk_item::build_chunk_unit_items(
            "c/0/0",
            vec![10, cols],
            vec![12, cols],
            indices.readonly(),
            &[2, 0],
            &[12, cols],
            &[inner, cols],
            &[0],
        )?;

        let got: Vec<_> = items
            .iter()
            .map(|i| {
                (
                    i.chunk_subset.start().to_vec(),
                    i.chunk_subset.end_exc(),
                    i.subset.start().to_vec(),
                    i.subset.end_exc(),
                    i.coords.as_ref().unwrap().to_vec(),
                    i.run_len,
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                // Rows 1, 1, 3 of the chunk are element offsets 3, 3, 9.
                (
                    vec![0, 0],
                    vec![4, 3],
                    vec![2, 0],
                    vec![5, 3],
                    vec![3, 3, 9],
                    3
                ),
                // The last chunk is short: rows 8..10.
                (vec![8, 0], vec![10, 3], vec![5, 0], vec![6, 3], vec![3], 3),
            ]
        );
        Ok(())
    })
}

/// A trailing selection must be one run per index.
#[test]
fn test_chunk_unit_items_refuses_mismatched_trailing_axes() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let indices = PyArray1::from_slice(py, &[0i64, 1]);
        // `X[rows, 0:2]` of 3 columns.
        let narrower = crate::chunk_item::build_chunk_unit_items(
            "c/0/0",
            vec![10, 3],
            vec![10, 2],
            indices.readonly(),
            &[0, 0],
            &[10, 2],
            &[4, 3],
            &[0],
        );
        assert!(
            narrower.is_ok(),
            "a contiguous column subset is served, not refused"
        );
        // 2 of 4 by 5 of 10 is two runs of 5 per index.
        let strided = crate::chunk_item::build_chunk_unit_items(
            "c/0/0/0",
            vec![10, 4, 10],
            vec![10, 2, 5],
            indices.readonly(),
            &[0, 0, 0],
            &[10, 2, 5],
            &[4, 4, 10],
            &[0, 0],
        );
        assert!(strided.is_err(), "a strided trailing box must be refused");
        // A run that walks off the end of its own row.
        let wraps = crate::chunk_item::build_chunk_unit_items(
            "c/0/0/0",
            vec![10, 4, 10],
            vec![10, 1, 4],
            indices.readonly(),
            &[0, 0, 0],
            &[10, 1, 4],
            &[4, 4, 10],
            &[0, 8],
        );
        assert!(
            wraps.is_err(),
            "a run leaving its own sub-row must be refused"
        );
        let ranks = crate::chunk_item::build_chunk_unit_items(
            "c/0",
            vec![10],
            vec![10, 2],
            indices.readonly(),
            &[0, 0],
            &[10, 2],
            &[4],
            &[0],
        );
        assert!(ranks.is_err());
        Ok(())
    })
}

#[test]
fn test_gather_copies_by_coordinate_and_refuses_the_rest() {
    let scratch: Vec<u8> = (0..12u8).collect(); // 6 elements of 2 bytes
    let mut out = vec![0u8; 6];

    crate::utils::gather(&scratch, &[0, 2, 5], 1, &mut out, 2).expect("in bounds");
    assert_eq!(out, vec![0, 1, 4, 5, 10, 11]);

    let mut out = vec![0u8; 2];
    assert!(crate::utils::gather(&scratch, &[6], 1, &mut out, 2).is_err());

    let mut out = vec![0u8; 4];
    assert!(crate::utils::gather(&scratch, &[0, 1, 2], 1, &mut out, 2).is_err());
}

#[test]
fn test_gather_copies_a_run_per_coordinate() {
    let scratch: Vec<u8> = (0..12u8).collect(); // 6 elements of 2 bytes, as 2 rows of 3
    let mut out = vec![0u8; 6];

    // Row 1 of 2 rows of 3.
    crate::utils::gather(&scratch, &[3], 3, &mut out, 2).expect("in bounds");
    assert_eq!(out, vec![6, 7, 8, 9, 10, 11]);

    let mut out = vec![0u8; 12];
    crate::utils::gather(&scratch, &[0, 3], 3, &mut out, 2).expect("in bounds");
    assert_eq!(out, (0..12u8).collect::<Vec<_>>());

    // In bounds at the start, not at the end.
    let mut out = vec![0u8; 6];
    assert!(crate::utils::gather(&scratch, &[4], 3, &mut out, 2).is_err());

    let mut out = vec![0u8; 0];
    assert!(crate::utils::gather(&scratch, &[0], 0, &mut out, 2).is_err());
}

/// `raw_runs` counts reads, not rows.
#[test]
fn test_raw_runs_counts_reads_not_rows() {
    let run_len = 2u64;
    let c = |v: &[u64]| v.iter().map(|r| r * run_len).collect::<Vec<u64>>();

    assert_eq!(crate::read_decode::raw_runs(&[], run_len), 0);
    assert_eq!(crate::read_decode::raw_runs(&c(&[7]), run_len), 1);
    let dense: Vec<u64> = (0..64).map(|r| r * run_len).collect();
    assert_eq!(crate::read_decode::raw_runs(&dense, run_len), 1);
    let strided: Vec<u64> = (0..64).step_by(2).map(|r| r * run_len).collect();
    assert_eq!(crate::read_decode::raw_runs(&strided, run_len), 32);
    assert_eq!(
        crate::read_decode::raw_runs(&c(&[0, 1, 2, 10, 11]), run_len),
        2
    );
    // A duplicate breaks the run.
    assert_eq!(crate::read_decode::raw_runs(&c(&[3, 3, 4]), run_len), 2);
}

/// The raw path copies stored bytes verbatim, so it needs them in this machine's order.
#[test]
fn test_raw_is_refused_for_a_foreign_byte_order() {
    let meta = |inner: &str| {
        format!(
            r#"{{"codecs":[{{"name":"sharding_indexed","configuration":{{"codecs":[{inner}]}}}}]}}"#
        )
    };
    let bytes_with = |endian: &str| {
        meta(&format!(
            r#"{{"name":"bytes","configuration":{{"endian":"{endian}"}}}}"#
        ))
    };

    assert_eq!(
        crate::inner_chunk_is_raw(&bytes_with("little")),
        cfg!(target_endian = "little")
    );
    assert_eq!(
        crate::inner_chunk_is_raw(&bytes_with("big")),
        cfg!(target_endian = "big")
    );
    // No `endian` means little.
    assert_eq!(
        crate::inner_chunk_is_raw(&meta(r#"{"name":"bytes"}"#)),
        cfg!(target_endian = "little")
    );
    assert!(!crate::inner_chunk_is_raw(&meta(
        r#"{"name":"bytes"},{"name":"crc32c"}"#
    )));
    assert!(!crate::inner_chunk_is_raw(&meta(r#"{"configuration":{}}"#)));
}

/// `coord + run_len` must not wrap into the next coordinate.
#[test]
fn test_coord_runs_do_not_wrap_at_the_top_of_the_range() {
    let run_len = 4u64;
    assert_eq!(
        crate::utils::coord_runs(&[u64::MAX - 1, 2], run_len).count(),
        2
    );
    assert_eq!(crate::utils::coord_runs(&[0, 4, 8], run_len).count(), 1);
    assert_eq!(crate::utils::coord_runs(&[], run_len).count(), 0);
}
