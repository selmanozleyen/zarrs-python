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

/// A box on two axes: one item per inner chunk it meets, a coordinate and an output offset
/// per run.
#[test]
fn test_push_box_splits_every_axis_at_inner_chunks() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let a = |v: &[i64]| PyArray1::from_slice(py, v).readonly();
        let mut handle = crate::chunk_item::ChunkItems::new();
        // A shard of 8 x 8 in inner chunks of 4 x 4: rows 2..6 cross a seam, columns 1..3 do
        // not; the output is 4 x 2.
        handle.push_box(
            "c/0/1",
            vec![8, 8],
            vec![4, 4],
            vec![4, 2],
            vec![a(&[2]), a(&[1])],
            vec![a(&[4]), a(&[2])],
            vec![a(&[0]), a(&[0])],
        )?;
        let got: Vec<_> = handle
            .as_slice()
            .iter()
            .map(|i| {
                (
                    i.key.as_str(),
                    i.chunk_subset.start().to_vec(),
                    i.coords.as_ref().unwrap().to_vec(),
                    i.out_offsets.as_ref().unwrap().to_vec(),
                    i.run_len,
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                ("c/0/1", vec![0, 0], vec![9, 13], vec![0, 2], 2),
                ("c/0/1", vec![4, 0], vec![1, 5], vec![4, 6], 2),
            ]
        );
        Ok(())
    })
}

/// A box of a decoded chunk lands in a strided box of the output, one run at a time.
#[test]
fn test_gather_pieces_copies_a_box() {
    // A 3 x 4 chunk of 1-byte elements: rows 1..3 by columns 1..3 are runs of 2 at 5 and 9.
    let scratch: Vec<u8> = (0..12u8).collect();
    let mut out = vec![0u8; 10];
    // A 2 x 5 output taking the box at columns 2..4.
    let (top, bottom) = out.split_at_mut(5);
    crate::utils::gather_pieces(&scratch, &[5, 9], 2, &mut [&mut top[2..4], &mut bottom[2..4]], 1)
        .expect("in bounds");
    assert_eq!(out, vec![0, 0, 5, 6, 0, 0, 0, 9, 10, 0]);

    // Whole rows 1..3 are one run of 8, spilling across two output rows.
    let mut out = vec![0u8; 8];
    let (a, b) = out.split_at_mut(4);
    crate::utils::gather_pieces(&scratch, &[4], 8, &mut [a, b], 1).expect("in bounds");
    assert_eq!(out, (4..12u8).collect::<Vec<_>>());
}

/// Whole trailing axes fold into the run: rows of a full-width box are one run each, and
/// consecutive rows one run together.
#[test]
fn test_push_box_folds_whole_trailing_axes_into_the_run() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let a = |v: &[i64]| PyArray1::from_slice(py, v).readonly();
        let mut handle = crate::chunk_item::ChunkItems::new();
        // A shard of 8 x 6 in inner chunks of 4 x 6: rows 1..3 and row 6, every column, out
        // to a 3 x 6 output.
        handle.push_box(
            "c/0/0",
            vec![8, 6],
            vec![4, 6],
            vec![3, 6],
            vec![a(&[1, 6]), a(&[0])],
            vec![a(&[2, 1]), a(&[6])],
            vec![a(&[0, 2]), a(&[0])],
        )?;
        let got: Vec<_> = handle
            .as_slice()
            .iter()
            .map(|i| {
                (
                    i.chunk_subset.start().to_vec(),
                    i.coords.as_ref().unwrap().to_vec(),
                    i.out_offsets.as_ref().unwrap().to_vec(),
                    i.run_len,
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (vec![0, 0], vec![6], vec![0], 12),
                (vec![4, 0], vec![12], vec![12], 6),
            ]
        );
        Ok(())
    })
}
