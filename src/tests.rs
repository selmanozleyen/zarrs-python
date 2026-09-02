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
            &[7],
            &[100],
            &[inner],
            crate::chunk_item::Offsets::Uniform(&[]),
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
                bad.readonly(), &[0], &[100],
                &[inner],
                crate::chunk_item::Offsets::Uniform(&[])
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
                over.readonly(), &[0], &[1],
                &[inner],
                crate::chunk_item::Offsets::Uniform(&[])
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
        handle.push_entry("c/0", vec![95], vec![100], a.readonly(), vec![0], vec![100], vec![10], vec![])?;
        handle.push_entry("c/1", vec![95], vec![100], b.readonly(), vec![2], vec![100], vec![10], vec![])?;

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
fn test_push_entry_leaves_overlap_to_the_vendor() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let mut handle = crate::chunk_item::ChunkItems::new();
        let a = PyArray1::from_slice(py, &[3i64, 20]);
        let b = PyArray1::from_slice(py, &[41i64]);
        handle.push_entry("c/0", vec![95], vec![100], a.readonly(), vec![0], vec![100], vec![10], vec![])?;

        // `a` produced two items covering output 0..2, so an entry starting at 1 gives two
        // items the same byte. `push_entry` no longer refuses that: the check it used could
        // not judge a banded entry, where two bands of one read share an axis-0 start and
        // overlap nothing, and a guard that quietly stops covering the newest shape is worse
        // than none. The overlap is refused at read instead, by the forward-only cursor in
        // `DisjointBytes` -- see `read_decode::tests::bytes_are_vended_once_and_forwards`,
        // which pins that directly.
        handle.push_entry("c/1", vec![95], vec![100], b.readonly(), vec![1], vec![100], vec![10], vec![])?;
        // Starting where the last one ended is exactly what zarr produces.
        handle.push_entry("c/1", vec![95], vec![100], b.readonly(), vec![2], vec![100], vec![10], vec![])?;
        assert_eq!(handle.as_slice().len(), 4);
        Ok(())
    })
}

/// The rank-2 case: the same grouping, with every column taken whole.
///
/// What is different from the 1-D test above and has to be pinned: the subsets carry the
/// full column range on axis 1, and the coordinates are SCALED by the row length -- a
/// coordinate addresses the start of a row inside the decoded chunk, not an element. Get
/// that scaling wrong and the read returns the right shape full of the wrong rows.
#[test]
fn test_chunk_unit_items_rank_two_takes_columns_whole() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let inner = 4u64;
        let cols = 3u64;
        // Chunk 0: rows 1 and 1 (a duplicate) and 3. Chunk 2: row 9.
        let indices = PyArray1::from_slice(py, &[1i64, 1, 3, 9]);
        let items = crate::chunk_item::build_chunk_unit_items(
            "c/0/0",
            vec![10, cols],
            vec![12, cols],
            indices.readonly(),
            &[2, 0],
            &[12, cols],
            &[inner, cols],
            crate::chunk_item::Offsets::Uniform(&[0]),
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
                // rows 0..4 of the chunk, all 3 columns; output rows 2..5, all 3 columns.
                // Rows 1, 1, 3 within the chunk are element offsets 3, 3, 9.
                (vec![0, 0], vec![4, 3], vec![2, 0], vec![5, 3], vec![3, 3, 9], 3),
                // hi is min(lo + inner, extent), so the last chunk is short: rows 8..10.
                (vec![8, 0], vec![10, 3], vec![5, 0], vec![6, 3], vec![3], 3),
            ]
        );
        Ok(())
    })
}

/// A trailing selection that is STRIDED within one index is refused, not silently trusted:
/// `gather` copies one contiguous run per coordinate, so a strided box would be filled with
/// whatever happened to sit consecutively after its start.
///
/// A merely NARROWER trailing axis is legal now -- 2 columns of a 3-column chunk is
/// `X[rows, 0:2]` -- so this no longer asserts that, and checks the shape rule instead.
#[test]
fn test_chunk_unit_items_refuses_mismatched_trailing_axes() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let indices = PyArray1::from_slice(py, &[0i64, 1]);
        // A narrower trailing axis alone is fine: 2 of 3 columns from 0 is `X[rows, 0:2]`.
        let narrower = crate::chunk_item::build_chunk_unit_items(
            "c/0/0",
            vec![10, 3],
            vec![10, 2],
            indices.readonly(),
            &[0, 0],
            &[10, 2],
            &[4, 3],
            crate::chunk_item::Offsets::Uniform(&[0]),
        );
        assert!(narrower.is_ok(), "a contiguous column subset is served, not refused");
        // 2 of 4 rows by 5 of 10 columns is 2 runs of 5 at a stride of 10 -- not one range.
        // A fused offset could not see this; the per-axis starts can.
        let strided = crate::chunk_item::build_chunk_unit_items(
            "c/0/0/0",
            vec![10, 4, 10],
            vec![10, 2, 5],
            indices.readonly(),
            &[0, 0, 0],
            &[10, 2, 5],
            &[4, 4, 10],
            crate::chunk_item::Offsets::Uniform(&[0, 0]),
        );
        assert!(strided.is_err(), "a strided trailing box must be refused");
        // A run that starts inside its own sub-row and walks off the end of it, likewise.
        let wraps = crate::chunk_item::build_chunk_unit_items(
            "c/0/0/0",
            vec![10, 4, 10],
            vec![10, 1, 4],
            indices.readonly(),
            &[0, 0, 0],
            &[10, 1, 4],
            &[4, 4, 10],
            crate::chunk_item::Offsets::Uniform(&[0, 8]),
        );
        assert!(wraps.is_err(), "a run leaving its own sub-row must be refused");
        // Differing ARITY is refused too -- a 1-D chunk against a 2-D output.
        let ranks = crate::chunk_item::build_chunk_unit_items(
            "c/0",
            vec![10],
            vec![10, 2],
            indices.readonly(),
            &[0, 0],
            &[10, 2],
            &[4],
            crate::chunk_item::Offsets::Uniform(&[0]),
        );
        assert!(ranks.is_err());
        Ok(())
    })
}

/// `gather` copies by coordinate, and refuses an out-of-range coord or a mismatched output.
#[test]
fn test_gather_copies_by_coordinate_and_refuses_the_rest() {
    let scratch: Vec<u8> = (0..12u8).collect(); // 6 elements of 2 bytes
    let mut out = vec![0u8; 6];

    crate::utils::gather(&scratch, &[0, 2, 5], 1, &mut out, 2).expect("in bounds");
    assert_eq!(out, vec![0, 1, 4, 5, 10, 11]);

    // A coordinate past the decoded buffer must not read adjacent elements.
    let mut out = vec![0u8; 2];
    assert!(crate::utils::gather(&scratch, &[6], 1, &mut out, 2).is_err());

    // An output region that does not match the coordinate count would write short or over.
    let mut out = vec![0u8; 4];
    assert!(crate::utils::gather(&scratch, &[0, 1, 2], 1, &mut out, 2).is_err());
}

/// With a run length, one coordinate is a whole row -- and the END of the run is what has to
/// be in bounds, which a start-only check would miss.
#[test]
fn test_gather_copies_a_run_per_coordinate() {
    let scratch: Vec<u8> = (0..12u8).collect(); // 6 elements of 2 bytes, as 2 rows of 3
    let mut out = vec![0u8; 6];

    // Row 1 of the chunk: coordinate 3 (element offset), 3 elements long.
    crate::utils::gather(&scratch, &[3], 3, &mut out, 2).expect("in bounds");
    assert_eq!(out, vec![6, 7, 8, 9, 10, 11]);

    // Both rows, in order.
    let mut out = vec![0u8; 12];
    crate::utils::gather(&scratch, &[0, 3], 3, &mut out, 2).expect("in bounds");
    assert_eq!(out, (0..12u8).collect::<Vec<_>>());

    // A coordinate INSIDE the buffer whose run walks off the end. The start alone is fine,
    // which is exactly why the check is on the end.
    let mut out = vec![0u8; 6];
    assert!(crate::utils::gather(&scratch, &[4], 3, &mut out, 2).is_err());

    // A zero run length would make the output region match at every coordinate count.
    let mut out = vec![0u8; 0];
    assert!(crate::utils::gather(&scratch, &[0], 0, &mut out, 2).is_err());
}

/// `push_points` is `#[pymethods]`, so its arguments are whatever Python passed. Two things
/// it must refuse rather than trust: a point whose offset leaves its own index's elements --
/// `gather` only knows the whole decoded buffer, so that would return the NEXT index's
/// element under this point's name -- and an offset array of the wrong length.
#[test]
fn test_push_points_refuses_offsets_that_leave_their_row() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let rows = PyArray1::from_slice(py, &[0i64, 1, 2]);
        // A chunk row holds 48 elements, so 48 is one past the end of index 0's own.
        let past = PyArray1::from_slice(py, &[0u64, 48, 2]);
        let mut handle = crate::chunk_item::ChunkItems::new();
        assert!(
            handle
                .push_points("c/0/0", vec![64, 48], vec![3], rows.readonly(),
                             past.readonly(), 0, 8)
                .is_err(),
            "a point past its own row must be refused"
        );

        let short = PyArray1::from_slice(py, &[0u64, 1]);
        let mut handle = crate::chunk_item::ChunkItems::new();
        assert!(
            handle
                .push_points("c/0/0", vec![64, 48], vec![3], rows.readonly(),
                             short.readonly(), 0, 8)
                .is_err(),
            "one offset per index, or the pairing is guesswork"
        );

        let ok = PyArray1::from_slice(py, &[0u64, 47, 2]);
        let mut handle = crate::chunk_item::ChunkItems::new();
        handle.push_points("c/0/0", vec![64, 48], vec![3], rows.readonly(),
                           ok.readonly(), 0, 8)?;
        Ok(())
    })
}

/// `push_grid` is `#[pymethods]` too. A column past the row it belongs to would have
/// `gather_columns` read the NEXT row's element under this column's name, so it is refused
/// here rather than trusted from the gate.
#[test]
fn test_push_grid_refuses_runs_outside_the_row() -> PyResult<()> {
    use numpy::{PyArray1, PyArrayMethods as _};

    Python::initialize();
    Python::attach(|py| {
        let rows = PyArray1::from_slice(py, &[0i64, 1, 2]);
        // A chunk row holds 48 elements, so 48 is one past the last.
        let past = PyArray1::from_slice(py, &[0u64, 48]);
        let mut handle = crate::chunk_item::ChunkItems::new();
        assert!(
            handle
                .push_grid("c/0/0", vec![64, 48], vec![3, 2], rows.readonly(),
                           past.readonly(), 1, 0, 8)
                .is_err(),
            "a run past the row must be refused"
        );

        // Repeats are legal and must be kept: a panel may ask for the same gene twice.
        let repeated = PyArray1::from_slice(py, &[5u64, 5, 47]);
        let mut handle = crate::chunk_item::ChunkItems::new();
        handle.push_grid("c/0/0", vec![64, 48], vec![3, 3], rows.readonly(),
                         repeated.readonly(), 1, 0, 8)?;
        Ok(())
    })
}

/// `raw_runs` counts READS, not rows, and that distinction is the whole gate.
///
/// Written because getting it wrong is not a crash: counting rows makes the gate refuse the
/// dense cases it should serve, and the read silently stays on the chunk path -- which reads
/// as "the raw path did not pay" rather than as "the raw path was never taken".
#[test]
fn test_raw_runs_counts_reads_not_rows() {
    let run_len = 2u64;
    let c = |v: &[u64]| v.iter().map(|r| r * run_len).collect::<Vec<u64>>();

    assert_eq!(crate::read_decode::raw_runs(&[], run_len), 0);
    assert_eq!(crate::read_decode::raw_runs(&c(&[7]), run_len), 1);
    // A whole 64-row chunk, consecutive: still ONE read, which is the point.
    let dense: Vec<u64> = (0..64).map(|r| r * run_len).collect();
    assert_eq!(crate::read_decode::raw_runs(&dense, run_len), 1);
    // Every other row: 32 reads, and the gate should refuse it.
    let strided: Vec<u64> = (0..64).step_by(2).map(|r| r * run_len).collect();
    assert_eq!(crate::read_decode::raw_runs(&strided, run_len), 32);
    // Two blocks.
    assert_eq!(crate::read_decode::raw_runs(&c(&[0, 1, 2, 10, 11]), run_len), 2);
    // A DUPLICATE steps by 0, so it breaks the run: the same row twice is two output
    // pieces and cannot be served by one read.
    assert_eq!(crate::read_decode::raw_runs(&c(&[3, 3, 4]), run_len), 2);
}

/// A coordinate near the top of the range ENDS a run instead of starting the next one.
///
/// The three walks `coord_runs` replaced did not agree: two added unchecked, so
/// `coord + run_len` wrapped in release -- and a wrapped sum that happens to equal the next
/// coordinate reads as CONSECUTIVE, which merges two reads that share no bytes. The checked
/// walk is the only behaviour this de-duplication changed, so it is the one thing pinned here.
#[test]
fn test_coord_runs_do_not_wrap_at_the_top_of_the_range() {
    let run_len = 4u64;
    // (u64::MAX - 1) + 4 wraps to 2. Unchecked, these two are one run.
    assert_eq!(crate::utils::coord_runs(&[u64::MAX - 1, 2], run_len).count(), 2);
    // The ordinary case still merges, and the empty one is still no runs at all.
    assert_eq!(crate::utils::coord_runs(&[0, 4, 8], run_len).count(), 1);
    assert_eq!(crate::utils::coord_runs(&[], run_len).count(), 0);
}
