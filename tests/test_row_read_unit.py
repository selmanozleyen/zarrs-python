"""The ROW read unit: a row read without the chunk around it.

Available when an inner chunk is a plain byte tiling -- no filter, no compressor -- so the
byte offset of a row inside it is arithmetic.

Values alone cannot test this. Both paths return the same bytes, so a gate that silently
refuses everything passes every correctness test in the suite and only the throughput moves --
which reads as "the optimisation did not pay" rather than "the optimisation never ran". That
has happened twice on this branch, so the path is asserted directly through a counter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import zarr

from zarrs._internal import read_unit_stats

if TYPE_CHECKING:
    from pathlib import Path

CHUNK_UNIT = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
SHAPE = (256, 64)
CHUNKS = (8, 64)
SHARDS = (32, 64)


def _write(path: Path, *, compressed: bool):
    values = np.arange(SHAPE[0] * SHAPE[1], dtype=np.float32).reshape(SHAPE)
    zarr.create_array(
        path,
        dtype=values.dtype,
        shape=values.shape,
        chunks=CHUNKS,
        shards=SHARDS,
        compressors=None if not compressed else "auto",
    )[:] = values
    return values


def _read(path: Path, selection) -> tuple[np.ndarray, int, int]:
    before = read_unit_stats()
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]
    after = read_unit_stats()
    return got, after[0] - before[0], after[1] - before[1]


def test_an_uncompressed_chunk_is_read_a_row_at_a_time(tmp_path: Path) -> None:
    """Scattered rows: one wanted row per chunk, so one run, so the gate admits it."""
    values = _write(tmp_path / "plain.zarr", compressed=False)
    rows = np.array([1, 40, 91, 200])
    got, row, chunk = _read(tmp_path / "plain.zarr", rows)

    np.testing.assert_array_equal(got, values[rows])
    assert row > 0, "an uncompressed scattered read should take the row path"
    assert chunk == 0, f"{chunk} jobs still read a whole chunk"


def test_a_compressed_chunk_is_never_read_a_row_at_a_time(tmp_path: Path) -> None:
    """The control. A compressor between an element and its bytes makes the offset of a row
    inside the chunk not arithmetic, so the row path must never engage -- at any gate."""
    values = _write(tmp_path / "cmp.zarr", compressed=True)
    rows = np.array([1, 40, 91, 200])
    got, row, chunk = _read(tmp_path / "cmp.zarr", rows)

    np.testing.assert_array_equal(got, values[rows])
    assert row == 0, "a compressed chunk cannot be read a row at a time"
    assert chunk > 0


def test_a_dense_run_of_rows_is_ONE_read(tmp_path: Path) -> None:
    """Consecutive rows coalesce. A whole 8-row chunk is one read, not eight.

    This is the half that makes the gate usable: counting rows rather than runs would refuse
    exactly the case the row path serves best, and refuse it silently.
    """
    values = _write(tmp_path / "run.zarr", compressed=False)
    got, row, chunk = _read(tmp_path / "run.zarr", slice(0, 8))

    np.testing.assert_array_equal(got, values[0:8])
    assert chunk == 0
    assert row == 1, f"8 consecutive rows should be ONE read, got {row}"


def test_a_scattered_chunk_declines_the_row_path(tmp_path: Path) -> None:
    """Every other row of a chunk is 4 runs, above the gate of 2, so the chunk is read whole.

    Eight requests for a fraction of the bytes is not better than one request for all of them
    when requests are the scarce resource -- measured at 0.95x-0.98x for 4-8 runs a chunk.
    """
    values = _write(tmp_path / "scat.zarr", compressed=False)
    rows = np.arange(0, 8, 2)  # rows 0,2,4,6 -- all in chunk 0, four runs
    got, row, chunk = _read(tmp_path / "scat.zarr", rows)

    np.testing.assert_array_equal(got, values[rows])
    assert row == 0, f"4 runs in one chunk is above the gate; got {row} row-unit jobs"
    assert chunk > 0


def test_the_gate_is_a_config_knob(tmp_path: Path):
    """`max_row_reads_per_chunk` must reach the read, and 0 must turn the path off.

    Values cannot see this -- both paths return the same bytes -- so it is asserted through
    the counter. The knob is honoured per call, unlike the two pool sizes, which only the
    first read in a process gets to set.
    """
    from zarrs._internal import read_unit_stats

    path = tmp_path / "gate.zarr"
    values = _write(path, compressed=False)
    array = zarr.open_array(path, mode="r")
    rows = np.arange(0, SHAPE[0], 8)  # scattered: one run per row

    with zarr.config.set({**CHUNK_UNIT, "codec_pipeline.max_row_reads_per_chunk": 0}):
        before = read_unit_stats()
        off = array.oindex[rows, :]
        after_off = read_unit_stats()
    assert after_off[0] == before[0], (
        f"the row path ran with the gate at 0: {before} -> {after_off}"
    )

    # High enough to admit a scattered draw the default would refuse.
    with zarr.config.set(
        {**CHUNK_UNIT, "codec_pipeline.max_row_reads_per_chunk": 1024}
    ):
        on = array.oindex[rows, :]
        after_on = read_unit_stats()
    assert after_on[0] > after_off[0], (
        f"the row path did not run with the gate at 1024: {after_off} -> {after_on}"
    )

    np.testing.assert_array_equal(off, values[rows, :])
    np.testing.assert_array_equal(on, values[rows, :])


def test_a_column_sub_box_is_read_by_row_from_the_right_columns(tmp_path: Path) -> None:
    """A partial trailing axis, which is the one live row arm no values test covered.

    The gate admits an item with ONE output piece, and a column sub-box gives one piece per
    row -- so the only banded shape that reaches the row path is a SINGLE row. There the
    offset arithmetic changes meaning: `element_offsets[0]` is the column offset inside the row rather
    than a whole row's start, and `run_len` is the BAND width rather than the row width.

    Every other row test reads full width, so both of those could have been wrong and still
    returned the right NUMBER of bytes -- which is all `decode_one`'s row branch checks.
    Values from the wrong columns, no error.
    """
    path = tmp_path / "band.zarr"
    values = np.arange(SHAPE[0] * SHAPE[1], dtype=np.float32).reshape(SHAPE)
    zarr.create_array(
        path,
        dtype=values.dtype,
        shape=values.shape,
        chunks=(8, 16),
        shards=(32, 64),
        compressors=None,
    )[:] = values

    selection = (np.array([11]), slice(20, 36))
    got, row, chunk = _read(path, selection)

    np.testing.assert_array_equal(got, values[np.ix_([11], np.arange(20, 36))])
    assert row > 0, "a single banded row is the one banded shape the gate admits"
    assert chunk == 0, f"{chunk} jobs still read a whole chunk"


def test_several_banded_rows_in_ONE_chunk_do_not_take_the_row_path(tmp_path: Path) -> None:
    """The other half of the gate, and the half that decides what "banded" costs.

    A banded item fills one output range PER ROW, and the row path takes an item's output as a
    single contiguous claim -- so it can serve a band only where the item holds one row. Rows
    scattered across chunks each give a one-row item, which is why the test above takes the
    row path; rows sharing a chunk give one item of several rows, which cannot.

    Named so that widening the gate to split a banded claim per row has to change this test
    rather than quietly pass it.
    """
    path = tmp_path / "bands.zarr"
    values = np.arange(SHAPE[0] * SHAPE[1], dtype=np.float32).reshape(SHAPE)
    zarr.create_array(
        path,
        dtype=values.dtype,
        shape=values.shape,
        chunks=(8, 32),
        shards=(32, 64),
        compressors=None,
    )[:] = values

    # All inside inner chunk (0, 0): one item, three rows, a 16-wide band of each.
    rows = np.array([1, 2, 3])
    got, row, chunk = _read(path, (rows, slice(8, 24)))

    np.testing.assert_array_equal(got, values[np.ix_(rows, np.arange(8, 24))])
    assert row == 0, "a multi-row band is not one contiguous output claim"
    assert chunk > 0


def test_the_row_path_on_a_shard_that_DIVIDES_a_trailing_axis(tmp_path: Path) -> None:
    """The one intersection no test covered: the row unit and a divided shard together.

    Every other test here uses a shard holding one inner chunk across the trailing axis, and
    the divided-shard tests all use compressed arrays, where this path is never eligible. So
    the arithmetic that has to hold in both at once -- an element offset reduced modulo the
    INNER extent on a trailing axis, then used as a byte offset inside a chunk that is a plain
    byte tiling -- was only ever exercised one half at a time.

    (16, 24) with shards (16, 24) and chunks (4, 12): two inner chunks across the row, so a
    read of columns 14:20 lands in the second one at an in-chunk offset of 2.
    """
    path = tmp_path / "divided.zarr"
    values = np.arange(16 * 24, dtype=np.float32).reshape(16, 24)
    zarr.create_array(
        path,
        dtype=values.dtype,
        shape=values.shape,
        chunks=(4, 12),
        shards=(16, 24),
        compressors=None,
    )[:] = values

    # One row per inner chunk on axis 0, so each item is a single banded row -- the only
    # banded shape the gate admits.
    rows = np.array([1, 5, 9, 13])
    got, row, chunk = _read(path, (rows, slice(14, 20)))

    np.testing.assert_array_equal(got, values[np.ix_(rows, np.arange(14, 20))])
    assert row > 0, "a divided shard must not by itself disqualify the row path"
