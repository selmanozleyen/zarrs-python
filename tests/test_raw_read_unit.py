"""Rows of an uncompressed chunk read as their own byte ranges.

Both paths return the same values, so the path taken is asserted through a counter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import zarr

from zarrs._internal import raw_path_stats

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
    before = raw_path_stats()
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]
    after = raw_path_stats()
    return got, after[0] - before[0], after[1] - before[1]


def test_an_uncompressed_chunk_is_read_a_row_at_a_time(tmp_path: Path) -> None:
    values = _write(tmp_path / "raw.zarr", compressed=False)
    rows = np.array([1, 40, 91, 200])
    got, raw, chunk = _read(tmp_path / "raw.zarr", rows)

    np.testing.assert_array_equal(got, values[rows])
    assert raw > 0, "an uncompressed scattered read should take the raw path"
    assert chunk == 0, f"{chunk} jobs still read a whole chunk"


def test_a_compressed_chunk_is_never_read_raw(tmp_path: Path) -> None:
    values = _write(tmp_path / "cmp.zarr", compressed=True)
    rows = np.array([1, 40, 91, 200])
    got, raw, chunk = _read(tmp_path / "cmp.zarr", rows)

    np.testing.assert_array_equal(got, values[rows])
    assert raw == 0, "a compressed chunk cannot be read a row at a time"
    assert chunk > 0


def test_a_dense_run_of_rows_is_one_read(tmp_path: Path) -> None:
    values = _write(tmp_path / "run.zarr", compressed=False)
    got, raw, chunk = _read(tmp_path / "run.zarr", slice(0, 8))

    np.testing.assert_array_equal(got, values[0:8])
    assert chunk == 0
    assert raw == 1, f"8 consecutive rows should be one read, got {raw}"


def test_a_scattered_chunk_declines_the_raw_path(tmp_path: Path) -> None:
    """Every other row of a chunk is 4 runs, above the limit of 2, so the chunk is read whole."""
    values = _write(tmp_path / "scat.zarr", compressed=False)
    rows = np.arange(0, 8, 2)
    got, raw, chunk = _read(tmp_path / "scat.zarr", rows)

    np.testing.assert_array_equal(got, values[rows])
    assert raw == 0, f"4 runs in one chunk is above the limit; got {raw} raw jobs"
    assert chunk > 0


def test_a_foreign_endian_chunk_is_never_read_raw(tmp_path: Path) -> None:
    """The raw path copies stored bytes verbatim, so it needs the platform's byte order."""
    from zarr.codecs import BytesCodec

    values = np.arange(SHAPE[0] * SHAPE[1], dtype=">f4").reshape(SHAPE)
    path = tmp_path / "big.zarr"
    zarr.create_array(
        path,
        dtype=values.dtype,
        shape=values.shape,
        chunks=CHUNKS,
        shards=SHARDS,
        serializer=BytesCodec(endian="big"),
        compressors=None,
    )[:] = values

    rows = np.arange(0, SHAPE[0], 8)
    got, raw, chunk = _read(path, (rows, slice(None)))

    np.testing.assert_array_equal(got, values[rows, :])
    assert raw == 0, f"a big-endian array took the raw path: {raw} raw jobs"
    assert chunk > 0
