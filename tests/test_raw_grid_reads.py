"""Grid selections on an UNCOMPRESSED array, where the exact-byte read path engages.

The raw path reuses chunk-unit items and reads `run_len` elements from each coordinate. That
is right for a plain item, whose coordinate IS a run. It is wrong for a GRID item, whose
coordinate is the START of an index's elements and whose wanted bytes are the runs its grid
names -- reading the front of each row instead of the selection, at the right shape, with no
error raised.

These arrays have no codec between an element and its bytes, which is the condition the raw
path tests for, so the values here are the ones that path produced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

CHUNK_UNIT = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
REFERENCE = {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}


@pytest.fixture
def raw_array(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Uncompressed and sharded: a plain byte tiling, so a row's offset is arithmetic."""
    values = np.arange(1024 * 48, dtype=np.float32).reshape(1024, 48)
    path = tmp_path / "raw"
    z = zarr.create_array(
        path,
        dtype=values.dtype,
        shape=values.shape,
        chunks=(8, 48),
        shards=(64, 48),
        compressors=None,
    )
    z[:] = values
    return path, values


@pytest.mark.parametrize(
    ("name", "read"),
    [
        ("whole rows", lambda a, r: a.oindex[r, :]),
        ("column sub-box", lambda a, r: a.oindex[r, 8:24]),
        ("grid, scattered columns", lambda a, r: a.oindex[r, np.array([0, 5, 5, 17, 40])]),
        ("grid, two columns far apart", lambda a, r: a.oindex[r, np.array([1, 47])]),
        ("paired points", lambda a, r: a[r, np.array([0, 5, 5, 17, 40, 44])]),
        ("contiguous slice", lambda a, r: a.oindex[10:200, :]),
        ("dense block", lambda a, r: a.oindex[np.arange(64, 128), :]),
    ],
)
def test_raw_reads_match_the_reference(
    raw_array: tuple[Path, np.ndarray], name: str, read
) -> None:
    """Every shape, against zarr-python, on a store where the raw path can engage.

    The grid rows are chosen to be scattered AND to include a repeat, because a repeat cannot
    join a run -- the same row twice is two output pieces and cannot be one read.
    """
    path, _ = raw_array
    rows = np.array([1, 3, 3, 40, 300, 900])
    with zarr.config.set(CHUNK_UNIT):
        got = read(zarr.open_array(path, mode="r"), rows)
    with zarr.config.set(REFERENCE):
        theirs = read(zarr.open_array(path, mode="r"), rows)
    np.testing.assert_array_equal(got, theirs)


def test_raw_grids_match_with_the_gate_wide_open(
    raw_array: tuple[Path, np.ndarray], monkeypatch
) -> None:
    """With the read budget raised, a grid definitely takes the raw path rather than falling
    back to the whole chunk -- so this asserts the raw grid code, not the fallback.

    The gate normally refuses a grid: three scattered columns from a hundred rows is three
    hundred requests where the chunk read is a handful, and requests are the scarce resource.
    """
    monkeypatch.setenv("ZARRS_RAW_MAX_READS_PER_CHUNK", "100000")
    path, _ = raw_array
    rows = np.array([1, 3, 3, 40, 300, 900])
    cols = np.array([0, 5, 5, 17, 40])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r").oindex[rows, cols]
    with zarr.config.set(REFERENCE):
        theirs = zarr.open_array(path, mode="r").oindex[rows, cols]
    np.testing.assert_array_equal(got, theirs)
