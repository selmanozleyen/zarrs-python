"""Arrays whose shard holds more than one inner chunk on a trailing axis.

This is an ordinary way to shard and the chunk-unit path refuses it, because that path
descends on one axis and vends each item's output as a single contiguous range. It used to be
served by the fused read path; removing that path in `38bfbed` left it with no fast path at
all, and 15 tests have failed since.

Declining is not wrong -- `pipeline.read` catches `DiscontiguousArrayError` and falls back to
zarr-python, which returns correct values. `open_strict` turns the decline into an error, so
these tests fail loudly on a geometry that merely reads slowly in normal use. That is the
point: they say whether the fast path serves it, not whether the answer is right.

Two geometries, and the rank-3 one is not decoration. An earlier attempt at this got rank 2
right and rank 3 wrong -- the coordinate stride ignored the axes BELOW the split -- and no
test in the suite had a third axis, so the suite said nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

ZARRS = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}


def open_strict(path):
    """Open with no fallback, so a declined selection raises instead of being rerouted.

    Without this every test here passes today: `pipeline.read` catches the decline and
    zarr-python returns the right values. The point is not whether the answer is right -- it
    is whether the fast path serves the geometry at all.
    """
    with zarr.config.set({**ZARRS, "codec_pipeline.strict": True}):
        return zarr.open_array(path, mode="r+")


def _write(path, values, chunks, shards):
    array = zarr.create_array(
        store=path, shape=values.shape, chunks=chunks, shards=shards, dtype=values.dtype
    )
    array[...] = values
    return array


@pytest.fixture
def divided_2d(tmp_path):
    """A shard 12 wide holding two 6-wide inner chunks, and two shards across the array."""
    values = np.arange(64 * 24, dtype=np.float64).reshape(64, 24)
    _write(tmp_path / "d2.zarr", values, (8, 6), (16, 12))
    return tmp_path / "d2.zarr", values


@pytest.fixture
def divided_3d(tmp_path):
    """The same division on axis 1, with a third axis below it taken whole.

    The axis below the split is what makes this different: a row of the decoded inner chunk
    is `band_width * 4` elements, not `band_width`.
    """
    values = np.arange(32 * 12 * 4, dtype=np.float64).reshape(32, 12, 4)
    _write(tmp_path / "d3.zarr", values, (8, 6, 4), (16, 12, 4))
    return tmp_path / "d3.zarr", values


@pytest.mark.parametrize(
    "rows",
    [
        np.array([0, 3, 4, 5, 17, 30]),
        np.arange(0, 64),
        np.array([0, 63]),
        np.array([11]),
        np.array([7, 8]),          # across an inner chunk boundary
        np.array([15, 16]),        # across a SHARD boundary
    ],
    ids=["scattered", "every-row", "endpoints", "single", "chunk-edge", "shard-edge"],
)
def test_2d_rows_match(divided_2d, rows):
    path, values = divided_2d
    array = open_strict(path)
    np.testing.assert_array_equal(array[rows], values[rows])


def test_2d_slice_matches(divided_2d):
    path, values = divided_2d
    array = open_strict(path)
    for lo, hi in [(0, 64), (3, 29), (7, 9), (15, 17)]:
        np.testing.assert_array_equal(array[lo:hi], values[lo:hi])


@pytest.mark.parametrize(
    "rows",
    [np.array([0, 3, 4, 5, 17, 30]), np.arange(0, 32), np.array([7, 8])],
    ids=["scattered", "every-row", "chunk-edge"],
)
def test_3d_rows_match(divided_3d, rows):
    """The case an earlier attempt got wrong, and that no existing test would have caught."""
    path, values = divided_3d
    array = open_strict(path)
    np.testing.assert_array_equal(array[rows], values[rows])
