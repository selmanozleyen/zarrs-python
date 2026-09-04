"""Arrays whose shard holds more than one inner chunk on a trailing axis.

An ordinary way to shard, and the one that needs a selection split into BANDS: the inner chunk
is the decode unit, so a trailing range crossing one of its boundaries is not a wide read but
one read per inner chunk.

Every test here uses `open_strict`, which turns a decline into an error. In normal use a
decline is not wrong -- `pipeline.read` catches `DiscontiguousArrayError` and falls back to
zarr-python, which returns correct values, slowly. Strict is what makes these tests say
whether the fast path SERVES the geometry rather than whether the answer is right.

The fixtures are chosen to separate defects rather than to cover shapes:

  banded_output_only  only the output is a sub-box; no shard divides
  divided_2d          both at once
  divided_3d          an axis BELOW the split -- an earlier attempt got rank 2 right and rank
                      3 wrong, because the coordinate stride ignored those axes
  short_final_band    a band NARROWER than its inner chunk. In the other three every band
                      fills a whole inner chunk, so a stride taken from the band width and one
                      taken from the inner chunk are the same number, and fourteen cases agree
                      with each other while both are wrong.
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
def banded_output_only(tmp_path):
    """The SHARD GRID halves the array's width, but each shard is one inner chunk wide.

    This isolates one of the two problems the 2-D fixture has at once. Here every entry
    covers columns 0-11 or 12-23 of a 24-wide output -- so an item's output is a sub-box
    rather than a contiguous range -- but no shard divides into several inner chunks, so no
    band split is needed. Fixing the output side alone should turn this green and leave
    `divided_2d` red.
    """
    values = np.arange(64 * 24, dtype=np.float64).reshape(64, 24)
    _write(tmp_path / "b2.zarr", values, (8, 12), (16, 12))
    return tmp_path / "b2.zarr", values


@pytest.fixture
def short_final_band(tmp_path):
    """The array ends MID inner chunk, so the last band is narrower than the chunk it sits in.

    20 columns over 12-wide shards: the second shard holds columns 12..20, and its inner
    chunks are 6 wide, so the bands are 6 and 2. A band of 2 inside an inner chunk of 6.

    This exists because the other three fixtures cannot catch a whole class of defect. In
    `divided_2d` and `divided_3d` every band happens to cover a WHOLE inner chunk, so
    `prod(band widths)` and `prod(inner[1:])` are the same number and a row stride taken from
    the band survives all fourteen of their cases -- which is a cousin of the exact defect the
    reverted attempt shipped. Modelled in Python first: with the stride taken from the widths,
    `divided_2d`, `divided_3d`, `banded_output_only` and a single divided shard all pass, and
    this is the first case that fails.
    """
    values = np.arange(64 * 20, dtype=np.float64).reshape(64, 20)
    _write(tmp_path / "sf.zarr", values, (8, 6), (16, 12))
    return tmp_path / "sf.zarr", values


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
        np.array([7, 8]),  # across an inner chunk boundary
        np.array([15, 16]),  # across a SHARD boundary
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
    [
        np.array([0, 3, 4, 5, 17, 30]),
        np.arange(0, 32),
        np.array([7, 8]),
        # SHORT, and low. A coordinate stride taken from the shard rather than the inner
        # chunk inflates every offset; `gather`'s bounds check catches that only once the
        # offsets run past the decoded buffer. A read of many rows fails loudly, a read of
        # two returns the wrong rows quietly. This is the case that fails silently.
        np.array([0, 1]),
    ],
    ids=["scattered", "every-row", "chunk-edge", "short-and-low"],
)
def test_3d_rows_match(divided_3d, rows):
    """The case an earlier attempt got wrong, and that no existing test would have caught."""
    path, values = divided_3d
    array = open_strict(path)
    np.testing.assert_array_equal(array[rows], values[rows])


@pytest.mark.parametrize(
    "rows",
    [
        np.array([0, 3, 4, 5, 17, 30]),
        np.arange(0, 64),
        np.array([11]),
        np.array([15, 16]),
    ],
    ids=["scattered", "every-row", "single", "shard-edge"],
)
def test_banded_output_matches(banded_output_only, rows):
    """Only the output is a sub-box; the shard is one inner chunk wide."""
    path, values = banded_output_only
    array = open_strict(path)
    np.testing.assert_array_equal(array[rows], values[rows])


@pytest.mark.parametrize(
    "rows",
    [
        np.array([0, 3, 4, 5, 17, 30]),
        np.arange(0, 64),
        np.array([11]),
        np.array([15, 16]),
        np.array([0, 1]),
    ],
    ids=["scattered", "every-row", "single", "shard-edge", "short-and-low"],
)
def test_short_final_band_matches(short_final_band, rows):
    """A band narrower than its inner chunk still has the inner chunk's row stride."""
    path, values = short_final_band
    array = open_strict(path)
    np.testing.assert_array_equal(array[rows], values[rows])


def test_short_final_band_slice_matches(short_final_band):
    path, values = short_final_band
    array = open_strict(path)
    for lo, hi in [(0, 64), (3, 29), (15, 17)]:
        np.testing.assert_array_equal(array[lo:hi], values[lo:hi])
