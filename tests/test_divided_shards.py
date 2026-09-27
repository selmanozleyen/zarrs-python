from __future__ import annotations

import numpy as np
import pytest
import zarr

ZARRS = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}


def open_strict(path):
    """Open with no fallback, so a decline raises rather than returning right values slowly."""
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
    """The shard grid halves the array's width, but each shard is one inner chunk wide."""
    values = np.arange(64 * 24, dtype=np.float64).reshape(64, 24)
    _write(tmp_path / "b2.zarr", values, (8, 12), (16, 12))
    return tmp_path / "b2.zarr", values


@pytest.fixture
def short_final_band(tmp_path):
    """The array ends mid inner chunk, so the last band is narrower than its inner chunk."""
    values = np.arange(64 * 20, dtype=np.float64).reshape(64, 20)
    _write(tmp_path / "sf.zarr", values, (8, 6), (16, 12))
    return tmp_path / "sf.zarr", values


@pytest.fixture
def divided_3d(tmp_path):
    """The same division on axis 1, with a third axis below it taken whole."""
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
        # Short and low: an inflated coordinate stride stays inside the decoded buffer here,
        # so `gather`'s bounds check does not fire and the wrong rows come back quietly.
        np.array([0, 1]),
    ],
    ids=["scattered", "every-row", "chunk-edge", "short-and-low"],
)
def test_3d_rows_match(divided_3d, rows):
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
