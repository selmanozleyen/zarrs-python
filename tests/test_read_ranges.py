from __future__ import annotations

import numpy as np
import pytest
import zarr
from zarr.core.sync import sync

import zarrs

LENGTH, INNER, SHARD = 5000, 64, 256


def _array(path, shards=(SHARD,), fill_value=0):
    values = np.arange(LENGTH, dtype=np.float32)
    a = zarr.create_array(
        store=path,
        shape=values.shape,
        chunks=(INNER,),
        shards=shards,
        dtype=values.dtype,
        fill_value=fill_value,
    )
    a[:] = values
    return zarr.open_array(path, mode="r"), values


@pytest.fixture
def arr(tmp_path):
    return _array(tmp_path / "a.zarr")


def read(a, starts, lengths, **kwargs):
    starts, lengths = np.asarray(starts, np.int64), np.asarray(lengths, np.int64)
    return sync(zarrs.aread_ranges(a, starts, lengths, **kwargs))


def expected(values, starts, lengths):
    return np.concatenate(
        [values[s : s + n] for s, n in zip(starts, lengths)] + [values[:0]]
    )


@pytest.mark.parametrize(
    ("starts", "lengths"),
    [
        ([0], [LENGTH]),  # the whole array
        ([10, 300, 900], [5, 1, 700]),  # the last one across three shards
        ([900, 10, 900], [30, 20, 30]),  # unordered and overlapping
        ([SHARD - 1, 2 * SHARD - 3], [2, 6]),  # across shard seams
        ([INNER - 1], [2]),  # across an inner-chunk seam
        ([5, 7], [0, 3]),  # an empty range among others
        ([], []),
        ([LENGTH - 1], [1]),
    ],
)
def test_ranges_match_numpy(arr, starts, lengths):
    a, values = arr
    np.testing.assert_array_equal(
        read(a, starts, lengths), expected(values, starts, lengths)
    )


def test_rows_match_a_coordinate_selection(arr):
    """The read anndata does today, and the one this replaces, side by side."""
    a, _ = arr
    rng = np.random.default_rng()
    starts = np.sort(rng.choice(LENGTH // 50, 60, replace=False)) * 50
    lengths = rng.integers(0, 50, starts.size)
    coords = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, lengths)])
    np.testing.assert_array_equal(
        read(a, starts, lengths), a.get_coordinate_selection(coords)
    )


def test_unsharded(tmp_path):
    a, values = _array(tmp_path / "u.zarr", shards=None)
    starts, lengths = [INNER - 3, 1000], [10, 3 * INNER]
    np.testing.assert_array_equal(
        read(a, starts, lengths), expected(values, starts, lengths)
    )


def test_missing_chunks_read_as_fill(tmp_path):
    a = zarr.create_array(
        store=tmp_path / "m.zarr",
        shape=(LENGTH,),
        chunks=(INNER,),
        shards=(SHARD,),
        dtype=np.float32,
        fill_value=7,
    )
    a[:10] = np.arange(10, dtype=np.float32)
    got = read(zarr.open_array(tmp_path / "m.zarr", mode="r"), [5, 3 * SHARD], [10, 4])
    np.testing.assert_array_equal(got, [5, 6, 7, 8, 9, 7, 7, 7, 7, 7, 7, 7, 7, 7])


def test_into_a_buffer(arr):
    a, values = arr
    out = np.empty(8, dtype=np.float32)
    assert read(a, [3], [8], out=out) is out
    np.testing.assert_array_equal(out, values[3:11])


def test_refusals_come_before_the_read(arr, tmp_path):
    a, _ = arr
    with pytest.raises(IndexError):
        zarrs.aread_ranges(a, np.array([LENGTH - 1]), np.array([2]))
    with pytest.raises(ValueError, match="out must be"):
        zarrs.aread_ranges(a, np.array([0]), np.array([4]), out=np.empty(3, np.float32))
    two_d = zarr.create_array(
        store=tmp_path / "2d.zarr", shape=(4, 4), chunks=(2, 2), dtype="f4"
    )
    with pytest.raises(zarrs.UnsupportedRangeReadError):
        zarrs.aread_ranges(two_d, np.array([0]), np.array([1]))


def test_consecutive_ranges_are_one_read(arr):
    """Rows that follow each other read as one span, so a chunk is not decoded once per row."""
    a, values = arr
    starts = np.arange(100, 100 + 64 * 5, 5)  # 64 touching ranges of 5, across a seam
    lengths = np.full(starts.size, 5)
    handle = zarrs._internal.ChunkItems()
    shard_ids = np.array([0, 1], dtype=np.int64)
    handle.push_ranges(["k0", "k1"], shard_ids, starts, lengths, SHARD, INNER)
    # 100..420 as one span is an item per inner chunk it crosses (1..6); per range it was 64+.
    assert len(handle) == 6, len(handle)
    np.testing.assert_array_equal(
        read(a, starts, lengths), expected(values, starts, lengths)
    )
