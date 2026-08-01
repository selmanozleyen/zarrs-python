"""Tests for the 1-D scattered vindex fast path.

These cover annbatch's CSR-component fetch shape: scattered integer
selections on 1-D arrays that zarr routes via CoordinateIndexer (i.e.
``arr[idx]`` and ``arr.vindex[idx]``).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import zarr
from zarr.storage import StorePath


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_vindex_1d_sorted_indices(store):
    sp = StorePath(store, path="vindex_1d_sorted")
    arr = zarr.create_array(
        sp,
        shape=(1000,),
        chunks=(50,),
        shards=(200,),
        dtype=np.float32,
        fill_value=0.0,
    )
    data = np.arange(1000, dtype=np.float32)
    arr[:] = data
    idx = np.array([3, 7, 12, 199, 201, 205, 700, 999], dtype=np.int64)
    np.testing.assert_array_equal(arr[idx], data[idx])
    np.testing.assert_array_equal(arr.vindex[idx], data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_vindex_1d_unsorted_indices(store):
    """Unsorted indices trigger zarr's sel_sort path: out_selection is
    a permuted ndarray rather than a contiguous slice.
    """
    sp = StorePath(store, path="vindex_1d_unsorted")
    arr = zarr.create_array(
        sp,
        shape=(1000,),
        chunks=(50,),
        shards=(200,),
        dtype=np.float32,
        fill_value=0.0,
    )
    data = np.arange(1000, dtype=np.float32)
    arr[:] = data
    idx = np.array([7, 3, 999, 201, 12, 700, 199, 205], dtype=np.int64)
    np.testing.assert_array_equal(arr[idx], data[idx])
    np.testing.assert_array_equal(arr.vindex[idx], data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_vindex_1d_with_duplicates(store):
    sp = StorePath(store, path="vindex_1d_dup")
    arr = zarr.create_array(
        sp,
        shape=(500,),
        chunks=(50,),
        dtype=np.float32,
        fill_value=0.0,
    )
    data = np.arange(500, dtype=np.float32)
    arr[:] = data
    idx = np.array([3, 3, 5, 7, 7, 7, 12], dtype=np.int64)
    np.testing.assert_array_equal(arr[idx], data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_vindex_1d_random(store):
    """200 random unsorted indices on a sharded array: the path that
    annbatch CSR loaders hit.
    """
    sp = StorePath(store, path="vindex_1d_random")
    arr = zarr.create_array(
        sp,
        shape=(2000,),
        chunks=(50,),
        shards=(500,),
        dtype=np.float32,
        fill_value=0.0,
    )
    data = np.arange(2000, dtype=np.float32)
    arr[:] = data
    rng = np.random.default_rng(0)
    idx = rng.permutation(2000)[:200].astype(np.int64)
    np.testing.assert_array_equal(arr[idx], data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_vindex_1d_stays_in_rust(store):
    """Patch the python fallback to fail and confirm scattered 1-D
    reads run through the Rust pipeline.
    """
    sp = StorePath(store, path="vindex_stays_in_rust")
    arr = zarr.create_array(
        sp,
        shape=(500,),
        chunks=(50,),
        dtype=np.float32,
        fill_value=0.0,
    )
    data = np.arange(500, dtype=np.float32)
    arr[:] = data
    idx = np.array([7, 3, 499, 100, 250], dtype=np.int64)

    from zarr.core import BatchedCodecPipeline

    async def _boom(*args, **kwargs):
        raise AssertionError("python fallback was unexpectedly invoked")

    with patch.object(BatchedCodecPipeline, "read", _boom):
        np.testing.assert_array_equal(arr[idx], data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_vindex_1d_write_falls_back(store):
    """Scattered writes fall back to the python pipeline (RMW race) but
    the fallback must still produce a correct result.
    """
    sp = StorePath(store, path="vindex_1d_write")
    arr = zarr.create_array(
        sp,
        shape=(100,),
        chunks=(10,),
        dtype=np.float32,
        fill_value=0.0,
    )
    arr[:] = np.zeros(100, dtype=np.float32)
    idx = np.array([3, 7, 50, 99, 11], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    arr.vindex[idx] = values
    expected = np.zeros(100, dtype=np.float32)
    expected[idx] = values
    np.testing.assert_array_equal(arr[:], expected)


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_vindex_1d_cached_shard_index_is_invalidated_after_write(store):
    """A write must invalidate persistent shard decoders and their indexes."""
    with zarr.config.set({"codec_pipeline.vindex_shard_index_cache_size": 16}):
        sp = StorePath(store, path="vindex_1d_cache_invalidation")
        arr = zarr.create_array(
            sp,
            shape=(4096,),
            chunks=(128,),
            shards=(1024,),
            dtype=np.int64,
            fill_value=0,
        )
        arr[:] = np.arange(arr.size, dtype=np.int64)
        idx = np.array([3, 127, 128, 900, 1023, 1024, 2050], dtype=np.int64)
        np.testing.assert_array_equal(arr[idx], idx)

        arr[0:2300] = np.arange(0, 2300, dtype=np.int64) + 10_000
        np.testing.assert_array_equal(arr[idx], idx + 10_000)


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_vindex_1d_shuffled_contiguous_runs(store):
    """Contiguous element ranges requested in shuffled order.

    This is the CSR shape a shuffling minibatch loader produces: each row is a
    contiguous range, but the rows arrive permuted. zarr sorts the coordinates
    and returns a permuted output mapping, so run grouping must key on chunk
    position alone -- keying on output position too collapses this to one run
    per element.
    """
    sp = StorePath(store, path="vindex_1d_shuffled_runs")
    n, row = 40_000, 97
    arr = zarr.create_array(
        sp, shape=(n,), chunks=(512,), shards=(4096,), dtype=np.float32, fill_value=0.0
    )
    data = np.arange(n, dtype=np.float32)
    arr[:] = data
    rng = np.random.default_rng(0)
    rows = rng.permutation(n // row)[:120]
    idx = np.concatenate(
        [np.arange(r * row, (r + 1) * row, dtype=np.int64) for r in rows]
    )
    np.testing.assert_array_equal(arr[idx], data[idx])
    np.testing.assert_array_equal(arr[idx[::-1]], data[idx[::-1]])
    np.testing.assert_array_equal(arr[np.sort(idx)], data[np.sort(idx)])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_vindex_1d_run_longer_than_scatter_slab(store):
    """A permuted run longer than one scratch slab must be decoded in pieces."""
    sp = StorePath(store, path="vindex_1d_long_run")
    n = 300_000
    arr = zarr.create_array(
        sp,
        shape=(n,),
        chunks=(8192,),
        shards=(65536,),
        dtype=np.float32,
        fill_value=0.0,
    )
    data = np.arange(n, dtype=np.float32)
    arr[:] = data
    # two long contiguous ranges, requested second-then-first so the output
    # positions are permuted and the scatter path is taken
    a = np.arange(10_000, 150_000, dtype=np.int64)
    b = np.arange(150_000, 290_000, dtype=np.int64)
    idx = np.concatenate([b, a])
    np.testing.assert_array_equal(arr[idx], data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_vindex_1d_separate_io_and_decode_targets(store):
    """The sparse path accepts independent outer-I/O and codec budgets."""
    with zarr.config.set(
        {
            "codec_pipeline.vindex_io_concurrent_target": 3,
            "codec_pipeline.vindex_decode_concurrent_target": 2,
            "codec_pipeline.vindex_shard_index_cache_size": 4,
        }
    ):
        sp = StorePath(store, path="vindex_1d_separate_targets")
        arr = zarr.create_array(
            sp,
            shape=(4096,),
            chunks=(128,),
            shards=(1024,),
            dtype=np.int64,
            fill_value=0,
        )
        data = np.arange(arr.size, dtype=np.int64)
        arr[:] = data
        idx = np.concatenate(
            [
                np.arange(900, 940, dtype=np.int64),
                np.arange(10, 30, dtype=np.int64),
                np.arange(2050, 2090, dtype=np.int64),
            ]
        )
    np.testing.assert_array_equal(arr[idx], data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_vindex_1d_sharded_blosc_partial_decode(store):
    """Sparse subsets remain correct through Blosc's partial decoder."""
    from zarr.codecs import BloscCodec

    with zarr.config.set({"codec_pipeline.vindex_shard_index_cache_size": 16}):
        sp = StorePath(store, path="vindex_1d_sharded_blosc")
        arr = zarr.create_array(
            sp,
            shape=(20_000,),
            chunks=(512,),
            shards=(4096,),
            dtype=np.int64,
            fill_value=0,
            compressors=[BloscCodec()],
        )
        data = np.arange(arr.size, dtype=np.int64)
        arr[:] = data
        rng = np.random.default_rng(42)
        idx = rng.choice(arr.size, size=1000, replace=False)
        np.testing.assert_array_equal(arr[idx], data[idx])
        np.testing.assert_array_equal(arr[np.sort(idx)], data[np.sort(idx)])
