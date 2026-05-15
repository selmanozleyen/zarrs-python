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
