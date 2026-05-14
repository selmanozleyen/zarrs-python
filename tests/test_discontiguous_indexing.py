"""Correctness tests for the discontiguous integer-array indexing path.

These tests exercise the Branch 1 work that splits a discontiguous integer
selection into multiple contiguous slice runs so that the Rust pipeline
(rather than the BatchedCodecPipeline fallback) handles fancy indexing.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from zarr.storage import StorePath


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_discontiguous_integer_selection_stays_in_rust(store):
    """Discontiguous integer selection should no longer fall back."""
    sp = StorePath(store, path="disc_int")
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

    # Indices that cross shard boundaries and have gaps within shards.
    idx = np.array([3, 7, 12, 199, 201, 205, 700, 701, 999], dtype=np.int64)
    result = arr.get_orthogonal_selection((idx,))
    np.testing.assert_array_equal(result, data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_discontiguous_integer_2d(store):
    sp = StorePath(store, path="disc_int_2d")
    arr = zarr.create_array(
        sp,
        shape=(100, 100),
        chunks=(10, 10),
        dtype=np.float32,
        fill_value=0.0,
    )
    data = np.arange(10000, dtype=np.float32).reshape(100, 100)
    arr[:] = data

    rows = np.array([1, 5, 50, 99], dtype=np.int64)
    cols = np.array([0, 3, 4, 90, 99], dtype=np.int64)
    result = arr.get_orthogonal_selection((rows, cols))
    np.testing.assert_array_equal(result, data[np.ix_(rows, cols)])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_fanout_cap_falls_back(store, monkeypatch):
    """If a single batch entry would explode into > MAX_FRAGMENTS items,
    we should fall back gracefully (or accept the explosion -- either is
    OK, but no crash and the result must still match numpy)."""
    from zarrs import utils

    monkeypatch.setattr(utils, "_MAX_FRAGMENTS_PER_ITEM", 4, raising=True)
    sp = StorePath(store, path="disc_cap")
    arr = zarr.create_array(
        sp,
        shape=(1000,),
        chunks=(10,),
        dtype=np.float32,
        fill_value=0.0,
    )
    data = np.arange(1000, dtype=np.float32)
    arr[:] = data
    idx = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], dtype=np.int64)
    result = arr.get_orthogonal_selection((idx,))
    np.testing.assert_array_equal(result, data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_contiguous_integer_run_within_chunk(store):
    """A contiguous integer ndarray should produce a single ChunkItem and
    still match numpy. This was already the fast path before Branch 1."""
    sp = StorePath(store, path="contig_int_run")
    arr = zarr.create_array(
        sp,
        shape=(50,),
        chunks=(50,),
        dtype=np.float32,
        fill_value=0.0,
    )
    data = np.arange(50, dtype=np.float32)
    arr[:] = data

    idx = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    np.testing.assert_array_equal(
        arr.get_orthogonal_selection((idx,)), data[idx]
    )


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_discontiguous_integer_with_drop_axis(store):
    """Combining a discontiguous int ndarray with an integer scalar dim
    (which gets dropped on output) must still match numpy."""
    sp = StorePath(store, path="disc_int_drop")
    arr = zarr.create_array(
        sp,
        shape=(10, 10),
        chunks=(5, 5),
        dtype=np.float32,
        fill_value=0.0,
    )
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    arr[:] = data

    rows = np.array([0, 3, 7], dtype=np.int64)
    np.testing.assert_array_equal(
        arr.get_orthogonal_selection((rows, 4)), data[rows, 4]
    )


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_unsorted_integer_selection_stays_in_rust(store):
    """An integer ndarray that is not sorted ascending must still stay in
    the Rust pipeline.

    annbatch's sparse-integer row fetch produces unsorted indices coming
    from CSR indptr ordering (arrival order, not sorted). Branch 1's
    fragmenting path used to require sorted ascending and raised
    DiscontiguousArrayError, which surfaced as a failure under
    config.codec_pipeline.strict=True.
    """
    sp = StorePath(store, path="unsorted_int")
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

    idx = np.array([7, 3, 12, 201, 199, 700, 999, 205], dtype=np.int64)
    result = arr.get_orthogonal_selection((idx,))
    np.testing.assert_array_equal(result, data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_duplicate_integer_selection_stays_in_rust(store):
    """An integer ndarray with duplicates must produce per-element output
    rows (matching numpy semantics), not collapse the duplicates."""
    sp = StorePath(store, path="dup_int")
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

    idx = np.array([3, 3, 5, 7, 7, 7, 12], dtype=np.int64)
    result = arr.get_orthogonal_selection((idx,))
    np.testing.assert_array_equal(result, data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_unsorted_integer_2d_stays_in_rust(store):
    """2D orthogonal selection with an unsorted row dim and a contiguous
    column dim. Mirrors the annbatch sparse-integer fetch shape."""
    sp = StorePath(store, path="unsorted_int_2d")
    arr = zarr.create_array(
        sp,
        shape=(100, 50),
        chunks=(10, 50),
        dtype=np.float32,
        fill_value=0.0,
    )
    data = np.arange(5000, dtype=np.float32).reshape(100, 50)
    arr[:] = data

    rows = np.array([7, 3, 99, 50, 1], dtype=np.int64)
    result = arr.get_orthogonal_selection((rows, slice(None)))
    np.testing.assert_array_equal(result, data[rows])
