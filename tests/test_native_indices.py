"""Correctness tests for the Phase 2 native integer-array indexing path.

These tests exercise the ChunkItem.chunk_subset / .subset DimSelector
schema where each per-dim selector is either a slice or a 1-D int64
ndarray. The Phase 2 path passes orthogonal-style integer arrays
straight through to Rust as a single ChunkItem and lets the Rust side
expand the cartesian product into a multi-region partial_decode call,
so no inner sub-chunk is fetched twice within the same ChunkItem.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from zarr.storage import StorePath

from zarrs.utils import (
    DiscontiguousArrayError,
    _is_simple_per_dim,
    make_chunk_info_for_rust_with_indices,
)


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_native_1d_sorted_indices(store):
    """Sorted ascending indices, 1-D array. Crosses shard boundaries."""
    sp = StorePath(store, path="native_1d_sorted")
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
    idx = np.array([3, 7, 12, 199, 201, 205, 700, 701, 999], dtype=np.int64)
    np.testing.assert_array_equal(arr.get_orthogonal_selection((idx,)), data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_native_1d_unsorted_indices(store):
    """Unsorted indices in arrival order (the annbatch CSR shape)."""
    sp = StorePath(store, path="native_1d_unsorted")
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
    np.testing.assert_array_equal(arr.get_orthogonal_selection((idx,)), data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_native_1d_reverse_sorted_indices(store):
    """Strictly decreasing indices."""
    sp = StorePath(store, path="native_1d_reverse")
    arr = zarr.create_array(
        sp,
        shape=(500,),
        chunks=(50,),
        dtype=np.float32,
        fill_value=0.0,
    )
    data = np.arange(500, dtype=np.float32)
    arr[:] = data
    idx = np.array([499, 400, 300, 200, 100, 0], dtype=np.int64)
    np.testing.assert_array_equal(arr.get_orthogonal_selection((idx,)), data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_native_1d_with_duplicates(store):
    """Duplicate indices must produce per-element output (numpy semantics)."""
    sp = StorePath(store, path="native_1d_dup")
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
    np.testing.assert_array_equal(arr.get_orthogonal_selection((idx,)), data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_native_2d_unsorted_rows_full_cols(store):
    """2-D selection: unsorted ndarray on rows, slice on cols (the
    annbatch dense fetch shape)."""
    sp = StorePath(store, path="native_2d_unsorted_rows")
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
    np.testing.assert_array_equal(
        arr.get_orthogonal_selection((rows, slice(None))), data[rows]
    )


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_native_2d_orthogonal_two_ndarrays(store):
    """Orthogonal selection with ndarray on both dims."""
    sp = StorePath(store, path="native_2d_two_arrays")
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
    np.testing.assert_array_equal(
        arr.get_orthogonal_selection((rows, cols)), data[np.ix_(rows, cols)]
    )


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_native_2d_with_drop_axis(store):
    """ndarray on rows, integer scalar on cols (drops the col dim)."""
    sp = StorePath(store, path="native_2d_drop")
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
def test_native_sharded_many_indices(store):
    """200 random sorted indices on a sharded array. Each shard typically
    receives a single ChunkItem with many ndarray indices, so this
    exercises the per-ChunkItem cartesian expansion in expand_to_subset_pairs
    plus the upstream sharding decoder's per-call inner-chunk dedup.
    """
    sp = StorePath(store, path="native_shard_many")
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
    idx = np.sort(rng.choice(2000, size=200, replace=False))
    np.testing.assert_array_equal(arr.get_orthogonal_selection((idx,)), data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_native_sharded_unsorted_indices_match_numpy(store):
    """Sharded array + unsorted indices: bit-for-bit match with numpy."""
    sp = StorePath(store, path="native_shard_unsorted")
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
    rng = np.random.default_rng(1)
    idx = rng.permutation(2000)[:200].astype(np.int64)
    np.testing.assert_array_equal(arr.get_orthogonal_selection((idx,)), data[idx])


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_native_empty_selection(store):
    """An ndarray of size 0 must produce no work and no error."""
    sp = StorePath(store, path="native_empty")
    arr = zarr.create_array(
        sp,
        shape=(100,),
        chunks=(10,),
        dtype=np.float32,
        fill_value=0.0,
    )
    arr[:] = np.arange(100, dtype=np.float32)
    idx = np.array([], dtype=np.int64)
    result = arr.get_orthogonal_selection((idx,))
    assert result.shape == (0,)


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_native_write_falls_back_to_python(store):
    """Writes with a discontiguous ndarray selection must fall back to
    the python pipeline (concurrent read-modify-write would race).
    The fallback must still produce the correct result.
    """
    sp = StorePath(store, path="native_write_fallback")
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
    arr.set_orthogonal_selection((idx,), values)
    expected = np.zeros(100, dtype=np.float32)
    expected[idx] = values
    np.testing.assert_array_equal(arr[:], expected)


# Unit tests on the python helper layer below this line ---------------------


def test_is_simple_per_dim_recognises_orthogonal():
    assert _is_simple_per_dim(
        (slice(0, 10), np.array([1, 2, 3], dtype=np.int64)),
        (slice(0, 10), np.array([0, 1, 2], dtype=np.int64)),
    )
    assert _is_simple_per_dim(
        (slice(0, 10), 4, np.array([1, 2, 3], dtype=np.int64)),
        (slice(0, 10), np.array([0, 1, 2], dtype=np.int64)),
    )


def test_is_simple_per_dim_rejects_coordinate_indexing():
    # Coordinate / vindex selectors arrive with out_selection as a single
    # slice, not a tuple, so the legacy path handles them.
    assert not _is_simple_per_dim(
        (np.array([1, 2, 3], dtype=np.int64), np.array([4, 5, 6], dtype=np.int64)),
        slice(0, 3),
    )


def test_native_path_emits_one_chunk_item_per_shard():
    """For an orthogonal-style discontiguous integer selection, the Phase 2
    path must emit exactly one ChunkItem per touched shard regardless of
    how scattered the indices are within each shard. This is the smoke
    test for the fragment-explosion fix.
    """

    class _ByteGetter:
        def __init__(self, path: str) -> None:
            self.path = path

    class _Config:
        write_empty_chunks = True

    class _ArraySpec:
        config = _Config()

        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    chunk_shape = (50,)
    chunk_spec = _ArraySpec(chunk_shape)
    # Two batch entries, each targeting a distinct shard, with within-
    # shard indices that the legacy slice-splitter would have split into
    # many ChunkItems. Phase 2 emits exactly one ChunkItem per entry.
    batch_info = [
        (
            _ByteGetter("c/0"),
            chunk_spec,
            (np.array([2, 7, 12, 30, 45], dtype=np.int64),),
            (np.array([0, 1, 2, 3, 4], dtype=np.int64),),
            False,
        ),
        (
            _ByteGetter("c/1"),
            chunk_spec,
            (np.array([1, 8, 22, 33], dtype=np.int64),),
            (np.array([5, 6, 7, 8], dtype=np.int64),),
            False,
        ),
    ]
    info = make_chunk_info_for_rust_with_indices(batch_info, (), (9,))
    assert len(info.chunk_info_with_indices) == 2


def test_native_path_skips_empty_ndarray_selection():
    class _ByteGetter:
        path = "c/0"

    class _Config:
        write_empty_chunks = True

    class _ArraySpec:
        config = _Config()
        shape = (50,)

    batch_info = [
        (
            _ByteGetter(),
            _ArraySpec(),
            (np.array([], dtype=np.int64),),
            (np.array([], dtype=np.int64),),
            False,
        )
    ]
    info = make_chunk_info_for_rust_with_indices(batch_info, (), (0,))
    assert info.chunk_info_with_indices == []


def test_native_path_write_raises_on_ndarray_dim():
    class _ByteGetter:
        path = "c/0"

    class _Config:
        write_empty_chunks = True

    class _ArraySpec:
        config = _Config()
        shape = (50,)

    batch_info = [
        (
            _ByteGetter(),
            _ArraySpec(),
            (np.array([2, 7, 12], dtype=np.int64),),
            (np.array([0, 1, 2], dtype=np.int64),),
            False,
        )
    ]
    with pytest.raises(DiscontiguousArrayError):
        make_chunk_info_for_rust_with_indices(
            batch_info, (), (3,), allow_fragmenting=False
        )
