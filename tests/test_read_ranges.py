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


def _hook(pipeline, a, runs, out):
    """Call the hook the way zarr does; fails if it hands the read back to zarr's default."""
    from zarr.abc.codec import CodecPipeline
    from zarr.core.buffer import default_buffer_prototype

    base = CodecPipeline.read_runs

    async def refuse(*args, **kwargs):
        raise AssertionError("zarrs handed the read back to zarr")

    CodecPipeline.read_runs = refuse
    try:
        sync(
            pipeline.read_runs(
                a.store_path,
                a.metadata,
                runs,
                out,
                config=a._async_array.config,
                chunk_grid=a._async_array._chunk_grid,
                prototype=default_buffer_prototype(),
            )
        )
    finally:
        CodecPipeline.read_runs = base


def read_runs(a, runs, out=None):
    """Runs on every axis through the pipeline's `read_runs` hook, which must serve them."""
    from zarr.core.buffer import default_buffer_prototype

    runs = tuple((np.asarray(s, np.int64), np.asarray(n, np.int64)) for s, n in runs)
    if out is None:
        shape = tuple(int(n.sum()) for _, n in runs)
        out = np.empty(shape, dtype=a.metadata.dtype.to_native_dtype())
    buffer = default_buffer_prototype().nd_buffer(out)
    _hook(a._async_array.codec_pipeline, a, runs, buffer)
    return out


def _rows(starts, lengths, *shape):
    """Runs on axis 0, every other axis whole."""
    return ((starts, lengths), *(([0], [n]) for n in shape))


def read(a, starts, lengths, out=None):
    return read_runs(a, _rows(starts, lengths, *a.shape[1:]), out)


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


def test_refusals_come_before_the_read(arr):
    a, _ = arr
    with pytest.raises(IndexError):
        read(a, [LENGTH - 1], [2])
    with pytest.raises(ValueError, match="out must be"):
        read(a, [0], [4], out=np.empty(3, np.float32))


def test_touching_ranges_read_each_chunk_once(arr):
    """Rows that follow each other meet the same inner chunks, each read and decoded once."""
    from zarrs._internal import raw_path_stats

    a, values = arr
    starts = np.arange(100, 100 + 64 * 5, 5)  # 64 touching ranges of 5, across a seam
    lengths = np.full(starts.size, 5)
    before = raw_path_stats()[1]
    np.testing.assert_array_equal(
        read(a, starts, lengths), expected(values, starts, lengths)
    )
    # 100..420 crosses six inner chunks.
    assert raw_path_stats()[1] - before == 6


def test_the_zarr_hook_serves_2d_and_hands_back_the_rest(arr, tmp_path):
    """`read_runs` as zarr's pipeline hook: reads what it serves, hands the rest to zarr."""
    from zarr.core.buffer import default_buffer_prototype

    a, values = arr
    out = default_buffer_prototype().nd_buffer.empty(
        shape=(7,), dtype=np.dtype("float32")
    )
    starts, lengths = np.array([300, 10]), np.array([4, 3])
    _hook(a._async_array.codec_pipeline, a, ((starts, lengths),), out)
    np.testing.assert_array_equal(
        out.as_ndarray_like(), expected(values, starts, lengths)
    )
    # Chunked on the second axis, which is served too.
    values2 = np.arange(16, dtype=np.float32).reshape(4, 4)
    two_d = zarr.create_array(
        store=tmp_path / "2d.zarr", shape=(4, 4), chunks=(2, 2), dtype="f4"
    )
    two_d[:] = values2
    runs = _rows(np.array([1, 3]), np.array([1, 1]), 4)
    np.testing.assert_array_equal(read_runs(two_d, runs), values2[[1, 3]])
    # A Fortran-ordered output is not, so zarr's default reads it.
    with pytest.raises(AssertionError, match="handed the read back"):
        _hook(
            two_d._async_array.codec_pipeline,
            two_d,
            tuple((np.asarray(s), np.asarray(n)) for s, n in runs),
            default_buffer_prototype().nd_buffer.empty(
                shape=(2, 4), dtype=np.dtype("float32"), order="F"
            ),
        )
    np.testing.assert_array_equal(
        two_d.get_range_selection([1, 3], [1, 1]), values2[[1, 3]]
    )


def test_zarr_range_selection_takes_the_hook(arr, monkeypatch):
    """Through zarr's own API, when the installed zarr has it: the hook serves the read."""
    if not hasattr(zarr.Array, "get_range_selection"):
        pytest.skip("this zarr has no get_range_selection")
    import zarrs.pipeline as pipeline_mod

    a, values = arr
    served = []
    original = pipeline_mod.ZarrsCodecPipeline.read_runs

    async def watched(self, *args, **kwargs):
        served.append(1)
        await original(self, *args, **kwargs)

    monkeypatch.setattr(pipeline_mod.ZarrsCodecPipeline, "read_runs", watched)
    starts, lengths = [900, 10, 900], [30, 20, 30]
    np.testing.assert_array_equal(
        a.get_range_selection(starts, lengths), expected(values, starts, lengths)
    )
    assert served == [1]


def test_ranges_sharing_a_chunk_read_it_once(arr):
    """Ranges that do not touch but share an inner chunk: one read and decode of it, not one each."""
    from zarrs._internal import raw_path_stats

    a, values = arr
    starts = np.arange(
        0, 4 * INNER, 8
    )  # 32 ranges of 3, eight in each of 4 inner chunks
    lengths = np.full(starts.size, 3)
    before = raw_path_stats()[1]
    np.testing.assert_array_equal(
        read(a, starts, lengths), expected(values, starts, lengths)
    )
    assert raw_path_stats()[1] - before == 4


@pytest.mark.parametrize("shards", [(SHARD, 7), None])
def test_rows_of_a_2d_array(tmp_path, shards):
    """Ranges of rows with the other axis whole, sharded or not."""
    values = np.arange(LENGTH * 7, dtype=np.float32).reshape(LENGTH, 7)
    a = zarr.create_array(
        store=tmp_path / "rows.zarr",
        shape=values.shape,
        chunks=(INNER, 7),
        shards=shards,
        dtype=values.dtype,
    )
    a[:] = values
    a = zarr.open_array(tmp_path / "rows.zarr", mode="r")
    starts, lengths = np.array([SHARD - 2, 10, 3000, 11]), np.array([5, 3, 1, 70])
    want = np.concatenate([values[s : s + n] for s, n in zip(starts, lengths)])
    np.testing.assert_array_equal(read(a, starts, lengths), want)


def _random_runs(rng, size, count):
    """Unordered runs that may overlap and repeat."""
    starts = rng.integers(0, size, count)
    lengths = rng.integers(1, size - starts + 1)
    return starts, lengths


def _indices(starts, lengths):
    return np.concatenate([np.arange(s, s + n) for s, n in zip(starts, lengths)])


@pytest.mark.parametrize("compressed", [True, False])
@pytest.mark.parametrize(
    ("shape", "chunks", "shards"),
    [
        ((1000,), (16,), (64,)),
        ((1000,), (16,), None),
        ((90, 70), (8, 5), (16, 20)),
        ((90, 70), (8, 5), None),
        ((30, 25, 20), (4, 5, 3), (8, 10, 9)),
        ((30, 25, 20), (4, 5, 3), None),
    ],
)
def test_runs_on_every_axis_match_numpy(tmp_path, shape, chunks, shards, compressed):
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    zarr.create_array(
        store=tmp_path / "n.zarr",
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype=values.dtype,
        compressors="auto" if compressed else None,
    )[:] = values
    a = zarr.open_array(tmp_path / "n.zarr", mode="r")
    rng = np.random.default_rng()
    for whole in [None, *range(len(shape))]:
        # Each axis in turn taken whole as one run, the rest as scattered runs.
        runs = [
            ([0], [n]) if axis == whole else _random_runs(rng, n, 4)
            for axis, n in enumerate(shape)
        ]
        want = values[np.ix_(*(_indices(s, n) for s, n in runs))]
        np.testing.assert_array_equal(read_runs(a, runs), want)


def test_runs_on_a_column_read_each_chunk_once(tmp_path):
    """Pieces meeting one inner chunk, out of output order, share its read and decode."""
    from zarrs._internal import raw_path_stats

    values = np.arange(64 * 16, dtype=np.float32).reshape(64, 16)
    zarr.create_array(
        store=tmp_path / "c.zarr",
        shape=values.shape,
        chunks=(16, 16),
        shards=(64, 16),
        dtype=values.dtype,
    )[:] = values
    a = zarr.open_array(tmp_path / "c.zarr", mode="r")
    runs = (([32, 0, 32], [16, 16, 16]), ([9, 1, 3], [2, 2, 4]))
    before = raw_path_stats()[1]
    got = read_runs(a, runs)
    np.testing.assert_array_equal(
        got, values[np.ix_(*(_indices(s, n) for s, n in runs))]
    )
    assert raw_path_stats()[1] - before == 2, "two inner chunks, each decoded once"


def test_whole_rows_of_an_uncompressed_shard_are_read_raw(tmp_path):
    from zarrs._internal import raw_path_stats

    values = np.arange(256 * 6, dtype=np.float32).reshape(256, 6)
    zarr.create_array(
        store=tmp_path / "r.zarr",
        shape=values.shape,
        chunks=(16, 6),
        shards=(64, 6),
        dtype=values.dtype,
        compressors=None,
    )[:] = values
    a = zarr.open_array(tmp_path / "r.zarr", mode="r")
    starts, lengths = np.array([3, 100]), np.array([5, 40])
    before = raw_path_stats()
    np.testing.assert_array_equal(
        read(a, starts, lengths), values[_indices(starts, lengths)]
    )
    after = raw_path_stats()
    assert after[0] > before[0]
    assert after[1] == before[1], "a run of whole rows decoded a chunk"


def test_zarr_run_selection_takes_the_hook(tmp_path, monkeypatch):
    """Through zarr's own API, with runs on the second axis."""
    from zarr.abc.codec import CodecPipeline

    if not hasattr(zarr.Array, "get_run_selection"):
        pytest.skip("this zarr has no get_run_selection")
    values = np.arange(24 * 36, dtype=np.int32).reshape(24, 36)
    zarr.create_array(
        store=tmp_path / "z.zarr",
        shape=values.shape,
        chunks=(4, 6),
        shards=(8, 12),
        dtype=values.dtype,
    )[:] = values
    a = zarr.open_array(tmp_path / "z.zarr", mode="r")

    async def refuse(*args, **kwargs):
        raise AssertionError("zarrs handed the read back to zarr")

    monkeypatch.setattr(CodecPipeline, "read_runs", refuse)
    got = a.get_run_selection((slice(3, 17), ([30, 1], [5, 12])))
    np.testing.assert_array_equal(got, values[3:17, [*range(30, 35), *range(1, 13)]])
