"""Integer-array selections served one inner chunk at a time.

A sorted integer selection is grouped by the unit the codec chain decodes -- the inner chunk --
so a chunk is read once however many of its elements are wanted; a coordinate list costs two
allocations and a partial-decode call PER ELEMENT. The path is narrow: one 1-D integer axis,
non-negative and non-decreasing, against a contiguous output slice. Anything else falls back and
must still be right, so both directions are asserted -- with `entries` saying which Rust entry
point served the read, since a silent fall back passes every values-only check.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

# Not a multiple of SHARD, so the last shard is only partly covered by the array and its
# later inner chunks hold no elements at all.
N = 40_000
CHUNK = 4_096
SHARD = 16_384

CHUNK_UNIT = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}


@pytest.fixture(params=["zstd", "none"])
def compressors(request):
    return "auto" if request.param == "zstd" else None


@pytest.fixture
def array(tmp_path: Path, compressors) -> tuple[Path, np.ndarray]:
    values = np.arange(N, dtype=np.float32)
    kwargs = {} if compressors == "auto" else {"compressors": compressors}
    path = tmp_path / "a"
    z = zarr.create_array(
        path,
        dtype=values.dtype,
        shape=values.shape,
        chunks=(CHUNK,),
        shards=(SHARD,),
        **kwargs,
    )
    z[:] = values
    return path, values


def selections() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        # Every element of one inner chunk: the whole-chunk subset is exactly what is wanted.
        "one whole chunk": np.arange(CHUNK, 2 * CHUNK),
        # Sparse across many chunks, which is what makes a per-element path expensive.
        "scattered": np.sort(rng.choice(N, size=2_000, replace=False)),
        # Non-decreasing, not strictly increasing. Duplicates are legal and must be kept.
        "with duplicates": np.repeat(
            np.sort(rng.choice(N, size=500, replace=False)), 3
        ),
        # The last shard, which the array covers only partly.
        "partly covered last shard": np.arange(N - 100, N),
        "single element": np.array([N - 1]),
        "every second": np.arange(0, N, 2),
    }


@pytest.mark.parametrize("name", list(selections()))
def test_selection_matches_and_takes_the_handle(
    array: tuple[Path, np.ndarray], entries: dict[str, int], name: str
) -> None:
    path, truth = array
    selection = selections()[name]

    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]

    np.testing.assert_array_equal(got, truth[selection])
    assert entries["handle"] > 0, "the batch was entirely chunk-unit but went as a list"
    assert entries["list"] == 0


@pytest.mark.parametrize(
    "selection",
    [
        # A backward step would mean one item per element, so the path declines it.
        pytest.param(np.array([9_000, 40, 8_000, 39]), id="decreasing"),
        # A STEP is not a contiguous run, so it stays on the ordinary route.
        pytest.param(slice(0, 4 * CHUNK, 3), id="strided slice"),
    ],
)
def test_ineligible_selections_decline_and_are_still_right(
    array: tuple[Path, np.ndarray], entries: dict[str, int], selection
) -> None:
    """Declined selections must still return the right data, down whichever path takes them.

    A contiguous slice is NOT here any more -- it is a sorted integer axis spelled
    differently, and it is served. Only a stepped one still declines.
    """
    path, truth = array

    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]

    np.testing.assert_array_equal(got, truth[selection])
    assert entries["handle"] == 0


@pytest.mark.parametrize(
    ("written", "name"),
    [
        # A shard written in part: some inner chunks of it exist, others never do.
        (slice(0, CHUNK), "partial shard"),
        # Whole shards written, whole shards missing.
        (slice(0, SHARD), "whole shard"),
    ],
)
def test_unwritten_chunks_read_as_fill(
    tmp_path: Path, entries: dict[str, int], written: slice, name: str
) -> None:
    """Chunks never written read back as the fill value.

    Every other test here writes in full, so that branch is otherwise never taken. `rows`
    straddles the boundary: written chunks, unwritten ones, and the chunk containing it."""
    values = np.arange(N, dtype=np.float32)
    path = tmp_path / f"sparse-{name.replace(' ', '-')}"
    z = zarr.create_array(
        path,
        dtype=values.dtype,
        shape=values.shape,
        chunks=(CHUNK,),
        shards=(SHARD,),
        fill_value=np.float32(-7),
    )
    z[written] = values[written]

    expected = np.full(N, -7, dtype=np.float32)
    expected[written] = values[written]

    rows = np.sort(
        np.concatenate(
            [
                np.arange(written.stop - 200, written.stop + 200),
                np.arange(N - 300, N - 100),
            ]
        )
    )
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows]

    np.testing.assert_array_equal(got, expected[rows])
    assert entries["handle"] > 0, "this selection should have taken the chunk-unit path"


def test_a_column_split_inner_chunk_is_served(
    tmp_path: Path, entries: dict[str, int]
) -> None:
    """A 2-D array whose inner chunk is NARROWER than the shard is served, one item per band.

    This test asserted the opposite until the band split landed, and the reasoning it gave was
    right about the mechanism and wrong about the remedy: with the columns divided, one
    selected row is not one contiguous run of the SHARD, and a descent that ignored the column
    axis would address the wrong subchunk. The answer is not to decline -- it is to describe
    one item per inner chunk, so that each item IS one run of the buffer that gets decoded.

    Values alone would pass a version that mis-grouped and got lucky, hence the entry check.
    """
    values = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = tmp_path / "two_d"
    z = zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(16, 16), shards=(32, 32)
    )
    z[:] = values

    rows = np.array([1, 5, 5, 40])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, :]

    np.testing.assert_array_equal(got, values[rows, :])
    assert entries["handle"] > 0, "a column-split 2-D selection should reach the chunk-unit path"
    assert entries["list"] == 0


@pytest.fixture
def full_width(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """A 2-D array whose inner chunk spans every column -- the dense-rep layout.

    64 rows per inner chunk, full width, is what `annbatch.write_sharded` produces for an
    obsm rep, and it is the only 2-D shape the rank-N path accepts.
    """
    values = np.arange(256 * 48, dtype=np.float32).reshape(256, 48)
    path = tmp_path / "full_width"
    z = zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(8, 48), shards=(64, 48)
    )
    z[:] = values
    return path, values


def test_full_width_two_dimensional_takes_the_path(
    full_width: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """The point of the rank-N change: rows grouped by inner chunk, one item per chunk.

    The rows are chosen so several share an inner chunk (8 rows each) and one repeats -- the
    axis only has to be non-DECREASING -- so a run length that addressed the wrong buffer, or
    a coordinate that was not scaled by it, would show up as wrong values rather than an error.
    """
    path, values = full_width
    rows = np.array([1, 3, 3, 9, 60, 61, 200])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, :]

    np.testing.assert_array_equal(got, values[rows, :])
    assert entries["handle"] > 0, "a full-width 2-D selection did not take the chunk-unit path"
    assert entries["list"] == 0


def test_a_partial_column_slice_takes_the_path(
    full_width: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """A column subset runs here too, which it did not used to.

    The inner chunk is decoded WHOLE either way -- it is the decode unit -- so a column
    subset narrows only which elements of each decoded row are copied out. That lives in the
    coordinates and the run length, not in the chunk subset.

    The rows deliberately share inner chunks and repeat one: a coordinate left unstepped by
    the column offset, or scaled by the output width instead of the chunk's own row, lands on
    the wrong elements and shows up here as wrong values rather than as an error.
    """
    path, values = full_width
    rows = np.array([1, 3, 3, 9, 60, 61, 200])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, 8:24]

    np.testing.assert_array_equal(got, values[rows, 8:24])
    assert entries["handle"] > 0, "a partial column slice did not take the chunk-unit path"
    assert entries["list"] == 0


def test_a_strided_column_slice_still_falls_back(
    full_width: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """Step 2 is not one contiguous run per row, and an item's output is vended as a single
    range. Declining is still the only correct answer -- widening took the contiguous case
    only."""
    path, values = full_width
    rows = np.array([1, 3, 9, 200])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, 8:24:2]

    np.testing.assert_array_equal(got, values[rows, 8:24:2])
    assert entries["handle"] == 0, "a strided column slice reached the chunk-unit path"


def test_a_column_slice_matches_zarr_python(
    full_width: tuple[Path, np.ndarray],
) -> None:
    """The widened case, byte for byte against the reference pipeline on the same store."""
    path, _ = full_width
    rows = np.sort(np.random.default_rng(0).choice(256, size=64, replace=False))
    with zarr.config.set(CHUNK_UNIT):
        mine = zarr.open_array(path, mode="r")[rows, 8:24]
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}):
        theirs = zarr.open_array(path, mode="r")[rows, 8:24]
    np.testing.assert_array_equal(mine, theirs)


def test_the_contiguity_rule() -> None:
    """A sub-box of one row is a single run only when every axis before the last PARTIAL one
    takes exactly one element. Getting this wrong copies strided data as if it were
    contiguous, which is silently wrong output rather than an error -- so it is asserted
    directly rather than only through the shapes the fixtures happen to build.
    """
    from zarrs.utils import _contiguous_offset

    # One trailing axis: any sub-range of it is contiguous.
    assert _contiguous_offset([8], [16], (48,)) == 8
    assert _contiguous_offset([0], [48], (48,)) == 0
    # Partial middle axis, last axis whole: still one run, offset in whole last-axis rows.
    assert _contiguous_offset([2, 0], [3, 10], (5, 10)) == 20
    # Partial LAST axis with a wider axis ahead of it: 5 blocks of 4, strided. Decline.
    assert _contiguous_offset([0, 2], [5, 4], (5, 10)) is None
    # The same partial last axis is fine once the axis ahead of it takes exactly one.
    assert _contiguous_offset([3, 2], [1, 4], (5, 10)) == 32


def test_full_width_matches_zarr_python(
    full_width: tuple[Path, np.ndarray],
) -> None:
    """Byte for byte against the reference pipeline, on the same store."""
    path, _ = full_width
    rows = np.sort(np.random.default_rng(0).choice(256, size=64, replace=False))
    with zarr.config.set(CHUNK_UNIT):
        mine = zarr.open_array(path, mode="r")[rows, :]
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}):
        theirs = zarr.open_array(path, mode="r")[rows, :]
    np.testing.assert_array_equal(mine, theirs)


# zarr warns that this layout disables partial reads. That IS the layout under test.
@pytest.mark.filterwarnings("ignore:Combining a `sharding_indexed` codec")
def test_a_codec_after_sharding_is_refused(
    tmp_path: Path, entries: dict[str, int]
) -> None:
    """A bytes-to-bytes codec AFTER the sharding codec compresses the whole shard, so the
    shard index's byte ranges no longer address the shard and one inner chunk cannot be read on
    its own. Whether the layout would then read wrongly or merely slowly has not been measured
    -- the refusal means it does neither."""
    from zarr.codecs import BytesCodec, ShardingCodec

    values = np.arange(N, dtype=np.float32)
    path = tmp_path / "outer_compressed"
    z = zarr.create_array(
        path,
        shape=values.shape,
        chunks=(SHARD,),
        dtype="float32",
        # An explicit sharding serializer leaves the default compressor OUTSIDE it, which is
        # the layout this refuses. `shards=` would nest the compressor inside instead.
        serializer=ShardingCodec(chunk_shape=(CHUNK,), codecs=[BytesCodec()]),
    )
    z[:] = values

    selection = np.sort(np.random.default_rng(0).choice(N, size=200, replace=False))
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]
    np.testing.assert_array_equal(got, values[selection])
    assert entries["handle"] == 0, "an outer codec reached the chunk-unit path"


def test_a_shard_holding_one_inner_chunk(
    tmp_path: Path, entries: dict[str, int]
) -> None:
    """`shards == chunks`, so a coords item's chunk subset IS the whole chunk -- the key never
    entered the partial-decoder cache and the read failed with "Partial decoder not found"."""
    values = np.arange(4096, dtype=np.float32)
    path = tmp_path / "one_per_shard"
    zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(1024,), shards=(1024,)
    )[:] = values

    rows = np.array([1, 5, 9, 300, 1025, 4095])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows]

    np.testing.assert_array_equal(got, values[rows])
    assert entries["handle"] > 0, "this selection should have taken the chunk-unit path"


def test_an_array_narrower_than_its_chunk_takes_the_path(
    tmp_path: Path, entries: dict[str, int]
) -> None:
    """The other case this widening admits, and probably the commoner one in real stores.

    The array is 30 columns wide but its chunk is 48, so the last column of every chunk is
    fill. `X[rows, :]` asks for the whole ARRAY and still selects only part of each decoded
    row -- which used to decline, because the test compared the selection against the chunk
    rather than against the array.
    """
    values = np.arange(256 * 30, dtype=np.float32).reshape(256, 30)
    path = tmp_path / "narrow"
    z = zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(8, 48), shards=(64, 48)
    )
    z[:] = values
    rows = np.array([1, 3, 3, 9, 60, 200])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, :]

    np.testing.assert_array_equal(got, values[rows, :])
    assert entries["handle"] > 0, "a narrow array did not take the chunk-unit path"
    assert entries["list"] == 0


@pytest.mark.parametrize(
    "selection",
    [
        pytest.param(slice(CHUNK - 10, 2 * CHUNK + 10), id="spanning chunks"),
        pytest.param(slice(None), id="everything"),
        pytest.param(slice(N - 100, N), id="partly covered last shard"),
    ],
)
def test_a_contiguous_slice_takes_the_path(
    array: tuple[Path, np.ndarray], entries: dict[str, int], selection
) -> None:
    """A sequential read is grouped like a scattered one, rather than falling to the fused path.

    It used to decline for a spelling reason -- axis 0 was a `slice` rather than an integer
    array -- not because anything about it is unservable.
    """
    path, truth = array

    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]

    np.testing.assert_array_equal(got, truth[selection])
    assert entries["handle"] > 0, "a contiguous slice did not take the chunk-unit path"
    assert entries["list"] == 0


def test_paired_points_take_the_path(
    full_width: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """`X[rows, cols]` -- one element per pair, and a flat result.

    zarr builds a `CoordinateIndexer` for this, so it arrives as one integer array per axis
    against a flat output slice. The columns deliberately do not ascend with the rows: an
    implementation that folded them into a single shared offset would still pass a test where
    they happened to.
    """
    path, values = full_width
    rows = np.array([1, 3, 3, 9, 60, 61, 200])
    cols = np.array([40, 0, 17, 5, 47, 2, 33])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, cols]

    assert got.shape == (rows.size,), "a point selection is flat"
    np.testing.assert_array_equal(got, values[rows, cols])
    assert entries["handle"] > 0, "a point selection did not take the chunk-unit path"
    assert entries["list"] == 0


def test_a_single_column_takes_the_path(
    full_width: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """`X[rows, 5]`, which is a point selection with a constant column, not a dropped axis."""
    path, values = full_width
    rows = np.array([1, 3, 3, 9, 60, 61, 200])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, 5]

    np.testing.assert_array_equal(got, values[rows, 5])
    assert entries["handle"] > 0, "a single column did not take the chunk-unit path"


def test_points_match_zarr_python(full_width: tuple[Path, np.ndarray]) -> None:
    """Byte for byte against the reference pipeline, on the same store."""
    path, _ = full_width
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(256, size=200, replace=True))
    cols = rng.choice(48, size=200, replace=True)
    with zarr.config.set(CHUNK_UNIT):
        mine = zarr.open_array(path, mode="r")[rows, cols]
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}):
        theirs = zarr.open_array(path, mode="r")[rows, cols]
    np.testing.assert_array_equal(mine, theirs)


def test_points_with_unsorted_rows_decline_and_are_still_right(
    full_width: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """Descending rows would make the output positions step backwards. Decline, stay correct."""
    path, values = full_width
    rows = np.array([200, 3, 61, 9])
    cols = np.array([1, 2, 3, 4])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, cols]

    np.testing.assert_array_equal(got, values[rows, cols])
    assert entries["handle"] == 0, "an unsorted point selection reached the chunk-unit path"


@pytest.fixture(params=["1d", "2d"])
def unsharded(tmp_path: Path, request) -> tuple[Path, np.ndarray]:
    """A plain chunked array with NO sharding codec at all."""
    if request.param == "1d":
        values = np.arange(4_000, dtype=np.float32)
        chunks = (256,)
    else:
        values = np.arange(1_024 * 24, dtype=np.float32).reshape(1_024, 24)
        chunks = (32, 24)
    path = tmp_path / f"plain_{request.param}"
    z = zarr.create_array(path, dtype=values.dtype, shape=values.shape, chunks=chunks)
    z[:] = values
    return path, values


def test_an_unsharded_array_takes_the_path(
    unsharded: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """No sharding codec: the chunk is its own decode unit and the store value is the chunk.

    That case is SIMPLER than the sharded one -- there is no index to read and nothing to
    descend -- and it declined only because `ShardInfo::from_codec_chain` returned None for
    an array with no levels.
    """
    path, values = unsharded
    rows = np.array([1, 3, 3, 40, 41, 300, 999])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows]

    np.testing.assert_array_equal(got, values[rows])
    assert entries["handle"] > 0, "an unsharded array did not take the chunk-unit path"
    assert entries["list"] == 0


def test_an_unsharded_array_matches_zarr_python(
    unsharded: tuple[Path, np.ndarray],
) -> None:
    path, _ = unsharded
    rng = np.random.default_rng(0)
    with zarr.config.set(CHUNK_UNIT):
        arr = zarr.open_array(path, mode="r")
        rows = np.sort(rng.choice(arr.shape[0], size=200, replace=False))
        mine = arr[rows]
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}):
        theirs = zarr.open_array(path, mode="r")[rows]
    np.testing.assert_array_equal(mine, theirs)


def test_an_unsharded_array_reads_unwritten_chunks_as_fill(
    tmp_path: Path, entries: dict[str, int]
) -> None:
    """A key that was never written is absent, not an error -- same as a never-written shard
    entry. `locate` hands back the whole value either way and the read finds nothing there."""
    path = tmp_path / "sparse_plain"
    z = zarr.create_array(
        path, dtype=np.float32, shape=(4_000,), chunks=(256,), fill_value=np.float32(-7)
    )
    z[0:256] = np.arange(256, dtype=np.float32)

    rows = np.array([1, 5, 2_000, 3_999])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows]

    np.testing.assert_array_equal(got, np.array([1, 5, -7, -7], dtype=np.float32))
    assert entries["handle"] > 0, "an unsharded array did not take the chunk-unit path"


def test_a_grid_selection_takes_the_path(
    full_width: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """`oindex[rows, cols]` -- the n x m grid, which is a gene panel across cells.

    zarr broadcasts the axes rather than pairing them, so this is NOT the point case: rows
    arrive (n,1) and cols (1,m).

    The columns REPEAT and are not consecutive: a gather that deduplicated them, or that took
    a contiguous span from the first to the last, would return the right shape full of wrong
    values. They do ascend, because an out-of-order list is a different case -- zarr then hands
    over an ndarray out-selection, and that declines (see below).
    """
    path, values = full_width
    rows = np.array([1, 3, 3, 9, 60, 61, 200])
    cols = np.array([0, 5, 17, 17, 40])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r").oindex[rows, cols]

    assert got.shape == (rows.size, cols.size)
    np.testing.assert_array_equal(got, values[np.ix_(rows, cols)])
    assert entries["handle"] > 0, "a grid selection did not take the chunk-unit path"
    assert entries["list"] == 0


def test_a_whole_column_panel_takes_the_path(
    full_width: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """`X[:, cols]` -- every row, a panel of columns. The row axis is a slice here."""
    path, values = full_width
    cols = np.array([2, 7, 44])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r").oindex[:, cols]

    np.testing.assert_array_equal(got, values[:, cols])
    assert entries["handle"] > 0, "a column panel did not take the chunk-unit path"


def test_a_grid_matches_zarr_python(full_width: tuple[Path, np.ndarray]) -> None:
    path, _ = full_width
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(256, size=64, replace=False))
    cols = np.sort(rng.choice(48, size=12, replace=True))
    with zarr.config.set(CHUNK_UNIT):
        mine = zarr.open_array(path, mode="r").oindex[rows, cols]
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}):
        theirs = zarr.open_array(path, mode="r").oindex[rows, cols]
    np.testing.assert_array_equal(mine, theirs)


def test_a_grid_with_unsorted_columns_declines_and_is_still_right(
    full_width: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """Out-of-order columns put the output rows at scattered positions.

    zarr stops describing the output as slices and hands over ndarray out-selections, which is
    the same thing that makes an unsorted ROW axis decline: an item's output is vended as one
    contiguous range and a scattered one cannot be expressed that way.
    """
    path, values = full_width
    rows = np.array([1, 3, 9, 60])
    cols = np.array([40, 0, 17, 5])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r").oindex[rows, cols]

    np.testing.assert_array_equal(got, values[np.ix_(rows, cols)])
    assert entries["handle"] == 0, "an unsorted grid reached the chunk-unit path"


@pytest.fixture
def volume(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """A rank-3 array whose inner chunk spans both trailing axes whole."""
    values = np.arange(256 * 8 * 16, dtype=np.float32).reshape(256, 8, 16)
    path = tmp_path / "vol"
    z = zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(8, 8, 16), shards=(64, 8, 16)
    )
    z[:] = values
    return path, values


@pytest.mark.parametrize(
    ("name", "sel"),
    [
        # The Cartesian product on all three axes.
        ("grid on every axis", (np.array([1, 3, 3, 9, 60]), np.array([1, 3, 6]), np.array([2, 9, 15]))),
        # A span on one trailing axis and a scattered list on the other.
        ("span and list", (np.array([1, 3, 9, 60]), slice(2, 5), np.array([0, 7, 15]))),
        # One plane: the middle axis takes a single element and is DROPPED from the result.
        ("dropped middle axis", (np.array([1, 3, 9, 60]), 3, slice(4, 12))),
        # Every row, a couple of planes.
        ("all rows", (slice(None), np.array([1, 5]), slice(None))),
    ],
)
def test_rank_three_grids_take_the_path(
    volume: tuple[Path, np.ndarray], entries: dict[str, int], name: str, sel
) -> None:
    """The grid generalises to rank N: the offset of an element inside its index's own
    elements is `sum(sel[axis][i] * stride[axis])`, and the product flattened row-major is the
    order the output row wants -- one list, whatever the rank.

    All four of these used to leave zarrs ENTIRELY, not merely miss the grouping.
    """
    path, values = volume
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r").oindex[sel]
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}):
        theirs = zarr.open_array(path, mode="r").oindex[sel]

    np.testing.assert_array_equal(got, theirs)
    assert entries["handle"] > 0, f"{name} did not take the chunk-unit path"
    assert entries["list"] == 0


def test_a_pure_slice_box_takes_the_path_as_runs(
    volume: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """A box of pure slices is served, because it is described as RUNS.

    It used to decline on the grounds that the ordinary route copies a contiguous n-D block
    blockwise where this path would gather element by element. That was true of the encoding,
    not of the data: `[rows, 2:5, 4:12]` of an (8,16) row is three runs of eight, and saying so
    turns twenty-four element copies into three memcpys.
    """
    path, values = volume
    rows = np.array([1, 3, 9, 60])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r").oindex[rows, 2:5, 4:12]

    np.testing.assert_array_equal(got, values[np.ix_(rows, range(2, 5), range(4, 12))])
    assert entries["handle"] > 0, "a pure-slice box did not take the chunk-unit path"
    assert entries["list"] == 0


def test_the_run_decomposition() -> None:
    """The runs a selection decomposes into, asserted directly.

    Values-only tests pass whatever the decomposition, so the thing that makes this path worth
    having is invisible to them: a box copied as one run per ELEMENT is correct and slow. Only
    an axis taken WHOLE lets the absorption continue outward past it, because a partial axis
    leaves a gap before the next one repeats.
    """
    from zarrs.utils import _as_contiguous

    # A slice-shaped index array is recognised as contiguous; a scattered one is not.
    assert _as_contiguous(np.array([4, 5, 6, 7])) == (4, 4)
    assert _as_contiguous(np.array([4])) == (4, 1)
    assert _as_contiguous(np.array([4, 6, 7])) is None
    # Descending is not contiguous either, however tempting the endpoints look.
    assert _as_contiguous(np.array([7, 6, 5, 4])) is None
def test_rank_four_grid_takes_the_path(tmp_path: Path, entries: dict[str, int]) -> None:
    """Nothing in the offset arithmetic knows the rank, so rank 4 is not a separate case --
    asserted rather than assumed, because "it generalises" is exactly the kind of claim that
    is wrong once."""
    values = np.arange(64 * 4 * 6 * 5, dtype=np.float32).reshape(64, 4, 6, 5)
    path = tmp_path / "hyper"
    z = zarr.create_array(
        path, dtype=values.dtype, shape=values.shape,
        chunks=(8, 4, 6, 5), shards=(32, 4, 6, 5),
    )
    z[:] = values
    sel = (np.array([1, 3, 3, 40]), np.array([0, 3]), slice(1, 4), np.array([0, 2, 4]))
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r").oindex[sel]
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}):
        theirs = zarr.open_array(path, mode="r").oindex[sel]

    np.testing.assert_array_equal(got, theirs)
    assert entries["handle"] > 0, "a rank-4 grid did not take the chunk-unit path"


@pytest.fixture
def nested(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Two levels of sharding: 256-row shard -> 32-row subshard -> 8-row inner chunk.

    `compressors=None` matters and is easy to omit: without it zarr puts a default compressor
    OUTSIDE the sharding codec, the shard index stops addressing the shard, and this path
    refuses the array for a reason that has nothing to do with nesting.
    """
    from zarr.codecs import BytesCodec, ShardingCodec

    values = np.arange(1024 * 48, dtype=np.float32).reshape(1024, 48)
    path = tmp_path / "nested2d"
    z = zarr.create_array(
        path,
        shape=values.shape,
        chunks=(256, 48),
        dtype="float32",
        compressors=None,
        serializer=ShardingCodec(
            chunk_shape=(32, 48),
            codecs=[ShardingCodec(chunk_shape=(8, 48), codecs=[BytesCodec()])],
        ),
    )
    z[:] = values
    return path, values


@pytest.mark.parametrize(
    ("name", "read"),
    [
        ("whole rows", lambda a, r: a.oindex[r, :]),
        ("column sub-box", lambda a, r: a.oindex[r, 8:24]),
        ("grid", lambda a, r: a.oindex[r, np.array([0, 5, 5, 17, 40])]),
        ("paired points", lambda a, r: a[r, np.array([0, 5, 5, 17, 40, 44])]),
        ("contiguous slice", lambda a, r: a.oindex[10:200, :]),
    ],
)
def test_every_shape_works_through_two_shard_levels(
    nested: tuple[Path, np.ndarray], entries: dict[str, int], name: str, read
) -> None:
    """The widening is orthogonal to how deep the sharding goes.

    `locate` walks one index per LEVEL and everything added here operates on the innermost
    decoded chunk, so nesting should not interact with it at all -- which is exactly the kind
    of "should" this repo has been wrong about, so it is asserted for every shape rather than
    argued.
    """
    path, _ = nested
    rows = np.array([1, 3, 3, 40, 300, 900])
    with zarr.config.set(CHUNK_UNIT):
        got = read(zarr.open_array(path, mode="r"), rows)
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}):
        theirs = read(zarr.open_array(path, mode="r"), rows)

    np.testing.assert_array_equal(got, theirs)
    assert entries["handle"] > 0, f"{name} did not take the chunk-unit path through two levels"
    assert entries["list"] == 0


# --- The description, checked against the bytes it claims -------------------------------
#
# Everything above reads a real array and compares values. That catches a description that is
# wrong AND reaches Rust; it does not catch one that is wrong and raises there, and it says
# nothing about which of the two forms `_chunk_unit_args` chose. Both regressions this exists
# for were in the description: an entry whose OUTPUT sub-box was strided, described as one
# run per index, and a `span` covering one column of a shard, described as all twelve. So the
# tuple itself is expanded here -- the way `build_chunk_unit_items` and `output_pieces` read
# it back -- and compared against the selection it came from, with no store and no Rust.


def _fold(starts, extents) -> int:
    """A coordinate as a row-major element offset."""
    offset, stride = 0, 1
    for axis in reversed(range(len(starts))):
        offset += int(starts[axis]) * stride
        stride *= int(extents[axis])
    return offset


def _batch(shard: tuple[int, ...], rows, boxes: list[tuple[int, int]]):
    """The entries zarr builds for `X[rows, lo:hi, ...]`, and the output shape.

    `rows` is either `(lo, hi)` -- a slice, which is what reaches the span form -- or a list
    of indices. The trailing axes are half-open ranges over the ARRAY, split across the shard
    grid exactly as zarr splits them, which is what produces a band when a range straddles a
    shard boundary.
    """
    import itertools
    import types

    sliced = isinstance(rows, tuple)
    row_list = list(range(*rows)) if sliced else list(rows)
    out_shape = (len(row_list), *[hi - lo for lo, hi in boxes])
    grids = [
        range(row_list[0] // shard[0], row_list[-1] // shard[0] + 1),
        *[range(lo // s, (hi - 1) // s + 1) for (lo, hi), s in zip(boxes, shard[1:])],
    ]
    entries = []
    for coord in itertools.product(*grids):
        take = [i for i, r in enumerate(row_list) if r // shard[0] == coord[0]]
        if not take:
            continue
        local = [row_list[i] - coord[0] * shard[0] for i in take]
        chunk_sel = [
            slice(local[0], local[-1] + 1) if sliced else np.array(local, dtype=np.int64)
        ]
        out_sel = [slice(take[0], take[-1] + 1)]
        for axis, ((lo, hi), extent) in enumerate(zip(boxes, shard[1:]), start=1):
            base = coord[axis] * extent
            a, b = max(lo, base), min(hi, base + extent)
            if a >= b:
                break
            chunk_sel.append(slice(a - base, b - base))
            out_sel.append(slice(a - lo, b - lo))
        else:
            entries.append(
                (
                    types.SimpleNamespace(path="c/" + "/".join(map(str, coord))),
                    types.SimpleNamespace(shape=tuple(shard)),
                    tuple(chunk_sel),
                    tuple(out_sel),
                    False,
                )
            )
    return entries, out_shape


def _wanted(entry, out_shape) -> list[tuple[int, int]]:
    """(shard element, output element) pairs the SELECTION asks for, paired in order."""
    import itertools

    _, spec, chunk_sel, out_sel, _ = entry
    axes = [
        list(s) if isinstance(s, np.ndarray) else list(range(s.start, s.stop))
        for s in chunk_sel
    ]
    src = [_fold(c, spec.shape) for c in itertools.product(*axes)]
    dst = [
        _fold(c, out_shape)
        for c in itertools.product(*[range(s.start, s.stop) for s in out_sel])
    ]
    assert len(src) == len(dst)
    return list(zip(src, dst))


def _described(args) -> list[tuple[int, int]]:
    """The same pairs, expanded the way Rust reads the description back.

    `span`: the trailing axes WHOLE on both sides -- `push_span` has nowhere to put anything
    else. `entry`: one run of `prod(out_widths[1:])` per index, at `starts` into the chunk row
    and at `out_starts[1:]` into the output row, which is `coords`/`run_len` and
    `output_pieces` respectively.
    """
    if args[0] == "span":
        _, _key, chunk_shape, shape, first, count, out_start, _inner = args
        row, out_row = prod(chunk_shape[1:]), prod(shape[1:])
        return [
            ((first + k) * row + j, (out_start + k) * out_row + j)
            for k in range(count)
            for j in range(row)
        ]
    _, _key, chunk_shape, shape, indices, out_starts, out_widths, _inner, starts = args
    row, out_row = prod(chunk_shape[1:]), prod(shape[1:])
    src_offset = _fold(starts, chunk_shape[1:])
    dst_offset = _fold(out_starts[1:], shape[1:])
    run = prod(out_widths[1:])
    return [
        (int(index) * row + src_offset + j, (out_starts[0] + i) * out_row + dst_offset + j)
        for i, index in enumerate(indices)
        for j in range(run)
    ]


@pytest.mark.parametrize(
    ("shard", "inner"),
    [
        ((4, 5), (4, 5)),
        ((2, 3, 4), (2, 3, 4)),
        # DIVIDED: the shard holds two inner chunks across, so a column range crossing the
        # boundary is described as several items rather than one. Without a case where
        # `inner` differs from `shard` this test cannot see a band at all.
        ((4, 12), (4, 6)),
        ((16, 12), (8, 6)),
        ((4, 12, 4), (4, 6, 4)),
        # An inner chunk that does NOT divide the shard, so the last band is short. A row
        # stride taken from the band width agrees with one taken from the inner chunk on
        # every other geometry here; this is where they part.
        ((4, 12), (4, 5)),
    ],
    ids=["rank-2", "rank-3", "divided", "divided-tall", "divided-rank-3", "ragged-inner"],
)
@pytest.mark.parametrize(
    "rows", [(0, 8), (1, 9), [0, 1, 3, 6], [5]], ids=["slice", "unaligned-slice", "list", "one"]
)
def test_a_description_names_exactly_its_own_bytes(shard, inner, rows) -> None:
    """Every column range over a shard grid, described and expanded back.

    Three things at once, and each has already been wrong: the pairs must match the selection
    (a `span` that covers one column of a shard fails here), the descriptions of one read must
    not claim a byte twice, and together they must cover the output exactly. A read that
    declines is fine -- this is about what is SAID, not about coverage of the fast path.
    """
    import itertools

    from zarrs.utils import _chunk_unit_args

    width = shard[1] * 3
    checked = 0
    for lo, hi in itertools.combinations(range(width + 1), 2):
        boxes = [(lo, hi), *[(0, shard[axis]) for axis in range(2, len(shard))]]
        entries, out_shape = _batch(shard, rows, boxes)
        pairs: list[tuple[int, int]] = []
        for entry in entries:
            pushes = _chunk_unit_args(entry, out_shape, (), tuple(inner))
            if pushes is None:
                break
            # One entry describes one item per band, so the entry's claim is the UNION of what
            # its pushes say -- an individual band names only its own columns.
            said = [pair for args in pushes for pair in _described(args)]
            assert sorted(said) == sorted(_wanted(entry, out_shape)), (
                f"{[a[0] for a in pushes]} for {entry[0].path} names bytes the selection did "
                f"not ask for: {entry[2]} -> {entry[3]} of {out_shape}"
            )
            pairs += said
        else:
            got = sorted(d for _, d in pairs)
            assert got == list(range(prod(out_shape))), (
                f"X[{rows}, {lo}:{hi}] of {shard}: the descriptions do not tile {out_shape}"
            )
            checked += 1
    assert checked, "every column range declined; this asserted nothing"


def test_a_strided_output_box_is_declined() -> None:
    """A partial LAST axis with a wider axis ahead of it is not one run per index.

    `output_pieces` models an item's output as ONE run per axis-0 index, so this must never be
    described -- it is the shape that failed 633 tests as "claims output bytes which run
    backwards". The chunk side of the same rule has its own test above; this is the OUTPUT
    side, which is a different question as soon as an entry stops spanning the whole extent.
    """
    import types

    from zarrs.utils import _chunk_unit_args

    entry = (
        types.SimpleNamespace(path="c/0/0/0"),
        types.SimpleNamespace(shape=(4, 5, 10)),
        (slice(0, 4), slice(0, 5), slice(0, 4)),
        (slice(0, 4), slice(0, 5), slice(0, 4)),
        False,
    )
    assert _chunk_unit_args(entry, (4, 5, 10), (), (4, 5, 10)) is None


@pytest.mark.parametrize(
    "selection",
    [
        pytest.param((5, 5), id="both-axes-scalar"),
        pytest.param((0, 0), id="both-axes-scalar-at-the-origin"),
        pytest.param((31, 23), id="both-axes-scalar-at-the-end"),
        pytest.param((slice(None), 3), id="scalar-on-a-trailing-axis"),
        pytest.param((5, slice(None)), id="scalar-row"),
        pytest.param((5, slice(6, 18)), id="scalar-row-and-a-column-band"),
        # A CoordinateIndexer with a constant column: `chunk_sel` arrives as
        # (array([1,5,20]), array([7,7,7])), which is the box `rows x 7:8` spelled as points.
        pytest.param((np.array([1, 5, 20]), 7), id="a-constant-column-is-a-scalar-axis"),
    ],
)
def test_a_scalar_axis_is_served(tmp_path: Path, entries: dict[str, int], selection) -> None:
    """zarr drops a scalar axis from the OUTPUT without saying so in `drop_axes`.

    Only axis 0 was rebuilt, so `X[5, 5]` -- both axes scalar, a 0-d output -- had no
    description and declined. Under `--strict` a decline is an error, which made a plain
    scalar read of a 2-D array fail outright; that is what `test_strict_mode` was catching.

    Rebuilding a dropped axis as an extent of one is exact rather than approximate: an axis of
    extent one contributes no stride, so the 0-d buffer of one element and the (1, 1) buffer
    of one element are the same bytes in the same order. The shard here is DIVIDED, so the
    rebuilt axis has to survive the band split too.
    """
    values = np.arange(32 * 24, dtype=np.float64).reshape(32, 24)
    path = tmp_path / "scalar.zarr"
    zarr.create_array(path, dtype=values.dtype, shape=values.shape, chunks=(8, 6),
                      shards=(16, 12))[:] = values

    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]

    np.testing.assert_array_equal(got, values[selection])
    assert entries["handle"] > 0, "a scalar axis should not send the batch to zarr-python"
    assert entries["list"] == 0


@pytest.mark.parametrize(
    "cols",
    [
        pytest.param(np.array([7]), id="one-column"),
        pytest.param(np.array([7, 7]), id="the-same-column-twice"),
    ],
)
def test_a_kept_constant_column_axis_is_not_rebuilt(tmp_path: Path, cols) -> None:
    """`oindex[rows, [7]]` keeps the column axis, so an extent of one would be a LIE.

    A constant trailing index array is read as a scalar axis, which is exact when the output
    dropped that axis and wrong when it kept it: the item would claim one output column against
    an output that has more, filling the right number of slots with the right bytes at the
    wrong stride. Wrong data, no error -- so the three-way length equality that refuses it is
    load-bearing, and this pins it.

    Verified by weakening the guard: with one operand dropped, both cases below are described
    with `out_widths=(3, 1)` against outputs of width 1 and 2.
    """
    values = np.arange(32 * 24, dtype=np.float64).reshape(32, 24)
    path = tmp_path / "kept.zarr"
    zarr.create_array(path, dtype=values.dtype, shape=values.shape, chunks=(8, 6),
                      shards=(16, 12))[:] = values

    rows = np.array([4, 5, 6])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r").oindex[rows, cols]

    np.testing.assert_array_equal(got, values[np.ix_(rows, cols)])
