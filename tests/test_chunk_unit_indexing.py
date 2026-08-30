"""Integer-array selections served one inner chunk at a time.

A sorted integer selection is grouped by the unit the codec chain decodes -- the inner chunk --
so a chunk is read once however many of its elements are wanted; a coordinate list costs two
allocations and a partial-decode call PER ELEMENT. The path is narrow: one 1-D integer axis,
non-negative and non-decreasing, against a contiguous output slice. Anything else falls back and
must still be right, so both directions are asserted -- with `entries` saying which Rust entry
point served the read, since a silent fall back passes every values-only check.
"""

from __future__ import annotations

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


def test_a_column_split_inner_chunk_falls_back(
    tmp_path: Path, entries: dict[str, int]
) -> None:
    """A 2-D array whose inner chunk is NARROWER than the array declines.

    The rank-N path takes axes after the first whole, and that is not a formality: with the
    columns divided, one selected row is no longer one contiguous run in the decoded chunk,
    its output rows are no longer one contiguous range, and the shard grid no longer holds a
    single subchunk on the column axis -- so the descent along axis 0 would address the wrong
    subchunk. Values alone would pass a version that mis-grouped and got lucky, hence the
    entry-point check.
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
    assert entries["handle"] == 0, "a column-split 2-D selection reached the chunk-unit path"


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
