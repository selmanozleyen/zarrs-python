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
        # Only integer arrays are grouped; a slice is already one subset.
        pytest.param(slice(CHUNK - 10, 2 * CHUNK + 10), id="slice"),
    ],
)
def test_ineligible_selections_decline_and_are_still_right(
    array: tuple[Path, np.ndarray], entries: dict[str, int], selection
) -> None:
    """Declined selections must still return the right data, down whichever path takes them."""
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


def test_two_dimensional_selection_falls_back(
    tmp_path: Path, entries: dict[str, int]
) -> None:
    """The grouping is the 1-D path: a 2-D array must decline rather than mis-group, and values
    alone would pass a version that mis-grouped and got lucky -- hence the entry-point check."""
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
    assert entries["handle"] == 0, "a 2-D selection reached the chunk-unit path"


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
