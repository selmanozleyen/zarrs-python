"""Integer-array selections served one inner chunk at a time.

A sorted integer selection is grouped by the unit the codec
chain actually decodes -- the inner chunk -- so a chunk is read once, decoded once and gathered
once however many of its elements are wanted. Handing zarrs a coordinate list instead costs two
allocations and a partial-decode call PER ELEMENT.

The path is narrow on purpose: one 1-D integer axis, non-negative and non-decreasing, against a
contiguous output slice. Anything else has to fall back and still return the right data, so both
directions are asserted here.

Which Rust entry point served the read is asserted too, not assumed. A batch that is entirely
chunk-unit goes over as one `ChunkItems` handle rather than as one Python object per item, and a
silent fall back to the list path would pass every correctness check while doing none of that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr


if TYPE_CHECKING:
    from pathlib import Path

# Not a multiple of CHUNK, so the last inner chunk is short and the subset has to be clamped.
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
        # Lands in the short final chunk, where lo + inner overruns the array.
        "short last chunk": np.arange(N - 100, N),
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


def test_decreasing_selection_falls_back_and_is_still_right(
    array: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """A backward step would mean one item per element, so the path declines it. The read still
    has to produce the right answer, down whichever path takes it."""
    path, truth = array
    selection = np.array([9_000, 40, 8_000, 39])

    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]

    np.testing.assert_array_equal(got, truth[selection])
    assert entries["handle"] == 0


def test_a_plain_slice_is_untouched(
    array: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """Only integer-array selections are grouped; a slice is already one subset."""
    path, truth = array

    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[CHUNK - 10 : 2 * CHUNK + 10]

    np.testing.assert_array_equal(got, truth[CHUNK - 10 : 2 * CHUNK + 10])
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
    """Chunks that were never written must read back as the fill value.

    Every other test here writes the array in full, so the branch that produces fill for a
    missing inner chunk -- no read, no decode, no worker -- is never taken. A partially
    written sharded array is not an edge case for this path, it is the shape a sparse dataset
    has, and the selection below deliberately spans both sides of the boundary."""
    values = np.arange(N, dtype=np.float32)
    path = tmp_path / f"sparse-{name.replace(' ', '-')}"
    z = zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(CHUNK,), shards=(SHARD,),
        fill_value=np.float32(-7),
    )
    z[written] = values[written]

    expected = np.full(N, -7, dtype=np.float32)
    expected[written] = values[written]

    # Straddles the boundary: elements from written chunks, from unwritten ones, and from
    # the chunk containing the boundary itself.
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


def test_two_dimensional_selection_falls_back(tmp_path: Path, entries: dict[str, int]) -> None:
    """The grouping is the 1-D path. A 2-D array must decline rather than mis-group.

    Asserting the values alone would pass for a version that mis-grouped and happened to be
    right, so the entry point is asserted too: none of these batches may take the handle."""
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
def test_a_codec_after_sharding_is_refused(tmp_path: Path) -> None:
    """Sharding with an OUTER compressor must not take the chunk-unit path.

    A bytes-to-bytes codec after the sharding codec compresses the whole shard, so a byte
    range into the file addresses compressed bytes rather than the shard the index describes.

    Without the refusal this raises `RuntimeError: the checksum is invalid` -- the crc32c in
    the default index codecs catches it, because the tail of a compressed shard does not
    checksum as an index. So it is loud, but it reads as data corruption rather than as an
    unsupported layout. It would only be SILENT if the index codecs carried no checksum,
    which is legal but not the default. Either way the refusal turns it into a clean fallback.
    """
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
