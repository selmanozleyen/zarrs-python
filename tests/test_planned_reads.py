"""The read planner must return exactly what the unplanned path returns.

It reads the shard index itself and issues its own coalesced byte ranges, so it bypasses the
sharding partial decoder entirely. Everything about a selection can change how it is planned --
which inner chunks are touched, whether their ranges merge, whether a chunk is absent from the
shard -- and none of it may change the bytes that come back.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

ZARRS = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
PLANNED = ZARRS | {
    "codec_pipeline.integer_array_indexing": True,
    "codec_pipeline.shard_index_cache_size": 8,
}


@pytest.fixture(
    params=[
        pytest.param(((4, 5), (16, 20)), id="2d"),
        # One inner chunk per shard: the index has a single entry and nothing can coalesce.
        pytest.param(((16, 20), (16, 20)), id="one-inner-chunk"),
        # Inner chunks narrower than the shard, so a row spans several of them.
        pytest.param(((4, 5), (8, 20)), id="wide-shard"),
    ]
)
def layout(request):
    return request.param


@pytest.fixture
def sharded(tmp_path: Path, layout) -> tuple[Path, np.ndarray]:
    chunks, shards = layout
    shape = (32, 40)
    expected = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    path = tmp_path / "foo.zarr"
    z = zarr.create_array(
        path, dtype=np.float64, shape=shape, chunks=chunks, shards=shards
    )
    z[:] = expected
    return path, expected


SELECTIONS = [
    pytest.param(np.s_[...], id="everything"),
    pytest.param(np.s_[3:7, 2:9], id="box"),
    pytest.param(np.s_[5, :], id="single-row"),
    pytest.param(np.s_[:, 7], id="single-col"),
    pytest.param(np.s_[0:1, 0:1], id="single-element"),
    pytest.param(np.s_[31:32, 39:40], id="last-element"),
]


@pytest.mark.parametrize("selection", SELECTIONS)
def test_planned_matches_unplanned(
    sharded: tuple[Path, np.ndarray], selection: object
) -> None:
    path, expected = sharded
    with zarr.config.set(PLANNED):
        planned = zarr.open_array(path, mode="r")[selection]
    np.testing.assert_array_equal(planned, expected[selection])


INTEGER_SELECTIONS = [
    pytest.param(np.array([0, 3, 4, 5, 17, 30]), id="scattered-rows"),
    pytest.param(np.arange(32), id="every-row"),
    pytest.param(np.array([31]), id="last-row"),
    pytest.param((np.array([1, 2, 30]), slice(None)), id="rows-full-slice"),
    pytest.param((np.array([1, 2, 30]), slice(3, 22)), id="rows-slice"),
    pytest.param((slice(None), np.array([0, 1, 9, 39])), id="cols"),
]


@pytest.mark.parametrize("selection", INTEGER_SELECTIONS)
def test_planned_matches_unplanned_for_integer_arrays(
    sharded: tuple[Path, np.ndarray], selection: object
) -> None:
    path, expected = sharded
    with zarr.config.set(PLANNED):
        planned = zarr.open_array(path, mode="r")[selection]
    np.testing.assert_array_equal(planned, expected[selection])


def test_planned_reads_absent_chunks_as_fill_value(tmp_path: Path) -> None:
    """An inner chunk that was never written has no byte range to read, and a shard that was
    never written has no index at all. Both must come back as the fill value."""
    path = tmp_path / "sparse.zarr"
    z = zarr.create_array(
        path,
        dtype=np.float64,
        shape=(32, 40),
        chunks=(4, 5),
        shards=(16, 20),
        fill_value=-7.0,
    )
    # One shard written, and inside it only part of one inner chunk.
    z[0:2, 0:2] = 1.0
    expected = np.full((32, 40), -7.0)
    expected[0:2, 0:2] = 1.0

    with zarr.config.set(PLANNED):
        planned = zarr.open_array(path, mode="r")
        np.testing.assert_array_equal(planned[...], expected)
        np.testing.assert_array_equal(
            planned[np.array([0, 5, 20])], expected[[0, 5, 20]]
        )


def test_uncompressed_reads_only_the_bytes_asked_for(tmp_path: Path) -> None:
    """With no compressor an element's position in an inner chunk is arithmetic, so a read of
    one element must fetch only that element -- not the chunk containing it.

    Timing cannot assert that. Truncation can: the shard index is put at the *start* so it
    survives, and the file is cut inside the last inner chunk. Reading an element before the cut
    succeeds only if the read is confined to it, while reading the whole chunk needs bytes that
    no longer exist.
    """
    shape, chunks = 64, 8
    itemsize = 8  # float64
    expected = np.arange(shape, dtype=np.float64)
    path = tmp_path / "flat_index_first.zarr"
    z = zarr.create_array(
        path,
        dtype=np.float64,
        shape=(shape,),
        # The shard is the chunk, and the sharding codec is the serializer, which is the only
        # way to ask for the index at the start.
        chunks=(shape,),
        serializer=zarr.codecs.ShardingCodec(
            chunk_shape=(chunks,),
            codecs=[zarr.codecs.BytesCodec()],
            index_location="start",
        ),
        compressors=None,
        filters=None,
    )
    z[:] = expected

    # Which inner chunk sits last in the file is the writer's choice, not the index order, so
    # read the index rather than assume it.
    shard = path / "c" / "0"
    raw = bytearray(shard.read_bytes())
    n_inner = shape // chunks
    pairs = [
        struct.unpack_from("<QQ", raw, 16 * i) for i in range(n_inner)
    ]  # index_location="start"
    last = max(range(n_inner), key=lambda i: pairs[i][0])
    offset, length = pairs[last]
    assert length == chunks * itemsize, (
        "uncompressed, so a chunk is exactly its elements"
    )
    # Keep the first half of that chunk, drop the rest.
    shard.write_bytes(raw[: offset + length // 2])

    with zarr.config.set(PLANNED):
        array = zarr.open_array(path, mode="r")
        lo = last * chunks
        surviving = slice(lo, lo + chunks // 2)
        # Inside the surviving half: only a sub-chunk read can answer this.
        np.testing.assert_array_equal(array[surviving], expected[surviving])
        # Spanning the cut: the bytes are genuinely gone, which is what makes the above a proof.
        with pytest.raises(Exception, match="fill whole buffer"):
            array[lo : lo + chunks]


def test_planning_is_skipped_when_the_array_is_not_sharded(tmp_path: Path) -> None:
    """Planning describes a shard layout, so an unsharded array must fall through to the
    normal path rather than misread its chunks."""
    path = tmp_path / "flat.zarr"
    expected = np.arange(32 * 40, dtype=np.float64).reshape(32, 40)
    z = zarr.create_array(path, dtype=np.float64, shape=(32, 40), chunks=(4, 5))
    z[:] = expected

    with zarr.config.set(PLANNED):
        planned = zarr.open_array(path, mode="r")
        np.testing.assert_array_equal(planned[...], expected)
        np.testing.assert_array_equal(planned[3:7, 2:9], expected[3:7, 2:9])
