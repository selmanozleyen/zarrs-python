"""The one question the whole read path hinges on, and its three possible answers.

Is this array sharded, and if so what is the innermost chunk shape? Getting it wrong does not
fail loudly: taking the SHARD shape as the decode unit builds items spanning several inner
chunks, which `locate` refuses -- and that refusal happens outside `read`'s `try`, so it is an
uncaught `PyRuntimeError` rather than a fallback.

It used to be answered TWICE -- once here by walking zarr's codec objects, once in Rust by
`ShardInfo::from_codec_chain` reading the bound codec chain -- and two derivations of one fact
can disagree. Rust owns it now. These pin the trichotomy directly, because every other test
reaches it only through a read, where `()` and a wrong tuple can look the same.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

ZARRS = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
SHAPE = (64, 48)


def _pipeline(path: Path):
    """The pipeline zarr built for this array, which is what holds the answer."""
    with zarr.config.set(ZARRS):
        return zarr.open_array(path, mode="r")._async_array.codec_pipeline


def test_an_unsharded_array_answers_the_empty_shape(tmp_path: Path) -> None:
    """`()` is NOT `None`. It means the chunk is its own decode unit, and only a batch entry
    knows what that chunk is -- so the read is served, it just takes the shape from elsewhere."""
    path = tmp_path / "plain.zarr"
    zarr.create_array(path, shape=SHAPE, dtype="float32", chunks=(8, 48))[:] = np.zeros(SHAPE, "float32")
    assert _pipeline(path)._inner_chunk_shape == ()


def test_a_sharded_array_answers_the_INNER_shape(tmp_path: Path) -> None:
    """Not the shard's. This is the distinction the whole path is built on."""
    path = tmp_path / "sharded.zarr"
    zarr.create_array(
        path, shape=SHAPE, dtype="float32", chunks=(8, 48), shards=(32, 48)
    )[:] = np.zeros(SHAPE, "float32")
    assert _pipeline(path)._inner_chunk_shape == (8, 48)


def test_a_nested_shard_answers_the_INNERMOST_shape(tmp_path: Path) -> None:
    """Two levels of sharding, and the answer is the bottom one -- the first `chunk_shape`
    found walking down is the outer shard's, which is what a naive descent returns."""
    from zarr.codecs import BytesCodec, ShardingCodec

    path = tmp_path / "nested.zarr"
    zarr.create_array(
        path,
        shape=SHAPE,
        dtype="float32",
        chunks=(32, 48),
        serializer=ShardingCodec(
            chunk_shape=(16, 48),
            codecs=[ShardingCodec(chunk_shape=(8, 48), codecs=[BytesCodec()])],
        ),
    )[:] = np.zeros(SHAPE, "float32")
    assert _pipeline(path)._inner_chunk_shape == (8, 48)


@pytest.mark.filterwarnings("ignore:Combining a `sharding_indexed` codec")
def test_a_codec_beside_the_sharding_codec_answers_None(tmp_path: Path) -> None:
    """`None` is refuse, and it is a different answer from `()`.

    A codec AFTER the sharding codec compresses the whole shard, so the shard index's byte
    ranges no longer address it. A codec BEFORE it reorders the elements inside an inner chunk.
    Both make one inner chunk unreadable on its own.
    """
    from zarr.codecs import BytesCodec, ShardingCodec, TransposeCodec

    after = tmp_path / "after.zarr"
    zarr.create_array(
        after,
        shape=SHAPE,
        dtype="float32",
        chunks=(32, 48),
        serializer=ShardingCodec(chunk_shape=(8, 48), codecs=[BytesCodec()]),
    )[:] = np.zeros(SHAPE, "float32")
    # `shards=` nests the compressor inside; an explicit serializer leaves it outside, which is
    # the layout under test.
    before = tmp_path / "before.zarr"
    zarr.create_array(
        before,
        shape=SHAPE,
        dtype="float32",
        chunks=(32, 48),
        filters=[TransposeCodec(order=(1, 0))],
        serializer=ShardingCodec(chunk_shape=(8, 48), codecs=[BytesCodec()]),
    )[:] = np.zeros(SHAPE, "float32")
    assert _pipeline(after)._inner_chunk_shape is None, "a codec after sharding"
    assert _pipeline(before)._inner_chunk_shape is None, "a codec before sharding"
