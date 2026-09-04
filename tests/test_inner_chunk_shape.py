"""The one question the whole read path hinges on, and its three possible answers.

Is this array sharded, and if so what is the innermost chunk shape? Getting it wrong does not
fail loudly: taking the SHARD shape as the decode unit builds items spanning several inner
chunks, which `locate` refuses -- and that refusal happens outside `read`'s `try`, so it is an
uncaught `PyRuntimeError` rather than a fallback.

It used to be answered TWICE -- once in Python by walking zarr's codec objects, once in Rust by
`ShardInfo::from_codec_chain` reading the bound codec chain -- and two derivations of one fact
can disagree. Rust owns it now. These pin the trichotomy directly, because every other test
reaches it only through a read, where `()` and a wrong tuple can look the same.

Geometries are borrowed from the tests that already exercise each layout, rather than invented:
`compressors=None` matters, because the default compressor lands OUTSIDE an explicit sharding
serializer, which is itself one of the refuse cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

ZARRS = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
N = 8_192
SHARD = 4_096
SUB = 1_024
INNER = 256


def _answer(path: Path):
    """What the pipeline zarr built for this array says the decode unit is."""
    with zarr.config.set(ZARRS):
        return zarr.open_array(path, mode="r")._async_array.codec_pipeline._inner_chunk_shape


def test_an_unsharded_array_answers_the_empty_shape(tmp_path: Path) -> None:
    """`()` is NOT `None`. It means the chunk is its own decode unit, and only a batch entry
    knows what that chunk is -- so the read is served, it just takes the shape from elsewhere."""
    path = tmp_path / "plain"
    zarr.create_array(
        path, shape=(N,), dtype="float32", chunks=(SHARD,)
    )[:] = np.arange(N, dtype="float32")
    assert _answer(path) == ()


def test_a_sharded_array_answers_the_INNER_shape(tmp_path: Path) -> None:
    """Not the shard's. That distinction is what the whole path is built on."""
    path = tmp_path / "sharded"
    zarr.create_array(
        path, shape=(N,), dtype="float32", chunks=(INNER,), shards=(SHARD,)
    )[:] = np.arange(N, dtype="float32")
    assert _answer(path) == (INNER,)


def test_a_nested_shard_answers_the_INNERMOST_shape(tmp_path: Path) -> None:
    """Two levels, and the answer is the bottom one -- the first `chunk_shape` found walking
    down is the OUTER shard's, which is what a naive descent returns."""
    from zarr.codecs import BytesCodec, ShardingCodec

    path = tmp_path / "nested"
    zarr.create_array(
        path,
        shape=(N,),
        dtype="float32",
        chunks=(SHARD,),
        compressors=None,
        serializer=ShardingCodec(
            chunk_shape=(SUB,),
            codecs=[ShardingCodec(chunk_shape=(INNER,), codecs=[BytesCodec()])],
        ),
    )[:] = np.arange(N, dtype="float32")
    assert _answer(path) == (INNER,)


# zarr warns that this layout disables partial reads. That IS the layout under test.
@pytest.mark.filterwarnings("ignore:Combining a `sharding_indexed` codec")
def test_a_codec_AFTER_sharding_answers_None(tmp_path: Path) -> None:
    """`None` is refuse, and it is a different answer from `()`.

    An explicit sharding serializer leaves the default compressor OUTSIDE it, which compresses
    the whole shard -- so the shard index's byte ranges no longer address the shard and one
    inner chunk cannot be read on its own.
    """
    from zarr.codecs import BytesCodec, ShardingCodec

    path = tmp_path / "after"
    zarr.create_array(
        path,
        shape=(N,),
        dtype="float32",
        chunks=(SHARD,),
        serializer=ShardingCodec(chunk_shape=(INNER,), codecs=[BytesCodec()]),
    )[:] = np.arange(N, dtype="float32")
    assert _answer(path) is None


@pytest.mark.filterwarnings("ignore:Combining a `sharding_indexed` codec")
def test_a_codec_BEFORE_sharding_answers_None(tmp_path: Path) -> None:
    """A `transpose` outside the sharding codec reorders the elements of the whole shard.

    `serializer=` is what puts it outside; the ordinary `shards=` + `filters=` spelling nests it
    INSIDE, where the inner chunk still decodes to the array's own order and is served.
    """
    from zarr.codecs import BytesCodec, ShardingCodec, TransposeCodec

    values = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    path = tmp_path / "transposed"
    zarr.create_array(
        path,
        shape=values.shape,
        dtype=values.dtype,
        chunks=(32, 32),
        filters=[TransposeCodec(order=(1, 0))],
        serializer=ShardingCodec(chunk_shape=(8, 32), codecs=[BytesCodec()]),
    )[:] = values
    assert _answer(path) is None
