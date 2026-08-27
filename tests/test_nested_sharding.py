"""Sharding inside sharding: the innermost chunk is still the decode unit.

A nested layout puts a shard inside a shard, so locating one innermost chunk means walking
TWO offset/size tables: the outer shard's index gives the subshard's extent, and the
subshard's own index -- which lives inside that extent -- gives the chunk's. The alternative
is to treat the subshard as the decode unit, which would decode many innermost chunks to keep
the elements of one, and that is the amplification this whole path exists to avoid.

These tests assert VALUES first and the path second, in that order of importance. Nested is
allowed to be slower than shallow; it is not allowed to be wrong, and shallow is not allowed
to regress because nested exists.

zarr only produces this layout when a sharding codec is passed as `serializer` with a nested
`ShardingCodec` in its `codecs`; `shards=` gives one level. `compressors=None` matters: the
default compressor would land OUTSIDE the sharding codec, which
`test_a_codec_after_sharding_is_refused` covers and which this path refuses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from zarrs._internal import CodecPipelineImpl

if TYPE_CHECKING:
    from pathlib import Path

N = 8_192
SHARD = 4_096
SUB = 1_024
INNER = 256

ZARRS = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}


@pytest.fixture
def entries(monkeypatch) -> dict[str, int]:
    """How many batches took each Rust entry point, so "which path served this" is asserted
    rather than assumed."""
    counts = {"handle": 0, "list": 0}
    for name, key in (
        ("retrieve_chunk_items_and_apply_index", "handle"),
        ("retrieve_chunks_and_apply_index", "list"),
    ):
        original = getattr(CodecPipelineImpl, name)

        def wrapper(self, *args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(CodecPipelineImpl, name, wrapper)
    return counts


@pytest.fixture
def nested(tmp_path: Path) -> tuple[Path, np.ndarray]:
    from zarr.codecs import BytesCodec, ShardingCodec

    values = np.arange(N, dtype=np.float32)
    path = tmp_path / "nested"
    z = zarr.create_array(
        path,
        shape=values.shape,
        chunks=(SHARD,),
        dtype="float32",
        compressors=None,
        serializer=ShardingCodec(
            chunk_shape=(SUB,),
            codecs=[ShardingCodec(chunk_shape=(INNER,), codecs=[BytesCodec()])],
        ),
    )
    z[:] = values
    return path, values


@pytest.fixture
def shallow(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """One level, for the comparison that matters: nested must not change this."""
    values = np.arange(N, dtype=np.float32)
    path = tmp_path / "shallow"
    z = zarr.create_array(
        path, shape=values.shape, chunks=(INNER,), shards=(SHARD,), dtype="float32"
    )
    z[:] = values
    return path, values


def selections() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        # Every element of one innermost chunk.
        "one inner chunk": np.arange(INNER, 2 * INNER),
        # Scattered across subshards AND across outer shards.
        "scattered": np.sort(rng.choice(N, size=500, replace=False)),
        # Inside a single subshard, so the outer index is consulted once.
        "one subshard": np.arange(SUB, SUB + 200),
        # Crosses an outer shard boundary.
        "across shards": np.arange(SHARD - 100, SHARD + 100),
        "with duplicates": np.repeat(
            np.sort(rng.choice(N, size=100, replace=False)), 3
        ),
        "every second": np.arange(0, N, 2),
    }


@pytest.mark.parametrize("name", list(selections()))
def test_nested_reads_are_correct(nested: tuple[Path, np.ndarray], name: str) -> None:
    """Values, whichever path serves them. This must hold before and after the descent
    exists, which is what makes it the test worth writing first."""
    path, truth = nested
    selection = selections()[name]
    with zarr.config.set(ZARRS):
        got = zarr.open_array(path, mode="r")[selection]
    np.testing.assert_array_equal(got, truth[selection])


def test_nested_matches_zarr_python_exactly(nested: tuple[Path, np.ndarray]) -> None:
    """The same selection through both pipelines, in case zarrs and zarr-python disagree
    about a nested layout in some way the truth array would not reveal."""
    path, _ = nested
    selection = np.sort(np.random.default_rng(1).choice(N, size=300, replace=False))

    with zarr.config.set(ZARRS):
        through_zarrs = zarr.open_array(path, mode="r")[selection]
    with zarr.config.set(
        {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}
    ):
        through_zarr_python = zarr.open_array(path, mode="r")[selection]
    np.testing.assert_array_equal(through_zarrs, through_zarr_python)


def test_shallow_still_takes_the_fast_path(
    shallow: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """The regression guard. Nested support must not cost the single-level path its handle."""
    path, truth = shallow
    selection = np.sort(np.random.default_rng(2).choice(N, size=300, replace=False))
    with zarr.config.set(ZARRS):
        got = zarr.open_array(path, mode="r")[selection]
    np.testing.assert_array_equal(got, truth[selection])
    assert entries["handle"] > 0, "the single-level path stopped using the handle"
    assert entries["list"] == 0
