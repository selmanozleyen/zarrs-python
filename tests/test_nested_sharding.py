"""Sharding inside sharding: the INNERMOST chunk is still the decode unit.

Locating one innermost chunk means walking TWO offset/size tables -- the outer shard's index
gives the subshard's extent, and the subshard's own index gives the chunk's. Treating the
subshard as the unit would decode many chunks to keep the elements of one.

zarr only produces this layout from a `ShardingCodec` serializer nesting another; `shards=`
gives one level, and `compressors=None` keeps the compressor from landing OUTSIDE the sharding
codec (which `test_a_codec_after_sharding_is_refused` covers, and this path refuses).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

N = 8_192
SHARD = 4_096
SUB = 1_024
INNER = 256

ZARRS = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}


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
    """One level of sharding."""
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
        # Every element of one innermost chunk, at depth 2.
        "one inner chunk": np.arange(INNER, 2 * INNER),
        # Scattered across subshards AND across outer shards, so both indexes are walked
        # repeatedly -- which is what a single-subshard and a shard-crossing case each did once.
        "scattered": np.sort(rng.choice(N, size=500, replace=False)),
    }


@pytest.mark.parametrize("name", list(selections()))
def test_nested_reads_are_correct(
    nested: tuple[Path, np.ndarray], entries: dict[str, int], name: str
) -> None:
    """Values first, then the path: the fallback got the values right before the descent existed."""
    path, truth = nested
    selection = selections()[name]
    with zarr.config.set(ZARRS):
        got = zarr.open_array(path, mode="r")[selection]
    np.testing.assert_array_equal(got, truth[selection])
    assert entries["handle"] > 0, "nested fell back instead of descending"


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
