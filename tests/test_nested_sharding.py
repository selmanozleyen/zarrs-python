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
        "one inner chunk": np.arange(INNER, 2 * INNER),
        # Across subshards AND outer shards, so both indexes are walked repeatedly.
        "scattered": np.sort(rng.choice(N, size=500, replace=False)),
    }


@pytest.mark.parametrize("name", list(selections()))
def test_nested_reads_are_correct(
    nested: tuple[Path, np.ndarray], entries: dict[str, int], name: str
) -> None:
    path, truth = nested
    selection = selections()[name]
    with zarr.config.set(ZARRS):
        got = zarr.open_array(path, mode="r")[selection]
    np.testing.assert_array_equal(got, truth[selection])
    assert entries["handle"] > 0, "nested fell back instead of descending"
    assert entries["list"] == 0


def test_shallow_still_takes_the_fast_path(
    shallow: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """Nested support must not cost the single-level path its handle."""
    path, truth = shallow
    selection = np.sort(np.random.default_rng(2).choice(N, size=300, replace=False))
    with zarr.config.set(ZARRS):
        got = zarr.open_array(path, mode="r")[selection]
    np.testing.assert_array_equal(got, truth[selection])
    assert entries["handle"] > 0, "the single-level path stopped using the handle"
    assert entries["list"] == 0
