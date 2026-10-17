"""Orthogonal reads that are not rows of one run go through `push_box`, never zarr-python."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

STRICT = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline", "codec_pipeline.strict": True}
LAYOUTS = {
    "2d": ((64, 48), (8, 6), (16, 12)),
    "3d-chunked-trailing": ((32, 24, 20), (8, 6, 5), (16, 12, 10)),
    "3d-whole-trailing": ((32, 24, 20), (8, 24, 20), (16, 24, 20)),
}


@pytest.fixture(params=list(LAYOUTS), ids=list(LAYOUTS))
def layout(request) -> str:
    return request.param


@pytest.fixture(params=[True, False], ids=["compressed", "uncompressed"])
def stored(tmp_path: Path, layout: str, request) -> tuple[Path, np.ndarray]:
    shape, chunks, shards = LAYOUTS[layout]
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    path = tmp_path / f"{layout}.zarr"
    zarr.create_array(
        path,
        dtype=values.dtype,
        shape=shape,
        chunks=chunks,
        shards=shards,
        compressors="auto" if request.param else None,
    )[:] = values
    return path, values


def _selectors(extent: int) -> dict[str, object]:
    return {
        "sorted": np.array([1, 2, 3, extent // 2, extent - 1]),
        "unsorted": np.array([extent - 1, 2, extent // 2, 1, 3]),
        "repeats": np.array([3, 1, 3, extent - 2]),
        "all": slice(None),
        "part": slice(2, extent - 3),
        "strided": slice(1, extent, 3),
    }


def test_every_orthogonal_mix_is_served_and_right(stored: tuple[Path, np.ndarray]) -> None:
    path, values = stored
    kinds = list(_selectors(4))
    with zarr.config.set(STRICT):
        z = zarr.open_array(path, mode="r")
        for combo in itertools.product(kinds, repeat=values.ndim):
            index = tuple(_selectors(n)[k] for k, n in zip(combo, values.shape, strict=True))
            axes = [np.arange(n)[i] for i, n in zip(index, values.shape, strict=True)]
            np.testing.assert_array_equal(
                z.oindex[index], values[np.ix_(*axes)], err_msg=str(combo)
            )


@pytest.mark.parametrize("compressed", [True, False], ids=["compressed", "uncompressed"])
@pytest.mark.parametrize("shards", [None, (16, 12, 10)], ids=["unsharded", "sharded"])
def test_whole_chunks_never_written_read_as_fill(tmp_path: Path, compressed, shards) -> None:
    """A whole-array read decodes each chunk into its box; a chunk never written is fill."""
    shape, chunks = (32, 24, 20), (8, 6, 5)
    path = tmp_path / "partial.zarr"
    z = zarr.create_array(
        path, dtype="f8", shape=shape, chunks=chunks, shards=shards, fill_value=-1.5,
        compressors="auto" if compressed else None,
    )
    values = np.full(shape, -1.5)
    values[:16, 6:18, :] = np.arange(16 * 12 * 20, dtype="f8").reshape(16, 12, 20)
    z[:16, 6:18, :] = values[:16, 6:18, :]
    with zarr.config.set(STRICT):
        np.testing.assert_array_equal(zarr.open_array(path, mode="r")[:], values)
