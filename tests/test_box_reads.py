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


@pytest.mark.parametrize("shards", [None, (16, 12, 10)], ids=["unsharded", "sharded"])
@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(np.array([1, 2, 9, 30]), id="sorted"),
        pytest.param(np.array([30, 2, 9, 1]), id="unsorted"),
        pytest.param(np.array([9, 9, 2, 2]), id="repeats"),
    ],
)
def test_slabs_of_a_compressed_array_are_copied_as_boxes(tmp_path: Path, shards, rows) -> None:
    """Only axis 0 fragmented: each chunk decodes once and each slab is copied as a box."""
    shape, chunks = (32, 24, 20), (8, 6, 5)
    path = tmp_path / "slabs.zarr"
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    zarr.create_array(path, dtype="f8", shape=shape, chunks=chunks, shards=shards)[:] = values
    with zarr.config.set(STRICT):
        z = zarr.open_array(path, mode="r")
        np.testing.assert_array_equal(z.oindex[rows], values[rows])
        np.testing.assert_array_equal(z.oindex[rows, 3:20, 1:18], values[rows][:, 3:20, 1:18])


def test_slabs_through_chunks_never_written_read_as_fill(tmp_path: Path) -> None:
    shape, chunks = (32, 24, 20), (8, 6, 5)
    path = tmp_path / "partial.zarr"
    z = zarr.create_array(path, dtype="f8", shape=shape, chunks=chunks, fill_value=-2.0)
    values = np.full(shape, -2.0)
    values[:8, 6:12, :] = 7.0
    z[:8, 6:12, :] = 7.0
    rows = np.array([20, 1, 5, 31])
    with zarr.config.set(STRICT):
        np.testing.assert_array_equal(zarr.open_array(path, mode="r").oindex[rows], values[rows])
