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


@pytest.mark.parametrize(
    "rows", [[6, 5], [7, 5, 6, 0], [31, 30, 29, 16, 15, 1, 0]], ids=str
)
def test_unsorted_rows_on_the_raw_path(tmp_path: Path, rows: list[int]) -> None:
    """An uncompressed array reads each row's own bytes; scattered rows land where asked."""
    values = np.arange(32 * 24, dtype=np.float64).reshape(32, 24)
    path = tmp_path / "raw.zarr"
    zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(8, 24), shards=(16, 24),
        compressors=None,
    )[:] = values
    with zarr.config.set(STRICT):
        got = zarr.open_array(path, mode="r")[np.array(rows)]
    np.testing.assert_array_equal(got, values[rows])
