"""A store opened read-only must refuse writes, as zarr-python's own pipeline does.

zarr-python enforces this in `StorePath.set`, which this pipeline bypasses: it is handed a
`StoreConfig` and builds its own Rust store, which is writable whatever mode the array was
opened in. Without the guard a write to a `mode="r"` array SUCCEEDS here and raises through
the default pipeline -- a silent divergence, and silent data loss for anyone relying on the
mode to protect an array.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def array(tmp_path: Path) -> tuple[Path, np.ndarray]:
    values = np.arange(64, dtype=np.float32)
    path = tmp_path / "a"
    zarr.create_array(path, dtype=values.dtype, shape=values.shape, chunks=(16,))[:] = values
    return path, values


def test_write_to_a_read_only_array_raises(array: tuple[Path, np.ndarray]) -> None:
    path, values = array
    z = zarr.open_array(path, mode="r")
    with pytest.raises(Exception, match="read-only"):
        z[0:16] = -1.0
    np.testing.assert_array_equal(zarr.open_array(path, mode="r")[:], values)


def test_a_writable_array_still_writes(array: tuple[Path, np.ndarray]) -> None:
    path, values = array
    z = zarr.open_array(path, mode="r+")
    z[0:16] = -1.0
    expected = values.copy()
    expected[0:16] = -1.0
    np.testing.assert_array_equal(zarr.open_array(path, mode="r")[:], expected)
