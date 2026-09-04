"""Opening an array must not start a thread pool. The read path builds its own, on first use."""

from __future__ import annotations

import os

import numpy as np
import pytest
import zarr

pytestmark = pytest.mark.skipif(
    not os.path.isdir("/proc/self/task"), reason="thread count needs /proc"
)


def _threads() -> int:
    return len(os.listdir("/proc/self/task"))


def test_opening_an_array_starts_no_threads(tmp_path) -> None:
    path = tmp_path / "a.zarr"
    zarr.create_array(
        store=str(path), shape=(64, 64), chunks=(16, 16), dtype="float32"
    )[:] = np.zeros((64, 64), dtype="float32")

    before = _threads()
    array = zarr.open(str(path), mode="r")
    assert array.shape == (64, 64)
    # `rayon::current_num_threads` used to be read here, which starts rayon's global pool:
    # one thread per core, in every process that opens an array, used by no read.
    assert _threads() - before < 4, "opening an array started a pool"

    # ... and the read path still has its own, so the read itself does work.
    np.testing.assert_array_equal(np.asarray(array[:]), np.zeros((64, 64), dtype="float32"))
    assert _threads() > before, "the read path built no pool at all"
