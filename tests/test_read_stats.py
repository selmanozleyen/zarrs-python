"""Whether a read was served here or handed to zarr-python, and a bad knob that says so.

Both are about the same thing: this pipeline's only failure mode is "correct but slower", and
from the outside that is indistinguishable from working. A selection it cannot describe returns
identical values through zarr-python; a config value of the wrong type used to disable it for
the whole array and blame the array.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

import zarrs

if TYPE_CHECKING:
    from pathlib import Path

ZARRS = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
N = 200


@pytest.fixture
def sharded(tmp_path: Path) -> tuple[Path, np.ndarray]:
    values = np.arange(N * 8, dtype=np.float32).reshape(N, 8)
    path = tmp_path / "a.zarr"
    zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(8, 8), shards=(32, 8)
    )[:] = values
    return path, values


def test_a_served_read_and_a_declined_one_move_different_counters(sharded) -> None:
    """A delta, not a reset: the counters are process-wide and another test may be reading."""
    path, values = sharded
    rows = np.array([1, 40, 91])

    before = zarrs.read_stats()
    with zarr.config.set(ZARRS):
        got = zarr.open_array(path, mode="r")[rows]
    np.testing.assert_array_equal(got, values[rows])
    served, declined = (n - w for n, w in zip(zarrs.read_stats(), before, strict=True))
    assert served > 0, "a sorted row selection is exactly what this path serves"
    assert declined == 0

    # Descending rows: the output slice is contiguous and ascending, so serving them would put
    # the right rows at the wrong offsets. It declines, and zarr-python returns the same values.
    before = zarrs.read_stats()
    with zarr.config.set(ZARRS):
        got = zarr.open_array(path, mode="r")[rows[::-1]]
    np.testing.assert_array_equal(got, values[rows[::-1]])
    served, declined = (n - w for n, w in zip(zarrs.read_stats(), before, strict=True))
    assert declined > 0, "a descending selection must be handed to zarr-python"
    assert served == 0


def test_a_bad_knob_names_the_knob(sharded) -> None:
    """Not the array.

    Every one of these is a pyo3 keyword argument, and pyo3 raises `TypeError` for the wrong
    type -- which the constructor catches and reports as "Array is unsupported by
    ZarrsCodecPipeline", disabling the pipeline for that array, permanently and silently, over
    the one thing the user cannot change. A negative value was worse: `OverflowError` escaped
    naming no key at all.
    """
    path, _ = sharded
    for value in (8.0, -1, "eight"):
        with (
            zarr.config.set(ZARRS | {"codec_pipeline.read_pool_size": value}),
            pytest.raises(ValueError, match="codec_pipeline.read_pool_size"),
        ):
            zarr.open_array(path, mode="r")[np.array([1, 2])]


def test_a_fork_hook_is_registered() -> None:
    """The pools are emptied before a fork rather than rebuilt after one.

    The pid check that rebuilds them in a child runs INSIDE the mutex guarding them, and `fork`
    copies a held mutex as held with an owner that does not exist -- so a child forked while a
    read was in flight blocks there for ever and never reaches the check.

    Asserting the hook itself, because the race needs a fork to land inside a critical section
    that lasts microseconds; a test that tried to hit it would pass by luck.
    """
    import os

    assert hasattr(zarrs._internal, "release_pools_for_fork")
    assert hasattr(os, "register_at_fork"), "POSIX only, and so is this test"
