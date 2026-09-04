"""A read after `fork()` must not hang.

The two pools are process-wide and persistent, which is the point -- but `fork()` copies the
parent's memory and only the CALLING thread. A child that inherits a built pool inherits
worker threads that do not exist in it, and the first `in_place_scope` parks on a latch
nothing will ever signal.

That is not a corner case. This path exists for minibatch loading, and
`torch.utils.data.DataLoader(num_workers > 0)` forks by default on Linux. It only hangs when
the parent read BEFORE forking, so it is data-dependent and invisible to every test that does
not fork -- which, before this file, was every test.

The child runs `os._exit` rather than returning: it must not run pytest's teardown, flush the
parent's buffers, or report a second time.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

import zarrs

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork"), reason="fork() is POSIX-only and this is a fork test"
)

PIPELINE = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
LENGTH = 200
SELECTION = np.array([1, 9, 17, 40, 41])
#: Generous: the failure is a PERMANENT hang, so any finite bound separates it from slow.
CHILD_TIMEOUT_SECONDS = 60.0


@pytest.fixture
def sharded_1d(tmp_path: Path) -> tuple[Path, np.ndarray]:
    values = np.arange(LENGTH, dtype=np.float64)
    path = tmp_path / "fork.zarr"
    array = zarr.create_array(
        store=path, shape=(LENGTH,), chunks=(8,), shards=(32,), dtype=values.dtype
    )
    array[:] = values
    return path, values


def _read(path: Path, values: np.ndarray) -> None:
    """One read through the chunk-unit path, which is what builds the pools."""
    got = zarr.open_array(path, mode="r")[SELECTION]
    np.testing.assert_array_equal(got, values[SELECTION])


def _wait(pid: int) -> int:
    """`waitpid` with a bound, because the bug under test is an unbounded wait."""
    deadline = time.monotonic() + CHILD_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return status
        time.sleep(0.05)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    pytest.fail(
        f"the child never finished within {CHILD_TIMEOUT_SECONDS}s -- it inherited pools "
        "whose worker threads do not exist in it"
    )


def test_a_read_in_a_forked_child_finishes(sharded_1d) -> None:
    path, values = sharded_1d
    with zarr.config.set(PIPELINE):
        # THE PARENT READS FIRST. Without this the child inherits no pools and builds its
        # own, which is the case that never hung and the reason this bug is data-dependent.
        _read(path, values)

        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                _read(path, values)
                code = 0
            finally:
                os._exit(code)

    status = _wait(pid)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, (
        f"the child read failed or crashed: waitpid status {status}"
    )


# Asking for a width the pools were not built at is the point of the second block, and saying
# so is what the warning is for -- so it is expected here, not a fault.
@pytest.mark.filterwarnings("ignore:.*pool_size = .* was ignored:UserWarning")
def test_the_pool_sizes_survive_a_fork(sharded_1d) -> None:
    """A fork must not silently resize the parent's pools.

    The before-fork hook empties the slot so a child cannot inherit a held lock. Written
    without remembering the sizes, that also RESIZED the parent: pools built at one width for
    the array that did the first read were rebuilt at whatever the next array asked for, so
    `pool_sizes()` began reporting the new width and the "was ignored" warning began firing for
    an array that had been honoured right up until an unrelated fork.

    The function whose whole job is to make "the knob arrived" checkable would have been the one
    lying, and only on a workload that forks -- which is the workload this path exists for.
    """
    path, values = sharded_1d
    absurd = 3  # small, specific, and nothing else in this file asks for it

    with zarr.config.set(PIPELINE | {"codec_pipeline.read_pool_size": absurd}):
        _read(path, values)
    built = zarrs.pool_sizes()
    assert built != (None, None), "a read must have built the pools"

    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)

    # The fork dropped them; the next read rebuilds. It must come back at the SAME width, not
    # at whatever this block asks for.
    with zarr.config.set(PIPELINE | {"codec_pipeline.read_pool_size": absurd + 40}):
        _read(path, values)
    assert zarrs.pool_sizes() == built, (
        "the fork resized the parent's pools, so pool_sizes() no longer describes the array "
        "that did the first read"
    )
