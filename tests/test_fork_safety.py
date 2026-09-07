"""A read after `fork()` must not hang.

The two worker pools are process-wide. `fork()` copies their memory but only the calling
thread, so a child that inherits a built pool inherits workers that do not exist, and its
first `in_place_scope` parks on a latch nothing will ever signal.

It only happens when the parent read BEFORE forking, so it is data-dependent and passes any
test that does not fork. `waitpid` with a bound, because the bug under test is an unbounded
wait: without the timeout a regression hangs the suite instead of failing it.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from zarrs._internal import pool_sizes

if TYPE_CHECKING:
    from pathlib import Path

PIPELINE = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
LENGTH = 200


@pytest.fixture
def sharded_1d(tmp_path: Path) -> tuple[Path, np.ndarray]:
    values = np.arange(LENGTH, dtype=np.float64)
    path = tmp_path / "1d.zarr"
    zarr.create_array(
        store=path, shape=(LENGTH,), chunks=(8,), shards=(32,), dtype=values.dtype
    )[:] = values
    return path, values


def _read(path: Path, values: np.ndarray) -> None:
    selection = np.array([1, 9, 17, 40, 41])
    np.testing.assert_array_equal(
        zarr.open_array(path, mode="r")[selection], values[selection]
    )


def _wait(pid: int, seconds: float = 60.0) -> int:
    """`waitpid` with a bound. The bug under test is an unbounded wait."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return status
        time.sleep(0.05)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    pytest.fail(f"the child did not finish in {seconds}s -- it inherited the pools")


@pytest.mark.skipif(
    not hasattr(os, "register_at_fork"), reason="fork() is POSIX-only and so is the hook"
)
def test_a_read_in_a_forked_child_finishes(sharded_1d) -> None:
    """The parent reads FIRST, which is what builds the pools and arms the bug."""
    path, values = sharded_1d
    with zarr.config.set(PIPELINE):
        _read(path, values)
        assert pool_sizes() != (None, None), "the parent must have built the pools"

        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                _read(path, values)
            except BaseException:
                code = 1
            os._exit(code)
        assert os.waitstatus_to_exitcode(_wait(pid)) == 0


@pytest.mark.skipif(
    not hasattr(os, "register_at_fork"), reason="fork() is POSIX-only and so is the hook"
)
def test_the_parent_can_still_read_after_a_fork(sharded_1d) -> None:
    """The hook empties the parent's pools too, so it must rebuild rather than break."""
    path, values = sharded_1d
    with zarr.config.set(PIPELINE):
        _read(path, values)
        before = pool_sizes()

        pid = os.fork()
        if pid == 0:
            os._exit(0)
        _wait(pid)

        _read(path, values)
        assert pool_sizes() == before, "the parent must rebuild at the width it had"
