"""A forked child must not deadlock on a pool whose threads it did not inherit (#171).

The parent decodes first on purpose: that is what builds the pool, and a pool that was never
built cannot be inherited broken. Each case runs in a child with a deadline, because the bug
under test is a hang -- an assertion would never be reached.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import zarr

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")


def _run_in_child(work, deadline: float = 30.0) -> None:
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            work()
        except BaseException:  # noqa: BLE001
            code = 1
        os._exit(code)

    end = time.monotonic() + deadline
    while time.monotonic() < end:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            assert os.waitstatus_to_exitcode(status) == 0, "the child raised"
            return
        time.sleep(0.02)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    pytest.fail(f"the child did not finish in {deadline}s -- it deadlocked")


def _seeded(path: str, *, shards: bool) -> zarr.Array:
    kwargs = {"shards": (128, 128)} if shards else {}
    array = zarr.create_array(
        store=path, shape=(256, 256), chunks=(32, 32), dtype="int16",
        zarr_format=3, **kwargs,
    )
    # The parent uses the codec, which is what builds the pool the child inherits.
    array[:] = np.arange(256 * 256, dtype="int16").reshape(256, 256)
    return array


@pytest.mark.parametrize("shards", [False, True], ids=["chunks", "sharded"])
def test_a_forked_child_can_read(tmp_path, shards: bool) -> None:
    path = str(tmp_path / "a.zarr")
    _seeded(path, shards=shards)
    expected = np.asarray(zarr.open_array(path, mode="r")[:])

    def read() -> None:
        got = np.asarray(zarr.open_array(path, mode="r")[:])
        np.testing.assert_array_equal(got, expected)

    _run_in_child(read)


def test_a_forked_child_can_write(tmp_path) -> None:
    """The write path took a different route to the same pool, and so has its own case."""
    path = str(tmp_path / "a.zarr")
    _seeded(path, shards=False)

    def write() -> None:
        zarr.open_array(path, mode="r+")[:64, :64] = np.zeros((64, 64), dtype="int16")

    _run_in_child(write)
    np.testing.assert_array_equal(
        np.asarray(zarr.open_array(path, mode="r")[:64, :64]),
        np.zeros((64, 64), dtype="int16"),
    )
