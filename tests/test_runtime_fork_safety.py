"""A read after `fork()` must not hang, and the parent must survive one.

The tokio runtime is process-wide. `fork()` copies its memory but only the calling thread, so
a child inherits worker and driver threads that do not exist and its first `block_on` waits on
a readiness nothing will ever signal.

Only the object-store and HTTP stores reach tokio -- `zarrs_filesystem` is sync -- so these
exercise it through an HTTP-backed array, and skip when there is no network rather than
pretending a local store proves anything.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import zarr

URL = "https://raw.githubusercontent.com/zarrs/zarrs/main/zarrs/tests/data/array_write_read.zarr/group/array"
PIPELINE = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}

pytestmark = pytest.mark.skipif(
    not hasattr(os, "register_at_fork"), reason="fork() is POSIX-only and so is the hook"
)


@pytest.fixture
def remote_array():
    """An HTTP-backed array, which is what routes through tokio."""
    try:
        with zarr.config.set(PIPELINE):
            array = zarr.open(URL)
            array[:]  # the read that builds the runtime
            return array
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no network for the tokio-backed store: {exc}")


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
    pytest.fail(f"the child did not finish in {seconds}s -- it inherited the runtime")


def test_a_read_in_a_forked_child_finishes(remote_array) -> None:
    """The parent reads FIRST, which is what builds the runtime and arms the bug.

    Delete the `register_at_fork` line in `__init__.py` and this hangs, which is the point:
    a test that cannot fail when the fix is removed is not testing the fix.
    """
    expected = np.asarray(remote_array[:])

    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            with zarr.config.set(PIPELINE):
                np.testing.assert_array_equal(
                    np.asarray(zarr.open(URL)[:]), expected, strict=False
                )
        except BaseException:  # noqa: BLE001
            code = 1
        os._exit(code)

    assert os.waitstatus_to_exitcode(_wait(pid)) == 0


def test_the_parent_can_still_read_after_a_fork(remote_array) -> None:
    """The hook empties the PARENT's slot too, so it must rebuild rather than break.

    A release that dropped the runtime out from under a live store would leave the parent with
    a handle to a shut-down reactor -- worse than the bug being fixed, and only visible here.
    """
    expected = np.asarray(remote_array[:])

    pid = os.fork()
    if pid == 0:
        os._exit(0)
    _wait(pid)

    with zarr.config.set(PIPELINE):
        np.testing.assert_array_equal(
            np.asarray(remote_array[:]), expected, strict=False
        )
