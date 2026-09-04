"""A remote store inherited across `fork()` is refused, rather than blocking."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import zarr

URL = "https://raw.githubusercontent.com/zarrs/zarrs/main/zarrs/tests/data/array_write_read.zarr/group/array"


def _child_exit_code(array: zarr.Array) -> int:
    """Read the inherited array in a forked child; 0 iff it was refused by name."""
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            array[:]
        except RuntimeError as exc:
            code = 0 if "inherited it across a fork" in str(exc) else 2
        except BaseException:  # noqa: BLE001
            code = 3
        os._exit(code)

    # A refusal is immediate. Anything that takes seconds is the block this prevents.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.05)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    pytest.fail("the child blocked instead of being refused")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
def test_a_forked_child_refuses_an_inherited_remote_array() -> None:
    try:
        array = zarr.open(URL)
        expected = np.asarray(array[:])
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no network: {exc}")

    assert _child_exit_code(array) == 0

    # The guard reads a pid; it does not disturb the process that opened the store.
    np.testing.assert_array_equal(np.asarray(array[:]), expected, strict=False)
