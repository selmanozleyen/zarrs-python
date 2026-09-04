"""Does a forked child read an INHERITED remote array? Diagnostic, not a contract."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import zarr

URL = "https://raw.githubusercontent.com/zarrs/zarrs/main/zarrs/tests/data/array_write_read.zarr/group/array"
PIPELINE = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}


@pytest.mark.skipif(not hasattr(os, "register_at_fork"), reason="POSIX only")
def test_a_forked_child_reads_the_inherited_remote_array() -> None:
    try:
        with zarr.config.set(PIPELINE):
            array = zarr.open(URL)
            expected = np.asarray(array[:])
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no network: {exc}")

    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            with zarr.config.set(PIPELINE):
                # the INHERITED array and its inherited connection pool
                np.testing.assert_array_equal(np.asarray(array[:]), expected, strict=False)
        except BaseException:
            code = 1
        os._exit(code)

    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            assert os.waitstatus_to_exitcode(status) == 0, "child errored"
            return
        time.sleep(0.05)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    pytest.fail("HUNG: the child could not read the inherited remote array in 45s")
