from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import zarr

from .conftest import SELECTIONS, ChildResult, run_in_child, seeded

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")


def assert_refused(result: ChildResult) -> None:
    assert result.code == 3, f"expected a refusal, got exit {result.code}: {result.error}"
    assert "RuntimeError" in result.error, result.error
    assert "forkserver" in result.error, result.error


@pytest.mark.parametrize("selection", SELECTIONS.values(), ids=SELECTIONS.keys())
@pytest.mark.parametrize("shards", [False, True], ids=["chunks", "sharded"])
def test_a_forked_child_is_refused_a_read(tmp_path, selection, *, shards: bool) -> None:
    path = str(tmp_path / "a.zarr")
    seeded(path, shards=shards)
    zarr.open_array(path, mode="r")[selection]

    assert_refused(run_in_child(lambda: zarr.open_array(path, mode="r")[selection]))


def test_a_forked_child_is_refused_a_write(tmp_path) -> None:
    path = str(tmp_path / "a.zarr")
    seeded(path)

    def write() -> None:
        zarr.open_array(path, mode="r+")[:64, :64] = np.full((64, 64), 7, dtype="int16")

    assert_refused(run_in_child(write))
    np.testing.assert_array_equal(
        np.asarray(zarr.open_array(path, mode="r")[:2, :2]),
        np.array([[0, 1], [256, 257]], dtype="int16"),
    )


def test_a_forked_child_is_refused_an_inherited_remote_store(http_url) -> None:
    array = zarr.open_array(http_url, mode="r")
    array[:8, :8]

    assert_refused(run_in_child(lambda: array[:]))


def test_a_forked_child_is_refused_a_remote_store_it_opens_itself(http_url) -> None:
    zarr.open_array(http_url, mode="r")[:8, :8]

    assert_refused(run_in_child(lambda: zarr.open_array(http_url, mode="r")[:]))


def test_an_unforked_process_is_unaffected(tmp_path) -> None:
    path = str(tmp_path / "a.zarr")
    seeded(path)
    expected = np.arange(256 * 256, dtype="int16").reshape(256, 256)
    np.testing.assert_array_equal(np.asarray(zarr.open_array(path, mode="r")[:]), expected)
    zarr.open_array(path, mode="r+")[:64, :64] = np.full((64, 64), 7, dtype="int16")


def test_a_child_that_forked_before_any_use_is_allowed(tmp_path) -> None:
    path = str(tmp_path / "a.zarr")
    seeded(path)
    total = int(np.arange(256 * 256, dtype="int16").sum())

    # A fresh interpreter, because this one has already used zarrs and so is armed for good.
    script = textwrap.dedent(f"""
        import os, sys
        pid = os.fork()
        if pid == 0:
            import numpy as np, zarr
            zarr.config.set({{"codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
                             "codec_pipeline.strict": True}})
            got = int(np.asarray(zarr.open_array({path!r}, mode="r")[:]).sum())
            os._exit(0 if got == {total} else 4)
        sys.exit(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
    """)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=120)
    assert done.returncode == 0, done.stderr.decode()
