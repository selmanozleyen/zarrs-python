from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import pytest
import zarr

SELECTIONS = {"whole-chunks": np.s_[:], "ragged": np.s_[3:130, 7:200]}


@dataclass
class ChildResult:
    code: int
    error: str


@pytest.fixture(autouse=True)
def _no_silent_fallback():
    # Without this, pipeline.py may hand the batch to zarr-python and the test passes on it.
    with zarr.config.set({"codec_pipeline.strict": True}):
        yield


def run_in_child(work, deadline: float = 30.0) -> ChildResult:
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        code = 0
        try:
            work()
        except BaseException as exc:  # noqa: BLE001
            os.write(write_fd, f"{type(exc).__name__}: {exc}".encode()[:4000])
            code = 3
        finally:
            os.close(write_fd)
        os._exit(code)

    os.close(write_fd)
    end = time.monotonic() + deadline
    status = None
    while time.monotonic() < end:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        time.sleep(0.02)
    else:
        # A child that finished inside the last sleep is not a hang, so look once more.
        done, status = os.waitpid(pid, os.WNOHANG)
        if not done:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            os.close(read_fd)
            pytest.fail(f"the child did not finish in {deadline}s, it deadlocked")

    message = os.read(read_fd, 4096).decode(errors="replace")
    os.close(read_fd)
    return ChildResult(os.waitstatus_to_exitcode(status), message)


def seeded(path: str, *, shards: bool = False) -> None:
    kwargs = {"shards": (128, 128)} if shards else {}
    array = zarr.create_array(
        store=path,
        shape=(256, 256),
        chunks=(32, 32),
        dtype="int16",
        zarr_format=3,
        **kwargs,
    )
    # The parent decodes here, which is what builds the pool a child would inherit.
    array[:] = np.arange(256 * 256, dtype="int16").reshape(256, 256)


@pytest.fixture
def http_url(tmp_path):
    seeded(str(tmp_path / "a.zarr"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
            "-d",
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), 0.1).close()
                break
            except OSError:
                time.sleep(0.05)
        else:
            pytest.skip("the local http server did not come up")
        yield f"http://127.0.0.1:{port}/a.zarr"
    finally:
        server.kill()
        server.wait()
