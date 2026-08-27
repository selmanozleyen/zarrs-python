"""The read/decode pool is one process-wide resource, so its widths are a process setting.

`zarr.config` is a context that each array snapshots when it is opened, so two arrays can
hold different `read_concurrency` values while only one of them can size a pool the whole
process shares. The first sharded array to ask sizes it; later arrays run at those widths
and are warned at OPEN, next to the config that set them, rather than failing on a read.

Each case runs in a FRESH INTERPRETER. The widths are fixed for the life of the process, so
a test sharing one with any other test could not control which array asks first -- and what
is asserted here is precisely about who asks first.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

PREAMBLE = """
import warnings
import numpy as np
import zarr

path = sys.argv[1]
values = np.arange(8192, dtype=np.float32)
narrow = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
          "codec_pipeline.read_concurrency": 8,
          "codec_pipeline.decode_concurrency": 4}
wide = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
        "codec_pipeline.read_concurrency": 64,
        "codec_pipeline.decode_concurrency": 16}
selection = np.sort(np.random.default_rng(0).choice(8192, size=500, replace=False))

with zarr.config.set(narrow):
    z = zarr.create_array(path, dtype=values.dtype, shape=values.shape,
                          chunks=(1024,), shards=(4096,))
    z[:] = values
"""


def run(body: str, tmp_path) -> str:
    """Run a snippet in a new interpreter, returning its stdout."""
    script = "import sys\n" + PREAMBLE + textwrap.dedent(body)
    done = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "a")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_the_first_sharded_array_sizes_the_pool(tmp_path) -> None:
    """And a later array asking for different widths is warned at open, then reads fine."""
    out = run(
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with zarr.config.set(narrow):
                first = zarr.open_array(path, mode="r")
                assert np.array_equal(first[selection], values[selection])
        assert not [w for w in caught if "already running" in str(w.message)], "warned early"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with zarr.config.set(wide):
                second = zarr.open_array(path, mode="r")
        pool = [w for w in caught if "already running" in str(w.message)]
        assert len(pool) == 1, [str(w.message) for w in caught]
        assert "8 readers and 4 decoders" in str(pool[0].message), str(pool[0].message)

        # It still reads correctly, at the process's widths.
        with zarr.config.set(wide):
            assert np.array_equal(second[selection], values[selection])
        print("OK")
        """,
        tmp_path,
    )
    assert out.strip().endswith("OK")


def test_matching_widths_do_not_warn(tmp_path) -> None:
    """Asking for what the process already runs is the ordinary case: config is normally set
    once, before anything is opened, and every array then agrees with the pool."""
    out = run(
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with zarr.config.set(narrow):
                for _ in range(3):
                    array = zarr.open_array(path, mode="r")
                    assert np.array_equal(array[:100], values[:100])
        assert not [w for w in caught if "already running" in str(w.message)], [
            str(w.message) for w in caught
        ]
        print("OK")
        """,
        tmp_path,
    )
    assert out.strip().endswith("OK")
