"""The read/decode pool is one process-wide resource, so its widths are a process setting.

`zarr.config` is a context that each array snapshots when it is opened, so two arrays can
hold different `read_concurrency` values while only one of them can size a pool that the
whole process shares. The first array to ask sizes it; later arrays run at those widths and
are warned at OPEN, next to the config that set them, rather than failing on a read later.

This is the only test in the suite that enables the pool, and it must stay that way: the
widths are fixed for the life of the process, so a second test enabling it would inherit
whatever this one set and its own request would be the one warned about.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

N = 8_192
CHUNK = 1_024
SHARD = 4_096

POOL = {
    "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
    "codec_pipeline.read_decode_pool": True,
    "codec_pipeline.chunk_unit_indexing": True,
}


def make_array(path: Path) -> np.ndarray:
    values = np.arange(N, dtype=np.float32)
    z = zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(CHUNK,), shards=(SHARD,)
    )
    z[:] = values
    return values


def test_the_widths_are_a_process_setting(tmp_path: Path) -> None:
    """Both orderings in one test, because the process can only be sized once."""
    truth = make_array(tmp_path / "a")
    selection = np.sort(np.random.default_rng(0).choice(N, size=500, replace=False))

    # First array to ask sizes the pool. Nothing to warn about yet.
    with zarr.config.set(
        POOL
        | {"codec_pipeline.read_concurrency": 8, "codec_pipeline.decode_concurrency": 4}
    ):
        first = zarr.open_array(tmp_path / "a", mode="r")
        np.testing.assert_array_equal(first[selection], truth[selection])

    # A second array asking for different widths is told, at open, that it will not get them.
    with (
        zarr.config.set(
            POOL
            | {
                "codec_pipeline.read_concurrency": 64,
                "codec_pipeline.decode_concurrency": 16,
            }
        ),
        pytest.warns(UserWarning, match="already running 8 readers and 4 decoders"),
    ):
        second = zarr.open_array(tmp_path / "a", mode="r")

    # And it still reads correctly, at the process's widths.
    with zarr.config.set(POOL):
        np.testing.assert_array_equal(second[selection], truth[selection])


def test_matching_widths_do_not_warn(tmp_path: Path) -> None:
    """Asking for what the process already runs is not worth a warning -- which is the
    ordinary case, since `zarr.config` is normally set once for the whole process."""
    truth = make_array(tmp_path / "b")
    widths = {
        "codec_pipeline.read_concurrency": 8,
        "codec_pipeline.decode_concurrency": 4,
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with zarr.config.set(POOL | widths):
            array = zarr.open_array(tmp_path / "b", mode="r")
            np.testing.assert_array_equal(array[:100], truth[:100])

    assert not [
        w for w in caught if "read/decode pool is already running" in str(w.message)
    ]
