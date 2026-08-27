"""Shard indexes are remembered for the life of the array, and dropped when it is written.

Reading a shard index is a full-latency round trip taken on the calling thread, before any
job reaches the reader pool, so a shard is worth paying for once per array rather than once
per call. The pipeline is built per array, so its lifetime is the array's.

The hazard is the write path: writing a chunk rewrites its shard's index, and a remembered
byte range then points at whatever now occupies those bytes. That does not raise -- it
returns the wrong data -- so the invalidation is what these tests are mostly about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

N = 8_192
CHUNK = 1_024
SHARD = 4_096

ZARRS = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}


@pytest.fixture
def array(tmp_path: Path) -> tuple[Path, np.ndarray]:
    values = np.arange(N, dtype=np.float32)
    path = tmp_path / "a"
    with zarr.config.set(ZARRS):
        z = zarr.create_array(
            path,
            dtype=values.dtype,
            shape=values.shape,
            chunks=(CHUNK,),
            shards=(SHARD,),
        )
        z[:] = values
    return path, values


def test_repeated_reads_agree(array: tuple[Path, np.ndarray]) -> None:
    """The second read of a shard uses the remembered index, and must not differ."""
    path, truth = array
    selection = np.sort(np.random.default_rng(0).choice(N, size=300, replace=False))

    with zarr.config.set(ZARRS):
        z = zarr.open_array(path, mode="r")
        first = z[selection]
        for _ in range(3):
            np.testing.assert_array_equal(z[selection], first)
    np.testing.assert_array_equal(first, truth[selection])


def test_a_write_through_this_pipeline_invalidates(
    array: tuple[Path, np.ndarray],
) -> None:
    """Read (caching the index), write through the same pipeline, read again.

    The write moves the inner chunks within the shard, so a remembered range would now
    address the wrong bytes. Nothing about that is loud, hence the assertion on values.
    """
    path, truth = array
    selection = np.arange(0, 2_000)

    with zarr.config.set(ZARRS):
        z = zarr.open_array(path, mode="r+")
        np.testing.assert_array_equal(z[selection], truth[selection])

        # Rewrite the whole array with different data, through this same pipeline.
        rewritten = (truth + 1_000.0).astype(np.float32)
        z[:] = rewritten

        np.testing.assert_array_equal(z[selection], rewritten[selection])
        np.testing.assert_array_equal(z[:], rewritten)


def test_a_partial_write_then_read(array: tuple[Path, np.ndarray]) -> None:
    """A write to one shard invalidates the whole array's remembered indexes, so reads of
    the untouched shards have to keep working too."""
    path, truth = array
    expected = truth.copy()

    with zarr.config.set(ZARRS):
        z = zarr.open_array(path, mode="r+")
        np.testing.assert_array_equal(z[np.arange(0, 500)], expected[np.arange(0, 500)])

        # One inner chunk, inside the first shard only.
        z[CHUNK : 2 * CHUNK] = 7.0
        expected[CHUNK : 2 * CHUNK] = 7.0

        # The written shard, and one that was not written.
        np.testing.assert_array_equal(
            z[np.arange(0, 3_000)], expected[np.arange(0, 3_000)]
        )
        np.testing.assert_array_equal(
            z[np.arange(N - 3_000, N)], expected[np.arange(N - 3_000, N)]
        )
        np.testing.assert_array_equal(z[:], expected)
