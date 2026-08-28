"""Shard indexes are remembered for the life of a READ-ONLY array, and not otherwise.

Reading a shard index is a full-latency round trip taken on the calling thread, before any
job reaches a reader, so a shard is worth paying for once per array rather than once
per call. The pipeline is built per array, so its lifetime is the array's.

Gated on the store being read-only rather than invalidated on write, because a remembered
byte range that has gone stale does not raise -- it returns plausible data from a valid
file. A read-only store rejects writes, so nothing this process does can move those bytes.
An external writer still can and no cache here can see that, the same limitation
`file_handle_cache_size` documents, except this one cannot be reached at all unless the
caller opened the array read-only.

`mode="r"` gives a read-only store; `mode="r+"` and `mode="a"` do not.

Every selection below is an integer array on purpose: that is the only path that consults
the cache, so a slice-based version of any of these would pass no matter what the cache did.
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


def test_the_gate_signal_still_means_what_it_did(
    array: tuple[Path, np.ndarray],
) -> None:
    """`store.read_only` is what decides whether anything is cached, so a zarr change that
    flipped it would silently turn the cache on for writable arrays."""
    path, _ = array
    assert zarr.open_array(path, mode="r").store.read_only is True
    assert zarr.open_array(path, mode="r+").store.read_only is False
    assert zarr.open_array(path, mode="a").store.read_only is False


def test_repeated_integer_reads_agree(array: tuple[Path, np.ndarray]) -> None:
    """The second read of a shard uses the remembered index and must not differ."""
    path, truth = array
    selection = np.sort(np.random.default_rng(0).choice(N, size=300, replace=False))

    with zarr.config.set(ZARRS):
        z = zarr.open_array(path, mode="r")
        first = z[selection]
        for _ in range(3):
            np.testing.assert_array_equal(z[selection], first)
    np.testing.assert_array_equal(first, truth[selection])


@pytest.mark.parametrize("mode", ["r+", "a"])
def test_a_writable_array_reads_correctly_after_writing(
    array: tuple[Path, np.ndarray], mode: str
) -> None:
    """Where a write is possible nothing is remembered, so a read after a write is fresh."""
    path, truth = array
    selection = np.sort(np.random.default_rng(1).choice(N, size=300, replace=False))

    with zarr.config.set(ZARRS):
        z = zarr.open_array(path, mode=mode)
        # Populate anything that might be cached, through the path that would cache it.
        np.testing.assert_array_equal(z[selection], truth[selection])

        rewritten = (truth + 1_000.0).astype(np.float32)
        z[:] = rewritten

        np.testing.assert_array_equal(z[selection], rewritten[selection])
        np.testing.assert_array_equal(z[:], rewritten)


@pytest.mark.parametrize("mode", ["r+", "a"])
def test_a_partial_write_then_an_integer_read(
    array: tuple[Path, np.ndarray], mode: str
) -> None:
    """One inner chunk rewritten, then read back through the caching path."""
    path, truth = array
    expected = truth.copy()
    touched = np.arange(CHUNK, CHUNK + 200)
    untouched = np.arange(N - 200, N)

    with zarr.config.set(ZARRS):
        z = zarr.open_array(path, mode=mode)
        np.testing.assert_array_equal(z[touched], expected[touched])

        z[CHUNK : 2 * CHUNK] = 7.0
        expected[CHUNK : 2 * CHUNK] = 7.0

        np.testing.assert_array_equal(z[touched], expected[touched])
        np.testing.assert_array_equal(z[untouched], expected[untouched])
        np.testing.assert_array_equal(z[:], expected)


def test_a_write_that_falls_back_to_zarr_python(array: tuple[Path, np.ndarray]) -> None:
    """A write zarrs cannot describe is performed by zarr-python's pipeline instead, so
    `store_chunks_with_indices` never runs. Nothing is remembered for a writable store, so
    that path needs no invalidation of its own."""
    path, truth = array
    expected = truth.copy()
    selection = np.sort(np.random.default_rng(2).choice(N, size=200, replace=False))

    with zarr.config.set(ZARRS):
        z = zarr.open_array(path, mode="r+")
        np.testing.assert_array_equal(z[selection], expected[selection])

        # Non-contiguous integer-array write: refused by make_slice_selection, so it falls
        # back to zarr-python's pipeline.
        rows = np.array([5, 900, 3_000])
        z[rows] = -1.0
        expected[rows] = -1.0

        np.testing.assert_array_equal(z[selection], expected[selection])
        np.testing.assert_array_equal(z[:], expected)
