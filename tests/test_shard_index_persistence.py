"""Shard indexes are remembered for the life of a READ-ONLY array, and not otherwise.

Reading a shard index is a full-latency round trip on the calling thread, so a shard is worth
paying for once per array. The pipeline is built per array, so its lifetime is the array's.

Gated on the store being read-only rather than invalidated on write: a stale byte range does
not raise, it returns plausible data from a valid file. `mode="r"` gives a read-only store;
`mode="r+"` and `mode="a"` do not. An external writer can still move the bytes, the same
limitation `file_handle_cache_size` documents. Every selection below is an integer array
because that is the only path that consults the cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from zarrs._internal import shard_index_cache_stats

if TYPE_CHECKING:
    from pathlib import Path

N = 8_192
CHUNK = 1_024
SHARD = 4_096

ZARRS = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}


def cache_delta(before: tuple[int, int, int]) -> tuple[int, int, int]:
    """`(call_hits, array_hits, builds)` since `before`.

    A DELTA rather than a reset, so these assertions survive anything else in the process
    touching the same counters -- another test, another array, a different worker. The
    row-unit counters are read the same way, and for the same reason.
    """
    return tuple(now - was for now, was in zip(shard_index_cache_stats(), before, strict=True))


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


def test_repeated_integer_reads_agree(
    array: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """The second read of a shard uses the remembered index and must not differ."""
    path, truth = array
    selection = np.sort(np.random.default_rng(0).choice(N, size=300, replace=False))

    before = shard_index_cache_stats()
    with zarr.config.set(ZARRS):
        z = zarr.open_array(path, mode="r")
        first = z[selection]
        after_first = cache_delta(before)
        for _ in range(3):
            np.testing.assert_array_equal(z[selection], first)
    np.testing.assert_array_equal(first, truth[selection])
    # Otherwise every test here would pass while the cache was never consulted.
    assert entries["handle"] > 0

    # AND SO WOULD EVERY TEST HERE if the cache never engaged: values and timing are both
    # identical whether an index is remembered or re-read. `N // SHARD` shards are touched, so
    # the first read builds each one once and the three after it must build nothing.
    _, _, first_builds = after_first
    _, array_hits, builds = cache_delta(before)
    assert first_builds == N // SHARD, (
        f"the first read built {first_builds} indexes for {N // SHARD} shards"
    )
    assert builds == first_builds, (
        f"{builds - first_builds} indexes were rebuilt on a read-only array"
    )
    assert array_hits > 0, "nothing was served from the per-array cache"


@pytest.mark.parametrize("mode", ["r+", "a"])
def test_a_partial_write_then_an_integer_read(
    array: tuple[Path, np.ndarray], mode: str
) -> None:
    """Where a write is possible nothing is remembered: one inner chunk rewritten, then the
    touched and untouched regions both read back through the path that would have cached."""
    path, truth = array
    expected = truth.copy()
    touched = np.arange(CHUNK, CHUNK + 200)
    untouched = np.arange(N - 200, N)

    before = shard_index_cache_stats()
    with zarr.config.set(ZARRS):
        z = zarr.open_array(path, mode=mode)
        np.testing.assert_array_equal(z[touched], expected[touched])

        z[CHUNK : 2 * CHUNK] = 7.0
        expected[CHUNK : 2 * CHUNK] = 7.0

        np.testing.assert_array_equal(z[touched], expected[touched])
        np.testing.assert_array_equal(z[untouched], expected[untouched])
        np.testing.assert_array_equal(z[:], expected)

    # The values above are right whether or not a stale index was used, because the write did
    # not move any bytes. What makes this test about the cache is that nothing was remembered
    # ACROSS the reads: the per-array cache is the one that could go stale, and on a writable
    # store it must never fill.
    _, array_hits, builds = cache_delta(before)
    assert array_hits == 0, (
        f"{array_hits} indexes came from the per-array cache on a mode={mode!r} array"
    )
    assert builds > 0, "no index was read at all, so the counter proves nothing"


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
