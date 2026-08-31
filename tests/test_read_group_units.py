"""Reads are counted in GROUPS, decodes in JOBS -- and the two are not the same number.

`batch_by_key` packs up to `GROUP_MAX_JOBS` same-key jobs into one group, and the job
channel carries groups. The widening loop's reader target must therefore be the group
count. It was the job count, which on any batch that grouped at all made
`live_readers < want_readers` true for the whole call: the loop could never reach its
target and polled at `WIDEN_POLL` from start to finish, ~5k `clock_nanosleep`/s per
in-flight call, buying nothing.

Values cannot test this -- the bytes are identical either way, only the syscall rate
moves. So the premise is asserted through the counter instead: on a read whose jobs
share a key, grouping must actually engage (jobs > groups). If it never engaged the two
targets would coincide and there would have been no bug to fix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import zarr

from zarrs._internal import read_merge_stats

if TYPE_CHECKING:
    from pathlib import Path

CHUNK_UNIT = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
SHAPE = (256, 64)
# Eight inner chunks to a shard, so a whole-shard read has eight same-key jobs to pack --
# exactly `GROUP_MAX_JOBS`, the case where jobs and groups differ most.
CHUNKS = (4, 64)
SHARDS = (32, 64)


def test_jobs_and_groups_are_different_counts(tmp_path: Path):
    values = np.arange(SHAPE[0] * SHAPE[1], dtype=np.float32).reshape(SHAPE)
    array = zarr.create_array(
        store=tmp_path / "a.zarr",
        shape=SHAPE,
        chunks=CHUNKS,
        shards=SHARDS,
        dtype="float32",
        compressors=None,
    )
    array[:] = values

    with zarr.config.set(CHUNK_UNIT):
        before_groups, before_jobs = read_merge_stats()
        # Rows spanning one shard, so every job carries the same key and packing applies.
        rows = np.arange(0, 32)
        got = array.oindex[rows, :]
        after_groups, after_jobs = read_merge_stats()

    np.testing.assert_array_equal(got, values[rows, :])

    groups = after_groups - before_groups
    jobs = after_jobs - before_jobs
    assert groups > 0, "the chunk-unit path did not run -- nothing was grouped"
    assert jobs > groups, (
        f"grouping never engaged: {jobs} jobs in {groups} groups. The reader target is the "
        "group count and the decoder target the job count; if these coincide the widening "
        "loop's units are untested."
    )


def test_scratch_pool_serves_a_later_call(tmp_path: Path):
    """The decode buffer must outlive the call, because the worker holding it does not.

    Same reasoning as above: values cannot see this. A pool that never serves and a pool
    that serves but buys nothing produce identical bytes and identical timings-within-noise,
    so the counter is the only thing that separates them.
    """
    from zarrs._internal import scratch_pool_stats

    values = np.arange(SHAPE[0] * SHAPE[1], dtype=np.float32).reshape(SHAPE)
    array = zarr.create_array(
        store=tmp_path / "b.zarr",
        shape=SHAPE,
        chunks=CHUNKS,
        shards=SHARDS,
        dtype="float32",
        compressors=None,
    )
    array[:] = values

    with zarr.config.set(CHUNK_UNIT):
        # The first read fills the pool as its workers exit; the second must be served by it.
        array.oindex[np.arange(0, 32), :]
        before_hits, _ = scratch_pool_stats()
        got = array.oindex[np.arange(32, 64), :]
        after_hits, _ = scratch_pool_stats()

    np.testing.assert_array_equal(got, values[32:64, :])
    assert after_hits > before_hits, (
        "no decode buffer was served from the pool on the second read -- the workers of the "
        "first call returned nothing, so every decoder is still allocating its own scratch"
    )
