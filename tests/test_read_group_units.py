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

    COMPRESSED, deliberately, and it caught a real change when it was not. An uncompressed
    inner chunk takes the raw path, where the read IS the answer and `decode_one` is a
    `copy_from_slice` with no scratch at all -- so this test built its array with
    `compressors=None` and then asserted that a buffer pool it could never reach had served
    one. It failed the moment raw jobs stopped being handed to the decode pool, which is
    exactly the behaviour that commit intended. The pool is for real decodes; test it on one.
    """
    from zarrs._internal import scratch_pool_stats

    values = np.arange(SHAPE[0] * SHAPE[1], dtype=np.float32).reshape(SHAPE)
    array = zarr.create_array(
        store=tmp_path / "b.zarr",
        shape=SHAPE,
        chunks=CHUNKS,
        shards=SHARDS,
        dtype="float32",
        compressors=zarr.codecs.BloscCodec(cname="lz4", clevel=1),
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


def test_the_raw_path_never_reaches_the_scratch_pool(tmp_path: Path):
    """An uncompressed chunk has nothing to decode, so no decode worker should be asked.

    The read IS the answer on the raw path -- `decode_one` is a `copy_from_slice` on bytes
    the reader just fetched, still in that core's cache. Handing it to the decode pool buys
    a queue push, a steal and a wake to run a memcpy, and it showed up as an uncompressed
    array getting monotonically SLOWER as the decode pool grew: 134.0 -> 129.2 -> 117.2
    M nnz/s at 32 -> 128 -> 512 threads.

    Asserted through the counter because values cannot see it: both paths return identical
    bytes and only the thread hand-off differs.
    """
    from zarrs._internal import scratch_pool_stats

    values = np.arange(SHAPE[0] * SHAPE[1], dtype=np.float32).reshape(SHAPE)
    array = zarr.create_array(
        store=tmp_path / "c.zarr",
        shape=SHAPE,
        chunks=CHUNKS,
        shards=SHARDS,
        dtype="float32",
        compressors=None,
    )
    array[:] = values

    with zarr.config.set(CHUNK_UNIT):
        before = scratch_pool_stats()
        got = array.oindex[np.arange(0, 32), :]
        after = scratch_pool_stats()

    np.testing.assert_array_equal(got, values[0:32, :])
    assert after == before, (
        f"the raw path touched the scratch pool: {before} -> {after}. An uncompressed chunk "
        "needs no scratch, so asking a decode worker for one is a hand-off bought for a memcpy"
    )
