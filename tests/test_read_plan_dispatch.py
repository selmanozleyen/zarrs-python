"""The planned path decodes each read as it lands, not once a chunk is whole.

Timing cannot assert this: a decode that waits for its siblings still finishes, just later, and on
a loaded machine "later" is indistinguishable from noise. The in-flight byte budget makes it
categorical instead.

A read is admitted only while `in_flight + bytes <= budget`, and its bytes are retired when *its
own* decode completes. Set the budget below one unit and every unit is admitted alone, which
separates the two designs cleanly:

  - decode per read (what this pipeline does): unit 1 is admitted, decodes, releases its bytes,
    unit 2 is admitted, and the call finishes.
  - decode per chunk (the barrier this chain removed): unit 1 arrives and is buffered until its
    chunk is whole, so its bytes are never retired, so unit 2 is never admitted, and the call
    hangs forever.

So a barrier reintroduced anywhere between the fetch pool and the decode pool turns these tests
into a deadlock rather than a slowdown, which is why the read runs on a thread with a deadline.

The two compressor settings exercise different halves. Uncompressed, a unit reads only the bytes
asked for and its decode is fused with its read. Compressed, the inner chunk is atomic, so the
read lands on the fetch pool and hands the decode to the decode pool -- that is the split with a
deadlock to have.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr
from zarr import create_array

if TYPE_CHECKING:
    from pathlib import Path

INNER = 64
SHARD = 4096

PLANNED = {
    "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
    "codec_pipeline.integer_array_indexing": True,
}
# Below any unit, so the budget admits exactly one at a time whatever a unit turns out to weigh.
# Sizing it in bytes would tie the test to whether a unit is a chunk or only the bytes wanted.
ONE_UNIT_AT_A_TIME = {"codec_pipeline.fetch_byte_budget": 1}


@pytest.fixture(params=["none", "zstd"])
def sharded_array(request, tmp_path: Path) -> Path:
    array = create_array(
        store=tmp_path,
        shape=(SHARD,),
        chunks=(INNER,),
        shards=(SHARD,),
        dtype="int32",
        compressors=None if request.param == "none" else "auto",
    )
    array[:] = np.arange(SHARD, dtype="int32")
    return tmp_path


def read_with_deadline(path: Path, config: dict, selection: np.ndarray) -> np.ndarray:
    """Read on a thread, so a pipeline that deadlocks fails the test instead of hanging it."""
    result: list[np.ndarray] = []
    error: list[BaseException] = []

    def read() -> None:
        try:
            with zarr.config.set(config):
                result.append(zarr.open_array(path, mode="r")[selection])
        except BaseException as err:  # noqa: BLE001 - reported on the main thread
            error.append(err)

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    thread.join(timeout=60)
    assert not thread.is_alive(), (
        "the planned read did not finish in 60s with a one-unit byte budget -- "
        "a decode is waiting for reads it should not need"
    )
    assert not error, error[0]
    return result[0]


# Deliberately non-adjacent inner chunks: chunk 0 and chunk 3. Adjacent ones share a read, which
# would leave a single unit and exercise the budget not at all.
SELECTION = np.array([0, 1, 3 * INNER, 3 * INNER + 1])


def test_a_read_decodes_before_its_chunk_is_whole(sharded_array: Path) -> None:
    got = read_with_deadline(sharded_array, PLANNED | ONE_UNIT_AT_A_TIME, SELECTION)
    np.testing.assert_array_equal(got, SELECTION.astype("int32"))


def test_a_one_unit_budget_returns_the_same_bytes_as_an_unbounded_one(
    sharded_array: Path,
) -> None:
    """Throttling admission changes when reads are issued, never what comes back."""
    selection = np.array([0, 1, 3 * INNER, 3 * INNER + 1, SHARD - 1])
    with zarr.config.set(PLANNED):
        unbounded = zarr.open_array(sharded_array, mode="r")[selection]
    throttled = read_with_deadline(
        sharded_array, PLANNED | ONE_UNIT_AT_A_TIME, selection
    )
    np.testing.assert_array_equal(throttled, unbounded)


def test_a_dense_read_completes_under_a_one_unit_budget(sharded_array: Path) -> None:
    """Every inner chunk at once is the case with the most units in flight to starve."""
    selection = np.arange(SHARD)
    got = read_with_deadline(sharded_array, PLANNED | ONE_UNIT_AT_A_TIME, selection)
    np.testing.assert_array_equal(got, selection.astype("int32"))


@pytest.mark.parametrize("threads", [1, 2, 8])
def test_a_single_fetch_thread_still_completes(
    sharded_array: Path, threads: int
) -> None:
    """One fetch thread plus a one-unit budget is the tightest configuration: if a decode ever
    needed the fetch pool to make progress, this is where it would hang."""
    got = read_with_deadline(
        sharded_array,
        PLANNED | ONE_UNIT_AT_A_TIME | {"codec_pipeline.fetch_threads": threads},
        SELECTION,
    )
    np.testing.assert_array_equal(got, SELECTION.astype("int32"))
