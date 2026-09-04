"""Whether a read was served here or handed to zarr-python, and a bad knob that says so.

Both are about the same thing: this pipeline's only failure mode is "correct but slower", and
from the outside that is indistinguishable from working. A selection it cannot describe returns
identical values through zarr-python; a config value of the wrong type used to disable it for
the whole array and blame the array.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

import zarrs

if TYPE_CHECKING:
    from pathlib import Path

ZARRS = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
N = 200


@pytest.fixture
def sharded(tmp_path: Path) -> tuple[Path, np.ndarray]:
    values = np.arange(N * 8, dtype=np.float32).reshape(N, 8)
    path = tmp_path / "a.zarr"
    zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(8, 8), shards=(32, 8)
    )[:] = values
    return path, values


def test_a_served_read_and_a_declined_one_move_different_counters(sharded) -> None:
    """A delta, not a reset: the counters are process-wide and another test may be reading."""
    path, values = sharded
    rows = np.array([1, 40, 91])

    before = zarrs.read_stats()
    with zarr.config.set(ZARRS):
        got = zarr.open_array(path, mode="r")[rows]
    np.testing.assert_array_equal(got, values[rows])
    served, declined = (n - w for n, w in zip(zarrs.read_stats(), before, strict=True))
    assert served > 0, "a sorted row selection is exactly what this path serves"
    assert declined == 0

    # Descending rows: the output slice is contiguous and ascending, so serving them would put
    # the right rows at the wrong offsets. It declines, and zarr-python returns the same values.
    before = zarrs.read_stats()
    with zarr.config.set(ZARRS):
        got = zarr.open_array(path, mode="r")[rows[::-1]]
    np.testing.assert_array_equal(got, values[rows[::-1]])
    served, declined = (n - w for n, w in zip(zarrs.read_stats(), before, strict=True))
    assert declined > 0, "a descending selection must be handed to zarr-python"
    assert served == 0


def test_a_bad_knob_names_the_knob(sharded) -> None:
    """Not the array.

    Every one of these is a pyo3 keyword argument, and pyo3 raises `TypeError` for the wrong
    type -- which the constructor catches and reports as "Array is unsupported by
    ZarrsCodecPipeline", disabling the pipeline for that array, permanently and silently, over
    the one thing the user cannot change. A negative value was worse: `OverflowError` escaped
    naming no key at all.
    """
    path, values = sharded
    rows = np.array([1, 2])

    # UNDER `strict`, an error naming the key.
    for value in (8.0, -1, "eight"):
        with (
            zarr.config.set(
                ZARRS
                | {"codec_pipeline.read_pool_size": value, "codec_pipeline.strict": True}
            ),
            pytest.raises(ValueError, match="codec_pipeline.read_pool_size"),
        ):
            zarr.open_array(path, mode="r")[rows]

    # WITHOUT it, a warning naming the key and the default -- not a dead pipeline. An
    # unconditional raise failed every array OPEN, including write-only workloads that never
    # touch this knob, which contradicts what non-strict means everywhere else here.
    with (
        zarr.config.set(ZARRS | {"codec_pipeline.read_pool_size": 8.0}),
        pytest.warns(UserWarning, match="codec_pipeline.read_pool_size"),
    ):
        got = zarr.open_array(path, mode="r")[rows]
    np.testing.assert_array_equal(got, values[rows])


def test_a_fork_drops_the_pools_in_the_PARENT(sharded) -> None:
    """Which is the only externally visible proof the `before` hook ran.

    The pid check inside `pools` would rebuild in a CHILD anyway, so a child seeing no pools
    proves nothing. The `before` handler empties the slot in the parent too -- so a parent that
    still has pools after forking is a parent whose hook did not run.

    The hook exists because the pid check runs INSIDE the mutex that `fork` copies as held: a
    child forked while a read was in flight blocks on it for ever and never reaches the check.
    """
    import os

    if not hasattr(os, "register_at_fork"):
        pytest.skip("POSIX only, and so is the hook")

    path, values = sharded
    rows = np.array([1, 40, 91])
    with zarr.config.set(ZARRS):
        got = zarr.open_array(path, mode="r")[rows]
    np.testing.assert_array_equal(got, values[rows])
    assert zarrs.pool_sizes() != (None, None), "a read must have built the pools"

    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)

    assert zarrs.pool_sizes() == (None, None), (
        "the fork left this process's pools in place, so the before-fork hook did not run"
    )


def test_an_empty_batch_cannot_fill_a_non_empty_output(sharded) -> None:
    """The guard the whole read path rests on is `covered() != output_len`, and the
    empty-batch early return skips it.

    That return exists so `X[[]]` is served rather than declined. Written unconditionally it
    also made `push nothing` against a real buffer a silent success, handing back whatever
    `np.empty` left in that memory. Nothing Python builds reaches it -- every describer refuses
    an empty index array -- but `ChunkItems` is a `#[pymethods]` boundary, which is the only
    reason the same hole was closed everywhere else.
    """
    import numpy as np

    from zarrs._internal import ChunkItems

    path, values = sharded
    with zarr.config.set(ZARRS):
        array = zarr.open_array(path, mode="r")
        impl = array._async_array.codec_pipeline.impl

    out = np.empty(64, dtype=values.dtype)
    with pytest.raises(RuntimeError, match="empty batch cannot fill"):
        impl.retrieve_chunk_items_and_apply_index(ChunkItems(), out)

    # The zero-length case is the one it exists to serve, and still is.
    impl.retrieve_chunk_items_and_apply_index(
        ChunkItems(), np.empty(0, dtype=values.dtype)
    )


def test_a_read_only_output_is_refused(sharded) -> None:
    """`from_raw_parts_mut` requires a writable target, and nothing established that.

    A read-only `out=` -- a read-only mmap, `flags.writeable = False`, a view over an immutable
    buffer -- was written through anyway: a segfault, or a silently diverging copy-on-write
    page. Neither is a Python exception. Contiguity and element size were both already checked
    two lines away, which is what made this read as an oversight.
    """
    import numpy as np

    from zarrs._internal import ChunkItems

    path, values = sharded
    with zarr.config.set(ZARRS):
        array = zarr.open_array(path, mode="r")
        impl = array._async_array.codec_pipeline.impl

    frozen = np.empty(0, dtype=values.dtype)
    frozen.flags.writeable = False
    with pytest.raises(ValueError, match="not writable"):
        impl.retrieve_chunk_items_and_apply_index(ChunkItems(), frozen)
