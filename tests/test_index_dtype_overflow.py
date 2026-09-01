"""What an index array IS, before anything tries to read with it.

Every array index in a batch is normalised to int64 positions on the way in. Three cases
decide what that means, and each of them read the wrong data or crashed before it:

  - an unsigned DECREASE wraps to +1 under a consecutive-difference test, so a descending
    selection reads as consecutive;
  - a uint64 selection arrives as float64, because uint64 - int64 promotes;
  - a boolean MASK is not an index array at all -- its positions are what it means.

These are dtype questions, so they are tested through the smallest read that reaches the
description, plus two direct calls on the collapse itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from zarrs.utils import (
    DiscontiguousArrayError,
    _as_int64_batch_info,
    make_slice_selection,
)

if TYPE_CHECKING:
    from pathlib import Path

# No fallback to hide behind: a selection zarrs cannot serve must raise rather than be
# served correctly by zarr-python and look like a passing test.
STRICT = {
    "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
    "codec_pipeline.strict": True,
}
# uint8 arrives as int64; uint64 alone arrives as float64. uint16/uint32 are uint8's path.
UNSIGNED = ["uint8", "uint64"]


@pytest.fixture
def sharded(tmp_path: Path) -> tuple[Path, np.ndarray]:
    path = tmp_path / "a.zarr"
    values = np.arange(32 * 40, dtype="float32").reshape(32, 40)
    zarr.create_array(
        path, shape=values.shape, dtype="float32", chunks=(4, 5), shards=(16, 20)
    )[:] = values
    return path, values


def test_wraparound_decrease_is_not_consecutive() -> None:
    """[3, 1] on uint8 differences to 254, not -2. Normalised, it is refused."""
    with pytest.raises(DiscontiguousArrayError):
        make_slice_selection((np.array([3, 1], dtype="uint8").astype(np.int64),))


@pytest.mark.parametrize("dtype", UNSIGNED)
def test_consecutive_unsigned_still_collapses(dtype: str) -> None:
    """The normalisation must not reject what was always valid."""
    (result,) = make_slice_selection((np.array([7, 8, 9], dtype=dtype).astype(np.int64),))
    assert result == slice(7, 10, 1)


@pytest.mark.parametrize("dtype", UNSIGNED)
def test_unsigned_rows_read_the_same_as_signed(dtype: str, sharded) -> None:
    """A selection's dtype is not part of its meaning."""
    path, values = sharded
    rows = [3, 4, 5, 6]
    with zarr.config.set(STRICT):
        array = zarr.open_array(path, mode="r")
        unsigned = array[np.array(rows, dtype=dtype), :]
        signed = array[np.array(rows, dtype="int64"), :]
    np.testing.assert_array_equal(unsigned, values[rows, :])
    np.testing.assert_array_equal(unsigned, signed)


def test_a_boolean_mask_reads_the_rows_it_marks(sharded) -> None:
    """A mask's POSITIONS are the selection.

    Cast elementwise instead, and an all-True mask becomes [1, 1, ...]: non-decreasing, the
    right length, and every row read is row 1. That returned wrong data with no error.
    """
    path, values = sharded
    mask = np.zeros(len(values), dtype=bool)
    mask[4:8] = True
    with zarr.config.set(STRICT):
        masked = zarr.open_array(path, mode="r")[mask, :]
    np.testing.assert_array_equal(masked, values[4:8, :])


def test_an_unreadable_index_dtype_is_declined() -> None:
    """Not guessed at: a dtype that is not an integer, unsigned or float position.

    Called directly. zarr's own `BasicIndexer` refuses a string selection before the
    pipeline is handed one, so no read can reach this branch -- and the branch still has to
    hold, because the next thing it would do is cast something that is not a position.
    """
    batch = [(None, None, np.array([3, 4], dtype="complex128"), slice(0, 2), False)]
    with pytest.raises(DiscontiguousArrayError):
        list(_as_int64_batch_info(batch))
