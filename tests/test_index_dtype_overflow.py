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
    # Spanning shards, so this is a scattered selection rather than one run -- the shape the
    # chunk-unit path exists for. A dtype must not change which of them is served.
    rows = [3, 4, 5, 11, 12, 27]
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


def test_a_uint64_selection_arrives_as_float_and_is_served() -> None:
    """The reason float is accepted at all, named rather than assumed.

    zarr subtracts the chunk offset before this pipeline sees an index array
    (`IntArrayDimIndexer.__iter__`: `self.dim_sel[start:stop] - dim_offset`, offset an `intp`).
    Under NEP 50 that promotes uint8/uint16/uint32 to int64 -- so those never arrive unsigned --
    and uint64 to float64. Measured on numpy 2.5.2.
    """
    arriving = np.array([7, 8, 9], dtype="uint64") - np.intp(4)
    assert arriving.dtype == np.float64, (
        "if numpy stops promoting uint64 to float64 here, this test measures nothing"
    )
    selection = (arriving,)
    ((_, _, chunk_selection, _, _),) = _as_int64_batch_info(
        [(None, None, selection, selection, True)]
    )
    assert chunk_selection[0].dtype == np.int64
    (result,) = make_slice_selection(chunk_selection)
    # The thing that used to raise `TypeError: slice indices must be integers`.
    assert result.indices(100) == (3, 6, 1)


@pytest.mark.parametrize(
    "values",
    [
        [3.7, 4.7],  # both fractional
        [3.0, 4.5, 5.0],  # one fractional among whole
        [np.nan, 1.0],
        [np.inf, 1.0],
        [2.0**63, 1.0],  # outside int64, and it CASTS without complaint
    ],
)
def test_a_float_index_that_is_not_a_whole_position_is_refused(values) -> None:
    """`astype(np.int64)` would turn 3.7 into 3 in silence, where zarr-python raises.

    Checked BEFORE the cast rather than by comparing against it: that comparison happens in
    float64, so `2.0**63` casts to `i64::MAX` and then compares equal to what it came from.
    """
    selection = (np.array(values, dtype="float64"),)
    with pytest.raises(DiscontiguousArrayError):
        next(iter(_as_int64_batch_info([(None, None, selection, selection, True)])))


def test_an_all_false_mask_declines_rather_than_raising_IndexError() -> None:
    """`flatnonzero` is the first construct here that can produce an empty index array.

    `dim_selection[0]` on one is an `IndexError`, which is not in `FALLBACK_TO_ZARR_PYTHON` --
    so it would escape `read` uncaught instead of declining. zarr skips chunks that select
    nothing, so nothing produces this today; the guard does not depend on that staying true.
    """
    selection = (np.zeros(8, dtype=bool),)
    ((_, _, chunk_selection, _, _),) = _as_int64_batch_info(
        [(None, None, selection, selection, True)]
    )
    assert chunk_selection[0].size == 0
    with pytest.raises(DiscontiguousArrayError):
        make_slice_selection(chunk_selection)


@pytest.mark.parametrize("dtype", UNSIGNED)
def test_unsigned_descending_rows_are_refused(dtype: str, sharded) -> None:
    """Rows 27 and 3 land in different shards, so each arrives alone and looks orderable.

    What refuses them is the negative bound: zarr makes 3 chunk-relative against shard 1
    and hands over [-13]. Signed dtypes are unaffected, which is why this is dtype-specific.
    """
    path, _ = sharded
    with zarr.config.set(STRICT), pytest.raises(DiscontiguousArrayError):
        zarr.open_array(path, mode="r")[np.array([27, 3], dtype=dtype), :]


# A negative index is refused in two places on the live path: `_chunk_unit_args` declines one
# (`utils.py`, `(indices < 0).any()`) and `build_chunk_unit_items` errors on one
# (`chunk_item.rs`, "index {} is negative"). Tested through the read below rather than by
# calling either directly, so the test survives a change of spelling.


def test_sorted_selections_never_produce_a_negative_bound(sharded) -> None:
    """The guard above must not be firing on ordinary reads."""
    path, values = sharded
    rng = np.random.default_rng(0)
    with zarr.config.set(STRICT):
        array = zarr.open_array(path, mode="r")
        for _ in range(50):
            rows = np.sort(rng.choice(32, size=rng.integers(1, 8), replace=False))
            np.testing.assert_array_equal(array[rows, :], values[rows, :])
