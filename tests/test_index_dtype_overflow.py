"""An index array's dtype must not change which selections are accepted.

WHAT ACTUALLY REACHES THIS PIPELINE. zarr subtracts the chunk offset before handing an index
array over -- `IntArrayDimIndexer.__iter__`, `self.dim_sel[start:stop] - dim_offset`, where the
offset is an `intp`. Under NEP 50 that promotes `uint8`, `uint16` and `uint32` to **int64**, so
those never arrive unsigned; `uint64` promotes to **float64**. Measured on numpy 2.5.2, not
assumed.

So the bug that bit was `uint64`: a float64 index array built `slice(np.float64(3.0),
np.float64(6.0), 1)`, and the first `.indices()` call on it raised `TypeError: slice indices
must be integers`. `TypeError` is not in `FALLBACK_TO_ZARR_PYTHON`, so it reached the user in
both strict and non-strict mode rather than declining to zarr-python.

The unsigned wrap is real but is NOT that bug, and this file used to say it was. On a raw
unsigned array a decrease wraps to a large positive step -- `np.diff(np.array([255, 0],
"uint8"))` is `[1]`, so the most extreme possible decrease reads as consecutive. It is
unreachable from zarr for the promotion reason above, and where it IS still live is upstream,
in zarr's own `Order.check`, which calls `np.diff` on the raw array and tests `>= 0`. This
pipeline guards its own arithmetic anyway, because `make_slice_selection` is module-public.
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
    """[255, 0] as uint8 differences to 1. It is a decrease of 255, not a step of 1.

    `make_slice_selection` differences directly, so go through the boundary that
    normalises: neither half is a guarantee alone.
    """
    selection = (np.array([255, 0], dtype="uint8"),)
    ((_, _, chunk_selection, _, _),) = _as_int64_batch_info(
        [(None, None, selection, selection, True)]
    )
    with pytest.raises(DiscontiguousArrayError):
        make_slice_selection(chunk_selection)


@pytest.mark.parametrize("dtype", UNSIGNED)
def test_consecutive_unsigned_still_collapses(dtype: str) -> None:
    """The fix must not reject what was always valid."""
    (result,) = make_slice_selection((np.array([7, 8, 9], dtype=dtype),))
    assert result == slice(7, 10, 1)


@pytest.mark.parametrize("dtype", UNSIGNED)
def test_unsigned_rows_read_the_same_as_signed(dtype: str, sharded) -> None:
    """A selection's dtype is not part of its meaning."""
    path, values = sharded
    rows = [3, 4, 5, 11, 12, 27]
    with zarr.config.set(STRICT):
        array = zarr.open_array(path, mode="r")
        unsigned = array[np.array(rows, dtype=dtype), :]
        signed = array[np.array(rows, dtype="int64"), :]
    np.testing.assert_array_equal(unsigned, values[rows, :])
    np.testing.assert_array_equal(unsigned, signed)


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


def test_a_uint64_selection_does_not_crash_on_a_float_slice() -> None:
    """THE BUG THIS FILE EXISTS FOR, named directly rather than reached through a read.

    zarr subtracts the chunk offset before this pipeline sees an index array
    (`IntArrayDimIndexer.__iter__`: `self.dim_sel[start:stop] - dim_offset`, where the offset
    is an `intp`). Under NEP 50 that promotes `uint8`/`uint16`/`uint32` to `int64` -- so those
    never arrive unsigned at all -- and `uint64` to **float64**. Measured on numpy 2.5.2.

    A float64 index array then builds `slice(np.float64(3.0), np.float64(6.0), 1)`, and the
    first thing that calls `.indices()` on it raises `TypeError: slice indices must be
    integers`. `TypeError` is not in `FALLBACK_TO_ZARR_PYTHON`, so it reached the user in both
    strict and non-strict mode rather than declining.
    """
    arriving_as_zarr_hands_it_over = np.array([7, 8, 9], dtype="uint64") - np.intp(4)
    assert arriving_as_zarr_hands_it_over.dtype == np.float64, (
        "if numpy stops promoting uint64 to float64 here, this test is measuring nothing"
    )

    selection = (arriving_as_zarr_hands_it_over,)
    ((_, _, chunk_selection, _, _),) = _as_int64_batch_info(
        [(None, None, selection, selection, True)]
    )
    (result,) = make_slice_selection(chunk_selection)
    assert result == slice(3, 6, 1)
    # The thing that used to raise.
    assert result.indices(100) == (3, 6, 1)


def test_a_fractional_index_is_refused_rather_than_truncated() -> None:
    """Float is accepted for uint64 and for nothing else.

    `astype(np.int64)` would turn 3.7 into 3 in silence, where zarr-python raises for a
    fractional index. The uint64 values that legitimately arrive as float are whole.
    """
    selection = (np.array([3.7, 4.7], dtype="float64"),)
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
