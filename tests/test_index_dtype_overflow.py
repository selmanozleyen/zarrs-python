"""An index array's dtype must not change which selections are accepted.

Subtracting in the incoming dtype inverts the comparison, because on an unsigned
array a decrease wraps to a large positive step:

    np.diff(np.array([255, 0], dtype="uint8"))   -> array([1], dtype=uint8)

the most extreme possible decrease reads as consecutive, and the slice built from it,
`slice(255, 0 + 1)`, is empty. uint64 is worse than wrong: mixed with a signed step it
promotes to float64 and loses exactness above 2**53.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from zarrs.utils import (
    DiscontiguousArrayError,
    UnsortedArrayIndexError,
    _as_int64_batch_info,
    make_slice_selection,
    split_selection_runs,
)

if TYPE_CHECKING:
    from pathlib import Path

SETTINGS = {
    "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
    "codec_pipeline.integer_array_indexing": True,
    # No fallback to hide behind: a selection zarrs cannot serve must raise rather
    # than be served correctly by zarr-python and look like a passing test.
    "codec_pipeline.strict": True,
}

UNSIGNED = ["uint8", "uint16", "uint32", "uint64"]


@pytest.fixture
def sharded(tmp_path: Path) -> tuple[Path, np.ndarray]:
    path = tmp_path / "a.zarr"
    values = np.arange(32 * 40, dtype="float32").reshape(32, 40)
    array = zarr.create_array(
        path,
        shape=values.shape,
        dtype="float32",
        chunks=(4, 5),
        shards=(16, 20),
        compressors=[zarr.codecs.BloscCodec(cname="lz4")],
    )
    array[:] = values
    return path, values


def test_wraparound_decrease_is_not_consecutive() -> None:
    """[255, 0] as uint8 differences to 1. It is a decrease of 255, not a step of 1.

    `make_slice_selection` differences its input directly, so it is the normalisation at the
    pipeline boundary that keeps this from reading as a step forward. Neither half is a
    guarantee alone, so go through the boundary rather than calling it with a bare array.
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
    with zarr.config.set(SETTINGS):
        array = zarr.open_array(path, mode="r")
        unsigned = array[np.array(rows, dtype=dtype), :]
        signed = array[np.array(rows, dtype="int64"), :]
    np.testing.assert_array_equal(unsigned, values[rows, :])
    np.testing.assert_array_equal(unsigned, signed)


@pytest.mark.parametrize("dtype", UNSIGNED)
def test_unsorted_unsigned_is_still_refused(dtype: str, sharded) -> None:
    """Descending rows must not be admitted just because the dtype hides the descent.

    Which refusal depends on chunking: zarr splits the read per chunk before the pipeline
    sees it, so with these small chunks each item carries a single (chunk-relative, possibly
    negative) index and the descent is no longer visible -- that lands on
    `DiscontiguousArrayError`. Coarser chunks keep the pair intact and raise
    `UnsortedArrayIndexError`. Either way it must not be served as though increasing.
    """
    path, _ = sharded
    with (
        zarr.config.set(SETTINGS),
        pytest.raises((UnsortedArrayIndexError, DiscontiguousArrayError)),
    ):
        zarr.open_array(path, mode="r")[np.array([27, 3], dtype=dtype), :]


def test_negative_chunk_relative_index_is_refused() -> None:
    """A negative index must never become a slice bound.

    zarr hands the pipeline chunk-relative indices, and its sharding indexer miscomputes
    them for unsigned dtypes -- `[27, 3]` as uint8 arrives as `[-13]`. Absolute slice
    bounds cannot express that: `slice(-13, -12)` reads as an empty subset, which reaches
    zarrs as a size mismatch rather than an error naming the cause.
    """
    with pytest.raises(DiscontiguousArrayError):
        list(
            split_selection_runs(
                (np.array([-13]), slice(0, 20, 1)),
                (slice(0, 1), slice(0, 20)),
            )
        )


def test_sorted_selections_never_produce_a_negative_bound(sharded) -> None:
    """The guard above must not be firing on ordinary reads.

    Every box a sorted selection yields has non-negative bounds, so the refusal is reserved
    for the miscomputed case and is not quietly rejecting work zarrs can serve.
    """
    path, values = sharded
    rng = np.random.default_rng(0)
    with zarr.config.set(SETTINGS):
        array = zarr.open_array(path, mode="r")
        for _ in range(50):
            rows = np.sort(
                rng.choice(values.shape[0], size=rng.integers(1, 8), replace=False)
            )
            np.testing.assert_array_equal(array[rows, :], values[rows, :])
