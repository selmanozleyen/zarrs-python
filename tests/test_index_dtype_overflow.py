"""An index array's dtype must not change which selections are accepted.

The ordering tests here used to be built on `np.diff`, which subtracts in the
incoming dtype. On an unsigned index array a decrease wraps to a large positive
step, so every such test silently inverted:

    np.diff(np.array([255, 0], dtype="uint8"))   -> array([1], dtype=uint8)

that is, the most extreme possible decrease reads as "consecutive". The slice built
from it, `slice(255, 0 + 1)`, is empty, so the read returned nothing for a selection
that asked for two rows. `np.uint64` is worse than merely wrong: mixing it with a
signed step promotes to float64 and loses exactness above 2**53.

Everything is compared in int64 now. Array indices cannot exceed int64, so the cast
is lossless, and the tests below are the ones that fail without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from zarrs.utils import DiscontiguousArrayError, make_slice_selection

if TYPE_CHECKING:
    from pathlib import Path

PLANNED = {
    "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
    "codec_pipeline.integer_array_indexing": True,
    "codec_pipeline.plan_reads": True,
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
    """[255, 0] as uint8 differences to 1. It is a decrease of 255, not a step of 1."""
    selection = (np.array([255, 0], dtype="uint8"),)
    with pytest.raises(DiscontiguousArrayError):
        make_slice_selection(selection)


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
    with zarr.config.set(PLANNED):
        array = zarr.open_array(path, mode="r")
        unsigned = array[np.array(rows, dtype=dtype), :]
        signed = array[np.array(rows, dtype="int64"), :]
    np.testing.assert_array_equal(unsigned, values[rows, :])
    np.testing.assert_array_equal(unsigned, signed)


@pytest.mark.parametrize("dtype", UNSIGNED)
def test_unsorted_unsigned_is_still_refused(dtype: str, sharded) -> None:
    """Descending rows must not be admitted just because the dtype hides the descent.

    With `strict`, being refused means raising; the point is that it does not get
    quietly served as though it were increasing.
    """
    path, _ = sharded
    with zarr.config.set(PLANNED), pytest.raises(Exception):  # noqa: B017, PT011
        zarr.open_array(path, mode="r")[np.array([27, 3], dtype=dtype), :]
