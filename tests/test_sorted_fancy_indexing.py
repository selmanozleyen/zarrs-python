from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from zarrs.utils import DiscontiguousArrayError, UnsupportedVIndexingError

if TYPE_CHECKING:
    from pathlib import Path

SHAPE = (32, 24)
CHUNKS = (8, 6)
SHARDS = (16, 12)
ZARRS = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}


def _write(path, values, chunks, shards) -> Path:
    zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=chunks, shards=shards
    )[:] = values
    return path


@pytest.fixture
def sharded(tmp_path: Path) -> tuple[Path, np.ndarray]:
    values = np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE)
    return _write(tmp_path / "2d.zarr", values, CHUNKS, SHARDS), values


@pytest.fixture
def sharded_1d(tmp_path: Path) -> tuple[Path, np.ndarray]:
    values = np.arange(64, dtype=np.float64)
    return _write(tmp_path / "1d.zarr", values, (8,), (16,)), values


def open_strict(path: Path) -> zarr.Array:
    """Open with no fallback, so an unsupported selection raises rather than being rerouted."""
    with zarr.config.set({**ZARRS, "codec_pipeline.strict": True}):
        return zarr.open_array(path, mode="r+")


# Each case spans chunk *and* shard boundaries, and mixes runs of length 1 with longer ones.
@pytest.mark.parametrize(
    "index",
    [
        pytest.param(np.array([0, 3, 4, 5, 17, 30]), id="rows"),
        pytest.param((np.array([2, 3, 20]), slice(4, 18)), id="rows-slice"),
        pytest.param((np.array([4, 5, 6, 29]), 7), id="rows-int"),
        pytest.param(np.array([0, 31]), id="rows-endpoints"),
        pytest.param(np.array([11]), id="single-row"),
        pytest.param(np.arange(0, 32), id="every-row"),
        pytest.param(np.array([3, 3, 3]), id="all-repeats"),
        pytest.param(np.array([0, 0, 1, 2, 2]), id="repeats-either-side-of-a-run"),
    ],
)
def test_sorted_integer_array_read(
    sharded: tuple[Path, np.ndarray], index: object
) -> None:
    path, expected = sharded
    np.testing.assert_array_equal(open_strict(path)[index], expected[index])


def test_sorted_vindex_1d(sharded_1d: tuple[Path, np.ndarray]) -> None:
    path, expected = sharded_1d
    index = np.array([0, 1, 5, 12, 13, 14, 63])
    z = open_strict(path)
    np.testing.assert_array_equal(z.vindex[index], expected[index])
    np.testing.assert_array_equal(z[index], expected[index])


# Indices within one shard, so one chunk item really gets several of them: spread across shards
# each item gets one, which is a box and was always supported.
@pytest.mark.parametrize(
    "index",
    [
        # Unsorted: zarr-python reorders the output, so a run's position in the selection is
        # not its position in the output.
        pytest.param(np.array([9, 2]), id="unsorted-rows"),
        pytest.param((np.array([1, 3]), np.array([0, 2])), id="two-array-axes"),
        pytest.param((slice(None), slice(None, None, 2)), id="strided"),
        # A column index array over a 24-wide array on 12-wide shards: each entry covers part
        # of the output width, which `_is_whole_axis` refuses on the output side.
        pytest.param((slice(None), np.array([0, 1, 7, 23])), id="cols"),
        pytest.param((slice(6, 9), np.array([5, 6, 13])), id="slice-cols"),
    ],
)
def test_unsupported_raises_strictly_but_falls_back_correctly(
    sharded: tuple[Path, np.ndarray], index: object
) -> None:
    path, expected = sharded
    with pytest.raises((DiscontiguousArrayError, UnsupportedVIndexingError)):
        open_strict(path)[index]
    with zarr.config.set(ZARRS):
        z = zarr.open_array(path, mode="r")
        np.testing.assert_array_equal(z[index], expected[index])


def test_a_write_of_rows_sharing_a_chunk(sharded: tuple[Path, np.ndarray]) -> None:
    """Splitting one write into several read-modify-writes of a chunk would race them."""
    path, expected = sharded
    index = np.array([1, 3, 4])
    with zarr.config.set(ZARRS):
        z = zarr.open_array(path, mode="r+")
        z[index, :] = np.full((len(index), SHAPE[1]), -1.0)
        expected[index, :] = -1.0
        np.testing.assert_array_equal(z[...], expected)


def test_contiguous_output_does_not_imply_sorted_input(
    sharded_1d: tuple[Path, np.ndarray],
) -> None:
    """`CoordinateIndexer` sorts only when the chunk-raveled order is wrong: 7 and 3 share one."""
    path, expected = sharded_1d
    index = np.array([7, 3])
    with pytest.raises((DiscontiguousArrayError, UnsupportedVIndexingError)):
        open_strict(path).vindex[index]
    with zarr.config.set(ZARRS):
        got = zarr.open_array(path, mode="r").vindex[index]
    np.testing.assert_array_equal(got, expected[index])


@pytest.mark.parametrize(
    "mask",
    [
        pytest.param(np.arange(SHAPE[0]) < 16, id="aligned-to-chunks"),
        pytest.param(np.ones(SHAPE[0], dtype=bool), id="every-row"),
        pytest.param(np.isin(np.arange(SHAPE[0]), [3, 17, 30]), id="scattered"),
        pytest.param(np.zeros(SHAPE[0], dtype=bool), id="no-rows"),
    ],
)
def test_boolean_mask_reads_the_positions_it_marks(
    sharded: tuple[Path, np.ndarray], mask: np.ndarray
) -> None:
    """`BoolArrayDimIndexer` hands over a boolean chunk selection, not an index array."""
    path, expected = sharded
    with zarr.config.set(ZARRS):
        got = zarr.open_array(path, mode="r")[mask, :]
    np.testing.assert_array_equal(got, expected[mask, :])
