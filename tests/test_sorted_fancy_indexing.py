"""Sorted integer-array reads go through the zarrs pipeline instead of falling back.

A scattered selection like ``z[[3, 7, 8], :]`` is not one rectangular subset, which is all a
chunk read used to be able to say, so it fell back to the ``zarr-python`` pipeline. It is a
stack of rectangular subsets, so the pipeline now emits one per run of consecutive indices.

`strict` mode is what makes these tests categorical rather than merely correct: with no
fallback available, an unsupported selection raises instead of quietly returning the right
answer via the other pipeline. It has to be set before the array is opened, since that is
when the pipeline decides whether it has a fallback, so every test opens its own handle.
"""

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


@pytest.fixture(autouse=True, params=[False, True], ids=["unplanned", "planned"])
def _split_reads(request):
    """Off by default -- see the README. Every case below must hold either way: planning changes
    how a selection is read, never what it returns."""
    with zarr.config.set(
        {
            "codec_pipeline.integer_array_indexing": True,
            "codec_pipeline.plan_reads": request.param,
        }
    ):
        yield


@pytest.fixture
def sharded(tmp_path: Path) -> tuple[Path, np.ndarray]:
    expected = np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE)
    path = tmp_path / "foo.zarr"
    z = zarr.create_array(
        path, dtype=np.float64, shape=SHAPE, chunks=CHUNKS, shards=SHARDS
    )
    z[:] = expected
    return path, expected


def open_strict(path: Path) -> zarr.Array:
    """Open with no fallback, so an unsupported selection raises instead of being rerouted."""
    with zarr.config.set(
        {
            "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
            "codec_pipeline.strict": True,
        }
    ):
        return zarr.open_array(path, mode="r+")


# Each case spans chunk *and* shard boundaries, and mixes runs of length 1 with longer ones.
@pytest.mark.parametrize(
    "index",
    [
        pytest.param(np.array([0, 3, 4, 5, 17, 30]), id="rows"),
        pytest.param((np.array([1, 9, 10, 31]), slice(None)), id="rows-full-slice"),
        pytest.param((np.array([2, 3, 20]), slice(4, 18)), id="rows-slice"),
        pytest.param((slice(None), np.array([0, 1, 7, 23])), id="cols"),
        pytest.param((slice(6, 9), np.array([5, 6, 13])), id="slice-cols"),
        pytest.param((np.array([4, 5, 6, 29]), 7), id="rows-int"),
        pytest.param(np.array([0, 31]), id="rows-endpoints"),
        pytest.param(np.array([11]), id="single-row"),
        pytest.param(np.arange(0, 32), id="every-row"),
    ],
)
def test_sorted_integer_array_read(
    sharded: tuple[Path, np.ndarray], index: object
) -> None:
    path, expected = sharded
    np.testing.assert_array_equal(open_strict(path)[index], expected[index])


def test_sorted_vindex_1d(tmp_path: Path) -> None:
    expected = np.arange(64, dtype=np.float64)
    path = tmp_path / "bar.zarr"
    z = zarr.create_array(
        path, dtype=np.float64, shape=(64,), chunks=(8,), shards=(16,)
    )
    z[:] = expected
    index = np.array([0, 1, 5, 12, 13, 14, 63])

    z = open_strict(path)
    np.testing.assert_array_equal(z.vindex[index], expected[index])
    np.testing.assert_array_equal(z[index], expected[index])


# Indices within one shard, so a single chunk item really does get several of them -- spread
# across shards each item gets one index, which is a box and was always supported.
UNSUPPORTED = [
    # Unsorted: zarr-python reorders the output, so a run's position in the selection is not
    # its position in the output.
    pytest.param(np.array([9, 2]), id="unsorted-rows"),
    # Two array axes: outer and coordinate indexing disagree on what this means.
    pytest.param((np.array([1, 3]), np.array([0, 2])), id="two-array-axes"),
    pytest.param((slice(None), slice(None, None, 2)), id="strided"),
]


@pytest.mark.parametrize("index", UNSUPPORTED)
def test_still_unsupported(sharded: tuple[Path, np.ndarray], index: object) -> None:
    path, _ = sharded
    with pytest.raises((DiscontiguousArrayError, UnsupportedVIndexingError)):
        open_strict(path)[index]


@pytest.mark.parametrize("index", UNSUPPORTED)
def test_unsupported_still_falls_back_correctly(
    sharded: tuple[Path, np.ndarray], index: object
) -> None:
    """Without strict mode the rejected selections must still return the right data."""
    path, expected = sharded
    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}):
        z = zarr.open_array(path, mode="r")
        np.testing.assert_array_equal(z[index], expected[index])


def test_writes_are_not_split(sharded: tuple[Path, np.ndarray]) -> None:
    """A split write would make several read-modify-writes of one chunk race, so writes still
    fall back. Rows 1, 3 and 4 share a chunk, which is the case that would lose data."""
    path, expected = sharded
    index = np.array([1, 3, 4])
    value = np.full((len(index), SHAPE[1]), -1.0)

    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}):
        z = zarr.open_array(path, mode="r+")
        z[index, :] = value
        expected[index, :] = value
        np.testing.assert_array_equal(z[...], expected)


def test_flag_off_rejects_the_same_read(sharded: tuple[Path, np.ndarray]) -> None:
    """The option is what enables this, so with it off the selection is unsupported again."""
    path, _ = sharded
    z = open_strict(path)
    index = np.array([0, 3, 4, 5, 17, 30])
    with (
        zarr.config.set({"codec_pipeline.integer_array_indexing": False}),
        pytest.raises(DiscontiguousArrayError),
    ):
        z[index]


@pytest.mark.parametrize(
    "index",
    [
        # Descending inside one chunk. `CoordinateIndexer` sorts only when the chunk-raveled
        # order is wrong, and both of these live in chunk 0, so out_selection comes back as
        # slice(0, 2) despite the indices descending. Building runs from that would give the
        # inverted box slice(7, 4).
        pytest.param(np.array([7, 3]), id="descending-in-one-chunk"),
    ],
)
def test_contiguous_output_does_not_imply_sorted_input(
    tmp_path: Path, index: np.ndarray
) -> None:
    """A rectangular output side is not evidence the input was ordered.

    `CoordinateIndexer` sorts only when the chunk-raveled order is wrong, and both
    indices live in chunk 0, so `out_selection` comes back `slice(0, 2)` while the
    indices descend. The ordering check refuses it; nothing about the output would.
    """
    expected = np.arange(64, dtype=np.float64)
    path = tmp_path / "one_d.zarr"
    z = zarr.create_array(
        path, dtype=np.float64, shape=(64,), chunks=(16,), shards=(32,)
    )
    z[:] = expected

    with pytest.raises((DiscontiguousArrayError, UnsupportedVIndexingError)):
        open_strict(path).vindex[index]

    # And the fallback must still answer it correctly.
    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}):
        got = zarr.open_array(path, mode="r").vindex[index]
    np.testing.assert_array_equal(got, expected[index])


def test_a_split_read_is_rejected_without_the_splitter(
    sharded: tuple[Path, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tests above must be exercising the splitter, not passing for some other reason."""
    import zarrs.utils

    monkeypatch.setattr(
        zarrs.utils,
        "split_selection_runs",
        lambda chunk_sel, out_sel: ((chunk_sel, out_sel),),
    )
    path, _ = sharded
    with pytest.raises(DiscontiguousArrayError):
        open_strict(path)[np.array([0, 3, 4, 5, 17, 30])]


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(np.array([3, 3, 4]), id="repeat-then-run"),
        pytest.param(np.array([3, 3, 3]), id="all-repeats"),
        pytest.param(np.array([2, 5, 5, 9]), id="repeat-mid-selection"),
        pytest.param(np.array([0, 0, 1, 2, 2]), id="repeats-either-side-of-a-run"),
    ],
)
def test_repeats_are_served_not_refused(
    sharded: tuple[Path, np.ndarray], index: np.ndarray
) -> None:
    """A repeat ends its run early and reads that index again into the next output slot.

    Through `open_strict`, so there is no fallback to answer correctly on zarrs'
    behalf: a right answer here is evidence zarrs served it.
    """
    path, values = sharded
    np.testing.assert_array_equal(open_strict(path)[index, :], values[index, :])
