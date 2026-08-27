"""Integer-array selections served one inner chunk at a time.

`codec_pipeline.chunk_unit_indexing` groups a sorted integer selection by the unit the codec
chain actually decodes -- the inner chunk -- so a chunk is read once, decoded once and gathered
once however many of its elements are wanted. Handing zarrs a coordinate list instead costs two
allocations and a partial-decode call PER ELEMENT.

The path is narrow on purpose: one 1-D integer axis, non-negative and non-decreasing, against a
contiguous output slice. Anything else has to fall back and still return the right data, so both
directions are asserted here.

Which Rust entry point served the read is asserted too, not assumed. A batch that is entirely
chunk-unit goes over as one `ChunkItems` handle rather than as one Python object per item, and a
silent fall back to the list path would pass every correctness check while doing none of that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from zarrs._internal import CodecPipelineImpl

if TYPE_CHECKING:
    from pathlib import Path

# Not a multiple of CHUNK, so the last inner chunk is short and the subset has to be clamped.
N = 40_000
CHUNK = 4_096
SHARD = 16_384

CHUNK_UNIT = {
    "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
    "codec_pipeline.chunk_unit_indexing": True,
}


@pytest.fixture(params=["zstd", "none"])
def compressors(request):
    return "auto" if request.param == "zstd" else None


@pytest.fixture
def array(tmp_path: Path, compressors) -> tuple[Path, np.ndarray]:
    values = np.arange(N, dtype=np.float32)
    kwargs = {} if compressors == "auto" else {"compressors": compressors}
    path = tmp_path / "a"
    z = zarr.create_array(
        path,
        dtype=values.dtype,
        shape=values.shape,
        chunks=(CHUNK,),
        shards=(SHARD,),
        **kwargs,
    )
    z[:] = values
    return path, values


@pytest.fixture
def entries(monkeypatch) -> dict[str, int]:
    """How many batches took each Rust entry point."""
    counts = {"handle": 0, "list": 0}
    for name, key in (
        ("retrieve_chunk_items_and_apply_index", "handle"),
        ("retrieve_chunks_and_apply_index", "list"),
    ):
        original = getattr(CodecPipelineImpl, name)

        def wrapper(self, *args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(CodecPipelineImpl, name, wrapper)
    return counts


def selections() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        # Every element of one inner chunk: the whole-chunk subset is exactly what is wanted.
        "one whole chunk": np.arange(CHUNK, 2 * CHUNK),
        # Sparse across many chunks, which is what makes a per-element path expensive.
        "scattered": np.sort(rng.choice(N, size=2_000, replace=False)),
        # Non-decreasing, not strictly increasing. Duplicates are legal and must be kept.
        "with duplicates": np.repeat(
            np.sort(rng.choice(N, size=500, replace=False)), 3
        ),
        # Lands in the short final chunk, where lo + inner overruns the array.
        "short last chunk": np.arange(N - 100, N),
        "single element": np.array([N - 1]),
        "every second": np.arange(0, N, 2),
    }


@pytest.mark.parametrize("name", list(selections()))
def test_selection_matches_and_takes_the_handle(
    array: tuple[Path, np.ndarray], entries: dict[str, int], name: str
) -> None:
    path, truth = array
    selection = selections()[name]

    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]

    np.testing.assert_array_equal(got, truth[selection])
    assert entries["handle"] > 0, "the batch was entirely chunk-unit but went as a list"
    assert entries["list"] == 0


def test_decreasing_selection_falls_back_and_is_still_right(
    array: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """A backward step would mean one item per element, so the path declines it. The read still
    has to produce the right answer, down whichever path takes it."""
    path, truth = array
    selection = np.array([9_000, 40, 8_000, 39])

    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]

    np.testing.assert_array_equal(got, truth[selection])
    assert entries["handle"] == 0


def test_a_plain_slice_is_untouched(
    array: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """Only integer-array selections are grouped; a slice is already one subset."""
    path, truth = array

    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[CHUNK - 10 : 2 * CHUNK + 10]

    np.testing.assert_array_equal(got, truth[CHUNK - 10 : 2 * CHUNK + 10])
    assert entries["handle"] == 0


def test_two_dimensional_selection_falls_back(tmp_path: Path) -> None:
    """The grouping is the 1-D path. A 2-D array must decline rather than mis-group."""
    values = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = tmp_path / "two_d"
    z = zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(16, 16), shards=(32, 32)
    )
    z[:] = values

    rows = np.array([1, 5, 5, 40])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, :]

    np.testing.assert_array_equal(got, values[rows, :])
