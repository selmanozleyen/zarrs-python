"""One-dimensional scattered gathers must reach the Rust pipeline.

Such a gather -- the shape a CSR row selection produces -- is a set of
contiguous runs arriving out of order. Read whole it is not a slice, so
without `split_1d_runs` the pipeline hands the entire read to zarr-python and
nothing on the Rust side is exercised. These check both that the split is
right and that the read actually takes the Rust path, since a silent fallback
still returns correct data and would otherwise go unnoticed.
"""

import numpy as np
import pytest
import zarr
from zarr.core import BatchedCodecPipeline

import zarrs  # noqa: F401
from zarrs.utils import split_1d_runs


class _Spec:
    shape = (100,)


def test_splits_out_of_order_runs():
    # chunk 10,11,12 -> out 5,6,7 ; then chunk 2,3,4 -> out 8,9,10
    chunk = np.array([10, 11, 12, 2, 3, 4])
    out = np.array([5, 6, 7, 8, 9, 10])
    runs = split_1d_runs([("bg", _Spec(), (chunk,), out, False)])
    assert [(r[2], r[3]) for r in runs] == [
        ((slice(10, 13, 1),), (slice(5, 8, 1),)),
        ((slice(2, 5, 1),), (slice(8, 11, 1),)),
    ]


def test_splits_where_only_the_output_jumps():
    # Contiguous in the chunk but not in the output: still two runs, or the
    # second run's elements would be written to the wrong place.
    chunk = np.array([0, 1, 2, 3])
    out = np.array([0, 1, 7, 8])
    runs = split_1d_runs([("bg", _Spec(), (chunk,), out, False)])
    assert [(r[2], r[3]) for r in runs] == [
        ((slice(0, 2, 1),), (slice(0, 2, 1),)),
        ((slice(2, 4, 1),), (slice(7, 9, 1),)),
    ]


def test_passes_through_non_fancy_selections():
    entry = ("bg", _Spec(), (slice(0, 5),), slice(0, 5), False)
    assert split_1d_runs([entry]) == [entry]


@pytest.mark.parametrize("size", [64, 500])
def test_gather_is_correct_and_skips_the_fallback(tmp_path, size, monkeypatch):
    array = zarr.create_array(
        store=str(tmp_path / "a.zarr"),
        shape=(size,),
        chunks=(size // 4,),
        dtype="int32",
        zarr_format=3,
    )
    array[:] = np.arange(size, dtype="int32")

    fallbacks = []
    original = BatchedCodecPipeline.read

    async def counted(self, *args, **kwargs):
        fallbacks.append(1)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(BatchedCodecPipeline, "read", counted)

    rng = np.random.default_rng(0)
    rows = rng.permutation(size)[: size // 2]
    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}):
        got = zarr.open_array(str(tmp_path / "a.zarr"), mode="r")[rows]

    assert np.array_equal(got, rows.astype("int32"))
    assert not fallbacks, "scattered 1-D gather fell back to zarr-python"
