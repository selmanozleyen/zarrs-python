"""One-dimensional gathers made of long runs should reach the Rust pipeline.

A CSR row selection is a set of contiguous runs, one per row. Read whole it is
not a slice, so without `split_1d_runs` the pipeline hands the entire read to
zarr-python -- silently, since the fallback returns correct data.

Whether splitting is worth doing depends on run length, and that is decided by
the caller. Sorted row indices give one long run per row; unsorted indices
interleave them into runs of a couple of elements, where splitting would issue
hundreds of thousands of tiny reads. So these check both that long runs split
and that short ones are declined rather than shattered.
"""

import numpy as np
import pytest
import zarr
from zarr.core import BatchedCodecPipeline

import zarrs  # noqa: F401
from zarrs.utils import MIN_COORDS_PER_RUN, split_1d_runs

RUN = MIN_COORDS_PER_RUN * 2


class _Spec:
    shape = (100_000,)


def _blocks(starts, length=RUN):
    """Indices covering `length` consecutive positions at each start."""
    return np.concatenate([np.arange(s, s + length) for s in starts])


def test_splits_long_runs_against_a_slice_output():
    # The sorted case: chunk side jumps between rows, output fills in order.
    chunk = _blocks([100, 5_000])
    runs = split_1d_runs([("bg", _Spec(), (chunk,), slice(0, chunk.size, 1), False)])
    assert [(r[2], r[3]) for r in runs] == [
        ((slice(100, 100 + RUN, 1),), (slice(0, RUN, 1),)),
        ((slice(5_000, 5_000 + RUN, 1),), (slice(RUN, 2 * RUN, 1),)),
    ]


def test_splits_long_runs_against_an_array_output():
    chunk = _blocks([10])
    out = np.arange(500, 500 + RUN)
    runs = split_1d_runs([("bg", _Spec(), (chunk,), out, False)])
    assert [(r[2], r[3]) for r in runs] == [
        ((slice(10, 10 + RUN, 1),), (slice(500, 500 + RUN, 1),))
    ]


def test_declines_to_shatter_short_runs():
    # Every index its own run. Splitting would emit one read per element, far
    # worse than letting zarr-python take the whole selection.
    scattered = np.arange(0, 400, 3)
    entry = ("bg", _Spec(), (scattered,), slice(0, scattered.size, 1), False)
    assert split_1d_runs([entry]) == [entry]


def test_passes_through_non_fancy_selections():
    entry = ("bg", _Spec(), (slice(0, 5),), slice(0, 5), False)
    assert split_1d_runs([entry]) == [entry]


@pytest.fixture
def array(tmp_path):
    a = zarr.create_array(
        store=str(tmp_path / "a.zarr"),
        shape=(100_000,),
        chunks=(25_000,),
        dtype="int32",
        zarr_format=3,
    )
    a[:] = np.arange(100_000, dtype="int32")
    return str(tmp_path / "a.zarr")


def _read_counting_fallbacks(path, rows, monkeypatch):
    fallbacks = []
    original = BatchedCodecPipeline.read

    async def counted(self, *args, **kwargs):
        fallbacks.append(1)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(BatchedCodecPipeline, "read", counted)
    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}):
        got = zarr.open_array(path, mode="r")[rows]
    return got, len(fallbacks)


def test_block_gather_reaches_rust(array, monkeypatch):
    rows = _blocks([0, 30_000, 61_000, 92_000], length=1_000)
    got, fallbacks = _read_counting_fallbacks(array, rows, monkeypatch)
    assert np.array_equal(got, rows.astype("int32"))
    assert not fallbacks, "a gather of long runs should not fall back"


def test_scattered_gather_falls_back_and_is_still_correct(array, monkeypatch):
    rng = np.random.default_rng(0)
    rows = np.sort(rng.permutation(100_000)[:500])
    got, fallbacks = _read_counting_fallbacks(array, rows, monkeypatch)
    assert np.array_equal(got, rows.astype("int32"))
    assert fallbacks, "runs of one element should be declined, not shattered"
