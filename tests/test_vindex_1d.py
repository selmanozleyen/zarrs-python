"""One-dimensional gathers made of long runs should reach the Rust pipeline.

A CSR row selection is a set of contiguous runs, one per row. Read whole it is
not a slice, so without `split_1d_runs` the pipeline hands the entire read to
zarr-python -- silently, since the fallback returns correct data.

Whether splitting is worth it depends on run length, and zarr already says
which case applies: `CoordinateIndexer` returns a plain output slice exactly
when the coordinates arrived in chunk order and it did not have to reorder
them. A scattered output array means the rows interleave, runs collapse to a
couple of elements, and splitting would be far worse than falling back.
"""

import numpy as np
import pytest
import zarr
from zarr.core import BatchedCodecPipeline

import zarrs  # noqa: F401
from zarrs.utils import split_1d_runs

RUN = 64


class _Spec:
    shape = (100_000,)


def _blocks(starts, length=RUN):
    """Indices covering `length` consecutive positions at each start."""
    return np.concatenate([np.arange(s, s + length) for s in starts])


def test_splits_runs_when_the_output_is_a_slice():
    chunk = _blocks([100, 5_000])
    runs = split_1d_runs([("bg", _Spec(), (chunk,), slice(0, chunk.size, 1), False)])
    assert [(r[2], r[3]) for r in runs] == [
        ((slice(100, 100 + RUN, 1),), (slice(0, RUN, 1),)),
        ((slice(5_000, 5_000 + RUN, 1),), (slice(RUN, 2 * RUN, 1),)),
    ]


def test_output_slice_offset_is_carried():
    chunk = _blocks([10])
    runs = split_1d_runs([("bg", _Spec(), (chunk,), slice(500, 500 + RUN), False)])
    assert [(r[2], r[3]) for r in runs] == [
        ((slice(10, 10 + RUN, 1),), (slice(500, 500 + RUN, 1),))
    ]


def test_declines_a_scattered_output():
    # An output index array is zarr saying it had to reorder the coordinates,
    # so the runs interleave. Splitting would fragment; fall back instead.
    chunk = _blocks([100, 5_000])
    out = np.concatenate([np.arange(RUN, 2 * RUN), np.arange(0, RUN)])
    entry = ("bg", _Spec(), (chunk,), out, False)
    assert split_1d_runs([entry]) == [entry]


def test_declines_when_no_two_indices_adjoin():
    # Every element its own run: one read per element buys nothing over
    # reading the chunk whole.
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
    assert fallbacks, "isolated indices should be declined, not shattered"


def test_unsorted_gather_falls_back_and_is_still_correct(array, monkeypatch):
    # Unsorted rows make zarr reorder, so the output comes back as an index
    # array and the runs interleave.
    rng = np.random.default_rng(0)
    rows = rng.permutation(_blocks([0, 30_000, 61_000, 92_000], length=1_000))
    got, fallbacks = _read_counting_fallbacks(array, rows, monkeypatch)
    assert np.array_equal(got, rows.astype("int32"))
    assert fallbacks, "a reordered gather should be declined"
