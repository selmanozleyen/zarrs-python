from __future__ import annotations

import numpy as np
import pytest
import zarr

# Several inner chunks per shard, so a span crosses boundaries within and between shards.
INNER = 8
SHARD = 32
LENGTH = 200


@pytest.fixture
def sharded_1d(tmp_path):
    values = np.arange(LENGTH, dtype=np.float64)
    array = zarr.create_array(
        store=tmp_path / "1d.zarr",
        shape=(LENGTH,),
        chunks=(INNER,),
        shards=(SHARD,),
        dtype=values.dtype,
    )
    array[:] = values
    return tmp_path / "1d.zarr", values


@pytest.mark.parametrize(
    ("start", "stop"),
    [
        (0, LENGTH),  # the whole array
        (0, INNER),  # exactly one inner chunk
        (0, INNER - 1),  # short of a boundary
        (1, INNER + 1),  # straddles one boundary
        (INNER, INNER * 2),  # a whole chunk, not the first
        (INNER - 1, INNER * 3 + 1),  # several chunks, ragged at both ends
        (SHARD - 1, SHARD + 1),  # across a SHARD boundary
        (LENGTH - 1, LENGTH),  # the very last element
        (LENGTH - INNER - 3, LENGTH),  # ragged, running to the end
        (7, 8),  # a single element, mid-chunk
    ],
)
def test_span_matches_the_values_it_describes(sharded_1d, start, stop):
    path, values = sharded_1d
    array = zarr.open_array(path, mode="r")
    np.testing.assert_array_equal(array[start:stop], values[start:stop])


def test_span_matches_an_explicit_index_array(sharded_1d):
    """The slice and the arange it replaced, side by side."""
    path, values = sharded_1d
    array = zarr.open_array(path, mode="r")
    for start, stop in [(0, LENGTH), (3, 97), (INNER - 1, SHARD + 5)]:
        as_span = array[start:stop]
        as_elements = array[np.arange(start, stop)]
        np.testing.assert_array_equal(as_span, as_elements)
        np.testing.assert_array_equal(as_span, values[start:stop])


def _axis0_runs(read) -> list[int]:
    """How many runs each box describes on axis 0 while `read` runs."""
    import zarrs.utils as utils

    seen, original = [], utils._box_args

    def watched(*args, **kwargs):
        out = original(*args, **kwargs)
        seen.extend(len(push[5][0]) for push in out or ())
        return out

    utils._box_args = watched
    try:
        read()
    finally:
        utils._box_args = original
    return seen


def test_a_span_is_one_run_per_entry(sharded_1d):
    path, _ = sharded_1d
    array = zarr.open_array(path, mode="r")
    runs = _axis0_runs(lambda: array[0:LENGTH])
    assert runs and set(runs) == {1}, runs


def test_coordinate_runs_stay_runs(tmp_path):
    """Whole CSR rows by coordinate: a long run is one run, scattered elements are not."""
    values = np.arange(4096, dtype=np.float32)
    array = zarr.create_array(
        store=tmp_path / "runs.zarr", shape=values.shape, chunks=(256,), shards=(1024,),
        dtype=values.dtype,
    )
    array[:] = values
    array = zarr.open_array(tmp_path / "runs.zarr", mode="r")
    # Rows crossing inner-chunk and shard boundaries: three rows, so at most three runs a shard.
    long_runs = np.concatenate(
        [np.arange(a, a + n) for a, n in [(10, 300), (900, 200), (2100, 90)]]
    )
    scattered = np.arange(0, 4096, 3)
    for coords, most in [(long_runs, 3), (scattered, None)]:
        got = []
        runs = _axis0_runs(lambda c=coords: got.append(array.get_coordinate_selection(c)))
        np.testing.assert_array_equal(got[0], values[coords])
        if most is None:
            assert sum(runs) == len(coords), runs
        else:
            assert max(runs) <= most, runs
