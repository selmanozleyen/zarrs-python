"""A ceiling that arrives after the pools are built must SAY so.

The two pools are process-wide and sized by the first read, so `zarr.config.set` around a
later read resizes nothing. That is the accepted cost of persistence. Doing it silently is
not: from the outside, "the knob did not pay" and "the knob never arrived" are the same
observation, and this project has published a number from each.

One test, in the order it has to run: the pools are `OnceLock`s, so the first read in the
process is the only one that can build them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import zarr

from zarrs._internal import pool_sizes

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
    return array, values


def test_a_ceiling_set_after_the_first_read_warns(sharded_1d):
    array, values = sharded_1d

    # Whatever the ambient config is, this read builds the pools if nothing else has.
    selection = np.array([1, 9, 17, 40, 41])
    np.testing.assert_array_equal(array[selection], values[selection])
    built_read, built_decode = pool_sizes()
    assert built_read is not None, "a read through the chunk-unit path must build the pools"

    # A size no earlier read can have built, so the mismatch does not depend on what ran first.
    absurd = built_read + 1_000
    with zarr.config.set({"codec_pipeline.read_worker_ceiling": absurd}):
        with pytest.warns(UserWarning, match=rf"read_worker_ceiling = {absurd} was ignored"):
            np.testing.assert_array_equal(array[selection], values[selection])

    # The warning is not a resize, and it must not have built the OTHER pool either.
    assert pool_sizes() == (built_read, built_decode)


def test_the_ceiling_actually_in_force_is_silent(sharded_1d):
    """No warning when the ask matches what was built -- or the signal is noise."""
    array, values = sharded_1d
    selection = np.array([2, 10, 18])
    np.testing.assert_array_equal(array[selection], values[selection])

    built_read, _ = pool_sizes()
    assert built_read is not None
    with zarr.config.set({"codec_pipeline.read_worker_ceiling": built_read}):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", UserWarning)
            np.testing.assert_array_equal(array[selection], values[selection])
        assert [str(w.message) for w in record] == []
