from __future__ import annotations

import numpy as np
import pytest
import zarr

# One shard holds several inner chunks, so a span crosses boundaries within a shard as well
# as between them.
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


def test_the_span_path_is_actually_taken(sharded_1d):
    """Otherwise the tests above pass by never reaching the code they are about."""
    from zarrs.utils import _chunk_unit_args

    path, _ = sharded_1d
    array = zarr.open_array(path, mode="r")
    seen = []
    original = _chunk_unit_args

    import zarrs.utils as utils

    def watched(*args, **kwargs):
        out = original(*args, **kwargs)
        # A list of pushes, one per band, so the kind is the first field of each.
        seen.extend(push[0] for push in out or ())
        return out

    utils._chunk_unit_args = watched
    try:
        array[0:LENGTH]
    finally:
        utils._chunk_unit_args = original
    assert "span" in seen, f"no entry took the span path; kinds seen: {set(seen)}"
