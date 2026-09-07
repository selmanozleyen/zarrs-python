"""A rank-3 read whose middle axis is partial must not be refused outright.

`Z[0:10]` on this geometry gives an item whose output box is contiguous per axis-0 index, but
in runs of the trailing axis rather than one run per index. Python's gate admits it; if Rust's
carve insists on exactly one run per index it raises, and the raise happens outside the
fall-back-to-zarr-python path, so the caller gets an exception instead of their data.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from zarrs._internal import pool_sizes


@pytest.fixture(autouse=True)
def _no_silent_fallback():
    """Without this the carve can refuse the read, zarr-python can serve it, and the test is
    green while never exercising the code under test."""
    with zarr.config.set({"codec_pipeline.strict": True}):
        yield


def _roundtrip(tmp_path, shape, chunks, shards, selection):
    path = str(tmp_path / "a.zarr")
    a = zarr.create_array(
        store=path, shape=shape, chunks=chunks, shards=shards, dtype="int16", zarr_format=3
    )
    expected = np.arange(int(np.prod(shape)), dtype="int16").reshape(shape)
    a[:] = expected
    got = np.asarray(zarr.open_array(path, mode="r")[selection])
    np.testing.assert_array_equal(got, expected[selection])
    # The I/O pool -- the FIRST element -- is built only by the chunk-unit read path. The CPU
    # pool is built by the constructor to read its width, so it is already there after an open
    # and says nothing. Without this the test proves the VALUES are right while saying nothing
    # about which code produced them.
    assert pool_sizes()[0] is not None, "the chunk-unit read path never ran"


def test_middle_axis_spans_two_shards(tmp_path) -> None:
    _roundtrip(tmp_path, (100, 32, 16), (8, 16, 16), (64, 16, 16), np.s_[0:10])


def test_middle_axis_split_into_bands(tmp_path) -> None:
    _roundtrip(tmp_path, (100, 16, 16), (8, 8, 16), (64, 16, 16), np.s_[0:10])
