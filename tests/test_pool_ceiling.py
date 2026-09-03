"""A ceiling that cannot be honoured must say so.

The two pools are process-wide and sized by the FIRST read in the process. The ceilings are
read when an ARRAY IS OPENED -- like `num_threads` and the chunk-concurrency bounds beside
them -- so an array opened later, asking for a different width, gets the pools that exist.

That is the accepted cost of persistence. Doing it silently is not: from the outside, "the
knob did not pay" and "the knob never arrived" are the same observation. So the read warns,
and under `codec_pipeline.strict` it raises -- the same switch that turns a decline into a
raise rather than a quiet fallback.

One file, because the pools are process-wide: the first read here builds them and everything
after is measured against that.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from zarrs._internal import pool_sizes

if TYPE_CHECKING:
    from pathlib import Path

PIPELINE = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
LENGTH = 200


@pytest.fixture
def sharded_1d(tmp_path: Path) -> tuple[Path, np.ndarray]:
    values = np.arange(LENGTH, dtype=np.float64)
    path = tmp_path / "1d.zarr"
    array = zarr.create_array(
        store=path, shape=(LENGTH,), chunks=(8,), shards=(32,), dtype=values.dtype
    )
    array[:] = values
    return path, values


def _read(path: Path, values: np.ndarray) -> None:
    """One read through the chunk-unit path, which is what builds the pools."""
    selection = np.array([1, 9, 17, 40, 41])
    np.testing.assert_array_equal(
        zarr.open_array(path, mode="r")[selection], values[selection]
    )


def test_a_ceiling_asked_for_after_the_pools_exist_warns(sharded_1d) -> None:
    path, values = sharded_1d
    with zarr.config.set(PIPELINE):
        _read(path, values)
    built_read, built_decode = pool_sizes()
    assert built_read is not None, (
        "a read through the chunk-unit path must build the pools"
    )

    # A size no earlier read can have built, so the mismatch does not depend on what ran
    # first. The array is opened INSIDE the block: that is when the ceiling is read.
    absurd = built_read + 1_000
    with (
        zarr.config.set(PIPELINE | {"codec_pipeline.read_worker_ceiling": absurd}),
        pytest.warns(UserWarning, match=rf"read_worker_ceiling = {absurd} was ignored"),
    ):
        _read(path, values)

    # The warning is not a resize, and it must not have built the OTHER pool either.
    assert pool_sizes() == (built_read, built_decode)


def test_strict_makes_it_an_error(sharded_1d) -> None:
    """`codec_pipeline.strict` is already "tell me rather than paper over it"."""
    path, values = sharded_1d
    with zarr.config.set(PIPELINE):
        _read(path, values)
    built_read, _ = pool_sizes()
    absurd = built_read + 1_000

    strict = PIPELINE | {
        "codec_pipeline.read_worker_ceiling": absurd,
        "codec_pipeline.strict": True,
    }
    with zarr.config.set(strict), pytest.raises(ValueError, match="was ignored"):
        _read(path, values)


def test_the_ceiling_actually_in_force_is_silent(sharded_1d) -> None:
    """No warning when the ask matches what was built -- or the signal is noise."""
    path, values = sharded_1d
    with zarr.config.set(PIPELINE):
        _read(path, values)
    built_read, _ = pool_sizes()

    with zarr.config.set(PIPELINE | {"codec_pipeline.read_worker_ceiling": built_read}):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", UserWarning)
            _read(path, values)
        assert [str(w.message) for w in record] == []


def test_the_decode_ceiling_is_checked_too(sharded_1d) -> None:
    """The two knobs are checked by one loop, and only one of them had a test.

    A copy-paste in the decode row -- reporting the READ pool's size against the decode ask --
    would have been invisible: every case here set `read_worker_ceiling`, so the row that was
    wrong was also the row that never ran.
    """
    path, values = sharded_1d
    with zarr.config.set(PIPELINE):
        _read(path, values)
    _, built_decode = pool_sizes()
    assert built_decode is not None

    absurd = built_decode + 1_000
    with (
        zarr.config.set(PIPELINE | {"codec_pipeline.decode_worker_ceiling": absurd}),
        pytest.warns(
            UserWarning, match=rf"decode_worker_ceiling = {absurd} was ignored"
        ),
    ):
        _read(path, values)
