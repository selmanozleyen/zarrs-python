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


def test_asking_above_what_the_process_builds_warns(sharded_1d) -> None:
    path, values = sharded_1d
    with zarr.config.set(PIPELINE):
        _read(path, values)
    built_read, built_decode = pool_sizes()
    assert built_read is not None, (
        "a read through the chunk-unit path must build the pools"
    )

    # Above `pool_max`, which is the one width masking still cannot give: the pools are
    # built once and cannot grow.
    absurd = built_read + 1_000
    with (
        zarr.config.set(PIPELINE | {"codec_pipeline.read_workers": absurd}),
        pytest.warns(UserWarning, match=rf"read_workers = {absurd} is above"),
    ):
        _read(path, values)

    # The warning is not a resize, and must not have built the other pool either.
    assert pool_sizes() == (built_read, built_decode)


def test_strict_makes_it_an_error(sharded_1d) -> None:
    path, values = sharded_1d
    with zarr.config.set(PIPELINE):
        _read(path, values)
    built_read, _ = pool_sizes()
    absurd = built_read + 1_000

    strict = PIPELINE | {
        "codec_pipeline.read_workers": absurd,
        "codec_pipeline.strict": True,
    }
    with zarr.config.set(strict), pytest.raises(ValueError, match="is above"):
        _read(path, values)


def test_a_width_below_the_maximum_is_honoured_after_the_pools_exist(sharded_1d) -> None:
    """The point of masking: a later array asking for FEWER workers gets them, silently.

    Under the previous design the pools were sized by the first read, so this asked for a
    width that could never arrive and warned. Now the pools are built wide once and each call
    takes the slice it asked for, so two arrays can differ.
    """
    path, values = sharded_1d
    with zarr.config.set(PIPELINE):
        _read(path, values)
    built_read, built_decode = pool_sizes()

    with zarr.config.set(PIPELINE | {"codec_pipeline.read_workers": 2}):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", UserWarning)
            _read(path, values)
        assert [str(w.message) for w in record] == []

    # Masking, not resizing: the pools are untouched by a call that used fewer of them.
    assert pool_sizes() == (built_read, built_decode)


def test_the_width_actually_in_force_is_silent(sharded_1d) -> None:
    """No warning when the ask is servable, or the signal is noise."""
    path, values = sharded_1d
    with zarr.config.set(PIPELINE):
        _read(path, values)
    built_read, _ = pool_sizes()

    with zarr.config.set(PIPELINE | {"codec_pipeline.read_workers": built_read}):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", UserWarning)
            _read(path, values)
        assert [str(w.message) for w in record] == []
