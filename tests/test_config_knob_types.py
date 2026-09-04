"""A config value of the wrong type must name the knob, not blame the array.

Every `codec_pipeline.*` option is handed to pyo3 as a keyword argument, and pyo3 raises
`TypeError` for a value of the wrong type. `get_codec_pipeline_impl` wraps the whole
construction in `except TypeError` -- which is there for bad array metadata -- so a mistyped
knob was reported as `UserWarning: Array is unsupported by ZarrsCodecPipeline` and the pipeline
was silently off for that array, from then on, over the one thing the user could not change.

A negative value was worse still: `OverflowError` is not a `TypeError`, so it escaped naming no
key at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

ZARRS = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}


@pytest.fixture
def array(tmp_path: Path) -> tuple[Path, np.ndarray]:
    values = np.arange(32, dtype="float32")
    path = tmp_path / "a.zarr"
    zarr.create_array(path, shape=values.shape, dtype=values.dtype, chunks=(8,))[:] = values
    return path, values


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        ("codec_pipeline.file_handle_cache_size", 8.0),
        ("codec_pipeline.file_handle_cache_size", -1),
        ("codec_pipeline.chunk_concurrent_minimum", "four"),
        ("threading.max_workers", 2.5),
        ("codec_pipeline.direct_io", "yes"),
        ("codec_pipeline.validate_checksums", "no"),
    ],
)
def test_a_bad_value_names_the_knob(array, knob: str, value) -> None:
    """Under `strict`, an error naming the key. Without it, a warning naming the key AND the
    default it fell back to -- and a pipeline that still works."""
    path, values = array

    with (
        zarr.config.set(ZARRS | {knob: value, "codec_pipeline.strict": True}),
        pytest.raises(ValueError, match=knob.replace(".", r"\.")),
    ):
        zarr.open_array(path, mode="r")[:]

    with (
        zarr.config.set(ZARRS | {knob: value}),
        pytest.warns(UserWarning, match=knob.replace(".", r"\.")),
    ):
        got = zarr.open_array(path, mode="r")[:]
    np.testing.assert_array_equal(got, values)


def test_a_bool_knob_refuses_a_true_string(array) -> None:
    """`"false"` is a true string. A knob that turns ON when its value is misspelled is worse
    than one that refuses, so these are checked with `isinstance`, not truthiness."""
    path, _ = array
    with (
        zarr.config.set(
            ZARRS
            | {"codec_pipeline.direct_io": "false", "codec_pipeline.strict": True}
        ),
        pytest.raises(ValueError, match="must be a bool"),
    ):
        zarr.open_array(path, mode="r")[:]


def test_validate_checksums_still_treats_None_as_True(array) -> None:
    """Documented old behaviour, normalised before the type check rather than warned about."""
    path, values = array
    with zarr.config.set(ZARRS | {"codec_pipeline.validate_checksums": None}):
        np.testing.assert_array_equal(zarr.open_array(path, mode="r")[:], values)
