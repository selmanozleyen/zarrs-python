"""A read decodes into a caller-supplied buffer, and that buffer must be writable.

`nparray_to_unsafe_cell_slice` builds a `&mut [u8]` over numpy's own memory with
`from_raw_parts_mut`, whose contract requires the memory be writable. Nothing checked it.

numpy hands out read-only arrays in ordinary situations -- a read-only mmap, a view over an
immutable `bytes`, anything with `flags.writeable = False`. Writing through one is a segfault
or a silently diverging copy-on-write page. Neither is a Python exception, so the caller sees
either a dead interpreter or, worse, a result that looks fine and is not.

The contiguity precondition of the same `unsafe` was already checked, four lines up.
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
def pipeline_impl(tmp_path: Path):
    """The Rust pipeline object zarr built for a small array."""
    path = tmp_path / "a.zarr"
    zarr.create_array(path, shape=(32,), dtype="float32", chunks=(8,))[:] = np.arange(
        32, dtype="float32"
    )
    with zarr.config.set(ZARRS):
        return zarr.open_array(path, mode="r")._async_array.codec_pipeline.impl


def test_a_read_only_output_is_refused(pipeline_impl) -> None:
    """An empty batch is enough: the buffer is converted before anything else happens, so this
    reaches the check without needing to describe a single chunk."""
    frozen = np.empty(8, dtype="float32")
    frozen.flags.writeable = False

    with pytest.raises(ValueError, match="not writable"):
        pipeline_impl.retrieve_chunks_and_apply_index([], frozen)


def test_a_writable_output_is_still_accepted(pipeline_impl) -> None:
    """The control. Without it the test above would also pass if the conversion refused
    everything."""
    pipeline_impl.retrieve_chunks_and_apply_index([], np.empty(8, dtype="float32"))
