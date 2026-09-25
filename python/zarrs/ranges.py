from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

from .pipeline import UnsupportedRangeReadError, ZarrsCodecPipeline

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import Any

    import zarr


def aread_ranges(
    array: zarr.Array | zarr.AsyncArray,
    starts: np.ndarray,
    lengths: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> Coroutine[Any, Any, np.ndarray]:
    """Read `lengths[i]` elements from `starts[i]` of a 1-D array, back to back, as a coroutine.

    A stand-in for a multi-range selection zarr does not have yet: the work is per range, where
    a coordinate selection of the same read is per element. Raises `UnsupportedRangeReadError`
    at once, before anything is read, if the array is not on the zarrs pipeline or is not a
    shape this serves.
    """
    a = getattr(array, "_async_array", array)
    if not isinstance(a.codec_pipeline, ZarrsCodecPipeline):
        raise UnsupportedRangeReadError("the array is not on the zarrs codec pipeline")
    if out is None:
        out = np.empty(int(np.sum(lengths)), dtype=a.metadata.dtype.to_native_dtype())
    read = a.codec_pipeline.plan_ranges(a.store_path, a.metadata, starts, lengths, out)

    async def run() -> np.ndarray:
        await asyncio.to_thread(read)
        return out

    return run()
