from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import pytest
from zarr import config
from zarr.storage import FsspecStore, LocalStore, MemoryStore, ZipStore

from zarrs._internal import CodecPipelineImpl

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import Any, Literal

    from zarr.abc.store import Store
    from zarr.core.common import MemoryOrder


@dataclass
class ArrayRequest:
    shape: tuple[int, ...]
    dtype: str
    order: MemoryOrder


@pytest.fixture
def entries(monkeypatch) -> dict[str, int]:
    """Counts of batches served by each Rust entry point.

    The whole point of this fixture is that a test asserting `handle == 0` -- a selection
    DECLINES -- means what it says. An earlier version looked the entry point up with
    `getattr(..., None)` and skipped it when absent, so a rename would have left every such
    assertion vacuously true: the counter no test could increment is also the counter every
    decline test compares against zero. That is this project's own "a knob that was set is not
    a knob that arrived", inside the fixture built to prevent it. So it patches, or it fails.

    There was a second entry point, `retrieve_chunks_and_apply_index` -- the fused path, which
    an audit of the public indexing surface found nothing reaching, so it was removed. Its
    counter is gone with it; that it no longer exists is asserted once, in
    `test_the_fused_path_is_gone`, rather than fifteen times as a zero that cannot move.
    """
    counts = {"handle": 0}
    name = "retrieve_chunk_items_and_apply_index"
    original = getattr(CodecPipelineImpl, name)

    def wrapper(self, *args, _original=original, **kwargs):
        counts["handle"] += 1
        return _original(self, *args, **kwargs)

    monkeypatch.setattr(CodecPipelineImpl, name, wrapper)
    return counts


@pytest.fixture(autouse=True)
def _setup_codec_pipeline():
    config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    pass


async def parse_store(
    store: Literal["local", "memory", "remote", "zip"], path: Path
) -> LocalStore | MemoryStore | FsspecStore | ZipStore:
    if store == "local":
        return await LocalStore.open(path)
    if store == "memory":
        return await MemoryStore.open()
    if store == "remote":
        return await FsspecStore.open(url=path)
    if store == "zip":
        return await ZipStore.open(path / "zarr.zip")
    raise AssertionError


@pytest.fixture(params=["local"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> Store:
    param = request.param
    return await parse_store(param, tmp_path)


@pytest.fixture
def array_fixture(request: pytest.FixtureRequest) -> npt.NDArray[Any]:
    array_request: ArrayRequest = request.param
    return (
        np.arange(np.prod(array_request.shape))
        .reshape(array_request.shape, order=array_request.order)
        .astype(array_request.dtype)
    )


# tests that also fail with zarr-python's default codec pipeline
zarr_python_default_codec_pipeline_failures = [
    # ellipsis weirdness, need to report, v3
    "test_roundtrip[oindex-2d-contiguous_in_chunk_array-ellipsis-v3]",
    "test_roundtrip[oindex-2d-discontinuous_in_chunk_array-ellipsis-v3]",
    "test_roundtrip[vindex-2d-contiguous_in_chunk_array-ellipsis-v3]",
    "test_roundtrip[vindex-2d-discontinuous_in_chunk_array-ellipsis-v3]",
    "test_roundtrip[oindex-2d-across_chunks_indices_array-ellipsis-v3]",
    "test_roundtrip[vindex-2d-ellipsis-across_chunks_indices_array-v3]",
    "test_roundtrip[vindex-2d-across_chunks_indices_array-ellipsis-v3]",
    "test_roundtrip[vindex-2d-ellipsis-contiguous_in_chunk_array-v3]",
    "test_roundtrip[vindex-2d-ellipsis-discontinuous_in_chunk_array-v3]",
    "test_roundtrip_read_only_zarrs[oindex-2d-contiguous_in_chunk_array-ellipsis-v3]",
    "test_roundtrip_read_only_zarrs[oindex-2d-discontinuous_in_chunk_array-ellipsis-v3]",
    "test_roundtrip_read_only_zarrs[vindex-2d-contiguous_in_chunk_array-ellipsis-v3]",
    "test_roundtrip_read_only_zarrs[vindex-2d-discontinuous_in_chunk_array-ellipsis-v3]",
    "test_roundtrip_read_only_zarrs[oindex-2d-across_chunks_indices_array-ellipsis-v3]",
    "test_roundtrip_read_only_zarrs[vindex-2d-ellipsis-across_chunks_indices_array-v3]",
    "test_roundtrip_read_only_zarrs[vindex-2d-across_chunks_indices_array-ellipsis-v3]",
    "test_roundtrip_read_only_zarrs[vindex-2d-ellipsis-contiguous_in_chunk_array-v3]",
    "test_roundtrip_read_only_zarrs[vindex-2d-ellipsis-discontinuous_in_chunk_array-v3]",
    # v2
    "test_roundtrip[oindex-2d-contiguous_in_chunk_array-ellipsis-v2]",
    "test_roundtrip[oindex-2d-discontinuous_in_chunk_array-ellipsis-v2]",
    "test_roundtrip[vindex-2d-contiguous_in_chunk_array-ellipsis-v2]",
    "test_roundtrip[vindex-2d-discontinuous_in_chunk_array-ellipsis-v2]",
    "test_roundtrip[oindex-2d-across_chunks_indices_array-ellipsis-v2]",
    "test_roundtrip[vindex-2d-ellipsis-across_chunks_indices_array-v2]",
    "test_roundtrip[vindex-2d-across_chunks_indices_array-ellipsis-v2]",
    "test_roundtrip[vindex-2d-ellipsis-contiguous_in_chunk_array-v2]",
    "test_roundtrip[vindex-2d-ellipsis-discontinuous_in_chunk_array-v2]",
    "test_roundtrip_read_only_zarrs[oindex-2d-contiguous_in_chunk_array-ellipsis-v2]",
    "test_roundtrip_read_only_zarrs[oindex-2d-discontinuous_in_chunk_array-ellipsis-v2]",
    "test_roundtrip_read_only_zarrs[vindex-2d-contiguous_in_chunk_array-ellipsis-v2]",
    "test_roundtrip_read_only_zarrs[vindex-2d-discontinuous_in_chunk_array-ellipsis-v2]",
    "test_roundtrip_read_only_zarrs[oindex-2d-across_chunks_indices_array-ellipsis-v2]",
    "test_roundtrip_read_only_zarrs[vindex-2d-ellipsis-across_chunks_indices_array-v2]",
    "test_roundtrip_read_only_zarrs[vindex-2d-across_chunks_indices_array-ellipsis-v2]",
    "test_roundtrip_read_only_zarrs[vindex-2d-ellipsis-contiguous_in_chunk_array-v2]",
    "test_roundtrip_read_only_zarrs[vindex-2d-ellipsis-discontinuous_in_chunk_array-v2]",
    # need to investigate this one - it seems to fail with the default pipeline
    # but it makes some sense that it succeeds with ours since we fall-back to numpy indexing
    # in the case of a collapsed dimension
    # "test_roundtrip_read_only_zarrs[vindex-2d-contiguous_in_chunk_array-contiguous_in_chunk_array]",
]


def pytest_collection_modifyitems(
    config: pytest.Config, items: Iterable[pytest.Item]
) -> None:
    for item in items:
        if item.name in zarr_python_default_codec_pipeline_failures:
            xfail_marker = pytest.mark.xfail(
                reason="This test fails with the zarr-python default codec pipeline."
            )
            item.add_marker(xfail_marker)
