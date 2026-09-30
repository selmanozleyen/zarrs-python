from __future__ import annotations

import asyncio
import itertools
import json
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypedDict
from warnings import warn

import numpy as np
from zarr.abc.codec import Codec, CodecPipeline
from zarr.codecs._v2 import V2Codec
from zarr.core import BatchedCodecPipeline
from zarr.core.config import config
from zarr.core.metadata import ArrayMetadata, ArrayV2Metadata, ArrayV3Metadata

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from typing import Self

    from zarr.abc.store import ByteGetter, ByteSetter, Store
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDArrayLike, NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.indexing import SelectorTuple
    from zarr.dtype import ZDType
    from zarr.storage import StorePath

from ._internal import ChunkItems, CodecPipelineImpl
from .utils import (
    DiscontiguousArrayError,
    FillValueNoneError,
    UnsupportedVIndexingError,
    chunk_info_for_read,
    make_chunk_info_for_rust_with_indices,
)


class UnsupportedDataTypeError(Exception):
    pass


class UnsupportedMetadataError(Exception):
    pass


class UnsupportedRangeReadError(Exception):
    pass


FALLBACK_TO_ZARR_PYTHON = (
    UnsupportedMetadataError,
    DiscontiguousArrayError,
    UnsupportedVIndexingError,
    UnsupportedDataTypeError,
    FillValueNoneError,
)


def _read_config() -> tuple[int | None, int | None, bool]:
    return (
        config.get("codec_pipeline.read_workers", None),
        config.get("codec_pipeline.decode_workers", None),
        config.get("codec_pipeline.strict", False),
    )


def get_codec_pipeline_impl(
    metadata: ArrayMetadata, store: Store, *, strict: bool
) -> CodecPipelineImpl | None:
    try:
        array_metadata_json = json.dumps(metadata.to_dict())
        # Maintain old behavior: https://github.com/zarrs/zarrs-python/tree/b36ba797cafec77f5f41a25316be02c718a2b4f8?tab=readme-ov-file#configuration
        validate_checksums = config.get("codec_pipeline.validate_checksums", True)
        if validate_checksums is None:
            validate_checksums = True
        return CodecPipelineImpl(
            array_metadata_json,
            store_config=store,
            validate_checksums=validate_checksums,
            chunk_concurrent_minimum=config.get(
                "codec_pipeline.chunk_concurrent_minimum", None
            ),
            chunk_concurrent_maximum=config.get(
                "codec_pipeline.chunk_concurrent_maximum", None
            ),
            num_threads=config.get("threading.max_workers", None),
            direct_io=config.get("codec_pipeline.direct_io", False),
            file_handle_cache_size=config.get(
                "codec_pipeline.file_handle_cache_size", 0
            ),
        )
    except TypeError as e:
        if strict:
            raise UnsupportedMetadataError() from e

        warn(
            f"Array is unsupported by ZarrsCodecPipeline: {e}",
            category=UserWarning,
        )
        return None


def get_codec_pipeline_fallback(
    metadata: ArrayMetadata, *, strict: bool
) -> BatchedCodecPipeline | None:
    if strict:
        return None
    else:
        codecs = array_metadata_to_codecs(metadata)
        return BatchedCodecPipeline.from_codecs(codecs)


class ZarrsCodecPipelineState(TypedDict):
    codec_metadata_json: str
    codecs: tuple[Codec, ...]


def array_metadata_to_codecs(metadata: ArrayMetadata) -> list[Codec]:
    if isinstance(metadata, ArrayV3Metadata):
        return metadata.codecs
    elif isinstance(metadata, ArrayV2Metadata):
        v2_codec = V2Codec(filters=metadata.filters, compressor=metadata.compressor)
        return [v2_codec]


def _shards_touched(
    starts: np.ndarray, lengths: np.ndarray, size: int
) -> np.ndarray:
    """The ascending ids of the shards of extent `size` that the runs touch."""
    keep = lengths > 0
    starts, lengths = starts[keep], lengths[keep]
    first, last = starts // size, (starts + lengths - 1) // size
    ids = np.union1d(first, last)
    if (wide := last - first > 1).any():
        between = [
            np.arange(f + 1, l) for f, l in zip(first[wide], last[wide], strict=True)
        ]
        ids = np.union1d(ids, np.concatenate(between))
    return ids.astype(np.int64)


@dataclass
class ZarrsCodecPipeline(CodecPipeline):
    metadata: ArrayMetadata
    store: Store
    impl: CodecPipelineImpl | None
    python_impl: BatchedCodecPipeline | None

    def __getstate__(self) -> ZarrsCodecPipelineState:
        return {"metadata": self.metadata, "store": self.store}

    def __setstate__(self, state: ZarrsCodecPipelineState):
        self.metadata = state["metadata"]
        self.store = state["store"]
        strict = config.get("codec_pipeline.strict", False)
        self.impl = get_codec_pipeline_impl(self.metadata, self.store, strict=strict)
        self.python_impl = get_codec_pipeline_fallback(self.metadata, strict=strict)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return self

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec]) -> Self:
        return BatchedCodecPipeline.from_codecs(codecs)

    @classmethod
    def from_array_metadata_and_store(
        cls, array_metadata: ArrayMetadata, store: Store
    ) -> Self:
        strict = config.get("codec_pipeline.strict", False)
        return cls(
            metadata=array_metadata,
            store=store,
            impl=get_codec_pipeline_impl(array_metadata, store, strict=strict),
            python_impl=get_codec_pipeline_fallback(array_metadata, strict=strict),
        )

    @cached_property
    def _inner_chunk_shape(self) -> tuple[int, ...] | None:
        """The inner chunk shape, `()` if unsharded, `None` if the read path does not serve it."""
        if self.impl is None:
            return None
        shape = self.impl.inner_chunk_shape()
        return None if shape is None else tuple(shape)

    @property
    def supports_partial_decode(self) -> bool:
        return False

    @property
    def supports_partial_encode(self) -> bool:
        return False

    def __iter__(self) -> Iterator[Codec]:
        yield from self.codecs

    def validate(
        self, *, shape: tuple[int, ...], dtype: ZDType, chunk_grid: ChunkGrid
    ) -> None:
        raise NotImplementedError("validate")

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        raise NotImplementedError("compute_encoded_size")

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        raise NotImplementedError("decode")

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        raise NotImplementedError("encode")

    async def read(
        self,
        batch_info: Iterable[
            tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]
        ],
        out: NDBuffer,  # type: ignore
        drop_axes: tuple[int, ...] = (),  # FIXME: unused
    ) -> None:
        # FIXME: Error if array is not in host memory
        if not out.dtype.isnative:
            raise RuntimeError("Non-native byte order not supported")
        try:
            if self.impl is None:
                raise UnsupportedMetadataError()
            self._raise_error_on_unsupported_batch_dtype(batch_info)
            chunks_desc = chunk_info_for_read(
                batch_info, drop_axes, out.shape, self._inner_chunk_shape
            )
        except FALLBACK_TO_ZARR_PYTHON:
            if self.python_impl is None:
                raise
            await self.python_impl.read(batch_info, out, drop_axes)
            return None
        else:
            out: NDArrayLike = out.as_ndarray_like()
            await asyncio.to_thread(
                self.impl.retrieve_chunk_items_and_apply_index,
                chunks_desc.chunk_info_with_indices,
                out,
                *_read_config(),
            )
            return None

    async def read_runs(
        self,
        store_path: StorePath,
        metadata: ArrayMetadata,
        runs: tuple[tuple[np.ndarray, np.ndarray], ...],
        out: NDBuffer | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Read the product of runs on every axis into `out`, or defer to zarr's default."""
        buffer = out.as_ndarray_like() if hasattr(out, "as_ndarray_like") else out
        try:
            read = self.plan_runs(store_path, metadata, runs, buffer)
        except UnsupportedRangeReadError:
            await super().read_runs(store_path, metadata, runs, out, **kwargs)
            return
        await asyncio.to_thread(read)

    def plan_runs(
        self,
        store_path: StorePath,
        metadata: ArrayMetadata,
        runs: tuple[tuple[np.ndarray, np.ndarray], ...],
        out: np.ndarray,
    ) -> Callable[[], None]:
        """Check the read now and return a call that does it.

        Raises `UnsupportedRangeReadError`, before anything is read, for an array not served.
        """
        inner = self._inner_chunk_shape
        grid = getattr(metadata, "chunk_grid", None)
        shape = tuple(int(v) for v in metadata.shape)
        if (
            self.impl is None
            or inner is None
            or not shape
            or not hasattr(grid, "chunk_shape")
        ):
            raise UnsupportedRangeReadError(
                "a chunked array with a regular grid is needed"
            )
        dtype = metadata.dtype.to_native_dtype()
        if dtype.kind in {"V", "S", "U", "M", "m", "O", "T"} or not dtype.isnative:
            raise UnsupportedRangeReadError(f"dtype {dtype} is not served")
        if not out.flags.c_contiguous:
            raise UnsupportedRangeReadError("out must be C contiguous")
        if len(runs) != len(shape):
            raise ValueError(
                f"one (starts, lengths) pair per axis is needed: {len(runs)} for {len(shape)}"
            )
        runs = [
            (np.asarray(s, dtype=np.int64), np.asarray(n, dtype=np.int64))
            for s, n in runs
        ]
        for axis, ((starts, lengths), size) in enumerate(zip(runs, shape, strict=True)):
            if starts.ndim != 1 or starts.shape != lengths.shape:
                raise ValueError("starts and lengths must be 1-D and the same length")
            if (lengths < 0).any() or (starts < 0).any() or (starts + lengths > size).any():
                raise IndexError(
                    f"a run falls outside axis {axis}, which has {size} elements"
                )
        want = tuple(int(lengths.sum()) for _, lengths in runs)
        if out.shape != want or out.dtype != dtype:
            raise ValueError(
                f"out must be a {dtype} array of shape {want}, not {out.dtype} {out.shape}"
            )
        shard_shape = [int(v) for v in grid.chunk_shape]
        ids = [
            _shards_touched(starts, lengths, size)
            for (starts, lengths), size in zip(runs, shard_shape, strict=True)
        ]
        keys = [
            (store_path / metadata.encode_chunk_key(c)).path
            for c in itertools.product(*(i.tolist() for i in ids))
        ]
        knobs = _read_config()
        impl = self.impl

        def read() -> None:
            if not out.size:
                return
            handle = ChunkItems()
            handle.push_runs(
                keys,
                ids,
                [starts for starts, _ in runs],
                [lengths for _, lengths in runs],
                shard_shape,
                list(inner or shard_shape),
            )
            impl.retrieve_chunk_items_and_apply_index(handle, out, *knobs)

        return read

    async def write(
        self,
        batch_info: Iterable[
            tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]
        ],
        value: NDBuffer,  # type: ignore
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        try:
            if self.impl is None:
                raise UnsupportedMetadataError()
            self._raise_error_on_unsupported_batch_dtype(batch_info)
            chunks_desc = make_chunk_info_for_rust_with_indices(
                batch_info, drop_axes, value.shape
            )
        except FALLBACK_TO_ZARR_PYTHON:
            if self.python_impl is None:
                raise
            await self.python_impl.write(batch_info, value, drop_axes)
            return None
        else:
            # FIXME: Error if array is not in host memory
            value_np: NDArrayLike | np.ndarray = value.as_ndarray_like()
            if not value_np.dtype.isnative:
                value_np = np.ascontiguousarray(
                    value_np, dtype=value_np.dtype.newbyteorder("=")
                )
            elif not value_np.flags.c_contiguous:
                value_np = np.ascontiguousarray(value_np)
            await asyncio.to_thread(
                self.impl.store_chunks_with_indices,
                chunks_desc.chunk_info_with_indices,
                value_np,
                chunks_desc.write_empty_chunks,
            )
            return None

    def _raise_error_on_unsupported_batch_dtype(
        self,
        batch_info: Iterable[
            tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]
        ],
    ):
        # https://github.com/LDeakin/zarrs/blob/0532fe983b7b42b59dbf84e50a2fe5e6f7bad4ce/zarrs_metadata/src/v2_to_v3.rs#L289-L293 for VSUMm
        # Further, our pipeline does not support variable-length objects due to limitations on decode_into, so object/np.dtypes.StringDType is also out
        if any(
            info.dtype.to_native_dtype().kind in {"V", "S", "U", "M", "m", "O", "T"}
            for (_, info, _, _, _) in batch_info
        ):
            raise UnsupportedDataTypeError()
