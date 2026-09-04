from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, TypedDict
from warnings import warn

import numpy as np
from zarr.abc.codec import Codec, CodecPipeline
from zarr.codecs._v2 import V2Codec
from zarr.core import BatchedCodecPipeline
from zarr.core.config import config
from zarr.core.metadata import ArrayMetadata, ArrayV2Metadata, ArrayV3Metadata

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Self

    from zarr.abc.store import ByteGetter, ByteSetter, Store
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDArrayLike, NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.indexing import SelectorTuple
    from zarr.dtype import ZDType

from ._internal import CodecPipelineImpl
from .utils import (
    DiscontiguousArrayError,
    FillValueNoneError,
    UnsupportedVIndexingError,
    chunk_info_for_read,
    chunk_info_for_write,
)


class UnsupportedDataTypeError(Exception):
    pass


class UnsupportedMetadataError(Exception):
    pass


# : What sends a batch to zarr-python instead. `read` and `write` must use the same set:
# : a member in one and not the other falls back on read and raises on write.
FALLBACK_TO_ZARR_PYTHON = (
    UnsupportedMetadataError,
    DiscontiguousArrayError,
    UnsupportedVIndexingError,
    UnsupportedDataTypeError,
    FillValueNoneError,
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
        """The innermost unit the codec chain decodes, as Rust reports it.

        A tuple is the inner chunk of a sharded array; `()` means the array is not sharded, so
        its chunk is its own decode unit; `None` means refuse. Asked rather than derived: this
        used to walk zarr's codec objects while Rust answered the same question from the bound
        chain, and two derivations of one fact can disagree.
        """
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
            desc = chunks_desc.chunk_info_with_indices
            # One entry point. `chunk_info_for_read` either produces a handle or raises, and the
            # raise is caught above as a fall back to zarr-python: there is no second Rust read
            # path to choose between any more.
            retrieve = self.impl.retrieve_chunk_items_and_apply_index
            # Per call because it IS a per-call decision: a threshold on how many byte-range
            # reads one chunk is worth, not a size that something was built at. The two pool
            # ceilings are not here -- they were read when the array was opened.
            # PER CALL, all of it. The widths are a mask on pools built once, so nothing
            # has to be fixed when the array is opened -- which means a `zarr.config.set`
            # block around a read is honoured, rather than silently too late.
            await asyncio.to_thread(
                retrieve,
                desc,
                out,
                config.get("codec_pipeline.read_workers", None),
                config.get("codec_pipeline.decode_workers", None),
                config.get("codec_pipeline.strict", False),
            )
            return None

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
            chunks_desc = chunk_info_for_write(batch_info, drop_axes, value.shape)
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
