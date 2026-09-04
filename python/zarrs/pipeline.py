from __future__ import annotations

import asyncio
import json
import threading
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


#: What sends a batch to zarr-python's pipeline instead of this one.
#:
#: Each means "zarrs cannot describe this", not "the read failed". `read` and `write` must
#: use the same set: a member in one and not the other falls back on read, raises on write.
FALLBACK_TO_ZARR_PYTHON = (
    UnsupportedMetadataError,
    DiscontiguousArrayError,
    UnsupportedVIndexingError,
    UnsupportedDataTypeError,
    FillValueNoneError,
)


#: `[served, declined]` read batches since import, under `_READ_LOCK`. See `read_stats`.
#:
#: A LOCK, because `_READ_COUNTS[i] += 1` is a read-modify-write and reads run on threads
#: (`asyncio.to_thread` in `read`). The GIL does not make that atomic -- it can be released
#: between the load and the store -- so concurrent reads would undercount, in a counter whose
#: whole purpose is to be trustworthy about how often the fast path was taken. It is taken
#: twice per BATCH, next to work measured in milliseconds.
_READ_COUNTS = [0, 0]
_READ_LOCK = threading.Lock()


def read_stats() -> tuple[int, int]:
    """`(served, declined)` read batches since import.

    THE OUTCOME NOTHING ELSE COUNTS. A selection this pipeline cannot describe is handed to
    zarr-python instead, which returns identical values more slowly. From the outside that is
    indistinguishable from working -- which, for a pipeline whose only failure mode is "correct
    but slow", is the one thing worth being able to ask.

    Process-wide and monotonic, like the other counters here, so a caller takes a delta around
    the read it cares about rather than resetting a global that another thread is using.

    `codec_pipeline.strict` answers a nearby question -- it turns a decline into an exception --
    but it is read when the array is opened and is all-or-nothing, so it cannot profile a mixed
    workload.
    """
    with _READ_LOCK:
        served, declined = _READ_COUNTS
    return served, declined


def _int_knob(key: str, default: int | None, *, strict: bool) -> int | None:
    """A config integer, checked HERE where its name is still in scope.

    THE FULL KEY IS PASSED IN, as a literal, rather than built from a short name. That is not
    style: this project's rule is that a knob which was set is not a knob that arrived, and the
    only way anything outside the process can check which knobs a build reads is to look for
    the key in the source. An earlier version took `"read_pool_size"` and wrote
    `f"codec_pipeline.{name}"`, so no literal key survived anywhere -- and two separate tools
    that scan for one concluded this build read no knobs at all, then measured it against
    another build at a different width.

    Every one of these is handed to pyo3 as a keyword argument, and pyo3 raises `TypeError`
    for a value of the wrong type -- which the caller catches and reports as "Array is
    unsupported by ZarrsCodecPipeline". So `read_pool_size: 8.0` disabled the whole pipeline
    for that array, permanently and silently, and blamed the one thing the user could not
    change. A negative value was worse: `OverflowError` is not a `TypeError`, so it escaped
    naming no key at all.
    """
    value = config.get(key, default)
    if value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0):
        return value
    message = f"{key} must be a non-negative integer or None, got {value!r}; using {default!r}"
    # RAISING ONLY UNDER `strict`, which is this file's standing convention: non-strict means
    # "do not fail a read over something this pipeline can work around". An unconditional raise
    # here failed every array OPEN, including write-only workloads that never touch the knob.
    # The warning still names the key, which was the whole point -- what it must not do is turn
    # a typo in one option into an unopenable array.
    if strict:
        raise ValueError(message)
    warn(message, category=UserWarning, stacklevel=2)
    return default


def get_codec_pipeline_impl(
    metadata: ArrayMetadata, store: Store, *, strict: bool
) -> CodecPipelineImpl | None:
    try:
        array_metadata_json = json.dumps(metadata.to_dict())
        # Maintain old behavior: https://github.com/zarrs/zarrs-python/tree/b36ba797cafec77f5f41a25316be02c718a2b4f8?tab=readme-ov-file#configuration
        validate_checksums = config.get("codec_pipeline.validate_checksums", True)
        max_workers = _int_knob("threading.max_workers", None, strict=strict)
        if validate_checksums is None:
            validate_checksums = True
        return CodecPipelineImpl(
            array_metadata_json,
            store_config=store,
            validate_checksums=validate_checksums,
            chunk_concurrent_minimum=_int_knob("codec_pipeline.chunk_concurrent_minimum", None, strict=strict),
            chunk_concurrent_maximum=_int_knob("codec_pipeline.chunk_concurrent_maximum", None, strict=strict),
            num_threads=max_workers,
            direct_io=config.get("codec_pipeline.direct_io", False),
            file_handle_cache_size=_int_knob("codec_pipeline.file_handle_cache_size", 0, strict=strict),
            # Read at OPEN, like `num_threads` and the chunk-concurrency bounds beside them.
            # They size process-wide pools that only the first read builds, so offering them
            # per call would be offering a choice that cannot be honoured.
            # FALLING BACK TO `threading.max_workers` when neither is set. That knob used to
            # bound everything this library did on the Rust side; the two pools took reads out
            # from under it, so a user who had tuned it found it silently governing writes
            # only. Each pool gets that many -- they are not interchangeable, and dividing one
            # budget between a reader that waits and a decoder that computes is the design
            # that was measured and lost (see `Pools` in `read_decode.rs`). The README says so.
            read_pool_size=_int_knob(
                "codec_pipeline.read_pool_size", max_workers, strict=strict
            ),
            decode_pool_size=_int_knob(
                "codec_pipeline.decode_pool_size", max_workers, strict=strict
            ),
            # Under `strict`, a size the process cannot give is an error rather than a
            # warning -- the same switch that turns a decline into a raise.
            strict=strict,
            store_is_read_only=store.read_only,
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
        """The shape of the INNERMOST unit the codec chain decodes.

        Three answers, and the difference matters: a tuple is the inner chunk of a sharded
        array; `()` means the array is NOT sharded, so its chunk is its own decode unit and
        only a batch entry knows that shape; `None` means refuse.

        ONE OWNER, and it is Rust. This used to walk zarr's codec objects here -- an
        `isinstance(codec, ShardingCodec)` descent reading `.chunk_shape` and `.codecs` --
        while `ShardInfo::from_codec_chain` answered the same question from the bound codec
        chain. Two derivations of one fact, and when they disagreed Python built a description
        Rust refuses, which surfaces as an uncaught `PyRuntimeError`: the retrieve call is in
        the `else:` branch of `read`, outside its `try`, so it does not fall back. A
        third-party codec registered for `sharding_indexed` reopens that gap however carefully
        this walk is written, because the two are reading different data.

        `self.impl` is `None` when the array was refused at construction, which is the same
        answer for the same reason.
        """
        if self.impl is None:
            return None
        shape = self.impl.inner_chunk_shape
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
            with _READ_LOCK:
                _READ_COUNTS[1] += 1
            if self.python_impl is None:
                raise
            await self.python_impl.read(batch_info, out, drop_axes)
            return None
        else:
            with _READ_LOCK:
                _READ_COUNTS[0] += 1
            out: NDArrayLike = out.as_ndarray_like()
            desc = chunks_desc.chunk_info_with_indices
            # One entry point. `chunk_info_for_read` either produces a handle or raises, and
            # the raise is caught above as a fall back to zarr-python -- there is no second
            # Rust read path to choose between any more.
            retrieve = self.impl.retrieve_chunk_items_and_apply_index
            await asyncio.to_thread(retrieve, desc, out)
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
