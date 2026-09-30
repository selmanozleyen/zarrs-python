from __future__ import annotations

import itertools
import operator
import os
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
from zarr.core.indexing import is_integer

from zarrs._internal import ChunkItem, ChunkItems

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import EllipsisType

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.indexing import SelectorTuple
    from zarr.dtype import ZDType

    BatchInfo = Iterable[
        tuple[ByteGetter | ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]
    ]


# adapted from https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
def get_max_threads() -> int:
    return (os.cpu_count() or 1) + 4


class DiscontiguousArrayError(Exception):
    pass


class UnsupportedVIndexingError(Exception):
    pass


class FillValueNoneError(Exception):
    pass


def _as_int64_batch_info(batch_info: BatchInfo) -> BatchInfo:
    """Normalise the batch's index arrays to int64 positions, lazily."""

    def cast(sel: SelectorTuple) -> SelectorTuple:
        if isinstance(sel, np.ndarray):
            if sel.dtype.kind == "b":
                return np.flatnonzero(sel).astype(np.int64, copy=False)
            if sel.dtype.kind not in "iuf":
                raise DiscontiguousArrayError(sel.dtype)
            # uint64 indices arrive as float64.
            if sel.dtype.kind == "f" and not (
                np.isfinite(sel).all()
                and (sel == np.rint(sel)).all()
                and (np.abs(sel) < 2.0**63).all()
            ):
                raise DiscontiguousArrayError(sel.dtype)
            return sel.astype(np.int64, copy=False)
        if isinstance(sel, tuple) and any(isinstance(s, np.ndarray) for s in sel):
            return tuple(map(cast, sel))
        return sel

    return (
        (byte_getter, chunk_spec, cast(chunk_sel), cast(out_sel), is_complete)
        for byte_getter, chunk_spec, chunk_sel, out_sel, is_complete in batch_info
    )


# This is a (mostly) copy of the function from zarr.core.indexing that fixes:
#   DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated
# TODO: Upstream this fix
def make_slice_selection(selection: tuple[np.ndarray | float]) -> list[slice]:
    ls: list[slice] = []
    for dim_selection in selection:
        if is_integer(dim_selection):
            ls.append(slice(int(dim_selection), int(dim_selection) + 1, 1))
        elif isinstance(dim_selection, np.ndarray):
            dim_selection = dim_selection.ravel()
            if len(dim_selection) == 0:
                raise DiscontiguousArrayError(dim_selection)
            if len(dim_selection) == 1:
                ls.append(
                    slice(int(dim_selection.item()), int(dim_selection.item()) + 1, 1)
                )
            else:
                # Signed, or an unsigned decrease wraps to a step of +1.
                steps = dim_selection[1:] - dim_selection[:-1]
                if (steps != 1).any() and (steps != 0).any():
                    raise DiscontiguousArrayError(steps)
                ls.append(slice(int(dim_selection[0]), int(dim_selection[-1]) + 1, 1))
        else:
            ls.append(dim_selection)
    return ls


def selector_tuple_to_slice_selection(selector_tuple: SelectorTuple) -> list[slice]:
    if isinstance(selector_tuple, slice):
        return [selector_tuple]
    if all(isinstance(s, slice) for s in selector_tuple):
        return list(selector_tuple)
    return make_slice_selection(selector_tuple)


def _as_selector_tuples(
    chunk_selection: SelectorTuple, out_selection: SelectorTuple
) -> tuple[tuple, tuple]:
    return (
        chunk_selection if isinstance(chunk_selection, tuple) else (chunk_selection,),
        out_selection if isinstance(out_selection, tuple) else (out_selection,),
    )


def _is_sorted_integer_axis(indices: Any, out_axis_sel: Any) -> bool:
    """Is this one non-decreasing 1-D integer axis written to a contiguous output slice?"""
    return (
        isinstance(indices, np.ndarray)
        and indices.ndim == 1
        and not (indices[1:] < indices[:-1]).any()
        and isinstance(out_axis_sel, slice)
        and out_axis_sel.step in (None, 1)
    )


def _output_run_matches(indices: np.ndarray, out_axis_sel: slice) -> bool:
    start = out_axis_sel.start or 0
    return out_axis_sel.stop - start == indices.size


def resulting_shape_from_index(
    array_shape: tuple[int, ...],
    index_tuple: tuple[int | slice | EllipsisType | np.ndarray],
    drop_axes: tuple[int, ...],
    *,
    pad: bool,
) -> tuple[int, ...]:
    result_shape = []
    advanced_index_shapes = [
        idx.shape for idx in index_tuple if isinstance(idx, np.ndarray)
    ]
    basic_shape_index = 0

    # Broadcast all advanced indices, if any
    if advanced_index_shapes:
        result_shape += np.broadcast_shapes(*advanced_index_shapes)
        # Consume dimensions from array_shape
        basic_shape_index += len(advanced_index_shapes)

    # Process each remaining index in index_tuple
    for idx in index_tuple:
        if isinstance(idx, int):
            # Integer index reduces dimension, so skip this dimension in array_shape
            basic_shape_index += 1
        elif isinstance(idx, slice):
            if idx.step is not None and idx.step > 1:
                raise DiscontiguousArrayError(
                    "Step size greater than 1 is not supported"
                )
            # Slice keeps dimension, adjust size accordingly
            start, stop, _ = idx.indices(array_shape[basic_shape_index])
            result_shape.append(stop - start)
            basic_shape_index += 1
        elif idx is Ellipsis:
            # Calculate number of dimensions that Ellipsis should fill
            num_to_fill = len(array_shape) - len(index_tuple) + 1
            result_shape += array_shape[
                basic_shape_index : basic_shape_index + num_to_fill
            ]
            basic_shape_index += num_to_fill
        elif not isinstance(idx, np.ndarray):
            raise ValueError(f"Invalid index type: {type(idx)}")

    # Step 4: Append remaining dimensions from array_shape if fewer indices were used
    if basic_shape_index < len(array_shape) and pad:
        result_shape += array_shape[basic_shape_index:]

    return tuple(size for idx, size in enumerate(result_shape) if idx not in drop_axes)


def prod_op(x: Iterable[int]) -> int:
    return reduce(operator.mul, x, 1)


def get_shape_for_selector(
    selector_tuple: SelectorTuple,
    shape: tuple[int, ...],
    *,
    pad: bool,
    drop_axes: tuple[int, ...] = (),
) -> tuple[int, ...]:
    if isinstance(selector_tuple, slice | np.ndarray):
        return resulting_shape_from_index(
            shape,
            (selector_tuple,),
            drop_axes,
            pad=pad,
        )
    return resulting_shape_from_index(shape, selector_tuple, drop_axes, pad=pad)


def get_implicit_fill_value(dtype: ZDType, fill_value: Any) -> Any:
    if fill_value is None:
        fill_value = dtype.default_scalar()
    return fill_value


@dataclass(frozen=True)
class RustChunkInfo:
    # A ChunkItems handle for a read, a list for a write.
    chunk_info_with_indices: list[ChunkItem] | ChunkItems
    write_empty_chunks: bool


def _step1_span(sel: Any, extent: int) -> tuple[int, int] | None:
    """A step-1 slice as (start, stop) within `extent`, or None if it is not one."""
    if not isinstance(sel, slice) or sel.step not in (None, 1):
        return None
    lo = sel.start or 0
    hi = extent if sel.stop is None else sel.stop
    if not 0 <= lo < hi <= extent:
        return None
    return lo, hi


def _is_one_run(widths: list[int], extents: tuple[int, ...]) -> bool:
    """Is a sub-box one run in row-major order: every axis before the last partial one takes one element?"""
    partial = [axis for axis, (w, e) in enumerate(zip(widths, extents, strict=True)) if w != e]
    return not partial or all(w == 1 for w in widths[: partial[-1]])


def _bands(lo: int, hi: int, inner: int, out_lo: int) -> list[tuple[int, int, int]]:
    """(chunk_start, width, out_start) per inner chunk the range `lo:hi` crosses."""
    out, at = [], lo
    while at < hi:
        end = min((at // inner + 1) * inner, hi)
        out.append((at, end - at, out_lo + (at - lo)))
        at = end
    return out


# Below this mean run length, one span per run costs more than naming the elements.
_MIN_MEAN_SPAN = 64


def _index_runs(indices: np.ndarray) -> list[tuple[int, int]] | None:
    """(first, stop) positions of each step-1 run in `indices`, or None if the runs are short."""
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    if (breaks.size + 1) * _MIN_MEAN_SPAN > indices.size:
        return None
    edges = [0, *breaks.tolist(), indices.size]
    return list(zip(edges[:-1], edges[1:]))


def _chunk_unit_args(
    entry, shape: tuple[int, ...], drop_axes: tuple[int, ...], inner_shape
) -> list[tuple] | None:
    """`ChunkItems` push arguments for one entry, or None if it is not served.

    Served: sorted non-negative indices or a step-1 slice on axis 0, and a step-1 slice on every
    other axis that is one run per row. A slice crossing inner chunks on a trailing axis gives
    one item per band.
    """
    byte_getter, chunk_spec, chunk_selection, out_selection, _ = entry
    if drop_axes or inner_shape is None:
        return None
    if any(int(v) <= 0 for v in inner_shape):
        return None
    chunk_sel_raw, out_sel_raw = _as_selector_tuples(chunk_selection, out_selection)
    # zarr drops a scalar axis from the output without saying so in `drop_axes`; put it back
    # as an extent of one.
    scalars: dict[int, int] = {}
    for axis, sel in enumerate(chunk_sel_raw):
        if isinstance(sel, (int, np.integer)):
            scalars[axis] = int(sel)
        elif (
            axis > 0
            and isinstance(sel, np.ndarray)
            and sel.ndim == 1
            and sel.size > 0
            and np.issubdtype(sel.dtype, np.integer)
            and bool((sel == sel[0]).all())
        ):
            scalars[axis] = int(sel[0])
    # An equality, so a constant array on an axis the output kept is not rebuilt.
    if (
        scalars
        and len(chunk_sel_raw) == len(chunk_spec.shape)
        and len(out_sel_raw) == len(shape) == len(chunk_spec.shape) - len(scalars)
    ):
        kept_out = iter(out_sel_raw)
        kept_extent = iter(shape)
        rebuilt = [
            (slice(scalars[axis], scalars[axis] + 1), slice(0, 1), 1)
            if axis in scalars
            else (sel, next(kept_out), next(kept_extent))
            for axis, sel in enumerate(chunk_sel_raw)
        ]
        chunk_selection = tuple(r[0] for r in rebuilt)
        out_selection = tuple(r[1] for r in rebuilt)
        shape = tuple(r[2] for r in rebuilt)
    # Not sharded: the chunk is the decode unit.
    if inner_shape == ():
        inner_shape = tuple(int(s) for s in chunk_spec.shape)
    chunk_sel, out_sel = _as_selector_tuples(chunk_selection, out_selection)
    rank = len(chunk_spec.shape)
    if not (rank == len(chunk_sel) == len(out_sel) == len(inner_shape) == len(shape)):
        return None
    widths: list[int] = []
    # The bands per trailing axis; the items are their product.
    lanes: list[list[tuple[int, int, int]]] = []
    for axis in range(1, rank):
        span = _step1_span(chunk_sel[axis], chunk_spec.shape[axis])
        if span is None:
            return None
        lo, hi = span
        out_span = _step1_span(out_sel[axis], shape[axis])
        if out_span is None or out_span[1] - out_span[0] != hi - lo:
            return None
        widths.append(int(hi - lo))
        lanes.append(_bands(int(lo), int(hi), int(inner_shape[axis]), int(out_span[0])))
    indices = chunk_sel[0]
    out_axis_sel = out_sel[0]
    whole = all(
        int(inner_shape[axis]) == int(chunk_spec.shape[axis])
        and int(widths[axis - 1]) == int(shape[axis]) == int(chunk_spec.shape[axis])
        for axis in range(1, rank)
    )
    if isinstance(indices, slice):
        span = _step1_span(indices, chunk_spec.shape[0])
        if span is None:
            return None
        # A span needs whole trailing axes; otherwise name the elements.
        if whole:
            out_span = _step1_span(out_axis_sel, shape[0])
            count = span[1] - span[0]
            if (
                out_span is not None
                and out_span[1] - out_span[0] == count
                and count > 0
            ):
                return [
                    (
                        "span",
                        byte_getter.path,
                        chunk_spec.shape,
                        shape,
                        int(span[0]),
                        int(count),
                        int(out_span[0]),
                        int(inner_shape[0]),
                    )
                ]
        indices = np.arange(span[0], span[1], dtype=np.int64)
    if not _is_sorted_integer_axis(indices, out_axis_sel) or indices.size == 0:
        return None
    indices = indices.astype(np.int64, copy=False)
    if (indices < 0).any():
        return None
    start = out_axis_sel.start or 0
    if not _output_run_matches(indices, out_axis_sel):
        return None
    # Long runs of consecutive indices, as a CSR row read gives, go as spans.
    if whole and (runs := _index_runs(indices)) is not None:
        return [
            (
                "span",
                byte_getter.path,
                chunk_spec.shape,
                shape,
                int(indices[a]),
                int(b - a),
                int(start + a),
                int(inner_shape[0]),
            )
            for a, b in runs
        ]

    pushes = []
    for combo in itertools.product(*lanes):
        band_starts = [b[0] for b in combo]
        band_widths = [b[1] for b in combo]
        band_out = [b[2] for b in combo]
        # One run per row in the output and in the inner chunk.
        if not _is_one_run(band_widths, tuple(shape[1:])) or not _is_one_run(
            band_widths, tuple(inner_shape[1:])
        ):
            return None
        pushes.append(
            (
                "entry",
                byte_getter.path,
                chunk_spec.shape,
                shape,
                indices,
                (int(start), *band_out),
                (int(shape[0]), *band_widths),
                tuple(int(v) for v in inner_shape),
                tuple(int(v) for v in band_starts),
            )
        )
    return pushes


def chunk_info_for_read(
    batch_info: BatchInfo,
    drop_axes: tuple[int, ...],
    shape: tuple[int, ...],
    inner_chunk_shape: tuple[int, ...] | None,
) -> RustChunkInfo:
    """A `ChunkItems` handle for a read batch, or `DiscontiguousArrayError` if one entry is not served."""
    entries = list(_as_int64_batch_info(batch_info))
    unit_args = [
        _chunk_unit_args(entry, shape, drop_axes, inner_chunk_shape)
        for entry in entries
    ]
    if unit_args and all(args is not None for args in unit_args):
        handle = ChunkItems()
        for kind, *args in itertools.chain.from_iterable(unit_args):
            if kind == "span":
                handle.push_span(*args)
            else:
                handle.push_entry(*args)
        return RustChunkInfo(handle, write_empty_chunks=True)

    raise DiscontiguousArrayError("this selection is not served by the chunk-unit path")


def make_chunk_info_for_rust_with_indices(
    batch_info: Iterable[
        tuple[ByteGetter | ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]
    ],
    drop_axes: tuple[int, ...],
    shape: tuple[int, ...],
) -> RustChunkInfo:
    batch_info = _as_int64_batch_info(batch_info)
    is_constant = shape == ()
    chunk_info_with_indices: list[ChunkItem] = []
    write_empty_chunks: bool = True
    for (
        byte_getter,
        chunk_spec,
        chunk_selection,
        out_selection,
        _,
    ) in batch_info:
        write_empty_chunks = chunk_spec.config.write_empty_chunks
        # Convert the selector tuples to ones that only have slices i.e., `i: int` replaced by slice(i, i+1)
        out_selection_as_slices = selector_tuple_to_slice_selection(out_selection)
        chunk_selection_as_slices = selector_tuple_to_slice_selection(chunk_selection)
        # Because `chunk_selection_as_slices` contains only slices, certain types of vindex-ing are not going to be able to be processed by the zarrs pipeline.
        # Thus we get the shapes of the input selector and the the converted-to-slices selector to check if they differ.
        # If they differ, then the indexing operation is not supported because it is not describe-able as slices.
        shape_chunk_selection_slices = get_shape_for_selector(
            tuple(chunk_selection_as_slices),
            chunk_spec.shape,
            pad=True,
            drop_axes=drop_axes,
        )
        shape_chunk_selection = get_shape_for_selector(
            chunk_selection, chunk_spec.shape, pad=True, drop_axes=drop_axes
        )
        if (chunk_size := prod_op(shape_chunk_selection)) != prod_op(
            shape_chunk_selection_slices
        ):
            raise UnsupportedVIndexingError(
                f"{shape_chunk_selection} != {shape_chunk_selection_slices}"
            )
        if not is_constant and chunk_size > prod_op(shape):
            raise IndexError(
                f"the size of the chunk subset {shape_chunk_selection} and input/output subset {shape} are incompatible"
            )
        io_array_shape = list(shape)
        out_selection_expanded = out_selection_as_slices
        # We need to have io_array_shape and out_selection_expanded with dimensionalities matching that of the underlying array.
        # `drop_axes`` is only triggered via fancy outer-indexing because applying `chunk_selection_as_slices` to the chunk array would not drop a dimension that the out-array thinks should be dropped, thus that dimension needs to be indicated.
        # However, other indexing operations can silently drop a dimension on input to match the output, like `z[1, ...]`.
        # In other words, applying the `chunk_selection_as_slices` to a chunk array would drop a dimension, but `out_selection` already encodes this dropped dimension because zarr-python constructs the out-array missing the dimension.
        # So if we detect that a dimension has been dropped silently like this after converting to slices, we update to handle the dropped dimension.
        scs_iter = iter(shape_chunk_selection)
        scs_current = next(scs_iter, None)
        for idx_shape, shape_chunk_from_slices in enumerate(
            shape_chunk_selection_slices
        ):
            # Detect if this dimension has been dropped on the io_array i.e., shape_chunk_selection has been exhausted so there is an extra 1-sized dimension at the end or has a mismatch with the "full" chunk shape `shape_chunk_selection_slices`.
            if shape_chunk_from_slices == 1 != scs_current:
                drop_axes += (idx_shape,)
            else:
                scs_current = next(scs_iter, None)
        if drop_axes:
            for axis in drop_axes:
                io_array_shape.insert(axis, 1)
                out_selection_expanded.insert(axis, slice(0, 1))
        chunk_info_with_indices.append(
            ChunkItem(
                key=byte_getter.path,
                chunk_subset=chunk_selection_as_slices,
                chunk_shape=chunk_spec.shape,
                subset=out_selection_expanded,
                shape=io_array_shape,
            )
        )
    return RustChunkInfo(chunk_info_with_indices, write_empty_chunks)
