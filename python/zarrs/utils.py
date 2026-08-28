from __future__ import annotations

import operator
import os
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
from zarr.core.indexing import is_integer

from zarrs._internal import ChunkItem, ChunkItems

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
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
    """Normalise the batch's integer-array indices to int64, lazily."""

    def cast(sel: SelectorTuple) -> SelectorTuple:
        if isinstance(sel, np.ndarray):
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
            if len(dim_selection) == 1:
                ls.append(
                    slice(int(dim_selection.item()), int(dim_selection.item()) + 1, 1)
                )
            else:
                # int64 (see `_as_int64`): an unsigned diff wraps a decrease into +1.
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
    """Both selections as tuples, so an axis can be addressed by position.

    zarr hands a bare slice or array for a one-axis selection and a tuple for anything else.
    The two eligibility tests below both need positional access, and both must agree about
    which axis is which, so they ask this once rather than each normalising for itself.
    """
    return (
        chunk_selection if isinstance(chunk_selection, tuple) else (chunk_selection,),
        out_selection if isinstance(out_selection, tuple) else (out_selection,),
    )


def _is_sorted_integer_axis(indices: Any, out_axis_sel: Any) -> bool:
    """Is this one sorted 1-D integer axis written to a contiguous output slice?

    The shared half of the two eligibility tests below, which ask the same question in two
    styles. Deliberately does NOT judge negative indices or the output length: on a negative
    index `split_selection_runs` raises and `_chunk_unit_args` declines, and the raise comes
    before the length check, so the order of those two decisions is part of each caller's
    behaviour. They stay at the call sites for that reason.
    """
    return (
        isinstance(indices, np.ndarray)
        and indices.ndim == 1
        # Non-decreasing only: any decrease would mean one box per element, a decode each.
        and not (indices[1:] < indices[:-1]).any()
        and isinstance(out_axis_sel, slice)
        and out_axis_sel.step in (None, 1)
    )


def split_selection_runs(
    chunk_selection: SelectorTuple, out_selection: SelectorTuple
) -> Iterator[tuple[SelectorTuple, SelectorTuple]]:
    """Split a selection with one non-consecutive integer-array axis into contiguous boxes.

    zarrs describes a chunk read as a rectangular subset, so ``z[[3, 7, 8], :]`` has no
    single-box description -- but it is a *stack* of boxes, one per run of consecutive
    indices. Only one array axis is split: with two, outer and coordinate indexing disagree
    on what the selection means. Anything not splittable is yielded unchanged.
    """
    chunk_sel, out_sel = _as_selector_tuples(chunk_selection, out_selection)
    unsplit = ((chunk_selection, out_selection),)

    array_axes = [
        axis for axis, sel in enumerate(chunk_sel) if isinstance(sel, np.ndarray)
    ]
    # Equal arity means no axis was dropped, so chunk axis `axis` is output axis `axis`.
    if len(array_axes) != 1 or len(chunk_sel) != len(out_sel):
        yield from unsplit
        return
    (axis,) = array_axes
    indices = chunk_sel[axis]
    out_axis_sel = out_sel[axis]
    if not _is_sorted_integer_axis(indices, out_axis_sel) or not all(
        isinstance(sel, slice) for sel in out_sel
    ):
        yield from unsplit
        return
    # this line can be removed once https://github.com/zarr-developers/zarr-python/issues/4285 is fixed
    if (indices < 0).any():
        raise DiscontiguousArrayError(indices)
    out_start = out_axis_sel.start or 0
    if out_axis_sel.stop - out_start != indices.size:
        yield from unsplit
        return

    # Always slices, even for one run: `resulting_shape_from_index` mis-orders a non-leading
    # advanced index, and the caller's element-count check then rejects the selection.
    boundaries = np.flatnonzero(indices[1:] != indices[:-1] + 1) + 1

    for start, stop in zip(
        np.concatenate(([0], boundaries)),
        np.concatenate((boundaries, [indices.size])),
        strict=True,
    ):
        rows = indices[start:stop]
        box_chunk_sel = list(chunk_sel)
        box_chunk_sel[axis] = slice(int(rows[0]), int(rows[-1]) + 1)
        box_out_sel = list(out_sel)
        box_out_sel[axis] = slice(out_start + int(start), out_start + int(stop))
        yield tuple(box_chunk_sel), tuple(box_out_sel)


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
    # A ChunkItems handle when the batch is entirely chunk-unit; a list otherwise. The
    # pipeline dispatches on which, because the two take different Rust entry points.
    chunk_info_with_indices: list[ChunkItem] | ChunkItems
    write_empty_chunks: bool


def _chunk_unit_args(
    entry, shape: tuple[int, ...], drop_axes: tuple[int, ...], inner_shape
) -> tuple | None:
    """Arguments for one item per inner chunk, or None if this entry is not that shape.

    The inner chunk is what the codec chain fetches and decodes; `chunk_spec.shape` is
    the SHARD. Grouping a selection by decode unit therefore needs the inner chunk
    shape, which the pipeline reads off the array metadata and passes in -- it was
    never unreachable, only never passed.

    Each group becomes a whole-inner-chunk subset plus the indices wanted from it, so
    the chunk is read once, decoded once and gathered once, however many of its
    elements the selection asks for. That is what zarr-python does, and it is why a
    run of consecutive indices and a strided scatter cost it the same.

    The checks here are vectorised numpy and cost nothing, so they stay in Python and
    hold the semantics; Rust does the grouping and the item construction, taking
    `indices` as a view. Returning the arguments rather than the items lets the caller
    choose the container: a `ChunkItems` handle when the whole batch is chunk-unit, a
    Python list when it has to meet entries that took another path.

    Narrow on purpose: one 1-D integer axis, non-negative and NON-DECREASING, with a
    contiguous output slice. Sorted is what makes it work -- each chunk's elements are
    then one contiguous run of the output, the only output shape a ChunkItem can
    describe.
    """
    byte_getter, chunk_spec, chunk_selection, out_selection, _ = entry
    if drop_axes or inner_shape is None or len(inner_shape) != 1:
        return None
    chunk_sel, out_sel = _as_selector_tuples(chunk_selection, out_selection)
    if len(chunk_sel) != 1 or len(out_sel) != 1 or len(chunk_spec.shape) != 1:
        return None
    (indices,) = chunk_sel
    (out_axis_sel,) = out_sel
    if not _is_sorted_integer_axis(indices, out_axis_sel) or indices.size == 0:
        return None
    indices = indices.astype(np.int64, copy=False)
    if (indices < 0).any():
        return None
    start = out_axis_sel.start or 0
    if out_axis_sel.stop - start != indices.size:
        return None

    return (
        byte_getter.path,
        chunk_spec.shape,
        shape,
        indices,
        start,
        int(inner_shape[0]),
    )


def chunk_info_for_write(
    batch_info: BatchInfo,
    drop_axes: tuple[int, ...],
    shape: tuple[int, ...],
) -> RustChunkInfo:
    """Describe a write batch to Rust, one item per entry.

    Neither the chunk-unit grouping nor the run splitting a READ gets: both can put two
    items on one chunk key, and the write path is a read-modify-write, so two items on one
    key race. A write therefore describes exactly what it was given.
    """
    return _chunk_items(_as_int64_batch_info(batch_info), drop_axes, shape)


def chunk_info_for_read(
    batch_info: BatchInfo,
    drop_axes: tuple[int, ...],
    shape: tuple[int, ...],
    inner_chunk_shape: tuple[int, ...] | None,
) -> RustChunkInfo:
    """Describe a read batch to Rust, grouped by decode unit where the selection allows.

    Tried in order: one item per inner chunk for the whole batch, which is the cheapest
    shape and the only one the concurrent read path can take; then one box per run of
    consecutive indices; then one item per entry, as a write does.
    """
    # A generator would be consumed by the eligibility test, and the ordinary route needs
    # to read the same entries again if that test fails.
    entries = list(_as_int64_batch_info(batch_info))

    # All or nothing. Eligibility turns almost entirely on per-array facts -- one 1-D axis,
    # no dropped axes, a known inner chunk shape -- so the entries of a batch come out
    # uniform in practice. Rather than carry a path for mixing grouped and ungrouped items,
    # which nothing has been able to produce, one ineligible entry sends the whole batch on:
    # slower for that read, and one less untested branch.
    unit_args = [
        _chunk_unit_args(entry, shape, drop_axes, inner_chunk_shape)
        for entry in entries
    ]
    if unit_args and all(args is not None for args in unit_args):
        # The whole batch crosses as one object; no ChunkItem is made in Python.
        handle = ChunkItems()
        for args in unit_args:
            handle.push_entry(*args)
        return RustChunkInfo(handle, write_empty_chunks=True)

    return _chunk_items(
        [
            (byte_getter, chunk_spec, box_chunk_sel, box_out_sel, is_complete)
            for (
                byte_getter,
                chunk_spec,
                chunk_selection,
                out_selection,
                is_complete,
            ) in entries
            for box_chunk_sel, box_out_sel in split_selection_runs(
                chunk_selection, out_selection
            )
        ],
        drop_axes,
        shape,
    )


def _chunk_items(
    batch_info: BatchInfo,
    drop_axes: tuple[int, ...],
    shape: tuple[int, ...],
) -> RustChunkInfo:
    """One ChunkItem per batch entry, the description both paths end at."""
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
