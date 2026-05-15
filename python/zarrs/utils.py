from __future__ import annotations

import operator
import os
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
from zarr.core.indexing import is_integer

from zarrs._internal import ChunkItem

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import EllipsisType

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.indexing import SelectorTuple
    from zarr.dtype import ZDType


# adapted from https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
def get_max_threads() -> int:
    return (os.cpu_count() or 1) + 4


class DiscontiguousArrayError(Exception):
    pass


class UnsupportedVIndexingError(Exception):
    pass


class FillValueNoneError(Exception):
    pass


# This is a (mostly) copy of the function from zarr.core.indexing that fixes:
#   DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated
# TODO: Upstream this fix
#
# Used by the legacy single-ChunkItem path for coordinate / mask / vindex
# selectors where chunk_selection arrives as a tuple of paired ndarrays
# rather than per-dim independent selectors. Folds a fully duplicated
# ndarray (e.g. [0, 0, 0]) into a single length-1 slice; raises
# DiscontiguousArrayError otherwise so the pipeline falls back.
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
                diff = np.diff(dim_selection)
                if (diff != 1).any() and (diff != 0).any():
                    raise DiscontiguousArrayError(diff)
                ls.append(slice(dim_selection[0], dim_selection[-1] + 1, 1))
        else:
            ls.append(dim_selection)
    return ls


def selector_tuple_to_slice_selection(selector_tuple: SelectorTuple) -> list[slice]:
    if isinstance(selector_tuple, slice):
        return [selector_tuple]
    if all(isinstance(s, slice) for s in selector_tuple):
        return list(selector_tuple)
    return make_slice_selection(selector_tuple)


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
    chunk_info_with_indices: list[ChunkItem]
    write_empty_chunks: bool


def _is_simple_per_dim(
    chunk_selection: Any, out_selection: Any
) -> bool:
    """Return True iff (chunk_selection, out_selection) align per-dim.

    Per-dim alignment means both are tuples of simple per-dim selectors
    (int, slice, or 1-D ndarray). This is what zarr's OrthogonalIndexer
    produces and is what the Phase 2 native path can pass straight
    through to Rust.

    Coordinate / mask / vindex selectors arrive with out_selection as a
    single slice or ndarray (not a tuple), or with non-simple selectors
    like Ellipsis; those paths still go through the legacy slice-only
    builder which raises DiscontiguousArrayError on non-contiguous
    arrays so the pipeline can fall back to the python pipeline.
    """
    if not isinstance(chunk_selection, tuple):
        return False
    if not isinstance(out_selection, tuple):
        return False
    for d in chunk_selection:
        if is_integer(d):
            continue
        if isinstance(d, slice):
            continue
        if isinstance(d, np.ndarray):
            continue
        return False
    for d in out_selection:
        if is_integer(d):
            continue
        if isinstance(d, slice):
            continue
        if isinstance(d, np.ndarray):
            continue
        return False
    return True


def _convert_per_dim_selector(sel: Any) -> slice | np.ndarray:
    """Normalize a per-dim selector to a slice or 1-D int64 ndarray.

    - integer scalar -> slice(v, v + 1)
    - slice -> slice (step != 1 raises DiscontiguousArrayError)
    - 0-D / size-1 ndarray -> slice(v, v + 1)
    - 1-D ndarray (size > 1) -> 1-D int64 ndarray (passes through to Rust
      as a DimSelector::Indices)

    The Rust ChunkItem constructor accepts slice or numpy.int64 1-D
    array per dim; everything else is rejected here.
    """
    if is_integer(sel):
        v = int(sel)
        return slice(v, v + 1)
    if isinstance(sel, slice):
        if sel.step is not None and sel.step != 1:
            raise DiscontiguousArrayError(
                f"slice with step != 1 is not supported: {sel}"
            )
        return sel
    if isinstance(sel, np.ndarray):
        arr = sel.ravel()
        if arr.size == 1:
            v = int(arr[0])
            return slice(v, v + 1)
        # Empty arrays are short-circuited by the caller; passing one
        # through would build a degenerate ChunkItem.
        return arr.astype(np.int64, copy=False)
    raise ValueError(f"unsupported per-dim selector: {type(sel).__name__}")


def _build_native_chunk_item(
    byte_getter: Any,
    chunk_spec: Any,
    chunk_selection: Any,
    out_selection: Any,
    drop_axes: tuple[int, ...],
    shape: tuple[int, ...],
    *,
    allow_fragmenting: bool,
    is_constant: bool,
) -> ChunkItem | None:
    """Build a single ChunkItem with per-dim slice or ndarray selectors.

    Returns None if any chunk-side ndarray dim is empty (no work to do).
    Raises DiscontiguousArrayError when ``allow_fragmenting`` is False
    and any chunk-side dim is an ndarray; this is the write-path
    fallback contract -- multiple read-modify-write of the same shard
    would race in store_chunks_with_indices.
    """
    # Walk per-dim. For integer scalars the chunk-side selector is a
    # length-1 slice; the dim index is added to the drop-axes set so
    # the corresponding output side gets a length-1 slot inserted at
    # the right position later. Non-integer dims (slices and ndarrays)
    # consume one entry of out_selection each.
    chunk_subset_selectors: list[slice | np.ndarray] = []
    out_subset_partial: list[slice | np.ndarray] = []
    int_scalar_axes: tuple[int, ...] = ()
    out_idx = 0
    for dim_idx, c_dim in enumerate(chunk_selection):
        if is_integer(c_dim):
            v = int(c_dim)
            chunk_subset_selectors.append(slice(v, v + 1))
            int_scalar_axes = int_scalar_axes + (dim_idx,)
            continue

        if out_idx >= len(out_selection):
            raise UnsupportedVIndexingError(
                "chunk_selection has more non-integer dims than out_selection"
            )
        o_dim = out_selection[out_idx]
        out_idx += 1

        c_conv = _convert_per_dim_selector(c_dim)
        if isinstance(c_conv, np.ndarray) and c_conv.size == 0:
            return None
        if isinstance(c_conv, np.ndarray) and not allow_fragmenting:
            raise DiscontiguousArrayError(
                "ndarray chunk-side dim selector requires allow_fragmenting=True; "
                "the write path falls back to the python pipeline because "
                "concurrent read-modify-write of the same shard would race"
            )
        chunk_subset_selectors.append(c_conv)
        out_subset_partial.append(_convert_per_dim_selector(o_dim))

    if out_idx != len(out_selection):
        raise UnsupportedVIndexingError(
            f"out_selection has {len(out_selection)} dims but only {out_idx} "
            "were consumed by non-integer chunk dims"
        )

    # Sanity check: total chunk-side elements must not exceed the
    # output buffer footprint. is_constant signals the broadcast-write
    # path where shape == () and the check does not apply.
    chunk_size = 1
    for dim_idx, sel in enumerate(chunk_subset_selectors):
        if isinstance(sel, slice):
            start, stop, _ = sel.indices(int(chunk_spec.shape[dim_idx]))
            chunk_size *= max(0, stop - start)
        else:
            chunk_size *= int(sel.size)
    if not is_constant and chunk_size > prod_op(shape):
        raise IndexError(
            f"the size of the chunk subset ({chunk_size} elements) and "
            f"input/output subset {shape} are incompatible"
        )

    # Build io_array_shape and out_subset by interleaving length-1
    # slots at every drop-axis position. The drop axes are the union
    # of caller-supplied drop_axes and the integer-scalar chunk dims.
    all_drop_axes = sorted(set(drop_axes) | set(int_scalar_axes))
    io_array_shape = list(shape)
    out_subset_expanded = list(out_subset_partial)
    for axis in all_drop_axes:
        io_array_shape.insert(axis, 1)
        out_subset_expanded.insert(axis, slice(0, 1))

    return ChunkItem(
        key=byte_getter.path,
        chunk_subset=chunk_subset_selectors,
        chunk_shape=chunk_spec.shape,
        subset=out_subset_expanded,
        shape=io_array_shape,
    )


def _emit_legacy_single_chunk_item(
    byte_getter: Any,
    chunk_spec: Any,
    chunk_selection: Any,
    out_selection: Any,
    drop_axes: tuple[int, ...],
    shape: tuple[int, ...],
    *,
    is_constant: bool,
) -> tuple[ChunkItem | None, tuple[int, ...]]:
    """Legacy single-ChunkItem builder used for selectors that the Phase 2
    native path cannot consume per-dim (coordinate / mask / vindex /
    Ellipsis / single bare selector). Raises DiscontiguousArrayError on
    non-contiguous ndarray selections so the pipeline falls back to the
    python pipeline.
    """
    out_selection_as_slices = selector_tuple_to_slice_selection(out_selection)
    chunk_selection_as_slices = selector_tuple_to_slice_selection(chunk_selection)

    shape_chunk_selection_slices = get_shape_for_selector(
        tuple(chunk_selection_as_slices),
        chunk_spec.shape,
        pad=True,
        drop_axes=drop_axes,
    )
    shape_chunk_selection = get_shape_for_selector(
        chunk_selection, chunk_spec.shape, pad=True, drop_axes=drop_axes
    )
    chunk_size = prod_op(shape_chunk_selection)
    if chunk_size != prod_op(shape_chunk_selection_slices):
        raise UnsupportedVIndexingError(
            f"{shape_chunk_selection} != {shape_chunk_selection_slices}"
        )
    if not is_constant and chunk_size > prod_op(shape):
        raise IndexError(
            f"the size of the chunk subset {shape_chunk_selection} and "
            f"input/output subset {shape} are incompatible"
        )

    io_array_shape = list(shape)
    out_selection_expanded = list(out_selection_as_slices)
    scs_iter = iter(shape_chunk_selection)
    scs_current = next(scs_iter, None)
    for idx_shape, shape_chunk_from_slices in enumerate(shape_chunk_selection_slices):
        if shape_chunk_from_slices == 1 and shape_chunk_from_slices != scs_current:
            drop_axes = drop_axes + (idx_shape,)
        else:
            scs_current = next(scs_iter, None)
    if drop_axes:
        for axis in drop_axes:
            io_array_shape.insert(axis, 1)
            out_selection_expanded.insert(axis, slice(0, 1))

    item = ChunkItem(
        key=byte_getter.path,
        chunk_subset=chunk_selection_as_slices,
        chunk_shape=chunk_spec.shape,
        subset=out_selection_expanded,
        shape=io_array_shape,
    )
    return item, drop_axes


def make_chunk_info_for_rust_with_indices(
    batch_info: Iterable[
        tuple[ByteGetter | ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]
    ],
    drop_axes: tuple[int, ...],
    shape: tuple[int, ...],
    *,
    allow_fragmenting: bool = True,
) -> RustChunkInfo:
    """Build ChunkItems for the Rust pipeline.

    For per-dim independent selectors (orthogonal-style indexing where
    both chunk_selection and out_selection are tuples of simple per-dim
    selectors), the Phase 2 native path passes the per-dim selectors --
    including 1-D int ndarrays -- straight through to a single
    ChunkItem. The Rust side then expands per-dim ndarray selectors
    into a multi-region partial_decode call via expand_to_subset_pairs
    and lets the upstream sharding decoder dedup inner-chunk fetches
    across cells.

    For coordinate / mask / vindex / Ellipsis selectors, the legacy
    single-ChunkItem path is used. That path raises
    DiscontiguousArrayError on non-contiguous integer arrays so the
    pipeline can fall back to the python pipeline.

    When ``allow_fragmenting`` is False (the write path), an ndarray
    chunk-side dim selector raises DiscontiguousArrayError so the
    pipeline falls back to the python pipeline. Multiple ChunkItems
    or a single multi-region ChunkItem for the same shard would each
    perform an independent read-modify-write in
    store_chunks_with_indices, which races.
    """
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

        # The Phase 2 native path requires per-dim alignment between
        # chunk_selection and out_selection. Constant writes (shape == ())
        # arrive with out_selection == () regardless of chunk_selection
        # rank, which breaks that alignment, so they always go through
        # the legacy single-ChunkItem path with an empty subset (the
        # broadcast convention the Rust constructor expects).
        if is_constant or not _is_simple_per_dim(chunk_selection, out_selection):
            item, drop_axes = _emit_legacy_single_chunk_item(
                byte_getter,
                chunk_spec,
                chunk_selection,
                out_selection,
                drop_axes,
                shape,
                is_constant=is_constant,
            )
            chunk_info_with_indices.append(item)
            continue

        item = _build_native_chunk_item(
            byte_getter,
            chunk_spec,
            chunk_selection,
            out_selection,
            drop_axes,
            shape,
            allow_fragmenting=allow_fragmenting,
            is_constant=is_constant,
        )
        if item is not None:
            chunk_info_with_indices.append(item)
    return RustChunkInfo(chunk_info_with_indices, write_empty_chunks)
