from __future__ import annotations

import itertools
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


def _split_array_into_runs(arr: np.ndarray) -> list[tuple[int, int]]:
    """Split a sorted ascending 1-D integer array into contiguous runs.

    Returns a list of (start, stop) half-open ranges in the underlying axis
    such that for each run every integer in [start, stop) is present in
    ``arr``. ``arr`` must be sorted ascending and 1-D; otherwise ValueError
    is raised. Empty arrays produce an empty list.
    """
    if arr.ndim != 1:
        raise ValueError(f"_split_array_into_runs expects a 1-D array, got ndim={arr.ndim}")
    if arr.size == 0:
        return []
    if arr.size == 1:
        v = int(arr[0])
        return [(v, v + 1)]
    diff = np.diff(arr)
    if (diff < 0).any():
        raise ValueError("_split_array_into_runs expects a sorted ascending array")
    boundaries = np.flatnonzero(diff != 1) + 1
    if boundaries.size == 0:
        return [(int(arr[0]), int(arr[-1]) + 1)]
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [arr.size]))
    return [(int(arr[s]), int(arr[e - 1]) + 1) for s, e in zip(starts, ends)]


def _dim_to_slice_runs(dim_sel: Any) -> list[slice]:
    """Convert a single dim selector to a list of contiguous slice runs.

    - slice -> as-is (single run; step != 1 raises DiscontiguousArrayError).
    - integer scalar -> [slice(v, v + 1)].
    - 1-D ndarray length 1 -> single slice.
    - 1-D ndarray contiguous -> single slice covering [arr[0], arr[-1] + 1).
    - 1-D ndarray discontiguous (sorted ascending) -> N slices via
      _split_array_into_runs.
    """
    if is_integer(dim_sel):
        v = int(dim_sel)
        return [slice(v, v + 1)]
    if isinstance(dim_sel, slice):
        if dim_sel.step is not None and dim_sel.step != 1:
            raise DiscontiguousArrayError(
                f"slice with step != 1 is not supported: {dim_sel}"
            )
        return [dim_sel]
    if isinstance(dim_sel, np.ndarray):
        arr = dim_sel.ravel()
        if arr.size == 0:
            return []
        if arr.size == 1:
            v = int(arr[0])
            return [slice(v, v + 1)]
        if (np.diff(arr) < 0).any():
            raise DiscontiguousArrayError(
                "ndarray dim selector is not sorted ascending"
            )
        runs = _split_array_into_runs(arr)
        return [slice(s, e) for (s, e) in runs]
    raise ValueError(f"unsupported dim selector type: {type(dim_sel).__name__}")


def selector_tuple_to_slice_run_tuples(
    selector_tuple: SelectorTuple,
) -> list[tuple[slice, ...]]:
    """Split a per-dim selector tuple into a list of slice-only tuples.

    For each dim, slice/int selectors yield exactly one slice; a contiguous
    integer ndarray yields one slice; a discontiguous integer ndarray
    yields N slices, one per contiguous run. The returned list is the
    cartesian product across dims.
    """
    if isinstance(selector_tuple, slice):
        return [(selector_tuple,)]
    if isinstance(selector_tuple, np.ndarray):
        runs = _dim_to_slice_runs(selector_tuple)
        return [(s,) for s in runs]
    per_dim_runs: list[list[slice]] = [_dim_to_slice_runs(s) for s in selector_tuple]
    return [tuple(combo) for combo in itertools.product(*per_dim_runs)]


# This is a (mostly) copy of the function from zarr.core.indexing that fixes:
#   DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated
# TODO: Upstream this fix
#
# This helper is intentionally distinct from selector_tuple_to_slice_run_tuples:
# the latter splits a discontiguous integer ndarray into multiple slice runs,
# while make_slice_selection keeps the legacy behavior of folding a fully
# duplicated ndarray (e.g. [0, 0, 0] in coordinate indexing) into a single
# length-1 slice. That fold is what allows the legacy single-ChunkItem path
# to express vindex / coordinate indexing without fanout.
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


def _slice_length(s: slice) -> int:
    """Length of a slice with concrete non-negative start/stop and step==1."""
    start = s.start if s.start is not None else 0
    stop = s.stop
    if stop is None:
        raise ValueError(f"slice without concrete stop: {s}")
    return max(0, int(stop) - int(start))


def _split_dim_jointly(
    chunk_dim: Any, out_dim: Any
) -> list[tuple[slice, slice]]:
    """Per-dim joint split: pair each chunk run with the corresponding out run.

    Used when chunk_selection has a discontiguous ndarray and out_selection
    has either a slice (sub-divided into consecutive sub-slices) or an
    ndarray (sliced piecewise; each piece must itself be contiguous).

    Three modes for an integer ndarray chunk_dim:

    - sorted strictly ascending: merge into contiguous runs and pair each
      run with a consecutive out sub-slice. Fewest fragments.
    - has duplicates or is unsorted: fall through to per-element pairs --
      each (chunk_idx, out_idx) is independent on the read path, so a
      length-1 chunk_run paired with the corresponding length-1 out_run
      is always correct. This is what zarr's OrthogonalIndexer produces
      for, e.g., annbatch's sparse-integer row fetch where CSR-derived
      indices arrive in arrival order rather than sorted.
    """
    # Compute chunk runs, tracking each run's length (number of elements
    # selected from the chunk axis).
    if is_integer(chunk_dim):
        v = int(chunk_dim)
        chunk_runs: list[tuple[slice, int]] = [(slice(v, v + 1), 1)]
    elif isinstance(chunk_dim, slice):
        if chunk_dim.step is not None and chunk_dim.step != 1:
            raise DiscontiguousArrayError(
                f"slice with step != 1 is not supported: {chunk_dim}"
            )
        chunk_runs = [(chunk_dim, _slice_length(chunk_dim))]
    elif isinstance(chunk_dim, np.ndarray):
        arr = chunk_dim.ravel()
        if arr.size == 0:
            return []
        if arr.size == 1:
            v = int(arr[0])
            chunk_runs = [(slice(v, v + 1), 1)]
        else:
            diff = np.diff(arr)
            if (diff > 0).all():
                # Sorted strictly ascending: merge into contiguous runs.
                runs = _split_array_into_runs(arr)
                # For a contiguous run [s, e), every integer in that range is
                # in arr, so the count of selected entries equals the run
                # width.
                chunk_runs = [(slice(s, e), e - s) for (s, e) in runs]
            else:
                # Unsorted or has duplicates: emit one length-1 chunk_run per
                # element, in the original order. The out side will be split
                # into matching length-1 pieces below, preserving the
                # element-to-output mapping. Fanout is then arr.size for
                # this dim. The caller does not cap fanout, so callers
                # building selectors that multiply across many dims
                # (orthogonal indexing with several large unsorted ndarrays)
                # are responsible for keeping the resulting ChunkItem count
                # within process memory.
                chunk_runs = [
                    (slice(int(v), int(v) + 1), 1) for v in arr
                ]
    else:
        raise ValueError(f"unsupported chunk_dim type: {type(chunk_dim).__name__}")

    # Now split out_dim to align with the chunk runs.
    if is_integer(out_dim):
        if len(chunk_runs) != 1 or chunk_runs[0][1] != 1:
            raise UnsupportedVIndexingError(
                "integer out_dim cannot be paired with multi-run chunk_dim"
            )
        v = int(out_dim)
        out_pieces: list[slice] = [slice(v, v + 1)]
    elif isinstance(out_dim, slice):
        if out_dim.step is not None and out_dim.step != 1:
            raise DiscontiguousArrayError(
                f"out slice with step != 1 is not supported: {out_dim}"
            )
        cur = out_dim.start if out_dim.start is not None else 0
        out_pieces = []
        for _, length in chunk_runs:
            out_pieces.append(slice(cur, cur + length))
            cur += length
        if out_dim.stop is not None and cur != out_dim.stop:
            # The chunk runs do not cover the full out slice. This should
            # not happen for a well-formed orthogonal/basic indexer pair.
            raise UnsupportedVIndexingError(
                f"chunk runs cover {cur - (out_dim.start or 0)} elements but "
                f"out slice is length {out_dim.stop - (out_dim.start or 0)}"
            )
    elif isinstance(out_dim, np.ndarray):
        arr_out = out_dim.ravel()
        cur = 0
        out_pieces = []
        for _, length in chunk_runs:
            sub = arr_out[cur : cur + length]
            if sub.size == 0:
                raise UnsupportedVIndexingError(
                    "out ndarray exhausted before chunk runs"
                )
            if sub.size == 1:
                v = int(sub[0])
                out_pieces.append(slice(v, v + 1))
            else:
                diff = np.diff(sub)
                if (diff != 1).any():
                    raise DiscontiguousArrayError(
                        "out ndarray sub-piece is not contiguous"
                    )
                out_pieces.append(slice(int(sub[0]), int(sub[-1]) + 1))
            cur += length
        if cur != arr_out.size:
            raise UnsupportedVIndexingError(
                f"out ndarray has {arr_out.size} entries but chunk runs only "
                f"cover {cur}"
            )
    else:
        raise ValueError(f"unsupported out_dim type: {type(out_dim).__name__}")

    return list(zip([c for c, _ in chunk_runs], out_pieces))


def _paired_chunk_and_out_subsets(
    chunk_selection: Any, out_selection: Any
) -> list[tuple[list[slice], list[slice]]]:
    """Jointly fragment chunk_selection and out_selection by contiguous runs.

    Returns a list of (chunk_subset_slices, out_subset_slices) pairs. The
    chunk side has one slice per dim of ``chunk_selection``; the out side
    has one slice per dim of ``out_selection`` (which may have fewer dims
    than chunk_selection when integer-valued chunk dims drop their out
    counterpart, as in zarr-python's OrthogonalIndexer).

    Raises DiscontiguousArrayError if any dim cannot be expressed as
    contiguous runs (e.g. a slice with step != 1). Raises
    UnsupportedVIndexingError if chunk_selection and out_selection
    structures cannot be aligned (e.g. coordinate / mask indexing where
    out_selection is not a tuple).
    """
    if not isinstance(chunk_selection, tuple):
        # Single-dim selectors arrive as a bare slice/ndarray for some
        # indexers; normalize to a 1-tuple.
        chunk_selection = (chunk_selection,)
    if not isinstance(out_selection, tuple):
        # Coordinate / mask indexers pass a single slice or ndarray here.
        # The per-dim chunk slices and the single out slot do not
        # correspond 1:1, so we cannot safely fragment.
        raise UnsupportedVIndexingError(
            "out_selection must be a tuple to support per-dim run splitting"
        )

    per_dim_pairs: list[list[tuple[slice, slice | None]]] = []
    out_idx = 0
    for c_dim in chunk_selection:
        if is_integer(c_dim):
            # Integer chunk dim: no corresponding out dim (zarr-python
            # drops it). Produce a single fragment with a length-1 slice
            # on the chunk side and None marker on the out side.
            v = int(c_dim)
            per_dim_pairs.append([(slice(v, v + 1), None)])
            continue
        if out_idx >= len(out_selection):
            raise UnsupportedVIndexingError(
                "chunk_selection has more non-integer dims than out_selection"
            )
        o_dim = out_selection[out_idx]
        out_idx += 1
        pairs = _split_dim_jointly(c_dim, o_dim)
        if not pairs:
            # Empty selection on this dim -> no fragments for the entry.
            return []
        per_dim_pairs.append(
            [(c_slice, o_slice) for c_slice, o_slice in pairs]
        )
    if out_idx != len(out_selection):
        raise UnsupportedVIndexingError(
            f"out_selection has {len(out_selection)} dims but only {out_idx} "
            "were consumed by non-integer chunk dims"
        )

    fragments: list[tuple[list[slice], list[slice]]] = []
    for combo in itertools.product(*per_dim_pairs):
        chunk_slices = [c for c, _ in combo]
        out_slices = [o for _, o in combo if o is not None]
        fragments.append((chunk_slices, out_slices))
    return fragments


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


def _detect_drop_axes_from_chunk_selection(
    chunk_selection: Any,
    chunk_spec_shape: tuple[int, ...],
    drop_axes: tuple[int, ...],
) -> tuple[int, ...]:
    """Mirror the existing shape-based drop_axes detection without needing
    a single concrete slice version of chunk_selection.

    Replicates the behavior of ``make_chunk_info_for_rust_with_indices``'s
    drop_axes loop using a synthetic slice-version shape where each dim of
    ``chunk_selection`` contributes the count of elements it would select
    (length 1 for ints, slice length for slices, len(arr) for ndarrays).
    """
    if not isinstance(chunk_selection, tuple):
        chunk_selection = (chunk_selection,)
    shape_chunk_selection = get_shape_for_selector(
        chunk_selection, chunk_spec_shape, pad=True, drop_axes=drop_axes
    )
    chunk_selection_padded_shape: list[int] = []
    for idx, sel in enumerate(chunk_selection):
        if is_integer(sel):
            chunk_selection_padded_shape.append(1)
        elif isinstance(sel, slice):
            start, stop, _ = sel.indices(chunk_spec_shape[idx])
            chunk_selection_padded_shape.append(max(0, stop - start))
        elif isinstance(sel, np.ndarray):
            chunk_selection_padded_shape.append(int(sel.size))
        else:
            chunk_selection_padded_shape.append(1)
    scs_iter = iter(shape_chunk_selection)
    scs_current = next(scs_iter, None)
    local_drop_axes = drop_axes
    for idx_shape, shape_chunk_from_slices in enumerate(chunk_selection_padded_shape):
        if shape_chunk_from_slices == 1 and shape_chunk_from_slices != scs_current:
            local_drop_axes = local_drop_axes + (idx_shape,)
        else:
            scs_current = next(scs_iter, None)
    return local_drop_axes


def _can_fragment_jointly(chunk_selection: Any, out_selection: Any) -> bool:
    """Return True iff the selector pair has the per-dim alignment that
    the joint run-splitter expects: both are tuples and chunk_selection
    has at least one discontiguous integer ndarray dim.

    Vectorized / coordinate indexing (where out_selection arrives as a
    single slice or ndarray rather than a tuple) is intentionally routed
    through the legacy single-ChunkItem path -- the per-dim chunk arrays
    are paired coordinates, not independent runs, so a per-dim cartesian
    fragmenting would produce wrong results.
    """
    if not isinstance(chunk_selection, tuple):
        return False
    if not isinstance(out_selection, tuple):
        return False
    for dim in chunk_selection:
        if isinstance(dim, np.ndarray) and dim.size > 1:
            arr = dim.ravel()
            if (np.diff(arr) != 1).any():
                return True
    return False


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
    """Build a single ChunkItem using the pre-Branch-1 logic.

    Used as the fallback path for selectors that the joint splitter
    cannot safely fragment (vindex / coordinate indexing) and for
    selectors that do not actually need fragmenting.
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

    When ``allow_fragmenting`` is True, a discontiguous integer-array
    selection in ``chunk_selection`` is split into multiple ChunkItems --
    one per contiguous run -- all sharing the same ``key``. This is safe
    on the read path because each ChunkItem writes to a disjoint region
    of the output array. It is NOT safe on the write path: multiple
    ChunkItems for the same shard each perform a read-modify-write of
    that shard, which races. Callers on the write path should pass
    ``allow_fragmenting=False`` so a discontiguous integer array still
    raises DiscontiguousArrayError and falls back to the python pipeline.
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

        if not allow_fragmenting or not _can_fragment_jointly(
            chunk_selection, out_selection
        ):
            # Legacy single-ChunkItem path. This handles all cases that
            # the pre-Branch-1 code handled, including vectorized /
            # coordinate indexing where out_selection is a single slice
            # or ndarray. If selector_tuple_to_slice_selection sees a
            # discontiguous integer ndarray here it raises
            # DiscontiguousArrayError, which the pipeline fallback
            # catches.
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

        # Branch-1 fragmenting path: chunk_selection has at least one
        # discontiguous integer ndarray dim and the joint splitter can
        # align it with out_selection.
        drop_axes = _detect_drop_axes_from_chunk_selection(
            chunk_selection, chunk_spec.shape, drop_axes
        )
        fragments = _paired_chunk_and_out_subsets(chunk_selection, out_selection)
        if not fragments:
            continue

        for chunk_subset_as_slices, out_subset_as_slices in fragments:
            shape_chunk_selection_slices = get_shape_for_selector(
                tuple(chunk_subset_as_slices),
                chunk_spec.shape,
                pad=True,
                drop_axes=drop_axes,
            )
            chunk_size = prod_op(shape_chunk_selection_slices)
            if not is_constant and chunk_size > prod_op(shape):
                raise IndexError(
                    f"the size of the chunk subset {shape_chunk_selection_slices} "
                    f"and input/output subset {shape} are incompatible"
                )

            io_array_shape = list(shape)
            out_subset_expanded = list(out_subset_as_slices)
            if drop_axes:
                for axis in drop_axes:
                    io_array_shape.insert(axis, 1)
                    out_subset_expanded.insert(axis, slice(0, 1))

            chunk_info_with_indices.append(
                ChunkItem(
                    key=byte_getter.path,
                    chunk_subset=chunk_subset_as_slices,
                    chunk_shape=chunk_spec.shape,
                    subset=out_subset_expanded,
                    shape=io_array_shape,
                )
            )
    return RustChunkInfo(chunk_info_with_indices, write_empty_chunks)
