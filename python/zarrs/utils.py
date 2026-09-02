from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass
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
    """Normalise the batch's array indices to int64 positions, lazily."""

    def cast(sel: SelectorTuple) -> SelectorTuple:
        if isinstance(sel, np.ndarray):
            # A boolean mask is not an index array; its positions are what it means.
            if sel.dtype.kind == "b":
                return np.flatnonzero(sel).astype(np.int64, copy=False)
            if sel.dtype.kind not in "iuf":
                raise DiscontiguousArrayError(sel.dtype)
            # THE ONE CAST: everything downstream assumes int64 positions. Float is accepted
            # only because uint64 arrives as float64 -- zarr subtracts an `intp` offset and
            # NEP 50 promotes. Checked BEFORE casting: `astype` truncates 3.7 in silence, and
            # comparing after casts `2.0**63` to `i64::MAX`, which compares equal.
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


# Modelled on `zarr.core.indexing.make_slice_selection`, and deliberately NOT replaceable by
# it. Upstream raises `ArrayIndexError` for any index array of more than one element; this
# turns a consecutive run into the slice it is, which every multi-row read and write here
# depends on. Importing upstream's would turn those into errors, so this is not a copy waiting
# to be deleted once a fix lands upstream -- it is a different function that happens to share
# a name and the single-element cases. It also avoids upstream's
# `DeprecationWarning: Conversion of an array with ndim > 0 to a scalar`.
def make_slice_selection(selection: tuple[np.ndarray | float]) -> list[slice]:
    ls: list[slice] = []
    for dim_selection in selection:
        if is_integer(dim_selection):
            ls.append(slice(int(dim_selection), int(dim_selection) + 1, 1))
        elif isinstance(dim_selection, np.ndarray):
            dim_selection = dim_selection.ravel()
            if len(dim_selection) == 0:
                # `dim_selection[0]` is an `IndexError` here, and `IndexError` is not in
                # `FALLBACK_TO_ZARR_PYTHON` -- it would escape `read` rather than decline.
                raise DiscontiguousArrayError(dim_selection)
            if len(dim_selection) == 1:
                ls.append(
                    slice(int(dim_selection.item()), int(dim_selection.item()) + 1, 1)
                )
            else:
                # Callers must normalise to int64 first: an unsigned diff wraps a decrease into +1.
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
    """Both selections as tuples."""
    return (
        chunk_selection if isinstance(chunk_selection, tuple) else (chunk_selection,),
        out_selection if isinstance(out_selection, tuple) else (out_selection,),
    )


def _is_sorted_integer_axis(indices: Any, out_axis_sel: Any) -> bool:
    """Is this one sorted 1-D integer axis written to a contiguous output slice?"""
    return (
        isinstance(indices, np.ndarray)
        and indices.ndim == 1
        # Non-decreasing only. When zarr DOES reorder for an unsorted selection the
        # out-selection is an ndarray, which the `isinstance(..., slice)` clause below rejects
        # first; what reaches this test is `CoordinateIndexer` with `sel_sort is None`, which
        # hands over a contiguous slice whose indices descend. `out_start + i` would then put
        # each element at the wrong output position.
        and not (indices[1:] < indices[:-1]).any()
        and isinstance(out_axis_sel, slice)
        and out_axis_sel.step in (None, 1)
    )


def _output_run_matches(indices: np.ndarray, out_axis_sel: slice) -> bool:
    """Does the output slice hold exactly one element per index."""
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
    return math.prod(x)


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
    # A ChunkItems handle when the batch is entirely chunk-unit; a list otherwise.
    chunk_info_with_indices: list[ChunkItem] | ChunkItems
    write_empty_chunks: bool


def _is_whole_axis(sel: Any, extent: int) -> bool:
    """Does this selector take the axis whole, start to finish, step 1?"""
    return (
        isinstance(sel, slice)
        and sel.step in (None, 1)
        and (sel.start or 0) == 0
        and sel.stop in (None, extent)
    )


def _step1_span(sel: Any, extent: int) -> tuple[int, int] | None:
    """A step-1 slice as (start, stop) within `extent`, or None if it is not one.

    Rejects rather than clamps: a stop past the extent means the caller and this path disagree
    about the array, and guessing which is right is how wrong data gets returned.
    """
    if not isinstance(sel, slice) or sel.step not in (None, 1):
        return None
    lo = sel.start or 0
    hi = extent if sel.stop is None else sel.stop
    if not 0 <= lo < hi <= extent:
        return None
    return lo, hi


def _contiguous_offset(
    starts: list[int], widths: list[int], extents: tuple[int, ...]
) -> int | None:
    """Element offset of a sub-box within ONE row, or None if that box is not contiguous.

    Row-major, the box is one unbroken range exactly when every axis before the last partial
    one selects a single element. Give a partial axis a wider axis ahead of it and the box
    takes `widths[k]` elements, skips the rest of that axis, and takes them again -- strided,
    and an item's output is vended as ONE range, which cannot express that.

    So `X[rows, a:b]` on a 2-D array is always contiguous (there is nothing before axis 1),
    which is the case this exists for; `X[rows, :, a:b]` on a rank-3 array is not.
    """
    last_partial = -1
    for axis, (width, extent) in enumerate(zip(widths, extents, strict=True)):
        if width != extent:
            last_partial = axis
    if last_partial > 0 and any(width != 1 for width in widths[:last_partial]):
        return None
    offset = 0
    stride = 1
    for axis in reversed(range(len(widths))):
        offset += starts[axis] * stride
        stride *= extents[axis]
    return offset


def _bands(lo: int, hi: int, inner: int, out_lo: int) -> list[tuple[int, int, int]]:
    """(chunk_start, width, out_start) per inner chunk the range `lo:hi` crosses.

    A shard may hold several inner chunks across a trailing axis, and the inner chunk is the
    decode unit -- so a selection straddling a boundary is not one read of a wide row, it is
    one read per inner chunk. Splitting here rather than in Rust is deliberate: it is what
    both previous attempts got wrong, and it is testable without a build.

    Always advances -- `(at // inner + 1) * inner > at` for any positive `inner` -- so it
    cannot loop. `inner <= 0` is refused by the caller before it gets here.
    """
    out, at = [], lo
    while at < hi:
        end = min((at // inner + 1) * inner, hi)
        out.append((at, end - at, out_lo + (at - lo)))
        at = end
    return out


def _chunk_unit_args(
    entry, shape: tuple[int, ...], drop_axes: tuple[int, ...], inner_shape
) -> list[tuple] | None:
    """Args for `ChunkItems.push_entry`, one per item, or None if this entry is not that shape.

    Eligible: an integer axis at AXIS 0 -- non-negative, non-decreasing, against a contiguous
    output slice -- with every axis after it taken whole or as a contiguous sub-box.

    A LIST, because one entry can describe several items. The inner chunk is the decode unit,
    so a trailing selection crossing an inner-chunk boundary is not one wide read: it is one
    read per inner chunk, and the items are the product of the per-axis bands. A shard holding
    one inner chunk on each trailing axis yields exactly one item, which is every geometry this
    path served before the split.

    What a band may not be is strided within one index, since an item's output is vended as a
    single range -- checked per band, on both the chunk and the output side.

    `chunk_spec.shape` is the SHARD, so `inner_shape` is passed in separately.
    """
    byte_getter, chunk_spec, chunk_selection, out_selection, _ = entry
    if drop_axes or inner_shape is None:
        return None
    # `_bands` divides by these. Metadata reaches here through its own parser, so a zero
    # extent declines rather than raising ZeroDivisionError out of the description builder.
    if any(int(v) <= 0 for v in inner_shape):
        return None
    chunk_sel_raw, out_sel_raw = _as_selector_tuples(chunk_selection, out_selection)
    # `X[5]` -- a SCALAR row. zarr passes a bare integer on axis 0 and an output one rank
    # shorter, and does not use `drop_axes` to say so. The buffer is laid out exactly as the
    # one-row selection `X[5:6]` produces, since an axis of extent one contributes no stride,
    # so the axis is synthesised back rather than given a path of its own.
    #
    # This was the LAST selection form reaching the fused read path: an audit of the public
    # indexing surface -- 44 forms over three layouts -- found 34 on the chunk-unit path,
    # 8 declining to zarr-python, and only this one on the fused one.
    #
    # EVERY scalar axis, not just axis 0. `X[5, 5]` drops both and hands over a 0-d output,
    # which the axis-0-only version could not describe -- and under `--strict` a decline is an
    # error, so a plain scalar read of a 2-D array failed outright. Rebuilding each dropped
    # axis as an extent of one is exact for the same reason it was for axis 0: an axis of
    # extent one contributes no stride, so a 0-d buffer of one element and a (1, 1) buffer of
    # one element have identical layouts. This also picks up `X[:, 3]`, where the dropped axis
    # is a trailing one.
    # A CONSTANT trailing index array is a scalar axis spelled as a point selection. `X[rows, 7]`
    # reaches the pipeline as a `CoordinateIndexer` -- one integer array per axis, paired
    # elementwise against a FLAT output -- so axis 1 arrives as [7, 7, 7], not as the integer the
    # syntax suggests. Every point then sits at the same column, which is the box `rows x 7:8`,
    # and rebuilding it is exact for the same reason the plain integer is.
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
    # The three-way length EQUALITY is the whole defence, and it must stay an equality. It is
    # what refuses a constant array whose axis the output KEPT -- `oindex[rows, [7]]`, whose
    # output is rank 2 -- where rebuilding it as an extent of one would claim a single output
    # column against an output that has more. That fills the right number of slots with the
    # right bytes at the wrong stride: wrong data, no error. Weaken it to an inequality, or drop
    # an operand from it, and this becomes a wrong-bytes path.
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
    # Not sharded: the chunk IS the decode unit, and the grid checks below then compare it
    # against itself, which is exactly right -- there is no subdivision to get wrong.
    if inner_shape == ():
        inner_shape = tuple(int(s) for s in chunk_spec.shape)
    chunk_sel, out_sel = _as_selector_tuples(chunk_selection, out_selection)
    rank = len(chunk_spec.shape)
    if not (rank == len(chunk_sel) == len(out_sel) == len(inner_shape) == len(shape)):
        return None
    # Every axis after the split: a contiguous step-1 slice of the chunk, held WHOLE in the
    # output, on a shard grid that keeps one subchunk there.
    #
    # `inner_shape[axis] == chunk_spec.shape[axis]` is the one that must stay: it is the
    # shard GRID, not the selection, and it is what lets `locate` descend on axis 0 alone.
    # The ENTRY's own box per trailing axis, before it is cut into bands. Only the span
    # gate below reads these; the items are described by `lanes`.
    widths: list[int] = []
    # One list of bands per trailing axis. A shard holding several inner chunks across an axis
    # turns one entry into one item per band, and the items are the product across axes.
    lanes: list[list[tuple[int, int, int]]] = []
    for axis in range(1, rank):
        span = _step1_span(chunk_sel[axis], chunk_spec.shape[axis])
        if span is None:
            return None
        lo, hi = span
        # The output holds exactly what was selected -- so an item filling all of it is one
        # contiguous output range, which is what the carve hands out.
        # The output axis need not be WHOLE -- only a contiguous band as wide as the chunk
        # selection. It stopped being whole as soon as the shard grid divided the array's
        # trailing axis: a two-shard-wide array gives every entry half the output width, and
        # `shape[axis] != hi - lo` refused it. That is the guard that actually kept this
        # geometry out; the inner-chunk one below never got a chance to.
        out_span = _step1_span(out_sel[axis], shape[axis])
        if out_span is None or out_span[1] - out_span[0] != hi - lo:
            return None
        widths.append(int(hi - lo))
        # The inner chunk is the decode unit, so a selection crossing one of its boundaries is
        # not a wide read -- it is one read per inner chunk. `inner_shape[axis] !=
        # chunk_spec.shape[axis]` used to decline outright here; it now splits instead.
        lanes.append(_bands(int(lo), int(hi), int(inner_shape[axis]), int(out_span[0])))
    indices = chunk_sel[0]
    out_axis_sel = out_sel[0]
    if isinstance(indices, slice):
        span = _step1_span(indices, chunk_spec.shape[0])
        if span is None:
            return None
        # KEEP THE RUN. With the trailing axes taken whole, `first..first + count` on axis 0
        # is one contiguous block per inner chunk, so Rust needs a coordinate and a length --
        # not a coordinate per element.
        #
        # This used to `np.arange(span[0], span[1])` on the grounds that "a contiguous slice
        # IS a sorted integer axis, spelled differently". True, and it costs one u64 per
        # ELEMENT: a chunk_size 64 preload describes ~130 runs with 11.9M numbers, ~95 MB,
        # measured at 98 ms to build and 112 ms to hand over against a ~317 ms preload. The
        # runs were already there -- anndata derives them from `indptr` and zarr carries them
        # through as slices -- and this is where they were being thrown away.
        # The span form has nowhere to put a trailing start or width: it says "the whole
        # trailing extent", on both sides, and Rust rechecks only `chunk_shape[1:] ==
        # shape[1:]` -- which is about the two EXTENTS and says nothing about this entry's
        # box within them. `starts == 0` was not the property needed: an entry whose chunk box
        # begins at column 0 and stops short of the extent passes it, and `X[a:b, 1:13]` on a
        # 12-wide shard grid builds exactly that -- one column of chunk, described as twelve.
        # The WIDTH is what has to be whole; a start of 0 then follows from it.
        # `inner_shape[axis] == chunk_spec.shape[axis]` is NOT redundant here, and it is the
        # edit that makes removing the gate above safe. `push_span` has nowhere to put a band:
        # it takes the whole trailing extent and derives its row stride from the shard. Let a
        # DIVIDED shard reach it and it builds an item spanning two inner chunks, which
        # `locate` refuses -- and the retrieve runs outside `pipeline.py`'s try, so that is an
        # uncaught PyRuntimeError where today there is a clean fallback.
        if all(
            int(inner_shape[axis]) == int(chunk_spec.shape[axis])
            and int(widths[axis - 1]) == int(shape[axis]) == int(chunk_spec.shape[axis])
            for axis in range(1, rank)
        ):
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
        # A sub-box on a trailing axis makes each index its own run, so the span form does
        # not describe it and the elements are named after all.
        indices = np.arange(span[0], span[1], dtype=np.int64)
    if not _is_sorted_integer_axis(indices, out_axis_sel) or indices.size == 0:
        return None
    indices = indices.astype(np.int64, copy=False)
    if (indices < 0).any():
        return None
    start = out_axis_sel.start or 0
    if not _output_run_matches(indices, out_axis_sel):
        return None

    pushes = []
    # One item per combination of bands across the trailing axes. Rank 1 has no lanes, so the
    # product is a single empty tuple and that path is unchanged.
    for combo in itertools.product(*lanes):
        band_starts = [b[0] for b in combo]
        band_widths = [b[1] for b in combo]
        band_out = [b[2] for b in combo]
        # Both one-run tests, per band. The OUTPUT one against the output extents: without it
        # `output_pieces` models an item as one run per axis-0 index, which is true only while
        # at most one trailing axis is partial -- that was 633 tests. The CHUNK one against the
        # INNER extents, with the band reduced into its own inner chunk, because the buffer
        # being addressed is the inner chunk and not the shard.
        if _contiguous_offset(band_out, band_widths, tuple(shape[1:])) is None:
            return None
        # Gate only. Rust re-derives the offset from these same starts and rechecks the shape,
        # because `push_entry` is reachable from Python with arbitrary arguments and a single
        # fused offset is not a checkable thing.
        within = [s % int(inner_shape[a + 1]) for a, s in enumerate(band_starts)]
        if _contiguous_offset(within, band_widths, tuple(inner_shape[1:])) is None:
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
                # The WHOLE inner chunk, not just the split extent. Every trailing stride Rust
                # computes is a product of these, and the decoded buffer is the inner chunk --
                # so handing it the shard's extents is right only while a shard holds one
                # inner chunk on each trailing axis, which is no longer required.
                tuple(int(v) for v in inner_shape),
                # SHARD-relative: this is what steers `locate` to the right inner chunk.
                # Rust reduces it into the inner chunk for the coordinate.
                tuple(int(v) for v in band_starts),
            )
        )
    return pushes


def _point_unit_args(
    entry, shape: tuple[int, ...], drop_axes: tuple[int, ...], inner_shape
) -> tuple | None:
    """Args for `ChunkItems.push_points`, or None if this entry is not a point selection.

    `X[rows, cols]` and `X[rows, 5]` both reach the pipeline as a `CoordinateIndexer`: one
    integer array per axis, paired element-wise, against a FLAT output slice -- not as a
    dropped axis, which is what you would guess from the syntax.

    Each point is a single element, so the run length is one and the only thing varying is
    where inside its own row each point sits. That is what the per-index offsets carry. The
    ordinary route spends two allocations and a partial-decode call per POINT, so grouping
    them by the chunk that actually gets decoded is worth more here than anywhere else.
    """
    byte_getter, chunk_spec, chunk_selection, out_selection, _ = entry
    if drop_axes or inner_shape is None:
        return None
    # Not sharded: the chunk IS the decode unit, and the grid checks below then compare it
    # against itself, which is exactly right -- there is no subdivision to get wrong.
    if inner_shape == ():
        inner_shape = tuple(int(s) for s in chunk_spec.shape)
    chunk_sel, out_sel = _as_selector_tuples(chunk_selection, out_selection)
    rank = len(chunk_spec.shape)
    # Rank 1 is the plain row case, which `_chunk_unit_args` already serves better.
    if rank < 2 or len(chunk_sel) != rank or len(inner_shape) != rank:
        return None
    # The output of a point selection is flat, however many axes were indexed.
    if len(out_sel) != 1 or len(shape) != 1:
        return None
    if not all(
        isinstance(sel, np.ndarray)
        and sel.ndim == 1
        and np.issubdtype(sel.dtype, np.integer)
        for sel in chunk_sel
    ):
        return None
    n = chunk_sel[0].size
    if n == 0 or any(sel.size != n for sel in chunk_sel):
        return None
    rows = chunk_sel[0].astype(np.int64, copy=False)
    out_axis_sel = out_sel[0]
    if not _is_sorted_integer_axis(rows, out_axis_sel):
        return None
    if (rows < 0).any() or not _output_run_matches(rows, out_axis_sel):
        return None
    # Still the GRID condition: one subchunk on every axis after the split, or `locate`
    # cannot keep walking axis 0 alone.
    for axis in range(1, rank):
        if inner_shape[axis] != chunk_spec.shape[axis]:
            return None
    # A point's offset inside its own index is the C-order ravel of the trailing axes, which
    # is `np.ravel_multi_index` -- bounds check included, negatives included. Out of bounds is
    # a DECLINE here, not an error: the ordinary route serves the read.
    try:
        offsets = np.ravel_multi_index(
            tuple(chunk_sel[1:]), tuple(chunk_spec.shape[1:])
        ).astype(np.uint64, copy=False)
    except ValueError:
        return None
    return (
        byte_getter.path,
        chunk_spec.shape,
        shape,
        rows,
        offsets,
        out_axis_sel.start or 0,
        int(inner_shape[0]),
    )


def _as_contiguous(idx: np.ndarray) -> tuple[int, int] | None:
    """`(start, length)` if these indices are consecutive and ascending, else None.

    A slice arrives already contiguous, but so does an index array that happens to be one --
    and the caller cannot tell them apart by then, because both were normalised to index
    arrays so the ranks could be handled uniformly.
    """
    if idx.size == 0:
        return None
    start = int(idx[0])
    if idx.size > 1 and not np.array_equal(idx, np.arange(start, start + idx.size)):
        return None
    return start, int(idx.size)


def _grid_unit_args(
    entry, shape: tuple[int, ...], drop_axes: tuple[int, ...], inner_shape
) -> tuple | None:
    """Args for `ChunkItems.push_grid`, or None if this entry is not a grid selection.

    A GRID is the Cartesian product: every selected index on axis 0 crossed with every
    selected position on each axis after it. `oindex[rows, cols]`, `X[:, cols]`, and in rank 3
    `oindex[rows, ys, zs]` or `X[rows, 3, 4:12]`.

    zarr BROADCASTS the axes rather than pairing them, so on rank 3 they arrive shaped
    (n,1,1), (1,m,1), (1,1,p). Any of them may instead be a step-1 slice, and an axis taking a
    single element may be dropped from the output entirely -- which changes nothing about the
    buffer, since an axis of extent one contributes no stride, so it is synthesised back.

    RANK-N, not rank 2. The offset of one element inside its index's own elements is
    `sum(sel[axis][i] * stride[axis])`, and flattening the product in row-major order gives
    exactly the order the output row wants -- one list, whatever the rank. An earlier version
    of this declined rank > 2 claiming the offsets "stop being a single list". They do not.

    The box is described as RUNS, not as elements. `X[rows, 2:5, 4:12]` of an (8,16) row is
    three runs of eight at 36, 52 and 68 -- three memcpys, not twenty-four element copies. A
    fully scattered selection degenerates to runs of one, which is the worst case rather than
    the only case, and a selection whose trailing axes are all whole collapses to a SINGLE run
    covering the entire index.
    """
    byte_getter, chunk_spec, chunk_selection, out_selection, _ = entry
    if inner_shape is None:
        return None
    if inner_shape == ():
        inner_shape = tuple(int(s) for s in chunk_spec.shape)
    chunk_sel, out_sel = _as_selector_tuples(chunk_selection, out_selection)
    rank = len(chunk_spec.shape)
    if rank < 2 or len(chunk_sel) != rank or len(inner_shape) != rank:
        return None
    # The split axis is what the grouping walks; it cannot be the one that disappears.
    if 0 in drop_axes or any(axis >= rank for axis in drop_axes):
        return None
    kept = [axis for axis in range(rank) if axis not in drop_axes]
    if len(out_sel) != len(kept) or len(shape) != len(kept):
        return None
    # Back to full rank, so every check below indexes by CHUNK axis.
    out_sel = tuple(
        out_sel[kept.index(a)] if a in kept else slice(0, 1) for a in range(rank)
    )
    shape = tuple(shape[kept.index(a)] if a in kept else 1 for a in range(rank))

    def axis_indices(sel, extent):
        """One axis's selection as a 1-D index array, or None."""
        if isinstance(sel, slice):
            span = _step1_span(sel, extent)
            return None if span is None else np.arange(span[0], span[1], dtype=np.int64)
        if isinstance(sel, np.ndarray) and np.issubdtype(sel.dtype, np.integer):
            return np.ravel(sel).astype(np.int64, copy=False)
        return None

    rows = axis_indices(chunk_sel[0], chunk_spec.shape[0])
    out_axis_sel = out_sel[0]
    if rows is None or rows.size == 0:
        return None
    if not _is_sorted_integer_axis(rows, out_axis_sel):
        return None
    if (rows < 0).any() or not _output_run_matches(rows, out_axis_sel):
        return None

    sels = []
    for axis in range(1, rank):
        idx = axis_indices(chunk_sel[axis], chunk_spec.shape[axis])
        if idx is None or idx.size == 0:
            return None
        if (idx < 0).any() or (idx >= chunk_spec.shape[axis]).any():
            return None
        # The output holds exactly what was selected, and the item fills all of it.
        if shape[axis] != idx.size or not _is_whole_axis(out_sel[axis], shape[axis]):
            return None
        # Still the GRID condition: one subchunk per axis after the split, or `locate` cannot
        # keep walking axis 0 alone.
        if inner_shape[axis] != chunk_spec.shape[axis]:
            return None
        sels.append(idx)
    # A sub-box is a set of RUNS in row-major order, not a set of elements, and saying so is
    # the difference between one memcpy per run and one copy per element. Absorb trailing axes
    # into the run from the inside out: an axis contributes its length, and only an axis taken
    # WHOLE lets the absorption continue past it, because a partial axis leaves a gap before
    # the next one repeats.
    extents = [int(e) for e in chunk_spec.shape]
    run, first_absorbed = 1, rank
    absorbed_start = [0] * rank
    for axis in reversed(range(1, rank)):
        span = _as_contiguous(sels[axis - 1])
        if span is None:
            break
        start_a, len_a = span
        absorbed_start[axis] = start_a
        run *= len_a
        first_absorbed = axis
        if len_a != extents[axis]:
            break

    # Whatever is left varies per run, enumerated row-major so the runs land in output order:
    # each varying axis takes its own indices, each absorbed one is pinned at its run start,
    # and the C-order ravel of that open mesh IS the run starts. numpy says all of it --
    # `np.ix_` builds the mesh and `np.ravel_multi_index` applies the strides, so there is no
    # stride table here to keep in step with row-major order, and no `reshape` broadcast to
    # rebuild the mesh zarr already handed over in that shape. It also bounds-checks, where
    # multiplying in uint64 wrapped an out-of-range index silently.
    cols = [
        sels[axis - 1] if axis < first_absorbed else np.array([absorbed_start[axis]])
        for axis in range(1, rank)
    ]
    starts = (
        np.ravel_multi_index(np.ix_(*cols), tuple(extents[1:]))
        .ravel()
        .astype(np.uint64, copy=False)
    )

    return (
        byte_getter.path,
        chunk_spec.shape,
        shape,
        rows,
        np.ascontiguousarray(starts),
        int(run),
        out_axis_sel.start or 0,
        int(inner_shape[0]),
    )


def chunk_info_for_write(
    batch_info: BatchInfo,
    drop_axes: tuple[int, ...],
    shape: tuple[int, ...],
) -> RustChunkInfo:
    """Describe a write batch to Rust, one item per entry.

    Never split: two items on one chunk key make the read-modify-writes race.
    """
    return _chunk_items(_as_int64_batch_info(batch_info), drop_axes, shape)


def chunk_info_for_read(
    batch_info: BatchInfo,
    drop_axes: tuple[int, ...],
    shape: tuple[int, ...],
    inner_chunk_shape: tuple[int, ...] | None,
) -> RustChunkInfo:
    """Describe a read batch to Rust, grouped by decode unit where the selection allows.

    One item per inner chunk if every entry is eligible; otherwise one box per run of
    consecutive indices, falling back to one item per entry.
    """
    # A generator would be consumed by the eligibility test, and the ordinary route needs
    # to read the same entries again if that test fails.
    entries = list(_as_int64_batch_info(batch_info))

    # All or nothing: one ineligible entry sends the whole batch down the ordinary route.
    unit_args = [
        _chunk_unit_args(entry, shape, drop_axes, inner_chunk_shape)
        for entry in entries
    ]
    if unit_args and all(args is not None for args in unit_args):
        handle = ChunkItems()
        # An entry straddling an inner-chunk boundary on a trailing axis describes one item
        # per band, so this is a list of lists.
        for kind, *args in itertools.chain.from_iterable(unit_args):
            # A span names a contiguous block; an entry names its elements. Both land in the
            # same handle and are served by the same path -- the difference is only how much
            # had to be said to describe the read.
            if kind == "span":
                handle.push_span(*args)
            else:
                handle.push_entry(*args)
        return RustChunkInfo(handle, write_empty_chunks=True)

    # A point selection is a different SHAPE of batch, not a failed row one, so it gets its
    # own all-or-nothing pass rather than being mixed in.
    point_args = [
        _point_unit_args(entry, shape, drop_axes, inner_chunk_shape)
        for entry in entries
    ]
    if point_args and all(args is not None for args in point_args):
        handle = ChunkItems()
        for args in point_args:
            handle.push_points(*args)
        return RustChunkInfo(handle, write_empty_chunks=True)

    # And a grid is a third shape: same columns from every row, scattered within the row.
    grid_args = [
        _grid_unit_args(entry, shape, drop_axes, inner_chunk_shape) for entry in entries
    ]
    if grid_args and all(args is not None for args in grid_args):
        handle = ChunkItems()
        for args in grid_args:
            handle.push_grid(*args)
        return RustChunkInfo(handle, write_empty_chunks=True)

    # Nothing else is served here. Every selection either produced a handle above or is
    # declined to zarr-python, which is a real fallback rather than a slower Rust path.
    #
    # `DiscontiguousArrayError` is what `pipeline.read` catches to fall back, so raising it is
    # how a decline is spelled.
    raise DiscontiguousArrayError("this selection is not served by the chunk-unit path")


def _chunk_items(
    batch_info: BatchInfo,
    drop_axes: tuple[int, ...],
    shape: tuple[int, ...],
) -> RustChunkInfo:
    """One ChunkItem per batch entry."""
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
