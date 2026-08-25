"""Minimal repro: zarr-python cannot read an unsorted fancy index of unsigned dtype.

Stock zarr, no zarrs involved.

    z[np.array([3, 0], dtype="int64"), :]   ->  works
    z[np.array([3, 0], dtype="uint8"),  :]  ->  IndexError

Same values, same array; only the index dtype differs.

Root cause is `Order.check` in zarr/core/indexing.py, which classifies a selection by
`diff = np.diff(a)` and calls it INCREASING when every step is >= 0. On an unsigned array
a subtraction cannot go negative, so it wraps: `np.diff(uint8([3, 0]))` is 253, not -3.
A descending selection is therefore classified INCREASING, `dim_out_sel` is left None, and
the indices are never reordered to group by chunk -- so index 3 is looked up in the chunk
that only holds rows 0..1.

    np.diff(np.array([3, 0],   dtype="uint8"))  ->  [253]  ->  Order.INCREASING
    np.diff(np.array([255, 0], dtype="uint8"))  ->  [1]    ->  Order.INCREASING

The one-line fix is to difference in a signed type, `np.diff(a.astype(np.int64,
copy=False))`, or to classify with comparisons (`a[1:] >= a[:-1]`), which do not wrap.

Reproducing needs all four of:
  * an unsigned index dtype -- int8/int32 are fine, so this is signedness, not width
  * an unsorted index -- np.sort(rows) always works
  * a second axis -- the same index on a 1-D array is fine
  * an index spanning more than one chunk -- a single-chunk array is fine

It fails loudly rather than silently: over 400 random unsorted uint8 selections, 292
raised and 108 succeeded, none returned wrong data.

    $ python repro_zarr_unsigned_index.py
"""

import tempfile
from pathlib import Path

import numpy as np
import zarr
from zarr.core.indexing import Order

ROWS = [3, 0]


def main() -> None:
    print(f"zarr {zarr.__version__}, numpy {np.__version__}\n")

    values = np.arange(4, dtype="float32").reshape(4, 1)
    store = Path(tempfile.mkdtemp()) / "a.zarr"
    array = zarr.create_array(store, shape=values.shape, dtype="float32", chunks=(2, 1))
    array[:] = values

    expected = values[ROWS, :]
    array = zarr.open_array(store, mode="r")

    for dtype in ("int64", "int8", "uint8", "uint16", "uint32", "uint64"):
        try:
            got = array[np.array(ROWS, dtype=dtype), :]
        except Exception as exc:  # noqa: BLE001 - the point is to show what escapes
            print(f"  {dtype:>6}: {type(exc).__name__}: {exc}")
        else:
            ok = "matches" if np.array_equal(got, expected) else "WRONG DATA"
            print(f"  {dtype:>6}: {ok}")

    # Sorting the same values sidesteps it entirely.
    got = array[np.array(sorted(ROWS), dtype="uint8"), :]
    sorted_ok = np.array_equal(got, values[sorted(ROWS), :])
    print(
        f"\n  uint8 sorted {sorted(ROWS)}: {'matches' if sorted_ok else 'WRONG DATA'}"
    )

    print("\nthe misclassification, directly:")
    for dtype in ("int64", "uint8"):
        a = np.array(ROWS, dtype=dtype)
        print(f"  np.diff({dtype:>6}{ROWS}) = {np.diff(a)}  ->  {Order.check(a)}")
    a = np.array([255, 0], dtype="uint8")
    print(f"  np.diff( uint8[255, 0]) = {np.diff(a)}  ->  {Order.check(a)}")


if __name__ == "__main__":
    main()
