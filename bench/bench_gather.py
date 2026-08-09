"""Minibatch gather from a backed CSR: zarr-python's pipeline against this one.

The workload is the one that motivated the work -- a few thousand random rows
out of a sparse matrix on parallel storage, which is what a minibatch loader
does every step. Rows are sorted, because that is what lets zarr describe the
selection as contiguous runs and is free for a loader to do.

Two modes:

  full        `sparse_dataset[rows]` end to end. What a user experiences,
              including zarr's indexing and anndata assembling the CSR.
  data-only   the same reads against the `data` array directly. Excludes
              anndata, so the number is about the read rather than about
              matrix assembly -- and is far less noisy for it.

Arms alternate in a shuffled order each round and every run draws fresh rows,
so neither pipeline inherits the other's page cache.

    python bench/bench_gather.py <collection> [--rows N] [--rounds N]
                                 [--fetch-threads N] [--data-only]
"""

from __future__ import annotations

import argparse
import random
import secrets
import statistics
import time

import numpy as np
import zarr

import zarrs  # noqa: F401

PIPELINES = {
    "zarr-python": "zarr.core.codec_pipeline.BatchedCodecPipeline",
    "zarrs": "zarrs.ZarrsCodecPipeline",
}


def configure(pipeline: str, fd_cache: int) -> None:
    cfg = {"codec_pipeline.path": PIPELINES[pipeline]}
    if pipeline == "zarrs":
        # Not a zarr-python option; setting it there raises.
        cfg["codec_pipeline.file_handle_cache_size"] = fd_cache
    zarr.config.set(cfg)


def run_full(collection: str, rows: int, reps: int, rng) -> tuple[float, int]:
    """`sparse_dataset[rows]`, as a user writes it."""
    from anndata._core.sparse_dataset import sparse_dataset

    root = zarr.open_group(collection, mode="r")
    names = sorted(k for k in root if k.startswith("dataset_"))
    datasets = [sparse_dataset(root[f"{n}/X"]) for n in names]
    n_obs = [d.shape[0] for d in datasets]

    times, nnz = [], 0
    for i in range(reps + 1):  # one warmup
        which = rng.integers(0, len(datasets), rows)
        picks = [
            np.sort(rng.integers(0, n_obs[j], int((which == j).sum())))
            for j in range(len(datasets))
        ]
        start = time.perf_counter()
        out = [d[r] for d, r in zip(datasets, picks) if len(r)]
        elapsed = time.perf_counter() - start
        if i:
            times.append(elapsed)
            nnz += sum(int(o.nnz) for o in out)
    return statistics.median(times), nnz // len(times)


def run_data_only(collection: str, rows: int, reps: int, rng) -> tuple[float, int]:
    """The same reads against `data`, without anndata assembling a matrix."""
    root = zarr.open_group(collection, mode="r")
    name = sorted(k for k in root if k.startswith("dataset_"))[0]
    data = root[f"{name}/X/data"]
    indptr = np.asarray(root[f"{name}/X/indptr"][:])

    times, nnz = [], 0
    for i in range(reps + 1):
        picks = np.sort(rng.integers(0, indptr.size - 1, rows))
        coords = np.concatenate(
            [np.arange(indptr[r], indptr[r + 1]) for r in picks.tolist()]
        )
        start = time.perf_counter()
        out = data[coords]
        elapsed = time.perf_counter() - start
        if i:
            times.append(elapsed)
            nnz += out.size
    return statistics.median(times), nnz // len(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", help="path to a .zarr holding dataset_*/X groups")
    parser.add_argument("--rows", type=int, default=9192)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--fetch-threads", type=int, default=64)
    parser.add_argument("--fd-cache", type=int, default=512)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()

    import os

    os.environ["ZARRS_PYTHON_FETCH_THREADS"] = str(args.fetch_threads)
    measure = run_data_only if args.data_only else run_full
    itemsize = 4  # float32 CSR data

    results: dict[str, list[float]] = {name: [] for name in PIPELINES}
    nnz_seen = 0
    for _ in range(args.rounds):
        for name in random.sample(list(PIPELINES), len(PIPELINES)):
            configure(name, args.fd_cache)
            # Fresh rows per run: a fixed seed would serve later runs from the
            # page cache and make every arm look identical.
            seconds, nnz = measure(
                args.collection, args.rows, args.reps, np.random.default_rng(secrets.randbits(63))
            )
            results[name].append(args.rows / seconds)
            nnz_seen = nnz

    mode = "data-only" if args.data_only else "full gather"
    print(
        f"\n{args.rows} sorted rows, {mode}, {args.rounds} rounds x {args.reps} reps, "
        f"fetch_threads={args.fetch_threads}\n"
    )
    print(f"{'pipeline':<14} {'rows/s':>10} {'MiB/s':>9}   runs")
    baseline = statistics.median(results["zarr-python"])
    for name, values in results.items():
        median = statistics.median(values)
        mib = nnz_seen * itemsize / 2**20 / (args.rows / median)
        runs = " ".join(f"{v:.0f}" for v in values)
        print(f"{name:<14} {median:>10.0f} {mib:>9.1f}   {runs}")
    print(f"\nzarrs is {statistics.median(results['zarrs']) / baseline:.1f}x zarr-python")


if __name__ == "__main__":
    main()
