from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import zarr

from scenarios import BATCH_SIZE, CHUNK_SIZE, EXPERIMENT_ARMS, PRELOAD_NCHUNKS, ExperimentArm


def get_process_stats() -> dict:
    proc = psutil.Process()
    ctx = proc.num_ctx_switches()
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_time": rusage.ru_utime,
        "sys_time": rusage.ru_stime,
        "max_rss_kb": rusage.ru_maxrss,
        "voluntary_ctx_switches": ctx.voluntary,
        "involuntary_ctx_switches": ctx.involuntary,
        "open_fds": proc.num_fds(),
    }


def run_arm_benchmark(
    arm: ExperimentArm,
    dataset_paths: list[Path],
    num_batches: int = 12,
    warmup_batches: int = 2,
) -> dict:
    import sys
    sys.path.insert(0, "/private/tmp/annbatch-sorted-indexing/src")
    from annbatch import Loader

    # Configure zarr / zarrs pipeline settings
    zarr_cfg = {
        "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
        **arm.zarrs_config,
    }

    os.environ["ZARRS_VINDEX_STATS"] = "1"
    start_proc_stats = get_process_stats()

    batch_times = []
    total_samples = 0

    with zarr.config.set(zarr_cfg):
        datasets = [ad.io.sparse_dataset(zarr.open(p)["X"]) for p in dataset_paths]

        loader = Loader(
            shuffle=True,
            chunk_size=CHUNK_SIZE,
            preload_nchunks=PRELOAD_NCHUNKS,
            batch_size=BATCH_SIZE,
            return_index=True,
            to=None,
            **arm.loader_kwargs,
        ).add_datasets(datasets)

        iterator = iter(loader)

        # Cold batch timing
        t0 = time.perf_counter()
        cold_batch = next(iterator)
        cold_ms = (time.perf_counter() - t0) * 1000.0

        # Warmup batches
        for _ in range(warmup_batches):
            _ = next(iterator)

        # Measured batches
        t_start = time.perf_counter()
        for _ in range(num_batches):
            t_b0 = time.perf_counter()
            batch = next(iterator)
            t_b1 = time.perf_counter()
            batch_times.append((t_b1 - t_b0) * 1000.0)
            total_samples += batch["X"].shape[0]

        total_wall = time.perf_counter() - t_start

    end_proc_stats = get_process_stats()

    p10 = float(np.percentile(batch_times, 10))
    median = float(np.median(batch_times))
    p90 = float(np.percentile(batch_times, 90))
    samples_per_sec = total_samples / total_wall if total_wall > 0 else 0.0

    return {
        "arm": arm.name,
        "input_order": arm.input_order,
        "zarrs_fetch": arm.zarrs_fetch,
        "cold_batch_ms": round(cold_ms, 2),
        "batch_times_ms": [round(t, 2) for t in batch_times],
        "median_ms": round(median, 2),
        "p10_ms": round(p10, 2),
        "p90_ms": round(p90, 2),
        "samples_per_sec": round(samples_per_sec, 2),
        "cpu_user_sec": round(end_proc_stats["user_time"] - start_proc_stats["user_time"], 3),
        "cpu_sys_sec": round(end_proc_stats["sys_time"] - start_proc_stats["sys_time"], 3),
        "peak_rss_mb": round(end_proc_stats["max_rss_kb"] / 1024.0, 2),
        "open_fds": end_proc_stats["open_fds"],
        "voluntary_ctx_switches": end_proc_stats["voluntary_ctx_switches"] - start_proc_stats["voluntary_ctx_switches"],
        "involuntary_ctx_switches": end_proc_stats["involuntary_ctx_switches"] - start_proc_stats["involuntary_ctx_switches"],
    }


def main():
    parser = argparse.ArgumentParser(description="Sorted per-inner-chunk fetch cluster benchmark runner")
    parser.add_argument("--arms", nargs="+", default=["B", "D"], help="Arms to run e.g. B D or A B C D")
    parser.add_argument("--dataset-paths", nargs="+", required=True, help="Paths to Zarr datasets")
    parser.add_argument("--num-batches", type=int, default=12, help="Number of measured batches")
    parser.add_argument("--output-json", type=str, default="benchmark_results.json", help="Output results file")
    args = parser.parse_args()

    arm_lookup = {arm.name.split()[1]: arm for arm in EXPERIMENT_ARMS}
    selected_arms = [arm_lookup[key] for key in args.arms if key in arm_lookup]

    results = []
    print(f"Running benchmark matrix for arms: {[a.name for a in selected_arms]}")

    for arm in selected_arms:
        print(f"\n--- Running {arm.name} ---")
        res = run_arm_benchmark(arm, [Path(p) for p in args.dataset_paths], num_batches=args.num_batches)
        print(f"  Cold: {res['cold_batch_ms']} ms | Median: {res['median_ms']} ms (p10: {res['p10_ms']}, p90: {res['p90_ms']})")
        print(f"  Throughput: {res['samples_per_sec']} samples/s | Peak RSS: {res['peak_rss_mb']} MB")
        results.append(res)

    out_file = Path(args.output_json)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved benchmark results to {out_file}")


if __name__ == "__main__":
    main()
