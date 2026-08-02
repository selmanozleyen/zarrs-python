#!/bin/bash
#SBATCH --job-name=sparse-chunk-fetch-bench
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --constraint=cpu_96
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

set -euo pipefail

mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Record provenance before run
python /Users/selman.ozleyen/.gemini/antigravity-ide/scratch/check_provenance.py

DATASETS=(
  "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr/dataset_0"
  "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr/dataset_1"
  "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr/dataset_2"
  "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr/dataset_3"
)

# Run Arms B and D (alternating order to remove fixed-order bias)
echo "=== Phase 1: Comparing Arm B vs Arm D ==="
python bench/run_cluster_benchmark.py \
  --arms B D \
  --dataset-paths "${DATASETS[@]}" \
  --num-batches 12 \
  --output-json "results_bd_${SLURM_JOB_ID}.json"

echo "=== Phase 2: Full A/B/C/D Matrix ==="
python bench/run_cluster_benchmark.py \
  --arms A B C D \
  --dataset-paths "${DATASETS[@]}" \
  --num-batches 12 \
  --output-json "results_abcd_${SLURM_JOB_ID}.json"

echo "End time: $(date)"
