#!/bin/bash
#SBATCH --job-name=anndata-native-access
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --constraint=cpu_96
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=logs/native_%j.out
#SBATCH --error=logs/native_%j.err

# Native anndata random access on Lustre -- no annbatch, no loader, no prefetch.
#
# Matrix: {main, pr2+vec} x {default, zarrs} x {sorted, unsorted}
#         x rows in {1, 8, 64, 512, 4096}
#
# Path discipline (theislab-hpc.md 3 and 6):
#   volatile -- env, uv cache, clone, TMPDIR -> /localscratch. Building the venv
#     from scratch there is cheap, imports are fast, and the metadata churn never
#     reaches the Lustre MDS.
#   durable  -- the datasets we read and the JSON we write -> Lustre.
# The datasets stay on Lustre deliberately: Lustre read latency is the thing
# under measurement, so staging them to scratch would destroy the experiment.

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONUNBUFFERED=1   # otherwise progress only lands in the log at exit
mkdir -p logs

STAGE_DIR="${STAGE_DIR:?set STAGE_DIR to the Lustre dir holding bench/}"
RESULTS_DIR="${RESULTS_DIR:-$STAGE_DIR/results}"
ANNDATA_FORK="${ANNDATA_FORK:-https://github.com/selmanozleyen/anndata.git}"
ANNDATA_UPSTREAM="${ANNDATA_UPSTREAM:-https://github.com/scverse/anndata}"
VEC_REF="${VEC_REF:-perf/vectorize-csr-coords}"
REPS="${REPS:-20}"
SRC=src/anndata/_core/sparse_dataset.py

SCRATCH="${SLURM_TMPDIR:-/localscratch/$USER}/native-${SLURM_JOB_ID:-manual}"
export UV_CACHE_DIR="$SCRATCH/uv-cache"
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$SCRATCH" "$TMPDIR" "$UV_CACHE_DIR" "$RESULTS_DIR"
trap 'rm -rf "$SCRATCH"' EXIT

echo "Job ID:   ${SLURM_JOB_ID:-none}"
echo "Node:     $(hostname)"
echo "Scratch:  $SCRATCH"
echo "Results:  $RESULTS_DIR"
echo "Start:    $(date -Is)"

DATASETS=(
  "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr/dataset_0"
  "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr/dataset_1"
  "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr/dataset_2"
  "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr/dataset_3"
)

# The fd cache is per array, so a few open arrays at cache 512 will exhaust the
# default limit -- see STATUS.md "Config that matters".
ulimit -n 65536 2>/dev/null || echo "WARN: RLIMIT_NOFILE stuck at $(ulimit -n)"
echo "RLIMIT_NOFILE: $(ulimit -n)"

# --- build everything on node-local scratch ---
cp -r "$STAGE_DIR/bench" "$SCRATCH/bench"
# The branch lives on the fork; `main` only exists upstream, so both remotes
# are needed to diff the two states.
git clone --quiet --depth 50 --branch "$VEC_REF" "$ANNDATA_FORK" "$SCRATCH/anndata"
git -C "$SCRATCH/anndata" remote add upstream "$ANNDATA_UPSTREAM"
git -C "$SCRATCH/anndata" fetch --quiet --depth 50 upstream main
echo "anndata @ $(git -C "$SCRATCH/anndata" log --oneline -1)"
echo "upstream main @ $(git -C "$SCRATCH/anndata" log --oneline -1 upstream/main)"

cd "$SCRATCH"
uv venv --quiet .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --quiet -e ./anndata scipy zarrs psutil

run_state() {  # label
  echo
  echo "############ anndata state: $1 ############"
  python "$SCRATCH/bench/run_cluster_benchmark.py" \
    --native \
    --label "$1" \
    --native-reps "$REPS" \
    --dataset-paths "${DATASETS[@]}" \
    --output-json "$RESULTS_DIR/native_${1}_${SLURM_JOB_ID:-manual}.json"
}

cd "$SCRATCH/bench"   # scenarios.py is imported as a top-level module
run_state "pr2+vec"

git -C "$SCRATCH/anndata" checkout --quiet upstream/main -- "$SRC"
echo "reverted $SRC to upstream/main"
run_state "main"

echo
echo "Results in $RESULTS_DIR:"
ls -1 "$RESULTS_DIR" | tail -4
echo "End: $(date -Is)"
