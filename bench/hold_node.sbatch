#!/bin/bash
#SBATCH --job-name=fetchpool-hold
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --output=logs/hold_%j.out
#SBATCH --error=logs/hold_%j.out

# Builds the environment on node-local scratch, publishes where it is, then
# holds the allocation so steps can be driven in with
# `srun --overlap --jobid=<id>`. That way a failed step is retried on the same
# node with the same env instead of costing another queue wait.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p logs

STAGE_DIR="${STAGE_DIR:?set STAGE_DIR}"
WHEEL_DIR="${WHEEL_DIR:?set WHEEL_DIR}"

# Volatile on scratch, durable on Lustre (theislab-hpc.md 3).
SCRATCH="${SLURM_TMPDIR:-/localscratch/$USER}/fetchpool-${SLURM_JOB_ID}"
export UV_CACHE_DIR="$SCRATCH/uv-cache" TMPDIR="$SCRATCH/tmp"
mkdir -p "$SCRATCH" "$UV_CACHE_DIR" "$TMPDIR"

# The synthetic array must live on Lustre: its read latency is the thing being
# measured, so staging it to scratch would defeat the experiment.
BENCH_DIR="$STAGE_DIR/fetchpool-data"
mkdir -p "$BENCH_DIR"

echo "job:      ${SLURM_JOB_ID}"
echo "node:     $(hostname)"
echo "scratch:  $SCRATCH"
echo "benchdir: $BENCH_DIR  (Lustre, deliberately)"
echo "start:    $(date -Is)"
ulimit -n 65536 2>/dev/null || true

cd "$SCRATCH"
uv venv --quiet .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --quiet numpy 'zarr>=3' scipy

# Publish the paths so driven steps can find them without guessing.
cat > "$STAGE_DIR/hold_${SLURM_JOB_ID}.env" <<EOF
SCRATCH=$SCRATCH
VENV=$SCRATCH/.venv
BENCH_DIR=$BENCH_DIR
WHEEL_DIR=$WHEEL_DIR
NODE=$(hostname)
EOF

echo "READY $(date -Is)"
echo "drive steps with: srun --overlap --jobid=${SLURM_JOB_ID} <cmd>"
sleep 7200
