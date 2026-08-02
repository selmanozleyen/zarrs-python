set -euo pipefail
STAGE="${STAGE_DIR:?}"; JOB="${JOB:?}"
source "$STAGE/hold_${JOB}.env"
export UV_CACHE_DIR="$SCRATCH/uv-cache" TMPDIR="$SCRATCH/tmp" PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"; source "$VENV/bin/activate"
R="$STAGE/data_results_${JOB}.jsonl"; : > "$R"
CS="$STAGE/bench_cold_shards.py"; SC="$STAGE/bench_scattered.py"
inst() { uv pip install --quiet --force-reinstall --no-deps "$(ls -1 "$WHEEL_DIR/$1"/*.whl|head -1)"; }

echo "### X/data geometry ###"
python - <<'PY'
import zarr
for a in ("data","indices"):
    z=zarr.open_array(f"/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr/dataset_0/X/{a}",mode="r")
    print(f"  {a:8} shape={z.shape[0]:>14,} dtype={z.dtype} shard={z.shards[0]:>12,} -> {z.shards[0]*z.dtype.itemsize/1024**2:7.1f} MiB/shard")
PY

# main first on each array so it gets the cold window, not the warmed one.
echo "### X/data cold shards ###"
inst main;      python "$CS" "main" - 16 data >>"$R"
inst fetchpool; python "$CS" "fetchpool" - 16 data >>"$R"

echo "### scattered rows (negative control) ###"
for n in 256 4096; do
  inst main;      python "$SC" "main" - "$n" >>"$R"
  inst fetchpool; python "$SC" "fetchpool" - "$n" >>"$R"
done
echo "### $R ###"; cat "$R"
