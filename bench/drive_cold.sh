set -euo pipefail
STAGE="${STAGE_DIR:?}"; JOB="${JOB:?}"
source "$STAGE/hold_${JOB}.env"
export UV_CACHE_DIR="$SCRATCH/uv-cache" TMPDIR="$SCRATCH/tmp" PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
source "$VENV/bin/activate"
R="$STAGE/cold_results_${JOB}.jsonl"; : > "$R"
B="$STAGE/bench_cold_shards.py"
inst() { uv pip install --quiet --force-reinstall --no-deps "$(ls -1 "$WHEEL_DIR/$1"/*.whl|head -1)"; }
for spr in 8 32; do
  inst main;      python "$B" "main"      -   "$spr" >>"$R"
  inst fetchpool; python "$B" "fetchpool" -   "$spr" >>"$R"
  inst fetchpool; python "$B" "fetchpool" 1   "$spr" >>"$R"
done
echo "### $R ###"; cat "$R"
