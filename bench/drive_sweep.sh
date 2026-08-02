set -euo pipefail
STAGE="${STAGE_DIR:?}"; JOB="${JOB:?}"
source "$STAGE/hold_${JOB}.env"
export UV_CACHE_DIR="$SCRATCH/uv-cache" TMPDIR="$SCRATCH/tmp" PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"; source "$VENV/bin/activate"
R="$STAGE/sweep_results_${JOB}.jsonl"; : > "$R"
B="$STAGE/bench_cold_shards.py"
inst() { uv pip install --quiet --force-reinstall --no-deps "$(ls -1 "$WHEEL_DIR/$1"/*.whl|head -1)"; }
inst main;      python "$B" "main" - 32 >>"$R"
inst fetchpool
for ft in 1 2 4 16 128; do python "$B" "fetchpool" "$ft" 32 >>"$R"; done
inst main;      python "$B" "main-repeat" - 32 >>"$R"
echo "### $R ###"; cat "$R"
