set -euo pipefail
STAGE="${STAGE_DIR:?}"; JOB="${JOB:?}"
source "$STAGE/hold_${JOB}.env"
export UV_CACHE_DIR="$SCRATCH/uv-cache" TMPDIR="$SCRATCH/tmp" PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"; source "$VENV/bin/activate"
R="$STAGE/flip_results_${JOB}.jsonl"; : > "$R"
SC="$STAGE/bench_scattered.py"
inst() { uv pip install --quiet --force-reinstall --no-deps "$(ls -1 "$WHEEL_DIR/$1"/*.whl|head -1)"; }
# Reversed order: if the advantage tracks position rather than wheel, it is cache.
inst fetchpool; python "$SC" "fetchpool-first" - 256 >>"$R"
inst main;      python "$SC" "main-second"     - 256 >>"$R"
inst fetchpool; python "$SC" "fetchpool-third" - 256 >>"$R"
inst main;      python "$SC" "main-fourth"     - 256 >>"$R"
