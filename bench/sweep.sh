#!/bin/bash
# Sweep fetch-pool depth at a fixed file-handle cache, one row order.
#
# The fd cache is not swept: 0 is upstream's default only because upstream has
# no opinion, and measured here it is worth ~5% while the pool is worth ~6x.
# Pin it on and vary the thing that matters.
#
# Config order is shuffled within each round because whichever runs later
# inherits the earlier ones' page cache; a fixed order parks that advantage on
# the same config every round and reads as a result.
#
#   sweep.sh <order> [rounds] [rows] [reps] [fetch-values...]
cd ~/zarrs-bench/zarrs-python || exit 1
export PATH=$HOME/.local/bin:$PATH
ORDER="${1:-sorted}"; ROUNDS="${2:-3}"; ROWS="${3:-1024}"; REPS="${4:-3}"
shift 4 2>/dev/null
FETCH=("${@:-0 32 128 256}")
[ $# -eq 0 ] && FETCH=(0 32 128 256)
FDC=512

for _ in $(seq 1 "$ROUNDS"); do
  printf '%s\n' "${FETCH[@]}" | shuf | while read -r f; do
    BENCH_ROW_ORDER="$ORDER" ZARRS_PYTHON_FETCH_THREADS=$f BENCH_FD_CACHE=$FDC \
      uv run --frozen --with anndata python bench/bench_vindex_pool.py "fetch$f" "$ROWS" "$REPS" 1 2>&1 \
      | grep -E '^fetch'
  done
done
