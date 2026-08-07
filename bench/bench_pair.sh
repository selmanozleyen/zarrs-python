#!/bin/bash
# Compare two zarrs-python builds on identical rows, main against ours.
#
# Each pair gets one fresh random seed, so both builds read the same rows and
# their checksums must agree. Whichever runs second is served partly from the
# first one's page cache, so the order alternates by pair -- over enough pairs
# that advantage lands on each build equally, which is the only way to compare
# on a node whose RAM exceeds the working set.
#
#   bench_pair.sh <main-dir> <ours-dir> <pairs> <rows> [reps] [warmup]
set -uo pipefail
MAIN="${1:?main checkout}"; OURS="${2:?ours checkout}"; PAIRS="${3:-6}"
ROWS="${4:-1024}"; REPS="${5:-3}"; WARMUP="${6:-1}"
SCRIPT=bench/bench_vindex_pool.py

run() { # dir label seed
  (cd "$1" && uv run --frozen --with anndata python "$SCRIPT" "$2" "$ROWS" "$REPS" "$WARMUP" "$3" 2>/dev/null)
}

for i in $(seq 1 "$PAIRS"); do
  seed=$(python3 -c 'import secrets;print(secrets.randbits(63))')
  if [ $((i % 2)) -eq 1 ]; then
    run "$MAIN" main "$seed"
    ZARRS_PYTHON_FETCH_THREADS=${FETCH:-32} BENCH_FD_CACHE=${FDC:-512} \
      run "$OURS" ours "$seed"
  else
    ZARRS_PYTHON_FETCH_THREADS=${FETCH:-32} BENCH_FD_CACHE=${FDC:-512} \
      run "$OURS" ours "$seed"
    run "$MAIN" main "$seed"
  fi
done
