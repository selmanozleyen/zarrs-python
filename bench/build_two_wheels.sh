#!/bin/bash
# Build one zarrs-python wheel per ref, on hpc-build01.
#
# Both refs sit on upstream/main, so Cargo.toml points at a released zarrs and
# no sibling checkout is needed. Everything compiles on /localscratch; only the
# wheels and their provenance land on Lustre.
set -euo pipefail

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export CARGO_TERM_COLOR=never

OUT_DIR="${OUT_DIR:?set OUT_DIR}"
FORK="${FORK:-https://github.com/selmanozleyen/zarrs-python.git}"
UPSTREAM="${UPSTREAM:-https://github.com/zarrs/zarrs-python}"
BUILD="/localscratch/$USER/two-wheel-build"

rm -rf "$BUILD"; mkdir -p "$BUILD" "$OUT_DIR"
export CARGO_HOME="$BUILD/cargo" UV_CACHE_DIR="$BUILD/uv-cache" TMPDIR="$BUILD/tmp"
mkdir -p "$CARGO_HOME" "$UV_CACHE_DIR" "$TMPDIR"

cd "$BUILD"
git clone --quiet "$FORK" repo
cd repo
git remote add upstream "$UPSTREAM"
git fetch --quiet upstream main

uv venv --quiet .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --quiet maturin

build_ref() {  # ref label
  local ref="$1" label="$2"
  echo "=== $label ($ref) ==="
  git checkout --quiet --detach "$ref"
  local sha; sha=$(git rev-parse HEAD)
  # Wheels from different refs share a filename, so isolate then rename.
  rm -rf "$BUILD/dist"; mkdir -p "$BUILD/dist"
  maturin build --release --out "$BUILD/dist" 2>&1 | tail -3
  local whl; whl=$(ls -1 "$BUILD/dist"/*.whl | head -1)
  local dest="$OUT_DIR/${label}-$(basename "$whl")"
  cp "$whl" "$dest"
  {
    echo "label:        $label"
    echo "ref:          $ref"
    echo "commit:       $sha"
    echo "built:        $(date -Is) on $(hostname)"
    echo "rustc:        $(rustc --version)"
    echo "sha256:       $(sha256sum "$dest" | awk '{print $1}')"
  } > "$dest.provenance.txt"
  cat "$dest.provenance.txt"
  echo
}

build_ref upstream/main main
build_ref origin/feat/fetch-pool fetchpool

echo "=== wheels in $OUT_DIR ==="
ls -1 "$OUT_DIR"
rm -rf "$BUILD"
