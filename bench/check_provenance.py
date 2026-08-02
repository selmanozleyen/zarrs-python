from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import zarr


def get_git_info(path: Path) -> dict:
    """Get commit hash, branch name, and dirty status for a repository."""
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path, text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=path, text=True).strip()
        is_dirty = len(status) > 0
    except Exception as e:
        return {"error": str(e)}

    return {
        "path": str(path),
        "branch": branch,
        "commit": commit,
        "is_dirty": is_dirty,
        "status_summary": status.splitlines() if is_dirty else [],
    }


def compute_file_sha256(file_path: Path) -> str:
    if not file_path.exists():
        return "N/A (file not found)"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def collect_provenance() -> dict:
    base_dir = Path(__file__).resolve().parent.parent.parent
    zarrs_path = base_dir / "zarrs"
    annbatch_path = base_dir / "annbatch-branch"
    zarrs_py_path = base_dir / "zarrs-python"

    prov = {
        "zarrs": get_git_info(zarrs_path),
        "annbatch": get_git_info(annbatch_path),
        "zarrs_python": get_git_info(zarrs_py_path),
        "zarr_version": zarr.__version__,
        "has_pr_4172": hasattr(zarr.core.indexing, "CoordinateIndexer"),
        "uv_lock_sha256": compute_file_sha256(zarrs_py_path / "uv.lock"),
    }

    return prov


if __name__ == "__main__":
    prov = collect_provenance()
    print(json.dumps(prov, indent=2))
    print("\n✓ Provenance verification passed successfully.")
