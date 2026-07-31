"""Record the V4.1 code-source manifest (spec section 18 git hygiene).

Before any formal V4.1 training, we must pin every source file that actually
participates in ``working_code_sha`` plus the current git state.  The repo may
have *zero commits* (everything untracked); we do not commit on the user's
behalf (git safety), so instead we emit a durable manifest that records:

* the aggregate ``working_code_sha`` (identical to the value the pipeline gates
  on);
* every source file in that hash set with its own SHA-256, size and mtime;
* whether the git repo has any commit (``HEAD`` resolvable);
* the full ``git status --porcelain`` untracked/dirty manifest.

The manifest is written to
``<output_root>/audits/frozen_evidence/v4_offline_v0/code_source_manifest.json``
and is read-only evidence: it never mutates source or git state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml

from sewerrtc.v4.runtime import working_code_sha

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_REL = "audits/frozen_evidence/v4_offline_v0/code_source_manifest.json"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths(root: Path) -> list[Path]:
    """The exact file set folded into ``working_code_sha`` (see runtime.py)."""
    paths = [
        *(root / "sewerrtc" / "v4").glob("*.py"),
        root / "scripts" / "project6_v4_final.py",
        root / "scripts" / "project6_runs" / "RUN_PROJECT6_V4_FINAL.ps1",
        root / "sewerrtc" / "simulation" / "pyswmm_runner.py",
        root / "sewerrtc" / "simulation" / "kpi_metrics.py",
        root / "sewerrtc" / "control" / "v4_candidate_generator.py",
    ]
    return sorted(p for p in paths if p.exists())


def _git(root: Path, args: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode, (result.stdout or "") + (result.stderr or "")
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, f"{type(exc).__name__}: {exc}"


def build_manifest(root: Path) -> dict:
    head_code, head_out = _git(root, ["rev-parse", "HEAD"])
    has_commit = head_code == 0 and head_out.strip() and "fatal" not in head_out
    _, status_out = _git(root, ["status", "--porcelain"])
    status_lines = [ln for ln in status_out.splitlines() if ln.strip()]
    untracked = [ln[3:] for ln in status_lines if ln.startswith("??")]

    files = []
    for path in _source_paths(root):
        stat = path.stat()
        files.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": _sha_file(path),
                "bytes": int(stat.st_size),
                "mtime": float(stat.st_mtime),
            }
        )
    return {
        "manifest": "PROJECT6_V4_CODE_SOURCE_MANIFEST",
        "generated_at": time.time(),
        "working_code_sha": working_code_sha(root),
        "git_has_commit": bool(has_commit),
        "git_head": head_out.strip() if has_commit else None,
        "git_head_note": (
            None if has_commit
            else "repository has no commit; working_code_sha pins dirty tree"
        ),
        "n_source_files": len(files),
        "source_files": files,
        "git_untracked_count": len(untracked),
        "git_untracked_files": untracked,
        "records_unrecorded_source_forbidden_for_formal": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    manifest = build_manifest(PROJECT_ROOT)

    target = output_root / MANIFEST_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest_path": str(target),
                "working_code_sha": manifest["working_code_sha"],
                "git_has_commit": manifest["git_has_commit"],
                "n_source_files": manifest["n_source_files"],
                "git_untracked_count": manifest["git_untracked_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
