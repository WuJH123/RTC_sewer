"""FreezeV42BrokenTrainingEvidence — archive old broken V4.2 training results.

This script copies the old (broken) V4.2 training checkpoints, histories,
audit results, and metadata into an immutable frozen archive.  The frozen
evidence must NOT be modified or overwritten after repair.

Archive location:
    outputs/project6_dual_reference_v4/final_v4/audits/frozen_evidence/
        v42_broken_training/<code_sha>/

Manifest fields:
    model_version = v4.2_broken_v0
    mechanical_defects_confirmed = true
    predictive_metrics_scientifically_invalid = true
    kpi_head_near_constant = true
    old_results_not_reusable_as_repaired_results = true
    immutable = true
    overwrite_prohibited = true
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_directory(directory: Path) -> str:
    """Compute a deterministic SHA over all files in a directory."""
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            rel = p.relative_to(directory)
            h.update(str(rel).encode())
            h.update(_sha256_file(p).encode())
    return h.hexdigest()


def freeze_v42_broken_training(
    output_root: Path | None = None,
) -> dict:
    """Freeze old broken V4.2 training evidence.

    Returns the manifest dict.
    """
    output_root = Path(output_root or PROJECT_ROOT / "outputs" /
                       "project6_dual_reference_v4" / "final_v4")
    model_dir = output_root / "models" / "v42_twin"
    audits_dir = output_root / "audits"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Compute code SHA (over sewerrtc/v4/models_v42/ directory)
    code_dir = PROJECT_ROOT / "sewerrtc" / "v4" / "models_v42"
    code_sha = _sha256_directory(code_dir) if code_dir.exists() else "unknown"

    # Frozen archive destination
    frozen_dir = (
        audits_dir / "frozen_evidence" / "v42_broken_training" / code_sha[:16]
    )
    if frozen_dir.exists():
        print(f"Frozen archive already exists: {frozen_dir}")
        print("REFUSING to overwrite (immutable=true, overwrite_prohibited=true)")
        with open(frozen_dir / "manifest.json") as f:
            return json.load(f)

    frozen_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy old checkpoints
    ckpt_dir = frozen_dir / "checkpoints"
    ckpt_dir.mkdir()
    for pt in sorted(model_dir.glob("v42_twin_model_seed*_fold*.pt")):
        shutil.copy2(pt, ckpt_dir / pt.name)
    n_ckpts = len(list(ckpt_dir.glob("*.pt")))

    # 2. Copy training histories
    hist_dir = frozen_dir / "histories"
    hist_dir.mkdir()
    for j in sorted(model_dir.glob("training_history_seed*_fold*.json")):
        shutil.copy2(j, hist_dir / j.name)
    # Also copy summary files
    for summary_name in ("training_history.json", "cv_metrics.json"):
        src = model_dir / summary_name
        if src.exists():
            shutil.copy2(src, frozen_dir / summary_name)
    n_hists = len(list(hist_dir.glob("*.json")))

    # 3. Copy audit results
    audit_dir = frozen_dir / "audits"
    audit_dir.mkdir()
    for audit_subdir in ("v42_head_activation", "v42_metric_semantics",
                         "v42_ranking_physics"):
        src = audits_dir / audit_subdir
        if src.exists():
            dst = audit_dir / audit_subdir
            shutil.copytree(src, dst)

    # 4. Compute SHA of trajectory dataset
    traj_dir = output_root / "v42" / "trajectory_dataset"
    traj_sha = "unknown"
    if traj_dir.exists():
        traj_sha = _sha256_directory(traj_dir)

    # 5. Read old cv_metrics summary
    cv_path = model_dir / "cv_metrics.json"
    cv_summary = {}
    if cv_path.exists():
        with open(cv_path) as f:
            cv_data = json.load(f)
        agg = cv_data.get("aggregate", {})
        cv_summary = {k: v for k, v in agg.items() if "r2" in k.lower()}

    # 6. Read head activation summary
    ha_path = audits_dir / "v42_head_activation" / "head_activation_summary.json"
    ha_summary = {}
    if ha_path.exists():
        with open(ha_path) as f:
            ha_summary = json.load(f)

    # 7. Build manifest
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": code_sha,
        "code_sha_short": code_sha[:16],
        "trajectory_dataset_sha256": traj_sha,
        "model_version": "v4.2_broken_v0",
        "mechanical_defects_confirmed": True,
        "predictive_metrics_scientifically_invalid": True,
        "kpi_head_near_constant": True,
        "old_results_not_reusable_as_repaired_results": True,
        "immutable": True,
        "overwrite_prohibited": True,
        "n_checkpoints_frozen": n_ckpts,
        "n_training_histories_frozen": n_hists,
        "cv_metrics_summary": cv_summary,
        "head_activation_summary": ha_summary,
        "defects": [
            "P0-1: actuator_action_encoder dead code (einsum path)",
            "P0-2: ranking hinge loss → zero gradient on correct side",
            "P0/P1-4: physics kpi_trajectory_consistency unit mismatch",
            "P1-6: KPI head near-constant (pfv_delta std≈4e-3)",
        ],
        "frozen_path": str(frozen_dir),
    }

    manifest_path = frozen_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Frozen {n_ckpts} checkpoints, {n_hists} histories")
    print(f"Archive: {frozen_dir}")
    print(f"Code SHA: {code_sha[:16]}")
    print(f"Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    manifest = freeze_v42_broken_training()
    print(json.dumps(manifest, indent=2))
