# -*- coding: utf-8 -*-
"""Freeze Pilot Dataset v2 evidence into audits/frozen_evidence/pilot_v2/<code_sha>/.

Read-only on all source artifacts.  Fail-closed: refuses to run if the freeze
root already contains pilot_v2_freeze.json.  Physical copies cover every small
evidence file (dataset_v2 manifests/audit/baseline report, gate v2 verdict,
extension + flat auxiliary planning/run manifests/datasets/gap catalog, all
sample completions and branch detail files, v2 stage status stamps); large
case.inp files are frozen by SHA256 manifest reference only.

The v2 verdict (scientific_fail) is immutable.  This freeze precedes Gate P3
feasibility mapping and must never be re-marked PASS.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

FINAL_V4 = Path(r"e:\RTC_sewer\Project6\outputs\project6_dual_reference_v4\final_v4")
DATASET_V2 = FINAL_V4 / "pilot" / "dataset_v2"
EVALUATION = FINAL_V4 / "pilot" / "evaluation"
EXTENSION = FINAL_V4 / "pilot_extension_v1"
FLAT_AUX = EXTENSION / "flat_auxiliary"
STAGE_STATUS = FINAL_V4 / "audits" / "stage_status"
EXPECTED_CODE_SHA = "9b21f44197c6e2176e3f2858bbe0a36bd3ba267c57b2ac6c647638867a16712c"
# Raw run completions carry the plan-provenance code sha inherited from the
# immutable v1 candidate plan base rows (the sha the SWMM runs executed under);
# it legitimately differs from the current chain sha after later code fixes.
EXPECTED_RAW_PROVENANCE_SHA = "b470463eec155d056404996a81e2ecaed1f95820e127811bba818f8bbc30eca6"
V2_STAGES = (
    "PlanPilotCoverageExtension",
    "AuditPilotCoverageExtensionPlan",
    "AuditPilotCoverageExtensionPreflight",
    "RunPilotCoverageExtension",
    "BuildPilotCoverageExtensionPartial",
    "AuditPilotCoverageExtensionPartial",
    "BuildPilotCoverageExtensionDataset",
    "AuditPilotCoverageExtensionDataset",
    "PlanPilotFlatAuxiliary",
    "AuditPilotFlatAuxiliaryPreflight",
    "RunPilotFlatAuxiliary",
    "BuildPilotFlatAuxiliaryDataset",
    "AuditPilotFlatAuxiliaryDataset",
    "BuildPilotDatasetV2",
    "AuditPilotDatasetV2",
    "TrainPilotBaselinesV2",
    "EvaluatePilotGateV2",
)
HARD_CHECKS = (
    "same_state_100pct",
    "readback_100pct",
    "hard_authenticity_100pct",
    "actual_duplicates_0",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_preserving(src: Path, base: Path, dest_root: Path) -> Path:
    rel = src.relative_to(base)
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def main() -> int:
    eval_stamp = json.loads((STAGE_STATUS / "EvaluatePilotGateV2.json").read_text("utf-8"))
    audit_stamp = json.loads((STAGE_STATUS / "AuditPilotDatasetV2.json").read_text("utf-8"))
    code_sha = str(eval_stamp["code_git_sha"])
    if code_sha != EXPECTED_CODE_SHA:
        print(f"FAIL: code_git_sha mismatch: {code_sha}")
        return 2
    if str(audit_stamp["code_git_sha"]) != EXPECTED_CODE_SHA:
        print(f"FAIL: audit code sha mismatch: {audit_stamp['code_git_sha']}")
        return 2
    freeze_root = FINAL_V4 / "audits" / "frozen_evidence" / "pilot_v2" / code_sha
    freeze_json = freeze_root / "pilot_v2_freeze.json"
    if freeze_json.exists():
        print(f"FAIL-CLOSED: freeze already exists: {freeze_json}")
        return 2
    freeze_root.mkdir(parents=True, exist_ok=True)

    # --- verify the immutable scientific_fail verdict before archiving ---
    verdict = json.loads((EVALUATION / "pilot_gate_v2_verdict.json").read_text("utf-8"))
    if verdict["status"] != "scientific_fail" or eval_stamp["status"] != "scientific_fail":
        print(f"FAIL: unexpected verdict: {verdict['status']} / {eval_stamp['status']}")
        return 2
    dataset_audit = json.loads((DATASET_V2 / "pilot_v2_dataset_audit.json").read_text("utf-8"))
    checks = dataset_audit["checks"]
    hard_pass = all(bool(checks[k]) for k in HARD_CHECKS)
    if not hard_pass:
        print(f"FAIL: hard authenticity checks not all true: {checks}")
        return 2

    # --- recompute frozen headline numbers from the immutable manifest ---
    samples = pd.read_csv(DATASET_V2 / "pilot_v2_sample_manifest.csv")
    if len(samples) != 479:
        print(f"FAIL: sample manifest rows={len(samples)}")
        return 2
    phase_counts = samples["source_phase"].value_counts().to_dict()
    if (
        int(phase_counts.get("primary400", 0)) != 400
        or int(phase_counts.get("joint_extension", 0)) != 60
        or int(phase_counts.get("flat_auxiliary", 0)) != 19
    ):
        print(f"FAIL: source phase counts={phase_counts}")
        return 2
    flat = samples[samples["confirmed_flat"].astype(bool)]
    flat_count = int(len(flat))
    flat_event_support = int(flat["event_id"].nunique()) if flat_count else 0
    responsive = samples[samples["checkpoint_role"] == "responsive"]
    checkpoint_joint = responsive.groupby(["event_id", "checkpoint_id"])[
        "joint_noninferior"
    ].any()
    joint_state_count = int(checkpoint_joint.sum())
    joint_state_fraction = float(checkpoint_joint.mean())
    expected = (14, 1, 9, 0.28125)
    got = (flat_count, flat_event_support, joint_state_count, round(joint_state_fraction, 6))
    if got != expected:
        print(f"FAIL: recomputed headline mismatch expected={expected} got={got}")
        return 2

    # --- false-safe of the HGB joint classifier on locked validation ---
    baseline = json.loads((DATASET_V2 / "baseline_models_report_v2.json").read_text("utf-8"))
    false_safe_hgb = (
        baseline["classification"]["joint_noninferior"]["hist_gradient_boosting"]
        ["splits"]["pilot_validation"]["false_safe_rate"]
    )
    if round(float(false_safe_hgb), 3) != 0.667:
        print(f"FAIL: unexpected hgb false_safe: {false_safe_hgb}")
        return 2

    # --- physical copies (small evidence) ---
    copied: list[Path] = []
    for src in sorted(DATASET_V2.glob("*")):
        if src.is_file():
            copied.append(copy_preserving(src, FINAL_V4, freeze_root))
    copied.append(copy_preserving(EVALUATION / "pilot_gate_v2_verdict.json", FINAL_V4, freeze_root))
    for folder in (
        EXTENSION / "planning",
        EXTENSION / "dataset",
        EXTENSION / "gaps",
        FLAT_AUX / "planning",
        FLAT_AUX / "dataset",
    ):
        for src in sorted(folder.rglob("*")):
            if src.is_file():
                copied.append(copy_preserving(src, FINAL_V4, freeze_root))
    for name in ("run_manifest.csv",):
        for base in (EXTENSION, FLAT_AUX):
            src = base / name
            if src.exists():
                copied.append(copy_preserving(src, FINAL_V4, freeze_root))
    for stage in V2_STAGES:
        for suffix in (".json", ".completion.json"):
            src = STAGE_STATUS / f"{stage}{suffix}"
            if src.exists():
                dest = freeze_root / "stage_status" / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                copied.append(dest)

    # 60 extension + 19 flat auxiliary completions and branch detail files;
    # verify uniform code sha across every raw completion.
    code_shas: set[str] = set()
    run_completions = {"extension": 0, "flat_auxiliary": 0}
    for key, runs_root in (("extension", EXTENSION / "runs"), ("flat_auxiliary", FLAT_AUX / "runs")):
        for comp in sorted(runs_root.glob("*/completion.json")):
            payload = json.loads(comp.read_text("utf-8"))
            value = payload.get("code_git_sha") or payload.get("code_sha256")
            if value:
                code_shas.add(str(value))
            copied.append(copy_preserving(comp, FINAL_V4, freeze_root))
            run_completions[key] += 1
        for detail in sorted(runs_root.glob("*/*/detail.csv")):
            copied.append(copy_preserving(detail, FINAL_V4, freeze_root))
    if run_completions != {"extension": 60, "flat_auxiliary": 19}:
        print(f"FAIL: run completions={run_completions}")
        return 2
    if code_shas != {EXPECTED_RAW_PROVENANCE_SHA}:
        print(f"FAIL: non-uniform raw provenance sha in completions: {sorted(code_shas)}")
        return 2

    # --- full SHA256 manifest over every v2 artifact (incl. case.inp) ---
    manifest_path = freeze_root / "pilot_v2_sha_manifest.csv"
    manifest_rows = 0
    total_bytes = 0
    started = time.time()
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        roots = (DATASET_V2, EVALUATION, EXTENSION)
        for root in roots:
            for src in sorted(root.rglob("*")):
                if not src.is_file():
                    continue
                size = src.stat().st_size
                writer.writerow([src.relative_to(FINAL_V4).as_posix(), size, sha256_file(src)])
                manifest_rows += 1
                total_bytes += size
    manifest_sha = sha256_file(manifest_path)
    print(f"manifest rows={manifest_rows} bytes={total_bytes} elapsed={time.time() - started:.1f}s")

    freeze_payload = {
        "freeze_name": "pilot_v2",
        "code_sha": code_sha,
        "config_sha": eval_stamp["config_sha"],
        "verdict": "scientific_fail",
        "hard_authenticity_pass": bool(hard_pass),
        "gate_checks": verdict["checks"],
        "dataset_checks": checks,
        "failed_dataset_checks": sorted(k for k, v in checks.items() if not v),
        "sample_count": int(len(samples)),
        "samples_by_source_phase": {str(k): int(v) for k, v in phase_counts.items()},
        "joint_state_count": joint_state_count,
        "joint_state_fraction": joint_state_fraction,
        "flat_count": flat_count,
        "flat_event_support": flat_event_support,
        "false_safe_hgb": float(false_safe_hgb),
        "immutable": True,
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_roots": [str(DATASET_V2), str(EVALUATION), str(EXTENSION)],
        "raw_run_provenance_code_sha": EXPECTED_RAW_PROVENANCE_SHA,
        "gate_eval_run_uuid": eval_stamp["run_uuid"],
        "gate_eval_input_sha": eval_stamp["input_sha"],
        "dataset_audit_run_uuid": audit_stamp["run_uuid"],
        "dataset_audit_input_sha": audit_stamp["input_sha"],
        "archive": {
            "copied_files": len(copied),
            "extension_run_completions": run_completions["extension"],
            "flat_auxiliary_run_completions": run_completions["flat_auxiliary"],
            "sha_manifest": manifest_path.name,
            "sha_manifest_rows": manifest_rows,
            "sha_manifest_total_bytes": total_bytes,
            "sha_manifest_sha256": manifest_sha,
            "large_files_policy": "case.inp frozen by sha manifest reference; all completions, plans, manifests, detail/result files physically copied",
        },
        "notes": "Pilot v2 verdict is immutable scientific_fail; Gate P3 feasibility mapping never rewrites v1/v2 evidence and never re-marks v2 as PASS.",
    }
    freeze_json.write_text(json.dumps(freeze_payload, indent=2), encoding="utf-8")
    print(f"FROZEN: {freeze_json}")
    print(f"copied={len(copied)} manifest_rows={manifest_rows} manifest_sha={manifest_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
