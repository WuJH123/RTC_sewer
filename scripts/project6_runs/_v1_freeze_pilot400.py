# -*- coding: utf-8 -*-
"""Freeze Pilot400 v1 evidence into audits/frozen_evidence/pilot400_v1/<code_sha>/.

Read-only on all source artifacts.  Fail-closed: refuses to run if the freeze
root already contains pilot400_v1_freeze.json.  Physical copies cover every
small evidence file (planning, dataset, stage status, sample completions,
reference completions, branch detail/result files); large case.inp files are
frozen by SHA256 manifest reference only.
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
PILOT = FINAL_V4 / "pilot"
STAGE_STATUS = FINAL_V4 / "audits" / "stage_status"
EXPECTED_CODE_SHA = "b470463eec155d056404996a81e2ecaed1f95820e127811bba818f8bbc30eca6"
PILOT_STAGES = (
    "PlanPilot400",
    "AuditPilotPlan",
    "AuditPilotPreflight",
    "RunPilot400",
    "BuildPilotPartial",
    "AuditPilotPartial",
    "BuildPilotDataset",
    "AuditPilotDataset",
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
    audit_stamp = json.loads((STAGE_STATUS / "AuditPilotDataset.json").read_text("utf-8"))
    code_sha = str(audit_stamp["code_git_sha"])
    if code_sha != EXPECTED_CODE_SHA:
        print(f"FAIL: code_git_sha mismatch: {code_sha}")
        return 2
    freeze_root = FINAL_V4 / "audits" / "frozen_evidence" / "pilot400_v1" / code_sha
    freeze_json = freeze_root / "pilot400_v1_freeze.json"
    if freeze_json.exists():
        print(f"FAIL-CLOSED: freeze already exists: {freeze_json}")
        return 2
    freeze_root.mkdir(parents=True, exist_ok=True)

    # --- verify verdict is the frozen scientific_fail before archiving ---
    checks = audit_stamp["evidence"]["checks"]
    failed_checks = sorted(k for k, v in checks.items() if not v)
    if audit_stamp["status"] != "scientific_fail" or failed_checks != [
        "flat_fraction_10_to_20pct",
        "joint_at_30pct_responsive_checkpoints",
    ]:
        print(f"FAIL: unexpected audit verdict: {audit_stamp['status']} {failed_checks}")
        return 2
    hard_checks = ("same_state_100pct", "readback_100pct", "actual_duplicates_0")
    hard_pass = all(bool(checks[k]) for k in hard_checks)

    # --- recompute frozen headline numbers from the immutable manifest ---
    samples = pd.read_csv(PILOT / "dataset" / "pilot_sample_manifest.csv")
    if len(samples) != 400:
        print(f"FAIL: sample manifest rows={len(samples)}")
        return 2
    flat_count = int(samples["confirmed_flat"].astype(bool).sum())
    flat_fraction = float(samples["confirmed_flat"].astype(bool).mean())
    responsive = samples[samples["checkpoint_role"] == "responsive"]
    checkpoint_joint = responsive.groupby(["event_id", "checkpoint_id"])[
        "joint_noninferior"
    ].any()
    joint_state_count = int(checkpoint_joint.sum())
    joint_state_fraction = float(checkpoint_joint.mean())
    expected = (10, 0.025, 9, 0.28125)
    got = (flat_count, round(flat_fraction, 6), joint_state_count, round(joint_state_fraction, 6))
    if got != expected:
        print(f"FAIL: recomputed headline mismatch expected={expected} got={got}")
        return 2

    # --- physical copies (small evidence) ---
    copied: list[Path] = []
    for src in sorted((PILOT / "planning").glob("*")):
        if src.is_file():
            copied.append(copy_preserving(src, PILOT, freeze_root))
    for src in sorted((PILOT / "dataset").glob("*")):
        if src.is_file():
            copied.append(copy_preserving(src, PILOT, freeze_root))
    for stage in PILOT_STAGES:
        for suffix in (".json", ".completion.json"):
            src = STAGE_STATUS / f"{stage}{suffix}"
            if src.exists():
                dest = freeze_root / "stage_status" / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                copied.append(dest)

    # 400 sample completions + branch detail files; verify uniform code sha
    code_shas: set[str] = set()
    sample_completions = 0
    for comp in sorted((PILOT / "runs").glob("*/completion.json")):
        payload = json.loads(comp.read_text("utf-8"))
        value = payload.get("code_git_sha") or payload.get("code_sha256")
        if value:
            code_shas.add(str(value))
        copied.append(copy_preserving(comp, PILOT, freeze_root))
        sample_completions += 1
    for detail in sorted((PILOT / "runs").glob("*/*/detail.csv")):
        copied.append(copy_preserving(detail, PILOT, freeze_root))
    if sample_completions != 400:
        print(f"FAIL: sample completions={sample_completions}")
        return 2
    if code_shas and code_shas != {EXPECTED_CODE_SHA}:
        print(f"FAIL: non-uniform code sha in completions: {sorted(code_shas)}")
        return 2

    # reference completions + branch detail/result files
    ref_completions = 0
    for comp in sorted((PILOT / "references").glob("*/*/reference_completion.json")):
        copied.append(copy_preserving(comp, PILOT, freeze_root))
        ref_completions += 1
    for pattern in ("*/*/*/detail.csv", "*/*/*/result.json"):
        for src in sorted((PILOT / "references").glob(pattern)):
            copied.append(copy_preserving(src, PILOT, freeze_root))
    if ref_completions != 40:
        print(f"FAIL: reference completions={ref_completions}")
        return 2

    # --- full SHA256 manifest over every pilot artifact (incl. case.inp) ---
    manifest_path = freeze_root / "pilot400_v1_sha_manifest.csv"
    manifest_rows = 0
    total_bytes = 0
    started = time.time()
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        for src in sorted(PILOT.rglob("*")):
            if not src.is_file():
                continue
            size = src.stat().st_size
            writer.writerow([src.relative_to(PILOT).as_posix(), size, sha256_file(src)])
            manifest_rows += 1
            total_bytes += size
    manifest_sha = sha256_file(manifest_path)
    print(f"manifest rows={manifest_rows} bytes={total_bytes} elapsed={time.time() - started:.1f}s")

    freeze_payload = {
        "freeze_name": "pilot400_v1",
        "code_sha": code_sha,
        "config_sha": audit_stamp["config_sha"],
        "status": "scientific_fail",
        "hard_authenticity_pass": bool(hard_pass),
        "failed_checks": failed_checks,
        "checks": checks,
        "flat_count": flat_count,
        "flat_fraction": flat_fraction,
        "joint_state_count": joint_state_count,
        "joint_state_fraction": joint_state_fraction,
        "sample_count": int(len(samples)),
        "immutable": True,
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_root": str(PILOT),
        "audit_run_uuid": audit_stamp["run_uuid"],
        "audit_input_sha": audit_stamp["input_sha"],
        "archive": {
            "copied_files": len(copied),
            "sample_completions": sample_completions,
            "reference_completions": ref_completions,
            "sha_manifest": manifest_path.name,
            "sha_manifest_rows": manifest_rows,
            "sha_manifest_total_bytes": total_bytes,
            "sha_manifest_sha256": manifest_sha,
            "large_files_policy": "case.inp frozen by sha manifest reference; all completions, plans, manifests, detail/result files physically copied",
        },
        "notes": "Pilot400 v1 verdict is immutable; v1 must never be re-marked PASS. Gate revision applies only to v2 (not_retroactive).",
    }
    freeze_json.write_text(json.dumps(freeze_payload, indent=2), encoding="utf-8")
    print(f"FROZEN: {freeze_json}")
    print(f"copied={len(copied)} manifest_rows={manifest_rows} manifest_sha={manifest_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
