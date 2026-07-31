"""Verify the full V4.2 prerequisite chain integrity."""
import json
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, "E:/RTC_sewer/Project6")
from sewerrtc.v4.runtime import working_code_sha

output_root = Path("E:/RTC_sewer/Project6/outputs/project6_dual_reference_v4/final_v4")
status_dir = output_root / "audits" / "stage_status"
project_root = Path("E:/RTC_sewer/Project6")

current_code_sha = working_code_sha(str(project_root))
config_path = Path("E:/RTC_sewer/Project6/configs/wuhan_project6_v4_final.yaml")
current_config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()

print(f"Current code_sha:  {current_code_sha}")
print(f"Current config_sha: {current_config_sha}")
print("=" * 80)

# Full prerequisite chain for TrainV42WaterBalanceBaseline
chain = [
    "AuditContracts",
    "FreezeV41ScientificFailure",
    "AuditV41ClassificationMetricSemantics",
    "BuildV42TrajectoryDataset",
    "AuditV42TrajectoryDataset",
    "TrainV42WaterBalanceBaseline",
    "EvaluateV42WaterBalanceBaseline",
]

# Evidence paths from STAGE_EVIDENCE_MAP
evidence_map = {
    "FreezeV41ScientificFailure": "audits/frozen_evidence/v41_scientific_failure/v41_freeze_manifest.json",
    "AuditV41ClassificationMetricSemantics": "audits/v42_metric_semantics/v41_metric_semantics_audit.json",
    "BuildV42TrajectoryDataset": "train1600_v3/trajectory_manifest_v42.parquet",
    "AuditV42TrajectoryDataset": "train1600_v3/trajectory_dataset_audit.json",
    "TrainV42WaterBalanceBaseline": "models/v42_water_balance/water_balance_baseline_cv.json",
    "EvaluateV42WaterBalanceBaseline": "models/v42_water_balance/water_balance_evaluation.json",
}

issues = []
for stage_name in chain:
    print(f"\n--- {stage_name} ---")
    status_path = status_dir / f"{stage_name}.json"
    completion_path = status_dir / f"{stage_name}.completion.json"

    if not status_path.exists():
        issues.append(f"{stage_name}: STATUS FILE MISSING")
        print("  [FAIL] STATUS FILE MISSING")
        continue
    if not completion_path.exists():
        issues.append(f"{stage_name}: COMPLETION FILE MISSING")
        print("  [FAIL] COMPLETION FILE MISSING")
        continue

    status = json.loads(status_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))

    # Check exit_code
    exit_code = int(status.get("exit_code", -1))
    if exit_code != 0:
        issues.append(f"{stage_name}: exit_code={exit_code} (not 0)")
        print(f"  [FAIL] exit_code={exit_code}")
    else:
        print(f"  [PASS] exit_code=0")

    # Check scope_complete
    scope_complete = bool(status.get("scope_complete", False))
    if not scope_complete:
        issues.append(f"{stage_name}: scope_complete=False")
        print(f"  [FAIL] scope_complete=False")
    else:
        print(f"  [PASS] scope_complete=True")

    # Check completion_valid (run_uuid match + status_sha256 match)
    uuid_match = completion.get("run_uuid") == status.get("run_uuid")
    actual_sha = hashlib.sha256(status_path.read_bytes()).hexdigest()
    sha_match = completion.get("status_sha256") == actual_sha
    if not uuid_match:
        issues.append(f"{stage_name}: run_uuid mismatch (status={status.get('run_uuid')}, completion={completion.get('run_uuid')})")
        print(f"  [FAIL] run_uuid mismatch")
    else:
        print(f"  [PASS] run_uuid match")
    if not sha_match:
        issues.append(f"{stage_name}: status_sha256 mismatch (expected={completion.get('status_sha256')[:16]}..., actual={actual_sha[:16]}...)")
        print(f"  [FAIL] status_sha256 mismatch")
    else:
        print(f"  [PASS] status_sha256 match")

    # Check config_sha
    cfg_sha = status.get("config_sha", "")
    if cfg_sha != current_config_sha:
        issues.append(f"{stage_name}: config_sha mismatch ({cfg_sha[:16]}... != {current_config_sha[:16]}...)")
        print(f"  [FAIL] config_sha mismatch")
    else:
        print(f"  [PASS] config_sha match")

    # Check code_git_sha
    code_sha = status.get("code_git_sha", "")
    if code_sha != current_code_sha:
        issues.append(f"{stage_name}: code_git_sha MISMATCH ({code_sha[:16]}... != {current_code_sha[:16]}...)")
        print(f"  [FAIL] code_git_sha MISMATCH")
    else:
        print(f"  [PASS] code_git_sha match")

    # Check evidence file existence
    if stage_name in evidence_map:
        ev_path = output_root / evidence_map[stage_name]
        if ev_path.exists():
            print(f"  [PASS] evidence exists: {evidence_map[stage_name]}")
        else:
            issues.append(f"{stage_name}: evidence file missing: {ev_path}")
            print(f"  [FAIL] evidence MISSING: {evidence_map[stage_name]}")

print("\n" + "=" * 80)
if issues:
    print(f"\nWARNING  FOUND {len(issues)} ISSUES:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n[PASS] ALL CHECKS PASSED - Chain integrity verified!")
