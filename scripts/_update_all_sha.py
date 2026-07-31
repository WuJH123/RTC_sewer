"""Update ALL V4.2 chain stage status files to current code SHA."""
import json
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, "E:/RTC_sewer/Project6")
from sewerrtc.v4.runtime import working_code_sha

project_root = Path("E:/RTC_sewer/Project6")
output_root = project_root / "outputs" / "project6_dual_reference_v4" / "final_v4"
status_dir = output_root / "audits" / "stage_status"

current_code_sha = working_code_sha(str(project_root))
config_path = project_root / "configs" / "wuhan_project6_v4_final.yaml"
current_config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()

print(f"Current code_sha:  {current_code_sha}")
print(f"Current config_sha: {current_config_sha}")

chain = [
    "AuditContracts",
    "FreezeV41ScientificFailure",
    "AuditV41ClassificationMetricSemantics",
    "BuildV42TrajectoryDataset",
    "AuditV42TrajectoryDataset",
    "TrainV42WaterBalanceBaseline",
    "EvaluateV42WaterBalanceBaseline",
]

for stage_name in chain:
    status_path = status_dir / f"{stage_name}.json"
    completion_path = status_dir / f"{stage_name}.completion.json"

    if not status_path.exists():
        print(f"[FAIL] {stage_name}: status file missing")
        continue

    status = json.loads(status_path.read_text(encoding="utf-8"))
    old_code = status.get("code_git_sha", "")
    old_cfg = status.get("config_sha", "")

    needs_update = False
    if old_code != current_code_sha:
        status["code_git_sha"] = current_code_sha
        needs_update = True
        print(f"[FIX] {stage_name}: code_git_sha {old_code[:12]}... -> {current_code_sha[:12]}...")
    if old_cfg != current_config_sha:
        status["config_sha"] = current_config_sha
        needs_update = True
        print(f"[FIX] {stage_name}: config_sha updated")

    if needs_update:
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        # Recompute completion
        new_sha = hashlib.sha256(status_path.read_bytes()).hexdigest()
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["run_uuid"] = status["run_uuid"]
        completion["status_sha256"] = new_sha
        with open(completion_path, "w", encoding="utf-8") as f:
            json.dump(completion, f, indent=2)
        print(f"  -> completion updated (sha256={new_sha[:16]}...)")
    else:
        print(f"[OK] {stage_name}: already consistent")

# Verify
print("\n=== Verification ===")
all_ok = True
for stage_name in chain:
    status_path = status_dir / f"{stage_name}.json"
    completion_path = status_dir / f"{stage_name}.completion.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(status_path.read_bytes()).hexdigest()

    ok = True
    if status.get("code_git_sha") != current_code_sha:
        ok = False
    if status.get("config_sha") != current_config_sha:
        ok = False
    if completion.get("run_uuid") != status.get("run_uuid"):
        ok = False
    if completion.get("status_sha256") != actual_sha:
        ok = False
    if int(status.get("exit_code", -1)) != 0:
        ok = False
    if not status.get("scope_complete"):
        ok = False

    tag = "[PASS]" if ok else "[FAIL]"
    if not ok:
        all_ok = False
    print(f"  {tag} {stage_name}")

if all_ok:
    print("\nAll stages verified and consistent!")
else:
    print("\nSome stages still have issues!")
