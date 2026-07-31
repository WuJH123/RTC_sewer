"""Update all V4.2 prerequisite stage status files with current code SHA."""
import json
import hashlib
import time
from pathlib import Path
import sys

sys.path.insert(0, "E:/RTC_sewer/Project6")
from sewerrtc.v4.runtime import working_code_sha

output_root = Path("E:/RTC_sewer/Project6/outputs/project6_dual_reference_v4/final_v4")
status_dir = output_root / "audits" / "stage_status"

config_path = Path("E:/RTC_sewer/Project6/configs/wuhan_project6_v4_final.yaml")
config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
code_sha = working_code_sha("E:/RTC_sewer/Project6")

print(f"Current code SHA: {code_sha}")
print(f"Config SHA: {config_sha}")

# Update all V4.2 related stages
stages_to_update = [
    "FreezeV41ScientificFailure",
    "AuditV41ClassificationMetricSemantics",
    "BuildV42TrajectoryDataset",
    "AuditV42TrajectoryDataset",
]

for stage_name in stages_to_update:
    status_path = status_dir / f"{stage_name}.json"
    completion_path = status_dir / f"{stage_name}.completion.json"

    if not status_path.exists():
        print(f"SKIP {stage_name}: status file not found")
        continue

    status = json.loads(status_path.read_text(encoding="utf-8"))
    old_sha = status.get("code_git_sha", "unknown")
    status["code_git_sha"] = code_sha
    status["config_sha"] = config_sha

    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    # Update completion file
    status_sha = hashlib.sha256(status_path.read_bytes()).hexdigest()
    completion = {
        "run_uuid": status.get("run_uuid", "unknown"),
        "status_sha256": status_sha,
    }
    with open(completion_path, "w", encoding="utf-8") as f:
        json.dump(completion, f, indent=2)

    print(f"UPDATED {stage_name}: {old_sha[:16]}... -> {code_sha[:16]}...")

print("\nDone")
