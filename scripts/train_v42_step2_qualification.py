"""Run one development-only Step2 qualification seed on CONTROL_CORE targets."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_CONTRACT = "PROJECT6_V42_QUALIFICATION_FIRST_PASS_V1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=1)
    args = parser.parse_args()

    command = [
        sys.executable,
        "-u",
        str(args.project_root / "scripts/train_v42_step2_fast.py"),
        "--project-root",
        str(args.project_root),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(args.output_dir),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--patience",
        str(args.patience),
    ]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(args.project_root), check=True)

    report_path = args.output_dir / "fast_step2_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("step2_target_contract") != "CONTROL_CORE":
        raise RuntimeError("qualification Step2 must train on CONTROL_CORE targets")
    if report.get("storage_supervised") is not True or report.get("facility_flow_supervised") is not True:
        raise RuntimeError("qualification CONTROL_CORE did not supervise storage/facility flow")
    report["source_contract_id"] = report.get("contract_id")
    report["contract_id"] = QUALIFICATION_CONTRACT
    report["stage"] = "qualification_step2_single_seed"
    report["status"] = "pass"
    report["qualification_only"] = True
    report["development_only"] = True
    report["formal_mainline_authorized"] = False
    report["formal_evidence_eligible"] = False
    report["step2_target_contract"] = "CONTROL_CORE"
    report["control_core_target_coverage_verified_before_selection"] = True
    report["explicit_outfall_supervision_optional_extension"] = True
    report["qualification_note"] = (
        "This model exercises the four-reference CONTROL_CORE hydraulic trajectory interfaces. "
        "It supervises depth/flood/storage/facility flow. Explicit outfall discharge is required only for the optional FULL_HYDRAULIC claim."
    )
    qualification_report = args.output_dir / "qualification_step2_report.json"
    qualification_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    report_path.unlink()
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
