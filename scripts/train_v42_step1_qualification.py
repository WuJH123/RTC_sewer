"""Run one Temporal Sparse GAT qualification seed without creating Formal evidence."""
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
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--sensor-layout-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min-train-groups", type=int, default=65)
    args = parser.parse_args()

    command = [
        sys.executable,
        "-u",
        str(args.project_root / "scripts/train_v42_step1_formal_f2.py"),
        "--project-root",
        str(args.project_root),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(args.output_dir),
        "--model-seed",
        str(args.model_seed),
        "--split-seed",
        str(args.split_seed),
        "--sensor-layout-seed",
        str(args.sensor_layout_seed),
        "--epochs",
        str(args.epochs),
        "--aux-epochs",
        "0",
        "--batch-size",
        str(args.batch_size),
        "--patience",
        str(args.patience),
        "--min-train-groups",
        str(args.min_train_groups),
    ]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(args.project_root), check=True)

    report_path = args.output_dir / "formal_step1_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_formal_generation_id"] = report.get("formal_generation_id")
    report["formal_generation_id"] = QUALIFICATION_CONTRACT
    report["contract_id"] = QUALIFICATION_CONTRACT
    report["stage"] = "qualification_step1_single_seed"
    report["qualification_only"] = True
    report["development_only"] = True
    report["formal_mainline_authorized"] = False
    report["formal_evidence_eligible"] = False
    report["qualification_note"] = (
        "One-epoch qualification model used only to exercise downstream interfaces. "
        "It cannot substitute for Formal production training."
    )
    qualification_report = args.output_dir / "qualification_step1_report.json"
    qualification_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    # Remove the misleading formal-named report from the isolated qualification root.
    report_path.unlink()
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
