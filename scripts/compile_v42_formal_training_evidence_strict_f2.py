"""Compile Formal Step1/Step2 paper evidence only after exact Calibration12."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_formal_strict import audit_calibration_completeness


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    ap.add_argument("--primary-seed", type=int, default=42)
    args = ap.parse_args()
    root = args.project_root.resolve()
    formal = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    audit = audit_calibration_completeness(formal)
    gate_path = formal / "calibration/FORMAL_F2_CALIBRATION12_GATE.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    if audit["status"] != "pass":
        print(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
        return 3
    cmd = [
        str(Path(sys.executable)),
        "-u",
        str(root / "scripts/compile_v42_formal_training_evidence_f2.py"),
        "--project-root",
        str(root),
        "--formal-root",
        str(formal),
        "--paper-root",
        str(root / "outputs/project6_dual_reference_v4/final_v4/v42_paper"),
        "--seeds",
        *[str(x) for x in args.seeds],
        "--primary-seed",
        str(args.primary_seed),
    ]
    subprocess.run(cmd, cwd=str(root), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
