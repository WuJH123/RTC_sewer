"""Run Formal F2 calibration and require exact frozen Calibration12.

The low-level Step1/Step2 calibrators keep their smaller ``--min`` arguments for
diagnostic reuse, but they cannot authorize the paper line through this entry
point unless both reports contain exactly the 12 rainfall SHA values frozen in
the current Formal ledger.
"""
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
    ap.add_argument("--primary-step1-seed", type=int, default=42)
    ap.add_argument("--sensor-layout-seed", type=int, default=42)
    args = ap.parse_args()
    root = args.project_root.resolve()
    cmd = [
        str(Path(sys.executable)),
        "-u",
        str(root / "scripts/run_v42_formal_f2.py"),
        "--stage",
        "calibration",
        "--seeds",
        *[str(x) for x in args.seeds],
        "--primary-step1-seed",
        str(args.primary_step1_seed),
        "--sensor-layout-seed",
        str(args.sensor_layout_seed),
    ]
    subprocess.run(cmd, cwd=str(root), check=True)
    formal = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    audit = audit_calibration_completeness(formal)
    out = formal / "calibration/FORMAL_F2_CALIBRATION12_GATE.json"
    out.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if audit["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
