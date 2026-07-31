from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract residual10 uncertainty calibration from a training report.")
    parser.add_argument("--report", default="outputs/models_hierarchical_residual10_h120_v2/raw_joint_residual10_core_h120_v1_train_report.json")
    parser.add_argument("--out-json", default="outputs/models_hierarchical_residual10_h120_v2/raw_joint_residual10_core_h120_v1_conformal_uncertainty.json")
    args = parser.parse_args()
    report_path = Path(args.report)
    out_path = Path(args.out_json)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    calibration = report.get("uncertainty_calibration") or report.get("calibration") or {}
    payload = {
        "source_report": str(report_path),
        "validation_gate_passed": bool(report.get("validation_gate_passed", False)),
        "rolling_horizon_smoke_eligibility": bool((report.get("rolling_horizon_smoke_eligibility") or {}).get("passed", False)),
        "uncertainty_calibration": calibration,
        "metrics": {
            "uncertainty_90pct_coverage": (report.get("metrics") or {}).get("uncertainty_90pct_coverage", {}),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out_json": str(out_path), "has_calibration": bool(calibration)}, indent=2))


if __name__ == "__main__":
    main()
