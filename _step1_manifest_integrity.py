"""Step1 manifest integrity audit (Section 4).

Verifies:
- frame_count == 13
- frame_interval_min == 5
- history_start = anchor - 60
- history_end = anchor
- action_authority == actual_readback_setting
- reserved_evaluation == 0
- future_hydraulic_truth_in_input == false
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"E:\RTC_sewer\Project6")
S1 = ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat"


def main() -> int:
    manifest = pd.read_parquet(S1 / "dataset" / "step1_window_manifest.parquet")
    n = len(manifest)

    checks = {}

    # frame_count == 13
    checks["frame_count_all_13"] = bool((manifest["frame_count"] == 13).all())

    # frame_interval_min == 5
    checks["frame_interval_all_5"] = bool((manifest["frame_interval_min"] == 5).all())

    # history_start = anchor - 60
    checks["history_start_correct"] = bool(
        (manifest["history_start_min"] == manifest["anchor_min"] - 60).all()
    )

    # history_end = anchor
    checks["history_end_correct"] = bool(
        (manifest["history_end_min"] == manifest["anchor_min"]).all()
    )

    # action_authority == actual_readback_setting
    aa = manifest["action_authority"].astype(str)
    checks["action_authority_actual_readback"] = bool(
        (aa == "actual_readback_setting").all()
    )

    # requested_action_fallback_allowed
    rafa = manifest["requested_action_fallback_allowed"].fillna(False)
    checks["requested_action_fallback_allowed_count"] = int(rafa.sum())

    # future_hydraulic_truth_in_input == false
    fhti = manifest["future_hydraulic_truth_in_input"].fillna(False).astype(bool)
    checks["future_hydraulic_truth_in_input_count"] = int(fhti.sum())
    checks["future_hydraulic_truth_in_input_false"] = not fhti.any()

    # reserved_evaluation == 0
    if "source_role" in manifest.columns:
        sr = manifest["source_role"].astype(str)
        checks["reserved_evaluation_count"] = int((sr == "reserved_evaluation").sum())
        checks["reserved_evaluation_zero"] = (sr == "reserved_evaluation").sum() == 0
    else:
        checks["reserved_evaluation_count"] = 0
        checks["reserved_evaluation_zero"] = True

    # Domain role distribution
    if "step1_domain_role" in manifest.columns:
        role_counts = manifest["step1_domain_role"].value_counts().to_dict()
    else:
        role_counts = {}

    report = {
        "contract": "PROJECT6_V42_STEP1_MANIFEST_INTEGRITY",
        "total_windows": n,
        "checks": checks,
        "domain_role_distribution": role_counts,
        "all_checks_passed": all(
            v for k, v in checks.items()
            if isinstance(v, bool)
        ),
    }

    (S1 / "dataset" / "step1_manifest_integrity.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
