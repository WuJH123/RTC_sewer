"""Compile Formal F2 Step3 PFV-first MPC evidence from real execution audit.

No caller can assert engineering legality by setting booleans here. The supplied
authoritative audit must come from executed/readback schedules and prove the
Engineering36/H12/K<=8 contract plus calibrated GAT/surrogate lineage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID, sha256_file
from sewerrtc.v4.paper_workflow_v42 import CONTRACT_ID


def _json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument(
        "--engineering-audit",
        type=Path,
        required=True,
        help="Authoritative executed/readback MPC audit produced on F2 calibration/validation runs.",
    )
    ap.add_argument(
        "--paper-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper",
    )
    args = ap.parse_args()

    step1 = _json(args.paper_root / "step1_gat/evidence.json")
    step2 = _json(args.paper_root / "step2_surrogate/evidence.json")
    audit = _json(args.engineering_audit)
    required_true = (
        "engineering_status_derived_from_execution",
        "changed_facilities_derived_from_executed_action",
        "readback_verified",
        "bounds_pass",
        "binary_pass",
        "rate_pass",
        "ramp_pass",
        "dwell_pass",
        "interlock_pass",
        "adaptive_k_pass",
    )
    missing = [key for key in required_true if audit.get(key) is not True]
    if missing:
        raise RuntimeError(f"authoritative Step3 execution audit is not all-pass: {missing}")
    if int(audit.get("facility_count", 0)) != 36:
        raise RuntimeError("Step3 execution audit does not prove Engineering36")
    if int(audit.get("horizon_steps", 0)) != 12:
        raise RuntimeError("Step3 execution audit does not prove H12")
    if int(audit.get("max_changed_facilities", 99)) > 8:
        raise RuntimeError("Step3 execution audit exceeds K<=8")
    if audit.get("pfv_reference") != "no_control" or audit.get("peak_reference") != "dynamic_internal" or audit.get("tfv_reference") != "dynamic_internal":
        raise RuntimeError("Step3 execution audit reference contract mismatch")
    if audit.get("uses_future_swmm_truth_online") is not False:
        raise RuntimeError("Step3 execution audit indicates future SWMM truth leakage")
    if audit.get("gat_model_sha256") != step1.get("gat_model_sha256"):
        raise RuntimeError("Step3 GAT hash does not match Formal Step1 evidence")
    if audit.get("surrogate_model_sha256") != step2.get("surrogate_model_sha256"):
        raise RuntimeError("Step3 surrogate hash does not match Formal Step2 evidence")
    payload = {
        "contract_id": CONTRACT_ID,
        "stage": "step3_pfvfirst_mpc",
        "status": "pass",
        "development_only": False,
        "formal_generation_id": FORMAL_GENERATION_ID,
        "selector": "decide_pfvfirst_mpc",
        "pfv_reference": "no_control",
        "peak_reference": "dynamic_internal",
        "tfv_reference": "dynamic_internal",
        "max_changed_facilities": 8,
        "horizon_steps": 12,
        "controllable_prefix_steps": 3,
        "execute_steps": 1,
        "facility_count": 36,
        "tfv_is_hard_safety_constraint": False,
        "engineering_status_derived_from_execution": True,
        "changed_facilities_derived_from_executed_action": True,
        "readback_verified": True,
        "uncertainty_and_ood_linked_to_calibrated_models": True,
        "gat_model_sha256": step1["gat_model_sha256"],
        "surrogate_model_sha256": step2["surrogate_model_sha256"],
        "safety_calibration_sha256": step2.get("safety_calibration_sha256"),
        "confidence_z": step2.get("confidence_z"),
        "uncertainty_limit": step2.get("uncertainty_limit"),
        "engineering_audit_sha256": sha256_file(args.engineering_audit),
        "uses_future_swmm_truth_online": False,
    }
    out = args.paper_root / "step3_mpc/evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
