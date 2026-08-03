"""Compile formal Step1/Step2 evidence only after real F2 calibration passes.

This script is intentionally unable to turn development artifacts into paper
evidence. It requires three-seed formal training, new-rainfall calibration,
causal GAT history, raw four-reference admission, full formal hydraulic target
supervision and zero rainfall overlap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID, read_table, sha256_file
from sewerrtc.v4.paper_workflow_v42 import CONTRACT_ID


FULL_STEP2_SUPERVISION_FLAGS = (
    "storage_supervised",
    "facility_flow_supervised",
    "outfall_supervised",
)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def _combined_model_hash(reports: list[dict], key: str) -> str:
    values = sorted(str(r.get(key, "")) for r in reports if str(r.get(key, "")))
    if len(values) != len(reports):
        raise RuntimeError(f"missing {key} in one or more formal model reports")
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _assert_same_split(reports: list[dict], keys: tuple[str, ...]) -> None:
    for key in keys:
        values = {tuple(map(str, r.get(key, []))) for r in reports}
        if len(values) != 1:
            raise RuntimeError(f"formal model seeds do not share frozen {key}")


def _assert_full_step2_target_supervision(reports: list[dict]) -> None:
    missing: dict[str, list[int]] = {}
    for flag in FULL_STEP2_SUPERVISION_FLAGS:
        bad = [int(r.get("seed", -1)) for r in reports if r.get(flag) is not True]
        if bad:
            missing[flag] = bad
    if missing:
        raise RuntimeError(
            "Formal Step2 cannot authorize paper evidence without full hydraulic "
            f"target supervision: {missing}. Missing targets must not be zero-filled; "
            "materialize authoritative storage/facility/outfall targets first."
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument(
        "--formal-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2",
    )
    ap.add_argument(
        "--paper-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    ap.add_argument("--primary-seed", type=int, default=42)
    args = ap.parse_args()

    step1_reports = [_json(args.formal_root / "step1" / f"seed_{seed}" / "formal_step1_report.json") for seed in args.seeds]
    step2_reports = [_json(args.formal_root / "step2" / "models" / f"seed_{seed}" / "formal_step2_report.json") for seed in args.seeds]
    step1_cal = _json(args.formal_root / "calibration" / "STEP1_UNCERTAINTY_OOD_CALIBRATION.json")
    step2_cal = _json(args.formal_root / "calibration" / "STEP2_SAFETY_CALIBRATION.json")
    raw_audit = _json(args.formal_root / "step2" / "FORMAL_F2_STEP2_RAW_ADMISSION_AUDIT.json")
    gat_audit = _json(args.formal_root / "step2" / "FORMAL_F2_STEP2_GAT_HISTORY_AUDIT.json")
    gat_manifest = args.formal_root / "step2" / "FORMAL_F2_STEP2_GAT_MANIFEST.parquet"
    ledger = read_table(args.formal_root / "prepare" / "FORMAL_F2_EVENT_LEDGER.csv")

    if any(r.get("status") != "pass" for r in step1_reports + step2_reports):
        raise RuntimeError("one or more Formal F2 model seed reports are not pass")
    _assert_full_step2_target_supervision(step2_reports)
    if step1_cal.get("status") != "pass" or step1_cal.get("uncertainty_calibrated") is not True or step1_cal.get("ood_calibrated") is not True:
        raise RuntimeError("Formal F2 Step1 uncertainty/OOD calibration has not passed on new calibration rainfalls")
    if step2_cal.get("status") != "pass" or step2_cal.get("safety_calibrated") is not True:
        raise RuntimeError("Formal F2 Step2 safety calibration has not passed on new calibration rainfalls")
    if raw_audit.get("status") != "pass" or raw_audit.get("raw_independent_oracle_all_pass") is not True:
        raise RuntimeError("Formal F2 raw Step2 admission has not passed")
    if gat_audit.get("status") != "pass" or gat_audit.get("current_frame_repetition_used") is not False or gat_audit.get("authoritative_swmm_history_used_as_online_input") is not False or gat_audit.get("realized_future_rainfall_used_online") is not False:
        raise RuntimeError("Formal F2 causal GAT-history audit has not passed")
    _assert_same_split(step1_reports, ("train_rainfall_groups", "validation_rainfall_groups", "model_calibration_rainfall_groups"))
    _assert_same_split(step2_reports, ("train_rainfall_groups", "validation_rainfall_groups", "calibration_rainfall_groups"))
    if min(int(r.get("train_rainfall_group_count", 0)) for r in step1_reports) < 65:
        raise RuntimeError("Formal Step1 train rainfall diversity below 65")
    if min(int(r.get("train_rainfall_group_count", 0)) for r in step2_reports) < 65:
        raise RuntimeError("Formal Step2 train rainfall diversity below 65")

    eval_roles = {role: set(ledger.loc[ledger.formal_f2_role.astype(str).eq(role), "rainfall_group_key"].astype(str)) for role in ("calibration", "locked_validation", "challenge", "formal_blind")}
    train1 = set(map(str, step1_reports[0]["train_rainfall_groups"])) | set(map(str, step1_reports[0]["validation_rainfall_groups"])) | set(map(str, step1_reports[0]["model_calibration_rainfall_groups"]))
    train2 = set(map(str, step2_reports[0]["train_rainfall_groups"])) | set(map(str, step2_reports[0]["validation_rainfall_groups"])) | set(map(str, step2_reports[0]["calibration_rainfall_groups"]))
    for role, groups in eval_roles.items():
        if role != "calibration" and (groups & (train1 | train2)):
            raise RuntimeError(f"Formal F2 {role} rainfall overlaps model development groups")
    new_cal_groups = set(map(str, step1_cal.get("calibration_rainfall_groups", []))) | set(map(str, step2_cal.get("calibration_rainfall_groups", [])))
    if not new_cal_groups.issubset(eval_roles["calibration"]):
        raise RuntimeError("calibration reports are not tied to F2 Calibration ledger")

    primary_idx = args.seeds.index(args.primary_seed) if args.primary_seed in args.seeds else None
    if primary_idx is None:
        raise ValueError("primary seed must be among --seeds")
    step1_primary = step1_reports[primary_idx]
    step2_primary = step2_reports[primary_idx]
    ensemble_gat_hash = _combined_model_hash(step1_reports, "gat_model_sha256")
    ensemble_surrogate_hash = _combined_model_hash(step2_reports, "surrogate_model_sha256")
    sample_lineage_sha = sha256_file(gat_manifest)

    step1_evidence = {
        "contract_id": CONTRACT_ID,
        "stage": "step1_sparse_state",
        "status": "pass",
        "development_only": False,
        "formal_generation_id": FORMAL_GENERATION_ID,
        "formal_reconstructor": "TemporalSparseGATReconstructorV42",
        "reconstructor_contract": "formal_temporal_v42",
        "new_formal_training": True,
        "rainfall_group_isolated_split": True,
        "action_authority": "actual_readback_setting",
        "uncertainty_calibrated": True,
        "ood_calibrated": True,
        "uses_future_hydraulic_truth": False,
        "gat_model_sha256": str(step1_primary["gat_model_sha256"]),
        "gat_ensemble_sha256": ensemble_gat_hash,
        "model_seed": args.primary_seed,
        "model_seeds": args.seeds,
        "train_rainfall_group_count": int(step1_primary["train_rainfall_group_count"]),
        "validation_rainfall_group_count": int(step1_primary["validation_rainfall_group_count"]),
        "calibration_evidence_sha256": sha256_file(args.formal_root / "calibration" / "STEP1_UNCERTAINTY_OOD_CALIBRATION.json"),
        "uncertainty_scale_95": step1_cal.get("uncertainty_scale_95"),
        "ood_limit_99": step1_cal.get("ood_limit_99"),
    }
    step2_evidence = {
        "contract_id": CONTRACT_ID,
        "stage": "step2_hydraulic_surrogate",
        "status": "pass",
        "development_only": False,
        "formal_generation_id": FORMAL_GENERATION_ID,
        "formal_model": "MultiReferenceHydraulicSurrogate",
        "four_reference_shared_model": True,
        "trajectory_first_kpi_derivation": True,
        "training_admission_authorized": True,
        "raw_independent_oracle_all_pass": True,
        "action_authority": "actual_readback_setting",
        "history_input_contract": "gat_compatible_causal_state",
        "rainfall_group_isolated_split": True,
        "formal_target_domain_only": True,
        "formal_target_coverage_complete": True,
        "storage_supervised": True,
        "facility_flow_supervised": True,
        "outfall_supervised": True,
        "sample_lineage_sha256": sample_lineage_sha,
        "surrogate_model_sha256": str(step2_primary["surrogate_model_sha256"]),
        "surrogate_ensemble_sha256": ensemble_surrogate_hash,
        "model_seed": args.primary_seed,
        "model_seeds": args.seeds,
        "train_rainfall_group_count": int(step2_primary["train_rainfall_group_count"]),
        "safety_calibrated": True,
        "safety_calibration_sha256": sha256_file(args.formal_root / "calibration" / "STEP2_SAFETY_CALIBRATION.json"),
        "confidence_z": step2_cal.get("confidence_z"),
        "pfv_false_safe_rate_calibration": step2_cal.get("pfv_false_safe_rate"),
        "peak_false_safe_rate_calibration": step2_cal.get("peak_false_safe_rate"),
        "joint_false_safe_rate_calibration": step2_cal.get("joint_false_safe_rate"),
        "uncertainty_limit": step2_cal.get("uncertainty_limit_99"),
    }
    step1_path = args.paper_root / "step1_gat" / "evidence.json"
    step2_path = args.paper_root / "step2_surrogate" / "evidence.json"
    step1_path.parent.mkdir(parents=True, exist_ok=True)
    step2_path.parent.mkdir(parents=True, exist_ok=True)
    step1_path.write_text(json.dumps(step1_evidence, indent=2, allow_nan=False), encoding="utf-8")
    step2_path.write_text(json.dumps(step2_evidence, indent=2, allow_nan=False), encoding="utf-8")
    result = {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "status": "pass",
        "step1_evidence": str(step1_path),
        "step2_evidence": str(step2_path),
        "primary_gat_model_sha256": step1_evidence["gat_model_sha256"],
        "primary_surrogate_model_sha256": step2_evidence["surrogate_model_sha256"],
        "formal_mainline_authorized_through_step2_only": True,
        "step3_and_paper_workflow_require_real_execution_evidence": True,
    }
    out = args.formal_root / "FORMAL_F2_TRAINING_EVIDENCE_COMPILE.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
