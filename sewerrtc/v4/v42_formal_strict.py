"""Defense-in-depth audit helpers for the Project6 V4.2 Formal F2 line.

The final paper-generation checks must not be bypassed by hand-written evidence:

* current Calibration is the complete frozen 12-rainfall holdout;
* CONTROL_CORE/FULL_HYDRAULIC supervision semantics are explicit;
* Step3 is PFV-budget + priority-depth hard-safe, TFV-primary, Peak-soft, with
  execution-derived Engineering36/readback evidence;
* stage21 contains authoritative SWMM Proposed/No-control/Internal/Hold runs;
* stage22 is an actually executed surrogate-state-feedback closed loop, not a
  metadata-only evidence stub;
* stage23 is the sparse-GAT-integrated authoritative SWMM loop;
* Policy-Lock and current-generation held-out evidence pass the frozen workflow.

Missing or inconsistent inputs fail closed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paper_workflow_v42 import audit_paper_workflow

EXPECTED_CALIBRATION_GROUPS = 12
CONTROL_OBJECTIVE_CONTRACT = "PROJECT6_V42_PFV_BUDGETED_TFV_MPC_V1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return (
        pd.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else pd.read_csv(path, low_memory=False)
    )


def _calibration_group_set(ledger: pd.DataFrame) -> set[str]:
    if "formal_f2_role" not in ledger or "rainfall_group_key" not in ledger:
        raise KeyError("Formal ledger missing formal_f2_role/rainfall_group_key")
    return set(
        ledger.loc[
            ledger["formal_f2_role"].astype(str).eq("calibration"),
            "rainfall_group_key",
        ].astype(str)
    )


def audit_calibration_completeness(formal_root: Path) -> dict[str, Any]:
    ledger = _read_table(formal_root / "prepare/FORMAL_F2_EVENT_LEDGER.csv")
    expected = _calibration_group_set(ledger)
    step1_path = formal_root / "calibration/STEP1_UNCERTAINTY_OOD_CALIBRATION.json"
    step2_path = formal_root / "calibration/STEP2_SAFETY_CALIBRATION.json"
    reasons: list[str] = []
    if len(expected) != EXPECTED_CALIBRATION_GROUPS:
        reasons.append(
            f"frozen_calibration_group_count_must_equal_{EXPECTED_CALIBRATION_GROUPS}:got_{len(expected)}"
        )
    try:
        step1 = _read_json(step1_path)
    except Exception as exc:
        step1 = {}
        reasons.append(f"step1_calibration_unreadable:{type(exc).__name__}")
    try:
        step2 = _read_json(step2_path)
    except Exception as exc:
        step2 = {}
        reasons.append(f"step2_calibration_unreadable:{type(exc).__name__}")

    for name, payload in (("step1", step1), ("step2", step2)):
        if payload:
            if payload.get("status") != "pass":
                reasons.append(f"{name}_calibration_status_not_pass")
            observed = set(map(str, payload.get("calibration_rainfall_groups", [])))
            if observed != expected:
                reasons.append(
                    f"{name}_calibration_does_not_equal_frozen_plan:missing={len(expected-observed)}:extra={len(observed-expected)}"
                )
            if int(payload.get("calibration_rainfall_group_count", -1)) != len(expected):
                reasons.append(f"{name}_calibration_reported_group_count_mismatch")

    return {
        "status": "pass" if not reasons else "fail",
        "expected_calibration_group_count": len(expected),
        "expected_calibration_rainfall_groups": sorted(expected),
        "step1_calibration_path": str(step1_path),
        "step2_calibration_path": str(step2_path),
        "reasons": reasons,
    }


def audit_step2_evidence_strict(paper_root: Path) -> dict[str, Any]:
    path = paper_root / "step2_surrogate/evidence.json"
    reasons: list[str] = []
    try:
        p = _read_json(path)
    except Exception as exc:
        return {
            "status": "fail",
            "path": str(path),
            "reasons": [f"step2_evidence_unreadable:{type(exc).__name__}"],
        }
    contract = str(p.get("step2_target_contract", ""))
    if contract not in {"CONTROL_CORE", "FULL_HYDRAULIC"}:
        reasons.append("step2_target_contract_missing_or_invalid")
    if p.get("control_core_target_coverage_complete") is not True:
        reasons.append("control_core_target_coverage_not_proven")
    if p.get("storage_supervised") is not True:
        reasons.append("storage_supervision_not_proven")
    if p.get("facility_flow_supervised") is not True:
        reasons.append("facility_flow_supervision_not_proven")
    if contract == "FULL_HYDRAULIC" and p.get("outfall_supervised") is not True:
        reasons.append("full_hydraulic_outfall_supervision_not_proven")
    if contract == "CONTROL_CORE" and p.get("outfall_claim_authorized") is True:
        reasons.append("control_core_must_not_claim_explicit_outfall_supervision")
    if p.get("no_control_all_open_verified") is not True:
        reasons.append("no_control_all_open_not_verified")
    if p.get("trajectory_first_kpi_derivation") is not True:
        reasons.append("trajectory_first_kpi_derivation_not_proven")
    if p.get("peak_is_hard_safety_constraint") is not False:
        reasons.append("peak_must_not_be_hard_safety_constraint")
    return {
        "status": "pass" if not reasons else "fail",
        "path": str(path),
        "step2_target_contract": contract,
        "reasons": reasons,
    }


def audit_step3_evidence_strict(paper_root: Path) -> dict[str, Any]:
    path = paper_root / "step3_mpc/evidence.json"
    reasons: list[str] = []
    try:
        p = _read_json(path)
    except Exception as exc:
        return {
            "status": "fail",
            "path": str(path),
            "reasons": [f"step3_evidence_unreadable:{type(exc).__name__}"],
        }
    if p.get("selector") != "decide_pfvfirst_mpc":
        reasons.append("wrong_step3_selector")
    if p.get("control_objective_contract") != CONTROL_OBJECTIVE_CONTRACT:
        reasons.append("wrong_control_objective_contract")
    if p.get("pfv_reference") != "no_control":
        reasons.append("wrong_pfv_reference")
    if float(p.get("pfv_absolute_allowance_m3", -1.0)) != 100.0:
        reasons.append("pfv_absolute_allowance_mismatch")
    if abs(float(p.get("pfv_relative_allowance_fraction", -1.0)) - 0.05) > 1.0e-12:
        reasons.append("pfv_relative_allowance_mismatch")
    if p.get("priority_depth_safety") is not True:
        reasons.append("priority_depth_safety_not_proven")
    if p.get("tfv_reference") != "dynamic_internal":
        reasons.append("wrong_tfv_reference")
    if p.get("tfv_is_primary_performance_objective") is not True:
        reasons.append("tfv_primary_objective_not_proven")
    if p.get("tfv_is_hard_safety_constraint") is not False:
        reasons.append("tfv_must_not_be_hard_gate")
    if p.get("peak_reference") != "dynamic_internal":
        reasons.append("wrong_peak_reference")
    if p.get("peak_is_hard_safety_constraint") is not False:
        reasons.append("peak_must_not_be_hard_gate")
    if p.get("peak_positive_excess_is_penalized") is not True:
        reasons.append("positive_peak_excess_penalty_not_proven")
    if int(p.get("facility_count", -1)) != 36:
        reasons.append("engineering36_not_proven")
    if int(p.get("max_changed_facilities", -1)) != 8:
        reasons.append("K_contract_mismatch")
    if int(p.get("horizon_steps", -1)) != 12:
        reasons.append("H12_contract_mismatch")
    for key in (
        "engineering_status_derived_from_execution",
        "changed_facilities_derived_from_executed_action",
        "readback_verified",
        "uncertainty_and_ood_linked_to_calibrated_models",
    ):
        if p.get(key) is not True:
            reasons.append(f"{key}_not_proven")
    return {
        "status": "pass" if not reasons else "fail",
        "path": str(path),
        "reasons": reasons,
    }


def audit_closed_loop_execution_strict(paper_root: Path) -> dict[str, Any]:
    """Require real execution semantics for Formal attribution stages 21-23."""
    reasons: list[str] = []
    paths = {
        "exact": paper_root / "exact_closed_loop/evidence.json",
        "surrogate": paper_root / "surrogate_closed_loop/evidence.json",
        "gat": paper_root / "gat_integrated_closed_loop/evidence.json",
    }
    payloads: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        try:
            payloads[name] = _read_json(path)
        except Exception as exc:
            payloads[name] = {}
            reasons.append(f"{name}_closed_loop_evidence_unreadable:{type(exc).__name__}")

    exact = payloads["exact"]
    if exact:
        if exact.get("status") != "pass" or exact.get("authoritative_engine") != "SWMM":
            reasons.append("stage21_not_authoritative_swmm")
        if exact.get("canonical_pfvfirst_mpc_v42") is not True:
            reasons.append("stage21_canonical_mpc_not_proven")
        refs = set(map(str, exact.get("authoritative_reference_strategies", [])))
        if refs != {"No-control", "Internal", "Hold"}:
            reasons.append("stage21_reference_strategy_set_mismatch")
        counts = exact.get("strategy_event_counts", {})
        event_count = int(exact.get("event_count", 0))
        if event_count != EXPECTED_CALIBRATION_GROUPS:
            reasons.append("stage21_must_cover_calibration12")
        if not isinstance(counts, dict) or any(
            int(counts.get(strategy, -1)) != event_count
            for strategy in ("Proposed", "No-control", "Internal", "Hold")
        ):
            reasons.append("stage21_reference_event_counts_incomplete")
        if exact.get("no_control_all_open_authoritative") is not True:
            reasons.append("stage21_no_control_all_open_not_proven")
        if exact.get("internal_native_rules_authoritative") is not True:
            reasons.append("stage21_internal_native_rules_not_proven")

    surrogate = payloads["surrogate"]
    if surrogate:
        if surrogate.get("status") != "pass":
            reasons.append("stage22_status_not_pass")
        if surrogate.get("surrogate_closed_loop_executed") is not True:
            reasons.append("stage22_metadata_only_stub_forbidden")
        if surrogate.get("surrogate_role") != "hydraulic_surrogate_not_policy":
            reasons.append("stage22_wrong_surrogate_role")
        if surrogate.get("pfvfirst_mpc_v42") is not True:
            reasons.append("stage22_canonical_mpc_not_proven")
        if int(surrogate.get("event_count", 0)) != EXPECTED_CALIBRATION_GROUPS:
            reasons.append("stage22_must_cover_calibration12")
        if surrogate.get("authoritative_hydraulic_truth_used_after_prefix") is not False:
            reasons.append("stage22_future_hydraulic_truth_feedback_detected")
        if surrogate.get("realized_future_rainfall_used_online") is not False:
            reasons.append("stage22_realized_future_rainfall_detected")
        if surrogate.get("dynamic_internal_future_action_used_online") is not False:
            reasons.append("stage22_future_internal_action_detected")

    gat = payloads["gat"]
    if gat:
        if gat.get("status") != "pass":
            reasons.append("stage23_status_not_pass")
        if gat.get("state_source") != "gat_sparse_reconstruction":
            reasons.append("stage23_not_sparse_gat_integrated")
        if int(gat.get("event_count", 0)) != EXPECTED_CALIBRATION_GROUPS:
            reasons.append("stage23_must_cover_calibration12")
        if gat.get("authoritative_swmm_outcome") is not True:
            reasons.append("stage23_authoritative_swmm_outcome_not_proven")
        if gat.get("authoritative_swmm_history_used_as_online_input") is not False:
            reasons.append("stage23_swmm_truth_history_leakage")
        if gat.get("current_frame_repetition_used") is not False:
            reasons.append("stage23_current_frame_repetition_detected")
        if gat.get("gat_uncertainty_used") is not True or gat.get("ood_gate_used") is not True:
            reasons.append("stage23_uncertainty_ood_not_active")

    return {
        "status": "pass" if not reasons else "fail",
        "paths": {name: str(path) for name, path in paths.items()},
        "reasons": reasons,
    }


def audit_formal_strict(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root)
    paper_root = output_root / "v42_paper"
    formal_root = paper_root / "formal_f2"
    calibration = audit_calibration_completeness(formal_root)
    step2 = audit_step2_evidence_strict(paper_root)
    step3 = audit_step3_evidence_strict(paper_root)
    closed_loop = audit_closed_loop_execution_strict(paper_root)
    workflow = audit_paper_workflow(output_root)
    reasons: list[str] = []
    for name, item in (
        ("calibration", calibration),
        ("step2", step2),
        ("step3", step3),
        ("closed_loop", closed_loop),
    ):
        if item.get("status") != "pass":
            reasons.append(f"{name}_strict_audit_not_pass")
    if not workflow.complete:
        reasons.append("paper_workflow_not_complete")
    return {
        "status": "pass" if not reasons else "fail",
        "strict_formal_complete": not reasons,
        "calibration": calibration,
        "step2": step2,
        "step3": step3,
        "closed_loop": closed_loop,
        "paper_workflow": workflow.as_dict(),
        "reasons": reasons,
    }
