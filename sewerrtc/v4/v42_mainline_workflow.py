"""End-to-end scientific gate for the V4.2 paper mainline.

This module connects the formal chain:

Phase R0 -> Step 1 temporal sparse GAT -> Step 2 four-reference hydraulic
surrogate -> Step 3 PFV-first MPC -> Step 4 closed-loop/lock/blind evidence.

It is an audit/orchestration gate, not a substitute for executing training or
SWMM. Missing evidence stops the chain at the first incomplete scientific stage.
The gate also verifies cross-stage model lineage so a different GAT/surrogate
cannot be substituted after training validation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .paper_workflow_v42 import (
    CONTRACT_ID,
    EVIDENCE_RELATIVE_PATHS,
    MODEL_LINE,
    audit_paper_workflow,
)


MAINLINE_ID = "PROJECT6_V42_MAINLINE_V1"
MAINLINE_STAGES = (
    "phase_r0",
    "step1_sparse_state",
    "step2_hydraulic_surrogate",
    "step3_pfvfirst_mpc",
    "step4_closed_loop_and_blind",
)


@dataclass(frozen=True)
class MainlineStageAudit:
    stage: str
    passed: bool
    reasons: tuple[str, ...]
    evidence: str


@dataclass(frozen=True)
class MainlineAudit:
    passed_through: str | None
    next_stage: str | None
    complete: bool
    stages: tuple[MainlineStageAudit, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mainline_id": MAINLINE_ID,
            "paper_contract_id": CONTRACT_ID,
            "model_line": MODEL_LINE,
            "passed_through": self.passed_through,
            "next_stage": self.next_stage,
            "complete": self.complete,
            "stages": [
                {
                    "stage": s.stage,
                    "passed": s.passed,
                    "reasons": list(s.reasons),
                    "evidence": s.evidence,
                }
                for s in self.stages
            ],
        }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _coerce_bool_series(series: pd.Series, *, column: str) -> pd.Series:
    """Parse persisted booleans without treating the string 'False' as True."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.fillna(0).astype(float).ne(0.0)
    text = series.fillna("").astype(str).str.strip().str.casefold()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", "", "none", "nan"}
    unknown = sorted(set(text.unique()) - true_values - false_values)
    if unknown:
        raise ValueError(f"boolean column {column!r} has unsupported values: {unknown[:10]}")
    return text.isin(true_values)


def _payload_common(payload: Mapping[str, Any], stage: str) -> list[str]:
    reasons: list[str] = []
    if payload.get("contract_id") != CONTRACT_ID:
        reasons.append("wrong_paper_contract")
    if payload.get("stage") != stage:
        reasons.append("stage_name_mismatch")
    if payload.get("status") != "pass":
        reasons.append("stage_status_not_pass")
    if payload.get("development_only") is True:
        reasons.append("development_only_evidence_cannot_authorize_mainline")
    return reasons


def audit_phase_r0(output_root: Path) -> MainlineStageAudit:
    root = output_root / "v42_paper" / "data_reuse"
    summary_path = root / "data_reuse_audit.json"
    reusable_path = root / "reusable_pool_summary.json"
    alignment_path = root / "case_alignment_audit.csv"
    split_path = root / "split_group_manifest.parquet"
    reasons: list[str] = []
    try:
        summary = _read_json(summary_path)
        if summary.get("full_finite_check") is not True:
            reasons.append("r0_full_finite_audit_not_complete")
        if summary.get("missing_targets_are_imputed") is not False:
            reasons.append("r0_missing_target_policy_invalid")
        if summary.get("strict_semantics_wrapper") is not True:
            reasons.append("r0_strict_semantics_not_proven")
        if summary.get("discovery_cache_current") is False:
            reasons.append("r0_scan_cache_does_not_cover_current_discovery")
    except Exception as exc:
        reasons.append(f"r0_audit_unreadable:{type(exc).__name__}")
    try:
        reusable = _read_json(reusable_path)
        if reusable.get("strict_scientific_admission") is not True:
            reasons.append("r0_reusable_pool_not_strict")
        if reusable.get("counterfactual_requires_all_four_roles_finite") is not True:
            reasons.append("r0_four_branch_finite_gate_missing")
        counts = reusable.get("task_counts", {})
        if int(counts.get("counterfactual_flood_cases", 0)) <= 0:
            reasons.append("r0_no_aligned_counterfactual_cases")
    except Exception as exc:
        reasons.append(f"r0_reusable_summary_unreadable:{type(exc).__name__}")
    try:
        align = _read_table(alignment_path)
        if align.empty:
            reasons.append("r0_alignment_empty")
        else:
            ok = _coerce_bool_series(
                align["same_state_numeric_pass"], column="same_state_numeric_pass"
            ) & _coerce_bool_series(
                align["same_forcing_pass"], column="same_forcing_pass"
            )
            if not bool(ok.any()):
                reasons.append("r0_no_numeric_same_state_same_forcing_case")
    except Exception as exc:
        reasons.append(f"r0_alignment_unreadable:{type(exc).__name__}")
    try:
        split = _read_table(split_path)
        if split.empty or "split_group_key" not in split.columns:
            reasons.append("r0_rainfall_split_groups_missing")
        if "reserved_evaluation" in split.columns and bool(
            _coerce_bool_series(
                split["reserved_evaluation"], column="reserved_evaluation"
            ).any()
        ):
            reasons.append("r0_reusable_split_contains_reserved_evaluation")
    except Exception as exc:
        reasons.append(f"r0_split_unreadable:{type(exc).__name__}")
    return MainlineStageAudit("phase_r0", not reasons, tuple(reasons), str(root))


def _audit_step1(output_root: Path) -> MainlineStageAudit:
    path = output_root / "v42_paper" / "step1_gat" / "evidence.json"
    reasons: list[str] = []
    try:
        p = _read_json(path)
        reasons.extend(_payload_common(p, "step1_sparse_state"))
        if p.get("formal_reconstructor") != "TemporalSparseGATReconstructorV42":
            reasons.append("wrong_step1_reconstructor")
        if p.get("reconstructor_contract") not in (None, "formal_temporal_v42"):
            reasons.append("step1_reconstructor_contract_not_formal_temporal")
        if p.get("new_formal_training") is not True:
            reasons.append("step1_new_training_not_proven")
        if p.get("rainfall_group_isolated_split") is not True:
            reasons.append("step1_rainfall_group_split_not_proven")
        if p.get("action_authority") != "actual_readback_setting":
            reasons.append("step1_action_authority_not_readback")
        if p.get("uncertainty_calibrated") is not True:
            reasons.append("step1_uncertainty_not_calibrated")
        if p.get("ood_calibrated") is not True:
            reasons.append("step1_ood_not_calibrated")
        if p.get("uses_future_hydraulic_truth") is not False:
            reasons.append("step1_future_truth_contract_violation")
        if not str(p.get("gat_model_sha256", "")).strip():
            reasons.append("step1_model_hash_missing")
    except Exception as exc:
        reasons.append(f"step1_evidence_unreadable:{type(exc).__name__}")
    return MainlineStageAudit("step1_sparse_state", not reasons, tuple(reasons), str(path))


def _audit_step2(output_root: Path) -> MainlineStageAudit:
    path = output_root / "v42_paper" / "step2_surrogate" / "evidence.json"
    reasons: list[str] = []
    try:
        p = _read_json(path)
        reasons.extend(_payload_common(p, "step2_hydraulic_surrogate"))
        if p.get("formal_model") != "MultiReferenceHydraulicSurrogate":
            reasons.append("wrong_step2_surrogate")
        if p.get("four_reference_shared_model") is not True:
            reasons.append("step2_four_reference_contract_not_proven")
        if p.get("trajectory_first_kpi_derivation") is not True:
            reasons.append("step2_kpi_shortcut_detected")
        if p.get("training_admission_authorized") is not True:
            reasons.append("step2_training_admission_not_authorized")
        if p.get("raw_independent_oracle_all_pass") is not True:
            reasons.append("step2_raw_oracle_not_all_pass")
        if p.get("action_authority") != "actual_readback_setting":
            reasons.append("step2_action_authority_not_readback")
        if p.get("history_input_contract") != "gat_compatible_causal_state":
            reasons.append("step2_history_input_not_gat_compatible")
        if p.get("rainfall_group_isolated_split") is not True:
            reasons.append("step2_rainfall_group_split_not_proven")
        if not str(p.get("surrogate_model_sha256", "")).strip():
            reasons.append("step2_model_hash_missing")
    except Exception as exc:
        reasons.append(f"step2_evidence_unreadable:{type(exc).__name__}")
    return MainlineStageAudit("step2_hydraulic_surrogate", not reasons, tuple(reasons), str(path))


def _audit_step3(output_root: Path) -> MainlineStageAudit:
    path = output_root / "v42_paper" / "step3_mpc" / "evidence.json"
    reasons: list[str] = []
    try:
        p = _read_json(path)
        reasons.extend(_payload_common(p, "step3_pfvfirst_mpc"))
        if p.get("selector") != "decide_pfvfirst_mpc":
            reasons.append("wrong_step3_selector")
        if p.get("pfv_reference") != "no_control":
            reasons.append("step3_wrong_pfv_reference")
        if p.get("peak_reference") != "dynamic_internal":
            reasons.append("step3_wrong_peak_reference")
        if p.get("tfv_reference") != "dynamic_internal":
            reasons.append("step3_wrong_tfv_reference")
        if int(p.get("max_changed_facilities", -1)) != 8:
            reasons.append("step3_K_contract_mismatch")
        if int(p.get("horizon_steps", 12)) != 12:
            reasons.append("step3_horizon_contract_mismatch")
        if int(p.get("facility_count", 36)) != 36:
            reasons.append("step3_engineering36_contract_mismatch")
        if p.get("tfv_is_hard_safety_constraint") is not False:
            reasons.append("step3_tfv_must_not_be_hard_gate")
        if p.get("engineering_status_derived_from_execution") is not True:
            reasons.append("step3_engineering_flags_not_authoritative")
        if p.get("changed_facilities_derived_from_executed_action") is not True:
            reasons.append("step3_K_not_derived_from_executed_action")
        if p.get("readback_verified") is not True:
            reasons.append("step3_readback_not_verified")
        if p.get("uncertainty_and_ood_linked_to_calibrated_models") is not True:
            reasons.append("step3_uncertainty_ood_lineage_missing")
    except Exception as exc:
        reasons.append(f"step3_evidence_unreadable:{type(exc).__name__}")
    return MainlineStageAudit("step3_pfvfirst_mpc", not reasons, tuple(reasons), str(path))


def _cross_stage_lineage_reasons(output_root: Path) -> list[str]:
    """Bind Step-1/Step-2 trained models to every formal downstream stage."""
    reasons: list[str] = []
    try:
        step1 = _read_json(output_root / "v42_paper" / "step1_gat" / "evidence.json")
        step2 = _read_json(output_root / "v42_paper" / "step2_surrogate" / "evidence.json")
        expected_gat = str(step1.get("gat_model_sha256", ""))
        expected_surrogate = str(step2.get("surrogate_model_sha256", ""))
        if not expected_gat or not expected_surrogate:
            return ["cross_stage_training_model_hash_missing"]

        checks = (
            ("true_state_offline_validation", "surrogate_model_sha256", expected_surrogate),
            ("surrogate_closed_loop", "surrogate_model_sha256", expected_surrogate),
            ("gat_integrated_closed_loop", "surrogate_model_sha256", expected_surrogate),
            ("gat_integrated_closed_loop", "gat_model_sha256", expected_gat),
            ("policy_lock", "model_sha256", expected_surrogate),
            ("policy_lock", "gat_model_sha256", expected_gat),
        )
        for stage, key, expected in checks:
            path = output_root / EVIDENCE_RELATIVE_PATHS[stage]
            payload = _read_json(path)
            observed = str(payload.get(key, ""))
            if observed != expected:
                reasons.append(f"{stage}_{key}_does_not_match_training_evidence")
    except Exception as exc:
        reasons.append(f"cross_stage_lineage_unreadable:{type(exc).__name__}")
    return reasons


def audit_v42_mainline(output_root: str | Path) -> MainlineAudit:
    root = Path(output_root)
    audits = [audit_phase_r0(root), _audit_step1(root), _audit_step2(root), _audit_step3(root)]
    passed_through: str | None = None
    final: list[MainlineStageAudit] = []
    for audit in audits:
        final.append(audit)
        if not audit.passed:
            return MainlineAudit(passed_through, audit.stage, False, tuple(final))
        passed_through = audit.stage

    paper = audit_paper_workflow(root)
    if not paper.complete:
        reasons = ()
        if paper.stage_audits:
            reasons = paper.stage_audits[-1].reasons
        final.append(
            MainlineStageAudit(
                "step4_closed_loop_and_blind",
                False,
                tuple(reasons) or (f"paper_workflow_next:{paper.next_stage}",),
                str(root / "v42_paper"),
            )
        )
        return MainlineAudit(
            passed_through,
            "step4_closed_loop_and_blind",
            False,
            tuple(final),
        )

    lineage_reasons = _cross_stage_lineage_reasons(root)
    if lineage_reasons:
        final.append(
            MainlineStageAudit(
                "step4_closed_loop_and_blind",
                False,
                tuple(lineage_reasons),
                str(root / "v42_paper"),
            )
        )
        return MainlineAudit(
            passed_through,
            "step4_closed_loop_and_blind",
            False,
            tuple(final),
        )

    final.append(
        MainlineStageAudit(
            "step4_closed_loop_and_blind", True, (), str(root / "v42_paper")
        )
    )
    return MainlineAudit(
        "step4_closed_loop_and_blind", None, True, tuple(final)
    )
