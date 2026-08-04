"""Fail-closed model-validity gate for the V4.2 paper workflow.

This module intentionally does **not** invent scientific accuracy thresholds.
Those thresholds must be frozen by the research team before Locked/Final
holdouts are inspected.  The code only verifies that a quantitative,
Calibration-only acceptance study exists, that it explicitly passed its frozen
contract, and that it is tied to the exact Step1/Step2 models being locked.

Expected evidence path:

    v42_paper/model_acceptance/evidence.json

The evidence producer is allowed to evolve separately, but Policy Lock must not
proceed until this structural contract passes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def audit_model_acceptance(paper_root: str | Path) -> dict[str, Any]:
    """Verify pre-Policy-Lock quantitative model acceptance evidence."""
    paper_root = Path(paper_root)
    evidence_path = paper_root / "model_acceptance/evidence.json"
    step1_path = paper_root / "step1_gat/evidence.json"
    step2_path = paper_root / "step2_surrogate/evidence.json"
    reasons: list[str] = []

    for path, label in (
        (evidence_path, "model_acceptance_evidence"),
        (step1_path, "step1_evidence"),
        (step2_path, "step2_evidence"),
    ):
        if not path.exists():
            reasons.append(f"{label}_missing")
    if reasons:
        return {
            "status": "fail",
            "model_accuracy_acceptance_pass": False,
            "evidence_path": str(evidence_path),
            "evidence_sha256": None,
            "reasons": reasons,
        }

    try:
        evidence = _read_json(evidence_path)
        step1 = _read_json(step1_path)
        step2 = _read_json(step2_path)
    except Exception as exc:
        return {
            "status": "fail",
            "model_accuracy_acceptance_pass": False,
            "evidence_path": str(evidence_path),
            "evidence_sha256": None,
            "reasons": [f"model_acceptance_evidence_unreadable:{type(exc).__name__}"],
        }

    if evidence.get("status") != "pass":
        reasons.append("model_acceptance_status_not_pass")
    if evidence.get("model_accuracy_acceptance_pass") is not True:
        reasons.append("model_accuracy_acceptance_not_pass")
    if evidence.get("quantitative_swmm_comparison_performed") is not True:
        reasons.append("quantitative_swmm_comparison_not_proven")
    if evidence.get("event_balanced_metrics_reported") is not True:
        reasons.append("event_balanced_metrics_not_proven")
    if evidence.get("thresholds_frozen_before_policy_lock") is not True:
        reasons.append("accuracy_thresholds_not_frozen_before_policy_lock")
    if evidence.get("uses_locked_or_final_for_threshold_tuning") is not False:
        reasons.append("locked_or_final_used_for_accuracy_threshold_tuning")
    if str(evidence.get("evaluation_role", "")) != "calibration":
        reasons.append("model_acceptance_must_use_calibration_role")
    if evidence.get("rainfall_group_isolated") is not True:
        reasons.append("model_acceptance_rainfall_isolation_not_proven")
    if int(evidence.get("training_rainfall_overlap_count", -1)) != 0:
        reasons.append("model_acceptance_overlaps_training_rainfall")
    if not str(evidence.get("acceptance_contract_sha256", "")).strip():
        reasons.append("acceptance_contract_sha256_missing")

    expected_step1 = str(step1.get("gat_model_sha256", "")).strip()
    expected_step2 = str(step2.get("surrogate_model_sha256", "")).strip()
    if not expected_step1 or str(evidence.get("gat_model_sha256", "")) != expected_step1:
        reasons.append("model_acceptance_gat_hash_mismatch")
    if not expected_step2 or str(evidence.get("surrogate_model_sha256", "")) != expected_step2:
        reasons.append("model_acceptance_surrogate_hash_mismatch")

    # Required metric families are semantic, not numerical thresholds. Their
    # threshold values and pass/fail directions belong to the separately frozen
    # acceptance contract referenced above.
    required_metric_families = {
        "step1_unobserved_depth",
        "step1_priority_depth",
        "step1_wet_or_high_depth",
        "step2_branch_depth",
        "step2_flooding_rate",
        "step2_pfv_budget_metric",
        "step2_tfv_delta",
        "step2_storage_volume",
        "step2_managed_facility_flow",
    }
    observed = {
        str(x) for x in evidence.get("accepted_metric_families", []) if str(x).strip()
    }
    missing_metric_families = sorted(required_metric_families - observed)
    if missing_metric_families:
        reasons.append(
            "model_acceptance_metric_families_missing:"
            + ",".join(missing_metric_families)
        )

    return {
        "status": "pass" if not reasons else "fail",
        "model_accuracy_acceptance_pass": not reasons,
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
        "acceptance_contract_sha256": evidence.get("acceptance_contract_sha256"),
        "gat_model_sha256": evidence.get("gat_model_sha256"),
        "surrogate_model_sha256": evidence.get("surrogate_model_sha256"),
        "reasons": reasons,
    }
