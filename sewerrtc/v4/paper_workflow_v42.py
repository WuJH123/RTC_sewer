"""Fail-closed stage gate for the final V4.2 paper workflow.

The gate owns the formal evidence sequence. It verifies that the formal
closed-loop stages are executed in order and that Challenge/Formal-Blind reuse
exactly the policy, surrogate, temporal-GAT and fallback hashes frozen at Policy
Lock.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CONTRACT_ID = "PROJECT6_V42_PAPER_WORKFLOW_V1"
MODEL_LINE = "v42_trajectory_first_multi_reference"

PAPER_STAGE_ORDER = (
    "true_state_offline_validation",
    "exact_swmm_closed_loop",
    "surrogate_closed_loop",
    "gat_integrated_closed_loop",
    "policy_lock",
    "challenge",
    "formal_blind",
)

EVIDENCE_RELATIVE_PATHS = {
    "true_state_offline_validation": "v42_paper/true_state_offline/evidence.json",
    "exact_swmm_closed_loop": "v42_paper/exact_closed_loop/evidence.json",
    "surrogate_closed_loop": "v42_paper/surrogate_closed_loop/evidence.json",
    "gat_integrated_closed_loop": "v42_paper/gat_integrated_closed_loop/evidence.json",
    "policy_lock": "v42_paper/policy_lock/evidence.json",
    "challenge": "v42_paper/challenge/evidence.json",
    "formal_blind": "v42_paper/formal_blind/evidence.json",
}

# Every model/policy component that can change closed-loop behaviour is locked.
# GAT used to be required at Policy Lock but was not compared for Challenge or
# Formal Blind, which allowed a changed state reconstructor to pass lineage.
LOCK_HASH_KEYS = (
    "policy_sha256",
    "model_sha256",
    "gat_model_sha256",
    "fallback_contract_sha256",
)


@dataclass(frozen=True)
class StageAudit:
    stage: str
    passed: bool
    reasons: tuple[str, ...]
    evidence_path: str
    evidence_sha256: str | None


@dataclass(frozen=True)
class WorkflowAudit:
    passed_through: str | None
    next_stage: str | None
    complete: bool
    stage_audits: tuple[StageAudit, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_id": CONTRACT_ID,
            "model_line": MODEL_LINE,
            "passed_through": self.passed_through,
            "next_stage": self.next_stage,
            "complete": self.complete,
            "stages": [
                {
                    "stage": item.stage,
                    "passed": item.passed,
                    "reasons": list(item.reasons),
                    "evidence_path": item.evidence_path,
                    "evidence_sha256": item.evidence_sha256,
                }
                for item in self.stage_audits
            ],
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evidence root must be a JSON object")
    return payload


def _common_reasons(stage: str, payload: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("contract_id") != CONTRACT_ID:
        reasons.append("wrong_or_legacy_workflow_contract")
    if payload.get("model_line") != MODEL_LINE:
        reasons.append("wrong_or_legacy_model_line")
    if payload.get("stage") != stage:
        reasons.append("stage_name_mismatch")
    if payload.get("status") != "pass":
        reasons.append("stage_status_not_pass")
    if payload.get("development_evidence_substituted") is True:
        reasons.append("development_evidence_cannot_substitute_formal_stage")
    if payload.get("legacy_locked_evidence_substituted") is True:
        reasons.append("legacy_locked_evidence_cannot_substitute_formal_stage")
    return reasons


def _require_hash(payload: Mapping[str, Any], key: str, reasons: list[str]) -> None:
    if not str(payload.get(key, "")).strip():
        reasons.append(f"missing_{key}")


def _stage_specific_reasons(stage: str, payload: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if stage == "true_state_offline_validation":
        if payload.get("state_source") != "true_state":
            reasons.append("true_state_validation_requires_true_state")
        if payload.get("four_reference_surrogate") is not True:
            reasons.append("four_reference_surrogate_not_verified")
        if payload.get("trajectory_first_kpi_derivation") is not True:
            reasons.append("trajectory_first_kpi_derivation_not_verified")
        if payload.get("training_admission_authorized") is not True:
            reasons.append("formal_step2_training_admission_not_proven")
        if payload.get("raw_independent_oracle_all_pass") is not True:
            reasons.append("raw_independent_oracle_not_proven")
        _require_hash(payload, "surrogate_model_sha256", reasons)
    elif stage == "exact_swmm_closed_loop":
        if payload.get("authoritative_engine") != "SWMM":
            reasons.append("exact_closed_loop_not_authoritative_swmm")
        if payload.get("online_future_hydraulic_truth_used") is True:
            reasons.append("future_hydraulic_truth_used_online")
        if payload.get("canonical_pfvfirst_mpc_v42") is not True:
            reasons.append("canonical_pfvfirst_mpc_not_used")
        if payload.get("engineering_status_derived_from_execution") is not True:
            reasons.append("engineering_guards_not_derived_from_execution")
        if payload.get("readback_verified") is not True:
            reasons.append("actual_readback_not_verified")
    elif stage == "surrogate_closed_loop":
        if payload.get("surrogate_role") != "hydraulic_surrogate_not_policy":
            reasons.append("surrogate_role_contract_violation")
        if payload.get("pfvfirst_mpc_v42") is not True:
            reasons.append("canonical_pfvfirst_mpc_not_used")
        _require_hash(payload, "surrogate_model_sha256", reasons)
    elif stage == "gat_integrated_closed_loop":
        if payload.get("state_source") != "gat_sparse_reconstruction":
            reasons.append("gat_integrated_loop_requires_sparse_gat_state")
        if payload.get("reconstructor_contract") != "formal_temporal_v42":
            reasons.append("gat_integrated_loop_requires_formal_temporal_reconstructor")
        if payload.get("gat_uncertainty_used") is not True:
            reasons.append("gat_uncertainty_not_used")
        if payload.get("ood_gate_used") is not True:
            reasons.append("ood_gate_not_used")
        if payload.get("uncertainty_calibrated") is not True:
            reasons.append("gat_uncertainty_not_calibrated")
        if payload.get("ood_calibrated") is not True:
            reasons.append("gat_ood_not_calibrated")
        _require_hash(payload, "gat_model_sha256", reasons)
        _require_hash(payload, "surrogate_model_sha256", reasons)
    elif stage == "policy_lock":
        for key in LOCK_HASH_KEYS:
            _require_hash(payload, key, reasons)
        if payload.get("post_lock_parameter_updates_allowed") is not False:
            reasons.append("policy_lock_allows_post_lock_updates")
    elif stage == "challenge":
        if payload.get("policy_locked_before_reveal") is not True:
            reasons.append("challenge_revealed_before_policy_lock")
        if payload.get("used_for_retraining") is True:
            reasons.append("challenge_used_for_retraining")
        for key in LOCK_HASH_KEYS:
            _require_hash(payload, key, reasons)
    elif stage == "formal_blind":
        try:
            event_count = int(payload.get("event_count", 0))
        except (TypeError, ValueError):
            event_count = 0
        if event_count < 24:
            reasons.append("formal_blind_event_count_below_24")
        if payload.get("policy_locked_before_reveal") is not True:
            reasons.append("formal_revealed_before_policy_lock")
        if payload.get("new_rainfall_sha_only") is not True:
            reasons.append("formal_contains_non_new_rainfall_sha")
        if payload.get("post_reveal_exclusion_used") is True:
            reasons.append("formal_post_reveal_exclusion_forbidden")
        if payload.get("used_for_retraining") is True:
            reasons.append("formal_used_for_retraining")
        rainfall_shas = payload.get("rainfall_sha256s")
        if not isinstance(rainfall_shas, list) or len(rainfall_shas) != event_count:
            reasons.append("formal_rainfall_sha_list_missing_or_count_mismatch")
        elif len({str(x) for x in rainfall_shas if str(x).strip()}) != event_count:
            reasons.append("formal_rainfall_sha_not_unique")
        if payload.get("revealed_rainfall_overlap_count") not in (0, "0"):
            reasons.append("formal_rainfall_overlaps_revealed_development")
        for key in LOCK_HASH_KEYS:
            _require_hash(payload, key, reasons)
    return reasons


def _policy_lineage_reasons(
    *,
    stage: str,
    payload: Mapping[str, Any],
    output_root: Path,
) -> list[str]:
    if stage not in {"challenge", "formal_blind"}:
        return []
    lock_path = output_root / EVIDENCE_RELATIVE_PATHS["policy_lock"]
    if not lock_path.exists():
        return ["policy_lock_evidence_missing_for_lineage"]
    try:
        lock = _read_json(lock_path)
    except Exception:
        return ["policy_lock_evidence_unreadable_for_lineage"]
    reasons: list[str] = []
    for key in LOCK_HASH_KEYS:
        expected = str(lock.get(key, ""))
        observed = str(payload.get(key, ""))
        if not expected or observed != expected:
            reasons.append(f"{key}_does_not_match_policy_lock")
    return reasons


def audit_stage_evidence(
    stage: str,
    evidence_path: Path,
    *,
    output_root: str | Path | None = None,
) -> StageAudit:
    if stage not in PAPER_STAGE_ORDER:
        raise KeyError(f"unknown paper stage: {stage}")
    path = Path(evidence_path)
    if not path.exists():
        return StageAudit(stage, False, ("evidence_missing",), str(path), None)
    try:
        payload = _read_json(path)
    except Exception as exc:
        return StageAudit(
            stage,
            False,
            (f"evidence_unreadable:{type(exc).__name__}",),
            str(path),
            _sha256(path),
        )
    reasons = _common_reasons(stage, payload)
    reasons.extend(_stage_specific_reasons(stage, payload))
    if output_root is not None:
        reasons.extend(
            _policy_lineage_reasons(
                stage=stage, payload=payload, output_root=Path(output_root)
            )
        )
    return StageAudit(stage, not reasons, tuple(reasons), str(path), _sha256(path))


def audit_paper_workflow(output_root: str | Path) -> WorkflowAudit:
    """Audit stages in order and stop at the first non-passing stage."""
    root = Path(output_root)
    audits: list[StageAudit] = []
    passed_through: str | None = None
    next_stage: str | None = None
    for stage in PAPER_STAGE_ORDER:
        evidence = root / EVIDENCE_RELATIVE_PATHS[stage]
        audit = audit_stage_evidence(stage, evidence, output_root=root)
        audits.append(audit)
        if not audit.passed:
            next_stage = stage
            break
        passed_through = stage
    complete = passed_through == PAPER_STAGE_ORDER[-1]
    return WorkflowAudit(
        passed_through=passed_through,
        next_stage=None if complete else next_stage,
        complete=complete,
        stage_audits=tuple(audits),
    )


def prerequisite_stage(stage: str) -> str | None:
    if stage not in PAPER_STAGE_ORDER:
        raise KeyError(stage)
    idx = PAPER_STAGE_ORDER.index(stage)
    return None if idx == 0 else PAPER_STAGE_ORDER[idx - 1]


def assert_stage_authorized(stage: str, output_root: str | Path) -> None:
    """Raise unless every preceding paper stage has valid V4.2 evidence."""
    if stage not in PAPER_STAGE_ORDER:
        raise KeyError(stage)
    root = Path(output_root)
    stage_idx = PAPER_STAGE_ORDER.index(stage)
    for prior in PAPER_STAGE_ORDER[:stage_idx]:
        audit = audit_stage_evidence(
            prior,
            root / EVIDENCE_RELATIVE_PATHS[prior],
            output_root=root,
        )
        if not audit.passed:
            raise RuntimeError(
                f"{stage} is not authorized because prerequisite {prior} failed: "
                + ",".join(audit.reasons)
            )


def write_stage_evidence(
    *,
    stage: str,
    output_root: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write evidence only after prerequisites pass and stamp V4.2 lineage."""
    assert_stage_authorized(stage, output_root)
    root = Path(output_root)
    path = root / EVIDENCE_RELATIVE_PATHS[stage]
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.update(
        {
            "contract_id": CONTRACT_ID,
            "model_line": MODEL_LINE,
            "stage": stage,
        }
    )
    path.write_text(json.dumps(body, indent=2, allow_nan=False), encoding="utf-8")
    return path
