"""Fail-closed stage gate for the final V4.2 paper workflow.

The current evaluation design uses independent-rainfall, current-generation
holdouts. Historical labels from older Project6 experiments are lineage metadata
only; the hard statistical requirement is zero rainfall-group overlap between
current model development and Calibration/Challenge/Locked/final held-out test.

The legacy stage name ``formal_blind`` is retained for path/API compatibility.
Its scientific meaning in this generation is ``final held-out test after Policy
Lock`` rather than a claim that the rainfall has never appeared anywhere in the
project's historical archive.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

CONTRACT_ID = "PROJECT6_V42_PAPER_WORKFLOW_V1"
MODEL_LINE = "v42_trajectory_first_multi_reference"
CAUSAL_HISTORY_CONTRACT = "PROJECT6_V42_CAUSAL_RECONSTRUCTED_HISTORY_V1"

PAPER_STAGE_ORDER = (
    "true_state_offline_validation",
    "exact_swmm_closed_loop",
    "surrogate_closed_loop",
    "gat_integrated_closed_loop",
    "policy_lock",
    "challenge",
    "locked_validation",
    "formal_blind",
)

EVIDENCE_RELATIVE_PATHS = {
    "true_state_offline_validation": "v42_paper/true_state_offline/evidence.json",
    "exact_swmm_closed_loop": "v42_paper/exact_closed_loop/evidence.json",
    "surrogate_closed_loop": "v42_paper/surrogate_closed_loop/evidence.json",
    "gat_integrated_closed_loop": "v42_paper/gat_integrated_closed_loop/evidence.json",
    "policy_lock": "v42_paper/policy_lock/evidence.json",
    "challenge": "v42_paper/challenge/evidence.json",
    "locked_validation": "v42_paper/locked_validation/evidence.json",
    "formal_blind": "v42_paper/formal_blind/evidence.json",
}

LOCK_HASH_KEYS = (
    "policy_sha256",
    "model_sha256",
    "gat_model_sha256",
    "fallback_contract_sha256",
)

REQUIRED_FORMAL_BLIND_STRATEGIES = (
    "Proposed",
    "EFD",
    "Auto-RBC",
    "All-close",
    "No-control",
    "Internal",
    "Hold",
)
FORMAL_AUTHORITY = "authoritative_swmm"


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
            "evaluation_semantics": "current_generation_rainfall_group_holdout",
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
        raise ValueError("JSON root must be an object")
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


def _nonempty_sha_set(value: Any) -> set[str] | None:
    if not isinstance(value, list):
        return None
    return {str(x).strip() for x in value if str(x).strip()}


def _event_count(payload: Mapping[str, Any]) -> int:
    try:
        return int(payload.get("event_count", 0))
    except (TypeError, ValueError):
        return 0


def _holdout_evidence_reasons(
    payload: Mapping[str, Any],
    *,
    stage_prefix: str,
    minimum_events: int,
) -> tuple[list[str], set[str]]:
    """Validate a current-generation rainfall-group holdout stage."""
    reasons: list[str] = []
    event_count = _event_count(payload)
    if event_count < minimum_events:
        reasons.append(f"{stage_prefix}_event_count_below_{minimum_events}")

    lock_flag = payload.get(
        "policy_locked_before_evaluation",
        payload.get("policy_locked_before_reveal"),
    )
    if lock_flag is not True:
        reasons.append(f"{stage_prefix}_evaluated_before_policy_lock")
    if payload.get("current_generation_holdout_only") is not True:
        reasons.append(f"{stage_prefix}_not_current_generation_holdout")
    if payload.get(
        "post_evaluation_exclusion_used", payload.get("post_reveal_exclusion_used")
    ) is True:
        reasons.append(f"{stage_prefix}_post_evaluation_exclusion_forbidden")
    if payload.get("used_for_retraining") is True:
        reasons.append(f"{stage_prefix}_used_for_retraining")

    rainfall_list = payload.get("rainfall_sha256s")
    rainfalls = _nonempty_sha_set(rainfall_list)
    if (
        rainfalls is None
        or not isinstance(rainfall_list, list)
        or len(rainfall_list) != event_count
    ):
        reasons.append(f"{stage_prefix}_rainfall_sha_list_missing_or_count_mismatch")
        rainfalls = set()
    elif len(rainfalls) != event_count:
        reasons.append(f"{stage_prefix}_rainfall_sha_not_unique")

    training_list = payload.get("training_rainfall_sha256s")
    training = _nonempty_sha_set(training_list)
    if training is None:
        reasons.append(f"{stage_prefix}_training_rainfall_sha_list_missing")
        training = set()
    overlap = rainfalls & training
    if overlap:
        reasons.append(f"{stage_prefix}_rainfall_overlaps_current_training")
    try:
        reported_overlap = int(payload.get("training_rainfall_overlap_count", -1))
    except (TypeError, ValueError):
        reported_overlap = -1
    if reported_overlap != len(overlap):
        reasons.append(f"{stage_prefix}_reported_training_overlap_count_mismatch")
    if reported_overlap != 0:
        reasons.append(f"{stage_prefix}_reported_training_overlap_not_zero")

    for key in LOCK_HASH_KEYS:
        _require_hash(payload, key, reasons)
    return reasons, rainfalls


def _formal_strategy_reasons(
    payload: Mapping[str, Any], event_count: int
) -> list[str]:
    reasons: list[str] = []
    authority = payload.get("strategy_authority")
    counts = payload.get("strategy_event_counts")
    if not isinstance(authority, Mapping):
        return ["formal_strategy_authority_missing"]
    if not isinstance(counts, Mapping):
        return ["formal_strategy_event_counts_missing"]
    for strategy in REQUIRED_FORMAL_BLIND_STRATEGIES:
        observed = str(authority.get(strategy, "")).strip().casefold()
        if observed != FORMAL_AUTHORITY:
            reasons.append(f"formal_strategy_not_authoritative_swmm:{strategy}")
        try:
            n = int(counts.get(strategy, -1))
        except (TypeError, ValueError):
            n = -1
        if n != event_count:
            reasons.append(f"formal_strategy_event_count_mismatch:{strategy}")
    extras = sorted(
        set(map(str, authority.keys())) - set(REQUIRED_FORMAL_BLIND_STRATEGIES)
    )
    if extras:
        reasons.append("formal_strategy_authority_contains_unexpected_entries")
    return reasons


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
        if payload.get("reconstructed_history_contract") != CAUSAL_HISTORY_CONTRACT:
            reasons.append("gat_integrated_loop_requires_causal_reconstructed_history")
        if payload.get("reconstructed_history_ready_before_mpc") is not True:
            reasons.append("gat_integrated_mpc_started_before_13_reconstructed_frames")
        if payload.get("authoritative_swmm_history_used_as_online_input") is not False:
            reasons.append("swmm_truth_history_used_in_gat_integrated_loop")
        if payload.get("current_frame_repetition_used") is not False:
            reasons.append("current_reconstructed_frame_repeated_as_history")
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
        if payload.get("control_objective_contract") != "PROJECT6_V42_PFV_BUDGETED_TFV_MPC_V1":
            reasons.append("wrong_control_objective_contract")
    elif stage == "challenge":
        stage_reasons, _ = _holdout_evidence_reasons(
            payload, stage_prefix="challenge", minimum_events=12
        )
        reasons.extend(stage_reasons)
    elif stage == "locked_validation":
        stage_reasons, _ = _holdout_evidence_reasons(
            payload, stage_prefix="locked", minimum_events=16
        )
        reasons.extend(stage_reasons)
    elif stage == "formal_blind":
        event_count = _event_count(payload)
        stage_reasons, _ = _holdout_evidence_reasons(
            payload, stage_prefix="formal_test", minimum_events=24
        )
        reasons.extend(stage_reasons)
        reasons.extend(_formal_strategy_reasons(payload, event_count))
    return reasons


def _policy_lineage_reasons(
    *, stage: str, payload: Mapping[str, Any], output_root: Path
) -> list[str]:
    if stage not in {"challenge", "locked_validation", "formal_blind"}:
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

    stage_rain = _nonempty_sha_set(payload.get("rainfall_sha256s")) or set()
    if stage == "locked_validation":
        earlier = (("challenge", "locked_rainfall_overlaps_challenge"),)
    elif stage == "formal_blind":
        earlier = (
            ("challenge", "final_test_rainfall_overlaps_challenge"),
            ("locked_validation", "final_test_rainfall_overlaps_locked_validation"),
        )
    else:
        earlier = ()
    for prior, reason in earlier:
        prior_path = output_root / EVIDENCE_RELATIVE_PATHS[prior]
        if not prior_path.exists():
            continue
        try:
            prior_payload = _read_json(prior_path)
            prior_shas = _nonempty_sha_set(prior_payload.get("rainfall_sha256s"))
            if prior_shas is not None and stage_rain & prior_shas:
                reasons.append(reason)
        except Exception:
            reasons.append(f"{prior}_rainfall_lineage_unreadable")
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
    *, stage: str, output_root: str | Path, payload: Mapping[str, Any]
) -> Path:
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
