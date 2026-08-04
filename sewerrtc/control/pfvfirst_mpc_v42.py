"""Canonical PFV-constrained TFV-minimising rolling MPC.

The controller separates *admission* from *performance*.

Hard admission
--------------
A Candidate must satisfy:

* priority flooding non-inferiority relative to No-control with a frozen budget
  ``100 m3 + 5% * PFV_no_control`` (evaluated on a one-sided UCB);
* Engineering36 K/bounds/rate/ramp/dwell/interlock constraints;
* finite PFV/TFV prediction and executability.

Performance inside the admitted set
-----------------------------------
The primary objective is total flooding volume relative to Dynamic Internal.
Global peak and priority-node depth are reporting/diagnostic channels only.
Uncertainty is already consumed by the one-sided PFV UCB and is not an
independent rejection gate or objective penalty.

If the admitted set is empty, or candidate selection fails, the frozen fallback
is executed. A performance improvement can never compensate a hard admission
violation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MPCWeights:
    """Legacy-compatible fields; V2 selection intentionally ignores them."""

    peak: float = 600.0
    action: float = 0.05
    terminal: float = 0.10
    uncertainty: float = 0.10


@dataclass(frozen=True)
class SafetyMargins:
    pfv_absolute_allowance_m3: float = 100.0
    pfv_relative_allowance_fraction: float = 0.05
    max_changed_facilities: int = 8
    # Backward-compatible API. V2 keeps the field for old evidence readers but
    # priority depth is never an admission gate.
    require_priority_depth: bool = False

    def pfv_allowance_m3(self, no_control_pfv_m3: float) -> float:
        ref = max(0.0, float(no_control_pfv_m3))
        return float(
            self.pfv_absolute_allowance_m3
            + self.pfv_relative_allowance_fraction * ref
        )


@dataclass(frozen=True)
class EngineeringStatus:
    bounds: bool
    rate: bool
    ramp: bool
    dwell: bool
    interlock: bool

    @property
    def passed(self) -> bool:
        return bool(self.bounds and self.rate and self.ramp and self.dwell and self.interlock)

    def failed_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in ("bounds", "rate", "ramp", "dwell", "interlock")
            if not bool(getattr(self, name))
        )


@dataclass(frozen=True)
class MPCandidate:
    candidate_id: str
    action_sequence: np.ndarray
    # Candidate-minus-No-control priority flooding UCB [m3].
    pfv_delta_ucb_m3: float
    # Candidate-minus-Dynamic-Internal peak rate [m3/s]. Kept under the legacy
    # field name for schema compatibility; it is now a performance term.
    peak_delta_ucb_m3s: float
    tfv_delta_di_m3: float
    action_cost: float
    terminal_cost: float
    uncertainty_cost: float
    changed_facilities: int
    engineering: EngineeringStatus
    uncertainty_pass: bool
    ood_pass: bool
    executable: bool
    # Additional diagnostic inputs retained for schema compatibility.
    pfv_no_control_m3: float = 0.0
    priority_depth_ucb_m: tuple[float, ...] = field(default_factory=tuple)
    priority_depth_limit_m: tuple[float, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FrozenFallback:
    fallback_id: str
    action_sequence: np.ndarray
    contract_hash: str
    legal: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateAudit:
    candidate_id: str
    safe: bool
    rejection_reasons: tuple[str, ...]
    objective: float | None
    pfv_allowance_m3: float | None = None
    maximum_priority_depth_exceedance_m: float | None = None


@dataclass(frozen=True)
class MPCDecision:
    selected_id: str
    execute_action: np.ndarray
    selected_sequence: np.ndarray
    used_fallback: bool
    reason: str
    objective: float | None
    audits: tuple[CandidateAudit, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


def _validate_action_sequence(name: str, sequence: np.ndarray) -> np.ndarray:
    arr = np.asarray(sequence, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name} action sequence must be [H,A]")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} action sequence contains NaN/Inf")
    return arr


def _priority_depth_exceedance(candidate: MPCandidate) -> float | None:
    depths = np.asarray(candidate.priority_depth_ucb_m, dtype=float).reshape(-1)
    limits = np.asarray(candidate.priority_depth_limit_m, dtype=float).reshape(-1)
    if depths.size == 0 and limits.size == 0:
        return None
    if depths.size == 0 or limits.size == 0 or depths.shape != limits.shape:
        raise ValueError("priority depth UCB/limit arrays must have the same non-zero shape")
    if not np.isfinite(depths).all() or not np.isfinite(limits).all():
        raise ValueError("priority depth UCB/limit arrays must be finite")
    return float(np.max(depths - limits))


def audit_candidate_safety(
    candidate: MPCandidate,
    *,
    margins: SafetyMargins,
) -> CandidateAudit:
    reasons: list[str] = []
    try:
        _validate_action_sequence(candidate.candidate_id, candidate.action_sequence)
    except ValueError:
        reasons.append("invalid_action_sequence")

    allowance: float | None = None
    if not np.isfinite(float(candidate.pfv_no_control_m3)):
        reasons.append("nonfinite_PFV_prediction")
    else:
        allowance = margins.pfv_allowance_m3(float(candidate.pfv_no_control_m3))

    if not np.isfinite(float(candidate.pfv_delta_ucb_m3)):
        reasons.append("nonfinite_PFV_prediction")
    elif allowance is not None and float(candidate.pfv_delta_ucb_m3) > allowance:
        reasons.append("PFV_budget_exceeded_vs_no_control")

    depth_exceedance: float | None = None
    try:
        depth_exceedance = _priority_depth_exceedance(candidate)
    except ValueError:
        # Diagnostic-only data cannot reject a candidate in V2.
        depth_exceedance = None

    if int(candidate.changed_facilities) < 0:
        reasons.append("negative_changed_facility_count")
    elif int(candidate.changed_facilities) > int(margins.max_changed_facilities):
        reasons.append("K_exceeded")

    for name in candidate.engineering.failed_names():
        reasons.append(f"engineering_{name}_violation")
    if not candidate.executable:
        reasons.append("candidate_not_executable")
    if not np.isfinite(float(candidate.tfv_delta_di_m3)):
        reasons.append("nonfinite_TFV_prediction")

    return CandidateAudit(
        candidate_id=str(candidate.candidate_id),
        safe=not reasons,
        rejection_reasons=tuple(reasons),
        objective=None,
        pfv_allowance_m3=allowance,
        maximum_priority_depth_exceedance_m=depth_exceedance,
    )


def performance_objective(candidate: MPCandidate, *, weights: MPCWeights) -> float:
    """Return only predicted TFV; ``weights`` remains for old callers."""
    del weights
    value = float(candidate.tfv_delta_di_m3)
    if not np.isfinite(value):
        raise ValueError(f"candidate {candidate.candidate_id} has non-finite TFV prediction")
    return value


def select_safe_candidate(
    candidates: Sequence[MPCandidate],
    *,
    margins: SafetyMargins,
    weights: MPCWeights,
) -> tuple[MPCandidate | None, tuple[CandidateAudit, ...], float | None]:
    audits: list[CandidateAudit] = []
    safe_scored: list[tuple[float, str, MPCandidate]] = []
    for candidate in candidates:
        audit = audit_candidate_safety(candidate, margins=margins)
        if not audit.safe:
            audits.append(audit)
            continue
        try:
            score = performance_objective(candidate, weights=weights)
        except ValueError as exc:
            audits.append(
                CandidateAudit(
                    candidate_id=candidate.candidate_id,
                    safe=False,
                    rejection_reasons=("nonfinite_TFV_prediction", str(exc)),
                    objective=None,
                    pfv_allowance_m3=audit.pfv_allowance_m3,
                    maximum_priority_depth_exceedance_m=audit.maximum_priority_depth_exceedance_m,
                )
            )
            continue
        audits.append(
            CandidateAudit(
                candidate_id=candidate.candidate_id,
                safe=True,
                rejection_reasons=(),
                objective=score,
                pfv_allowance_m3=audit.pfv_allowance_m3,
                maximum_priority_depth_exceedance_m=audit.maximum_priority_depth_exceedance_m,
            )
        )
        safe_scored.append((score, str(candidate.candidate_id), candidate))

    if not safe_scored:
        return None, tuple(audits), None
    score, _, selected = min(safe_scored, key=lambda item: (item[0], item[1]))
    return selected, tuple(audits), float(score)


def execute_frozen_fallback(
    fallback: FrozenFallback,
    *,
    audits: tuple[CandidateAudit, ...],
    reason: str,
    expected_contract_hash: str | None = None,
) -> MPCDecision:
    if not fallback.legal:
        raise RuntimeError("frozen fallback is not legal; fail closed without an active candidate")
    if not str(fallback.contract_hash).strip():
        raise RuntimeError("frozen fallback has no contract hash")
    if expected_contract_hash is not None and str(fallback.contract_hash) != str(expected_contract_hash):
        raise RuntimeError("frozen fallback contract hash mismatch")
    sequence = _validate_action_sequence(fallback.fallback_id, fallback.action_sequence)
    return MPCDecision(
        selected_id=fallback.fallback_id,
        execute_action=sequence[0].copy(),
        selected_sequence=sequence.copy(),
        used_fallback=True,
        reason=reason,
        objective=None,
        audits=audits,
        metadata={
            "fallback_contract_hash": fallback.contract_hash,
            "safety_and_performance_separated": True,
            "control_objective_contract": "PROJECT6_V42_PFV_ONLY_TFV_MIN_MPC_V2",
        },
    )


def decide_pfvfirst_mpc(
    *,
    candidates: Sequence[MPCandidate],
    fallback: FrozenFallback,
    margins: SafetyMargins | None = None,
    weights: MPCWeights | None = None,
    expected_fallback_contract_hash: str | None = None,
) -> MPCDecision:
    """Return the first action of the best admitted candidate or fallback."""
    margins = margins or SafetyMargins()
    weights = weights or MPCWeights()
    try:
        selected, audits, objective = select_safe_candidate(
            candidates,
            margins=margins,
            weights=weights,
        )
    except Exception as exc:
        return execute_frozen_fallback(
            fallback,
            audits=(),
            reason=f"candidate_selection_error:{type(exc).__name__}",
            expected_contract_hash=expected_fallback_contract_hash,
        )

    if selected is None:
        return execute_frozen_fallback(
            fallback,
            audits=audits,
            reason="safe_set_empty",
            expected_contract_hash=expected_fallback_contract_hash,
        )

    sequence = _validate_action_sequence(selected.candidate_id, selected.action_sequence)
    return MPCDecision(
        selected_id=selected.candidate_id,
        execute_action=sequence[0].copy(),
        selected_sequence=sequence.copy(),
        used_fallback=False,
        reason="minimum_tfv_objective_within_pfv_budget_and_engineering_safe_set",
        objective=objective,
        audits=audits,
        metadata={
            "objective": "minimize_TFV_subject_to_PFV_budget",
            "hard_constraints": [
                "PFV_budget_vs_no_control",
                "K",
                "bounds",
                "rate",
                "ramp",
                "dwell",
                "interlock",
                "executability",
            ],
            "pfv_absolute_allowance_m3": margins.pfv_absolute_allowance_m3,
            "pfv_relative_allowance_fraction": margins.pfv_relative_allowance_fraction,
            "peak_role": "reporting_only",
            "priority_depth_role": "diagnostic_only",
            "uncertainty_role": "PFV_UCB_only",
            "OOD_role": "diagnostic_only",
            "peak_is_hard_safety_constraint": False,
            "global_peak_objective_term": False,
            "peak_penalty_weight": 0.0,
            "action_penalty_weight": 0.0,
            "terminal_penalty_weight": 0.0,
            "uncertainty_penalty_weight": 0.0,
            "independent_OOD_gate": False,
            "independent_uncertainty_gate": False,
            "tfv_is_hard_safety_constraint": False,
            "safety_and_performance_separated": True,
            "control_objective_contract": "PROJECT6_V42_PFV_ONLY_TFV_MIN_MPC_V2",
        },
    )
