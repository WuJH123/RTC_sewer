"""Canonical PFV-first rolling-MPC decision logic for the V4.2 paper line.

Safety and performance are deliberately separated:

Safety set
~~~~~~~~~~
Candidate must pass PFV non-inferiority vs No-control, Peak non-inferiority vs
Dynamic Internal, K/bounds/rate/ramp/dwell/interlock, uncertainty/OOD and
executability checks.

Performance objective inside the safety set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ΔTFV_DI + λ1 J_action + λ2 J_terminal + λ3 J_uncertainty

TFV is not allowed to compensate a safety violation because unsafe candidates
are removed before the objective is evaluated.  If the safe set is empty, or
selection raises an exception, the frozen fallback is executed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MPCWeights:
    action: float = 0.05
    terminal: float = 0.10
    uncertainty: float = 0.10


@dataclass(frozen=True)
class SafetyMargins:
    pfv_delta_ucb_max_m3: float = 0.0
    peak_delta_ucb_max_m3s: float = 0.0
    max_changed_facilities: int = 8


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
    pfv_delta_ucb_m3: float
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

    if not np.isfinite(float(candidate.pfv_delta_ucb_m3)):
        reasons.append("pfv_uncertainty_not_finite")
    elif float(candidate.pfv_delta_ucb_m3) > float(margins.pfv_delta_ucb_max_m3):
        reasons.append("pfv_safety_violation_vs_no_control")

    if not np.isfinite(float(candidate.peak_delta_ucb_m3s)):
        reasons.append("peak_uncertainty_not_finite")
    elif float(candidate.peak_delta_ucb_m3s) > float(margins.peak_delta_ucb_max_m3s):
        reasons.append("peak_safety_violation_vs_dynamic_internal")

    if int(candidate.changed_facilities) < 0:
        reasons.append("negative_changed_facility_count")
    elif int(candidate.changed_facilities) > int(margins.max_changed_facilities):
        reasons.append("K_exceeded")

    for name in candidate.engineering.failed_names():
        reasons.append(f"engineering_{name}_violation")
    if not candidate.uncertainty_pass:
        reasons.append("uncertainty_gate_failed")
    if not candidate.ood_pass:
        reasons.append("ood_gate_failed")
    if not candidate.executable:
        reasons.append("candidate_not_executable")

    safe = not reasons
    return CandidateAudit(
        candidate_id=str(candidate.candidate_id),
        safe=safe,
        rejection_reasons=tuple(reasons),
        objective=None,
    )


def performance_objective(candidate: MPCandidate, *, weights: MPCWeights) -> float:
    """Evaluate performance only after hard-safety admission."""
    terms = (
        float(candidate.tfv_delta_di_m3),
        float(candidate.action_cost),
        float(candidate.terminal_cost),
        float(candidate.uncertainty_cost),
    )
    if not all(np.isfinite(v) for v in terms):
        raise ValueError(f"candidate {candidate.candidate_id} has non-finite objective term")
    return float(
        terms[0]
        + float(weights.action) * terms[1]
        + float(weights.terminal) * terms[2]
        + float(weights.uncertainty) * terms[3]
    )


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
                    rejection_reasons=("objective_not_finite", str(exc)),
                    objective=None,
                )
            )
            continue
        audits.append(
            CandidateAudit(
                candidate_id=candidate.candidate_id,
                safe=True,
                rejection_reasons=(),
                objective=score,
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
    """Return first action of the best safe candidate or the frozen fallback.

    This function is deliberately exception-safe around candidate scoring.  A
    candidate/solver-side error cannot promote an unsafe action; it falls back
    to the frozen safety policy.
    """
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
        reason="minimum_tfv_objective_within_hard_safety_set",
        objective=objective,
        audits=audits,
        metadata={
            "objective": "delta_tfv_di + lambda_action*J_action + lambda_terminal*J_terminal + lambda_uncertainty*J_uncertainty",
            "hard_constraints": [
                "PFV_vs_no_control",
                "Peak_vs_dynamic_internal",
                "K",
                "bounds",
                "rate",
                "ramp",
                "dwell",
                "interlock",
                "uncertainty",
                "OOD",
                "executability",
            ],
            "tfv_is_hard_safety_constraint": False,
            "safety_and_performance_separated": True,
        },
    )
