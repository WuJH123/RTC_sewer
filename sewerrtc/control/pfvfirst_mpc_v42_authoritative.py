"""Authoritative adapters around the V4.2 PFV-first selector.

Formal V4.2 use must derive K/engineering status from the projected action,
retain post-write readback evidence, and enforce the frozen H12 x Engineering36
shape. Arbitrary caller booleans or differently shaped action arrays cannot
stand in for the engineering contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .pfvfirst_mpc_v42 import EngineeringStatus, MPCandidate, MPCDecision


REQUIRED_ENGINEERING_CHECKS = ("bounds", "rate", "ramp", "dwell", "interlock")
FORMAL_HORIZON_STEPS = 12
FORMAL_FACILITY_COUNT = 36


@dataclass(frozen=True)
class ProjectionGuardEvidence:
    checks: Mapping[str, bool]
    contract_sha256: str
    authority: str = "projected_action_guard"

    def engineering_status(self) -> EngineeringStatus:
        missing = [key for key in REQUIRED_ENGINEERING_CHECKS if key not in self.checks]
        if missing:
            raise KeyError(f"projection guard missing checks: {missing}")
        if self.authority != "projected_action_guard":
            raise ValueError("projection guard authority is not canonical")
        if not str(self.contract_sha256).strip():
            raise ValueError("projection guard contract hash missing")
        return EngineeringStatus(
            bounds=bool(self.checks["bounds"]),
            rate=bool(self.checks["rate"]),
            ramp=bool(self.checks["ramp"]),
            dwell=bool(self.checks["dwell"]),
            interlock=bool(self.checks["interlock"]),
        )


@dataclass(frozen=True)
class ExecutionReadbackAudit:
    passed: bool
    reasons: tuple[str, ...]
    changed_facilities_written: int
    changed_facilities_readback: int
    max_written_readback_abs_error: float

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "changed_facilities_written": self.changed_facilities_written,
            "changed_facilities_readback": self.changed_facilities_readback,
            "max_written_readback_abs_error": self.max_written_readback_abs_error,
            "engineering_status_derived_from_execution": True,
            "changed_facilities_derived_from_executed_action": True,
            "readback_verified": self.passed,
            "horizon_steps": FORMAL_HORIZON_STEPS,
            "facility_count": FORMAL_FACILITY_COUNT,
        }


def _vector(name: str, value: np.ndarray, n: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.size < 1:
        raise ValueError(f"{name} must be a 1-D facility vector")
    if n is not None and arr.size != n:
        raise ValueError(f"{name} facility count mismatch: {arr.size}!={n}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def _sequence(
    value: np.ndarray,
    *,
    expected_horizon_steps: int = FORMAL_HORIZON_STEPS,
    expected_facility_count: int = FORMAL_FACILITY_COUNT,
) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError("projected action sequence must be [H,A]")
    expected = (int(expected_horizon_steps), int(expected_facility_count))
    if arr.shape != expected:
        raise ValueError(
            f"formal V4.2 projected action sequence must be {expected}, got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("projected action sequence contains NaN/Inf")
    return arr


def _changed(action: np.ndarray, anchor: np.ndarray, atol: float) -> int:
    return int(np.sum(np.abs(action - anchor) > float(atol)))


def build_authoritative_mpc_candidate(
    *,
    candidate_id: str,
    projected_action_sequence: np.ndarray,
    anchor_action: np.ndarray,
    guard_evidence: ProjectionGuardEvidence,
    pfv_delta_ucb_m3: float,
    peak_delta_ucb_m3s: float,
    tfv_delta_di_m3: float,
    action_cost: float,
    terminal_cost: float,
    uncertainty_cost: float,
    uncertainty_pass: bool,
    ood_pass: bool,
    executable: bool,
    changed_atol: float = 1.0e-9,
    metadata: Mapping[str, object] | None = None,
) -> MPCandidate:
    """Build a formal H12 x Engineering36 candidate from guard evidence."""
    sequence = _sequence(projected_action_sequence)
    anchor = _vector("anchor_action", anchor_action, FORMAL_FACILITY_COUNT)
    changed = _changed(sequence[0], anchor, changed_atol)
    engineering = guard_evidence.engineering_status()
    meta = dict(metadata or {})
    meta.update(
        {
            "candidate_authority": "projected_action_after_engineering_guard",
            "engineering_guard_contract_sha256": guard_evidence.contract_sha256,
            "changed_facilities_authority": "derived_from_projected_first_action_vs_anchor",
            "anchor_action": anchor.tolist(),
            "formal_horizon_steps": FORMAL_HORIZON_STEPS,
            "formal_facility_count": FORMAL_FACILITY_COUNT,
        }
    )
    return MPCandidate(
        candidate_id=str(candidate_id),
        action_sequence=sequence,
        pfv_delta_ucb_m3=float(pfv_delta_ucb_m3),
        peak_delta_ucb_m3s=float(peak_delta_ucb_m3s),
        tfv_delta_di_m3=float(tfv_delta_di_m3),
        action_cost=float(action_cost),
        terminal_cost=float(terminal_cost),
        uncertainty_cost=float(uncertainty_cost),
        changed_facilities=changed,
        engineering=engineering,
        uncertainty_pass=bool(uncertainty_pass),
        ood_pass=bool(ood_pass),
        executable=bool(executable),
        metadata=meta,
    )


def audit_executed_decision_readback(
    *,
    decision: MPCDecision,
    anchor_action: np.ndarray,
    written_action: np.ndarray,
    readback_action: np.ndarray,
    max_changed_facilities: int = 8,
    atol: float = 1.0e-6,
) -> ExecutionReadbackAudit:
    """Verify that the H12/Engineering36 first action was written and read back."""
    selected_sequence = _sequence(decision.selected_sequence)
    execute = _vector("decision.execute_action", decision.execute_action, FORMAL_FACILITY_COUNT)
    if not np.allclose(execute, selected_sequence[0], atol=atol, rtol=0.0):
        raise ValueError("decision.execute_action differs from selected_sequence[0]")
    anchor = _vector("anchor_action", anchor_action, FORMAL_FACILITY_COUNT)
    written = _vector("written_action", written_action, FORMAL_FACILITY_COUNT)
    readback = _vector("readback_action", readback_action, FORMAL_FACILITY_COUNT)
    reasons: list[str] = []
    if not np.allclose(execute, written, atol=atol, rtol=0.0):
        reasons.append("selected_action_differs_from_written_action")
    if not np.allclose(written, readback, atol=atol, rtol=0.0):
        reasons.append("written_action_differs_from_readback")
    changed_written = _changed(written, anchor, atol)
    changed_readback = _changed(readback, anchor, atol)
    if changed_written > int(max_changed_facilities):
        reasons.append("written_action_K_exceeded")
    if changed_readback > int(max_changed_facilities):
        reasons.append("readback_action_K_exceeded")
    max_error = float(np.max(np.abs(written - readback)))
    return ExecutionReadbackAudit(
        passed=not reasons,
        reasons=tuple(reasons),
        changed_facilities_written=changed_written,
        changed_facilities_readback=changed_readback,
        max_written_readback_abs_error=max_error,
    )
