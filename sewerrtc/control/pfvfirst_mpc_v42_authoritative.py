"""Authoritative adapters around the V4.2 PFV-first selector.

Formal V4.2 use must derive K/engineering status from the projected action,
retain post-write readback evidence, enforce the frozen H12 x Engineering36
shape, and derive PFV/Peak UCB + uncertainty/OOD admission from calibrated
prediction evidence rather than caller-supplied pass flags.
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
class CalibratedSafetyPrediction:
    """Calibrated model evidence used to deterministically build hard-safety UCBs."""

    pfv_delta_mean_m3: float
    pfv_delta_std_m3: float
    peak_delta_mean_m3s: float
    peak_delta_std_m3s: float
    confidence_z: float
    uncertainty_score: float
    uncertainty_limit: float
    ood_score: float
    ood_limit: float
    gat_model_sha256: str
    surrogate_model_sha256: str
    uncertainty_calibration_sha256: str
    ood_calibration_sha256: str

    def _validate(self) -> None:
        numeric = np.asarray(
            [
                self.pfv_delta_mean_m3,
                self.pfv_delta_std_m3,
                self.peak_delta_mean_m3s,
                self.peak_delta_std_m3s,
                self.confidence_z,
                self.uncertainty_score,
                self.uncertainty_limit,
                self.ood_score,
                self.ood_limit,
            ],
            dtype=float,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("calibrated safety prediction contains NaN/Inf")
        if self.pfv_delta_std_m3 < 0 or self.peak_delta_std_m3s < 0:
            raise ValueError("calibrated standard deviations cannot be negative")
        if self.confidence_z < 0:
            raise ValueError("confidence_z cannot be negative")
        if self.uncertainty_limit < 0 or self.ood_limit < 0:
            raise ValueError("uncertainty/OOD limits cannot be negative")
        for name, value in (
            ("gat_model_sha256", self.gat_model_sha256),
            ("surrogate_model_sha256", self.surrogate_model_sha256),
            ("uncertainty_calibration_sha256", self.uncertainty_calibration_sha256),
            ("ood_calibration_sha256", self.ood_calibration_sha256),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} missing")

    @property
    def pfv_delta_ucb_m3(self) -> float:
        self._validate()
        return float(self.pfv_delta_mean_m3 + self.confidence_z * self.pfv_delta_std_m3)

    @property
    def peak_delta_ucb_m3s(self) -> float:
        self._validate()
        return float(self.peak_delta_mean_m3s + self.confidence_z * self.peak_delta_std_m3s)

    @property
    def uncertainty_pass(self) -> bool:
        self._validate()
        return bool(self.uncertainty_score <= self.uncertainty_limit)

    @property
    def ood_pass(self) -> bool:
        self._validate()
        return bool(self.ood_score <= self.ood_limit)

    def metadata(self) -> dict[str, object]:
        self._validate()
        return {
            "safety_ucb_authority": "calibrated_prediction_mean_plus_z_std",
            "pfv_delta_mean_m3": float(self.pfv_delta_mean_m3),
            "pfv_delta_std_m3": float(self.pfv_delta_std_m3),
            "peak_delta_mean_m3s": float(self.peak_delta_mean_m3s),
            "peak_delta_std_m3s": float(self.peak_delta_std_m3s),
            "confidence_z": float(self.confidence_z),
            "uncertainty_score": float(self.uncertainty_score),
            "uncertainty_limit": float(self.uncertainty_limit),
            "ood_score": float(self.ood_score),
            "ood_limit": float(self.ood_limit),
            "gat_model_sha256": str(self.gat_model_sha256),
            "surrogate_model_sha256": str(self.surrogate_model_sha256),
            "uncertainty_calibration_sha256": str(self.uncertainty_calibration_sha256),
            "ood_calibration_sha256": str(self.ood_calibration_sha256),
        }


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
    """Compatibility builder; formal paper runs should use calibrated builder below."""
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
            "safety_ucb_authority": meta.get("safety_ucb_authority", "caller_supplied_development_only"),
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


def build_calibrated_authoritative_mpc_candidate(
    *,
    candidate_id: str,
    projected_action_sequence: np.ndarray,
    anchor_action: np.ndarray,
    guard_evidence: ProjectionGuardEvidence,
    safety_prediction: CalibratedSafetyPrediction,
    tfv_delta_di_m3: float,
    action_cost: float,
    terminal_cost: float,
    uncertainty_cost: float,
    executable: bool,
    changed_atol: float = 1.0e-9,
    metadata: Mapping[str, object] | None = None,
) -> MPCandidate:
    """Formal builder: derive hard-safety quantities from calibrated evidence."""
    meta = dict(metadata or {})
    meta.update(safety_prediction.metadata())
    meta["formal_candidate_builder"] = "build_calibrated_authoritative_mpc_candidate"
    return build_authoritative_mpc_candidate(
        candidate_id=candidate_id,
        projected_action_sequence=projected_action_sequence,
        anchor_action=anchor_action,
        guard_evidence=guard_evidence,
        pfv_delta_ucb_m3=safety_prediction.pfv_delta_ucb_m3,
        peak_delta_ucb_m3s=safety_prediction.peak_delta_ucb_m3s,
        tfv_delta_di_m3=tfv_delta_di_m3,
        action_cost=action_cost,
        terminal_cost=terminal_cost,
        uncertainty_cost=uncertainty_cost,
        uncertainty_pass=safety_prediction.uncertainty_pass,
        ood_pass=safety_prediction.ood_pass,
        executable=executable,
        changed_atol=changed_atol,
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
