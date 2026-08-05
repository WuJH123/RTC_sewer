"""Minimal operational contract for Project6 V4.2 real-time control.

Scientific contract
-------------------
The only hydraulic safety constraint is

    UCB(PFV_candidate - 1.05 * PFV_no_control) <= 100 m3

and the admitted candidate with the smallest predicted TFV is selected.

Engineering36 is treated as the controllable plant, not as an additional set of
scientific safety gates.  The runtime therefore keeps only what is necessary to
issue a meaningful SWMM command: finite H12 action arrays, settings in [0, 1],
valid binary semantics for the two verified binary pumps, H3 executable-prefix
semantics, and target-setting write/readback. K/rate/ramp/dwell/interlock are
diagnostics only and cannot empty the PFV-safe set.

This module deliberately patches the existing production modules at runtime so
legacy evidence readers can coexist while the active control path stays simple.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from sewerrtc.control.pfvfirst_mpc_v42 import (
    CandidateAudit,
    EngineeringStatus,
    FrozenFallback,
    MPCDecision,
    MPCWeights,
    SafetyMargins,
)
from sewerrtc.v4 import v42_formal_runtime as base_runtime


CONTRACT_ID = "PROJECT6_V42_SIMPLE_RTC_CORE_V1"
PFV_ABSOLUTE_ALLOWANCE_M3 = 100.0
PFV_RELATIVE_ALLOWANCE_FRACTION = 0.05
BINARY_IDS = {"ADD301.2", "ADD301.3"}


def project_candidate_sequence_minimal(
    sequence: np.ndarray,
    current_action: np.ndarray,
    actuators: pd.DataFrame,
) -> tuple[np.ndarray, EngineeringStatus, int, bool]:
    """Project to the minimal physical SWMM action domain.

    The first H3 is controllable; H4-H12 are held at the current readback because
    the controller replans every 10 minutes. Only ADD301.2/ADD301.3 are forced
    to binary settings; all other managed facilities remain continuous in [0,1].
    No K/rate/ramp/dwell/interlock rejection is applied.
    """
    ids = actuators["actuator_id"].astype(str).tolist()
    current = np.asarray(current_action, dtype=np.float32).reshape(-1)
    seq = np.asarray(sequence, dtype=np.float32).copy()
    if seq.shape != (base_runtime.HORIZON_STEPS, len(ids)) or current.size != len(ids):
        failed = EngineeringStatus(False, True, True, True, True)
        return seq, failed, 999, False
    if not np.isfinite(seq).all() or not np.isfinite(current).all():
        failed = EngineeringStatus(False, True, True, True, True)
        return seq, failed, 999, False

    seq = np.clip(seq, 0.0, 1.0)
    seq[base_runtime.CONTROLLABLE_PREFIX_STEPS :] = current[None, :]
    for aid in BINARY_IDS:
        if aid in ids:
            idx = ids.index(aid)
            seq[: base_runtime.CONTROLLABLE_PREFIX_STEPS, idx] = np.where(
                seq[: base_runtime.CONTROLLABLE_PREFIX_STEPS, idx] >= 0.5,
                1.0,
                0.0,
            )

    bounds = bool(np.isfinite(seq).all() and np.all((seq >= 0.0) & (seq <= 1.0)))
    binary = True
    for aid in BINARY_IDS:
        if aid in ids:
            values = seq[: base_runtime.CONTROLLABLE_PREFIX_STEPS, ids.index(aid)]
            binary = binary and bool(
                np.all(np.isclose(values, 0.0, atol=1e-6) | np.isclose(values, 1.0, atol=1e-6))
            )
    physical_validity = bool(bounds and binary)
    changed = np.any(
        np.abs(seq[: base_runtime.CONTROLLABLE_PREFIX_STEPS] - current[None, :]) > 1e-6,
        axis=0,
    )
    changed_count = int(changed.sum())
    status = EngineeringStatus(
        bounds=physical_validity,
        rate=True,
        ramp=True,
        dwell=True,
        interlock=True,
    )
    return seq.astype(np.float32), status, changed_count, physical_validity


def _candidate_audit(
    candidate: Any,
    *,
    margins: SafetyMargins,
) -> CandidateAudit:
    reasons: list[str] = []
    action = np.asarray(candidate.action_sequence, dtype=float)
    if action.ndim != 2 or not np.isfinite(action).all():
        reasons.append("invalid_action_sequence")
    if not bool(candidate.executable):
        reasons.append("action_not_physically_writable")

    budget = candidate.pfv_budget_metric_ucb_m3
    if budget is None:
        no_control = float(candidate.pfv_no_control_m3)
        legacy_ucb = float(candidate.pfv_delta_ucb_m3)
        if not np.isfinite(no_control) or not np.isfinite(legacy_ucb):
            reasons.append("nonfinite_PFV_prediction")
            budget_value = None
        else:
            budget_value = legacy_ucb - margins.pfv_relative_allowance_fraction * max(no_control, 0.0)
    else:
        budget_value = float(budget)

    if budget_value is None or not np.isfinite(float(budget_value)):
        if "nonfinite_PFV_prediction" not in reasons:
            reasons.append("nonfinite_PFV_prediction")
    elif float(budget_value) > float(margins.pfv_absolute_allowance_m3):
        reasons.append("PFV_budget_exceeded_vs_no_control")

    tfv = float(candidate.tfv_delta_di_m3)
    if not np.isfinite(tfv):
        reasons.append("nonfinite_TFV_prediction")

    return CandidateAudit(
        candidate_id=str(candidate.candidate_id),
        safe=not reasons,
        rejection_reasons=tuple(reasons),
        objective=None,
        pfv_allowance_m3=(
            float(margins.pfv_absolute_allowance_m3)
            + float(margins.pfv_relative_allowance_fraction)
            * max(0.0, float(candidate.pfv_no_control_m3))
            if np.isfinite(float(candidate.pfv_no_control_m3))
            else None
        ),
        maximum_priority_depth_exceedance_m=None,
        pfv_budget_metric_ucb_m3=(
            float(budget_value)
            if budget_value is not None and np.isfinite(float(budget_value))
            else None
        ),
    )


def decide_simple_pfv_tfv_mpc(
    *,
    candidates: Sequence[Any],
    fallback: FrozenFallback,
    margins: SafetyMargins | None = None,
    weights: MPCWeights | None = None,
    expected_fallback_contract_hash: str | None = None,
) -> MPCDecision:
    """Select minimum predicted TFV subject only to the PFV-UCB budget.

    Minimal action-domain validity is required so the command can be written to
    SWMM. K/rate/ramp/dwell/interlock are not candidate rejection reasons.
    """
    del weights
    margins = margins or SafetyMargins(
        pfv_absolute_allowance_m3=PFV_ABSOLUTE_ALLOWANCE_M3,
        pfv_relative_allowance_fraction=PFV_RELATIVE_ALLOWANCE_FRACTION,
        max_changed_facilities=36,
    )
    audits: list[CandidateAudit] = []
    admitted: list[tuple[float, str, Any]] = []
    for candidate in candidates:
        audit = _candidate_audit(candidate, margins=margins)
        if not audit.safe:
            audits.append(audit)
            continue
        score = float(candidate.tfv_delta_di_m3)
        audits.append(
            CandidateAudit(
                candidate_id=audit.candidate_id,
                safe=True,
                rejection_reasons=(),
                objective=score,
                pfv_allowance_m3=audit.pfv_allowance_m3,
                maximum_priority_depth_exceedance_m=None,
                pfv_budget_metric_ucb_m3=audit.pfv_budget_metric_ucb_m3,
            )
        )
        admitted.append((score, str(candidate.candidate_id), candidate))

    if not admitted:
        sequence = np.asarray(fallback.action_sequence, dtype=float)
        if sequence.ndim != 2 or not np.isfinite(sequence).all():
            raise RuntimeError("fallback action is invalid")
        if expected_fallback_contract_hash is not None and str(fallback.contract_hash) != str(
            expected_fallback_contract_hash
        ):
            raise RuntimeError("fallback contract hash mismatch")
        return MPCDecision(
            selected_id=fallback.fallback_id,
            execute_action=sequence[0].copy(),
            selected_sequence=sequence.copy(),
            used_fallback=True,
            reason="pfv_safe_set_empty",
            objective=None,
            audits=tuple(audits),
            metadata={
                "control_objective_contract": CONTRACT_ID,
                "hydraulic_hard_constraints": ["PFV_budget_vs_no_control"],
                "operational_gate_policy": "minimal_physical_validity_only",
            },
        )

    score, _, selected = min(admitted, key=lambda item: (item[0], item[1]))
    sequence = np.asarray(selected.action_sequence, dtype=float)
    return MPCDecision(
        selected_id=str(selected.candidate_id),
        execute_action=sequence[0].copy(),
        selected_sequence=sequence.copy(),
        used_fallback=False,
        reason="minimum_TFV_within_PFV_UCB_budget",
        objective=float(score),
        audits=tuple(audits),
        metadata={
            "objective": "minimize_TFV_subject_to_PFV_budget",
            "hydraulic_hard_constraints": ["PFV_budget_vs_no_control"],
            "PFV_rule": "UCB(PFV_candidate - 1.05 * PFV_no_control) <= 100 m3",
            "operational_gate_policy": "minimal_physical_validity_only",
            "K_rate_ramp_dwell_interlock_role": "diagnostic_only",
            "peak_role": "reporting_only",
            "priority_depth_role": "diagnostic_only",
            "control_objective_contract": CONTRACT_ID,
        },
    )


def _no_dwell_guard(
    command: np.ndarray,
    current: np.ndarray,
    ids: list[str],
    *,
    decision_step: int,
    last_change_step: dict[str, int],
) -> tuple[np.ndarray, bool, list[str]]:
    """Compatibility hook: cross-decision dwell is diagnostic, not a gate."""
    del current, ids, decision_step, last_change_step
    return np.asarray(command, dtype=np.float32).copy(), True, []


def apply_simple_rtc_contract() -> None:
    """Patch the active Formal runtime to the simplified RTC contract."""
    from sewerrtc.v4 import v42_formal_runtime_safe as safe_runtime
    from sewerrtc.v4 import v42_pfv_tfv_runtime_patch as production_selector

    base_runtime.project_candidate_sequence = project_candidate_sequence_minimal
    production_selector.decide_pfvfirst_mpc = decide_simple_pfv_tfv_mpc
    safe_runtime._runtime_dwell_guard = _no_dwell_guard
