from __future__ import annotations

"""Dual-reference safety contracts for Project6 V4.

No-control and Passive define the PFV safety envelope. Internal defines the
TFV/Peak performance envelope.  The module is deliberately independent of
PySWMM so it can be unit-tested and used by both online and audit paths.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence
import math

import numpy as np


class HydraulicPhase(str, Enum):
    PRE_RAIN = "pre_rain"
    RISING = "rising"
    PEAK = "peak"
    RECESSION = "recession"
    RECOVERED = "recovered"


@dataclass(frozen=True)
class DualReferenceLimits:
    pfv_abs_margin_m3: float = 0.0
    pfv_rel_margin: float = 0.0
    tfv_abs_margin_m3: float = 0.0
    tfv_rel_margin: float = 0.0
    peak_abs_margin: float = 0.0
    peak_rel_margin: float = 0.0
    quantile: float = 0.95
    readback_tolerance: float = 1.0e-4
    action_deadband: float = 0.02
    max_k: int = 8


@dataclass(frozen=True)
class ReferenceEnvelope:
    no_control_pfv: float
    passive_pfv: float
    internal_tfv: float
    internal_peak: float

    def pfv_cap(self, limits: DualReferenceLimits) -> float:
        nc = float(self.no_control_pfv) + max(
            float(limits.pfv_abs_margin_m3),
            max(0.0, float(self.no_control_pfv)) * float(limits.pfv_rel_margin),
        )
        passive = float(self.passive_pfv) + max(
            float(limits.pfv_abs_margin_m3),
            max(0.0, float(self.passive_pfv)) * float(limits.pfv_rel_margin),
        )
        return min(nc, passive)

    def tfv_cap(self, limits: DualReferenceLimits) -> float:
        return float(self.internal_tfv) + max(
            float(limits.tfv_abs_margin_m3),
            max(0.0, float(self.internal_tfv)) * float(limits.tfv_rel_margin),
        )

    def peak_cap(self, limits: DualReferenceLimits) -> float:
        return float(self.internal_peak) + max(
            float(limits.peak_abs_margin),
            max(0.0, float(self.internal_peak)) * float(limits.peak_rel_margin),
        )


@dataclass(frozen=True)
class CandidateEnvelopePrediction:
    candidate_id: str
    pfv_quantile_m3: float
    tfv_ucb_m3: float
    peak_ucb: float
    changed_facilities: int
    total_abs_variation: float
    reversal_count: int = 0
    uncertainty_pass: bool = True
    ood_pass: bool = True
    safety_classifier_pass: bool = True


@dataclass(frozen=True)
class CandidateGateResult:
    accepted: bool
    reasons: tuple[str, ...]
    action_cost: float
    benefit: float
    benefit_cost_ratio: float
    pfv_cap_m3: float
    tfv_cap_m3: float
    peak_cap: float


@dataclass
class EventQuantilePfvBudget:
    """Event-level PFV quantile ledger without an online hydraulic oracle.

    The safety cap is forecast causally from the No-control/Passive envelope and
    is only allowed to tighten as the event unfolds.  The live Proposed branch
    debits the cap with observed priority flooding volume.  A candidate is
    accepted only when its predicted full-event PFV quantile remains under the
    frozen/tightened cap.
    """

    no_control_event_pfv_quantile_m3: float = 0.0
    passive_event_pfv_quantile_m3: float = 0.0
    abs_margin_m3: float = 0.0
    rel_margin: float = 0.0
    observed_proposed_pfv_m3: float = 0.0
    inflight_candidate_event_pfv_m3: float = 0.0
    frozen_cap_m3: float | None = None
    ledger: list[dict[str, float | str]] = field(default_factory=list)

    def _forecast_cap(self) -> float:
        ref = min(
            max(0.0, float(self.no_control_event_pfv_quantile_m3)),
            max(0.0, float(self.passive_event_pfv_quantile_m3)),
        )
        return ref + max(float(self.abs_margin_m3), ref * float(self.rel_margin))

    @property
    def event_cap_m3(self) -> float:
        return float(self.frozen_cap_m3 if self.frozen_cap_m3 is not None else self._forecast_cap())

    @property
    def remaining_m3(self) -> float:
        return self.event_cap_m3 - max(0.0, float(self.observed_proposed_pfv_m3))

    def freeze_or_tighten(self, no_control_quantile_m3: float, passive_quantile_m3: float) -> float:
        self.no_control_event_pfv_quantile_m3 = max(0.0, float(no_control_quantile_m3))
        self.passive_event_pfv_quantile_m3 = max(0.0, float(passive_quantile_m3))
        candidate_cap = self._forecast_cap()
        self.frozen_cap_m3 = candidate_cap if self.frozen_cap_m3 is None else min(float(self.frozen_cap_m3), candidate_cap)
        self.ledger.append({"kind": "reference_cap", "cap_m3": float(self.frozen_cap_m3)})
        return float(self.frozen_cap_m3)

    def update_observed(self, proposed_cumulative_pfv_m3: float, safety_reference_cumulative_pfv_m3: float | None = None) -> None:
        del safety_reference_cumulative_pfv_m3
        self.observed_proposed_pfv_m3 = max(0.0, float(proposed_cumulative_pfv_m3))
        self.ledger.append({"kind": "observed_proposed", "debit_m3": self.observed_proposed_pfv_m3})

    def set_inflight(self, predicted_event_pfv_quantile_m3: float) -> None:
        self.inflight_candidate_event_pfv_m3 = max(0.0, float(predicted_event_pfv_quantile_m3))
        self.ledger.append({"kind": "candidate_event_quantile", "predicted_m3": self.inflight_candidate_event_pfv_m3})

    def allows(self, predicted_event_pfv_quantile_m3: float) -> bool:
        predicted = max(0.0, float(predicted_event_pfv_quantile_m3))
        return (
            self.observed_proposed_pfv_m3 <= self.event_cap_m3 + 1.0e-9
            and predicted <= self.event_cap_m3 + 1.0e-9
        )


@dataclass(frozen=True)
class PhaseFallbackDecision:
    fallback_id: str
    reason: str
    phase: HydraulicPhase


def classify_phase(
    *,
    current_rainfall: float,
    future_rainfall: Sequence[float],
    priority_depth_ratio: float,
    elapsed_since_rain_end_min: float | None,
) -> HydraulicPhase:
    future = np.asarray(list(future_rainfall), dtype=float)
    future = np.nan_to_num(future, nan=0.0)
    rain = max(0.0, float(current_rainfall))
    if elapsed_since_rain_end_min is not None and elapsed_since_rain_end_min >= 60 and priority_depth_ratio <= 0.01:
        return HydraulicPhase.RECOVERED
    if rain <= 1.0e-9 and (future.size == 0 or float(future.max()) <= 1.0e-9):
        return HydraulicPhase.RECESSION if elapsed_since_rain_end_min is not None else HydraulicPhase.PRE_RAIN
    future_peak = float(future.max()) if future.size else rain
    if priority_depth_ratio >= 0.90 or (rain >= 0.95 * max(future_peak, 1.0e-9) and rain > 0.0):
        return HydraulicPhase.PEAK
    return HydraulicPhase.RISING


def choose_phase_aware_fallback(
    *,
    phase: HydraulicPhase,
    pfv_budget_remaining_m3: float,
    internal_predicted_pfv_quantile_m3: float,
    pfv_cap_m3: float,
    internal_legal: bool,
    passive_legal: bool,
) -> PhaseFallbackDecision:
    """Freeze fallback before candidate scoring.

    Passive is the executable PFV-safe fallback. Internal is used only when PFV
    headroom exists and the event is at peak/recession, where TFV/Peak relief is
    needed. No-control remains the PFV reference twin rather than an in-branch
    executable policy because native SWMM rules cannot be removed mid-event.
    """
    pfv_tight = float(pfv_budget_remaining_m3) <= 0.0 or float(internal_predicted_pfv_quantile_m3) > float(pfv_cap_m3)
    if passive_legal and (pfv_tight or phase in {HydraulicPhase.PRE_RAIN, HydraulicPhase.RISING}):
        return PhaseFallbackDecision("passive_anchor", "pfv_safety_phase_or_budget_tight", phase)
    if internal_legal and phase in {HydraulicPhase.PEAK, HydraulicPhase.RECESSION}:
        return PhaseFallbackDecision("internal_rules", "tfv_peak_relief_with_pfv_headroom", phase)
    if passive_legal:
        return PhaseFallbackDecision("passive_anchor", "passive_only_legal_safe_fallback", phase)
    if internal_legal and not pfv_tight:
        return PhaseFallbackDecision("internal_rules", "internal_only_legal_with_pfv_headroom", phase)
    raise ValueError("no legal fallback satisfies the phase-aware PFV safety contract")


def evaluate_candidate(
    prediction: CandidateEnvelopePrediction,
    *,
    references: ReferenceEnvelope,
    limits: DualReferenceLimits,
    event_budget: EventQuantilePfvBudget,
    minimum_material_benefit: float,
    minimum_benefit_cost_ratio: float,
    changed_facility_penalty: float,
    variation_penalty: float,
    reversal_penalty: float,
) -> CandidateGateResult:
    reasons: list[str] = []
    pfv_cap = min(references.pfv_cap(limits), event_budget.event_cap_m3)
    tfv_cap = references.tfv_cap(limits)
    peak_cap = references.peak_cap(limits)
    if prediction.pfv_quantile_m3 > pfv_cap + 1.0e-9:
        reasons.append("pfv_quantile_worse_than_no_control_passive_envelope")
    if not event_budget.allows(prediction.pfv_quantile_m3):
        reasons.append("event_pfv_budget_exhausted")
    if prediction.tfv_ucb_m3 > tfv_cap + 1.0e-9:
        reasons.append("tfv_ucb_worse_than_internal")
    if prediction.peak_ucb > peak_cap + 1.0e-9:
        reasons.append("peak_ucb_worse_than_internal")
    if prediction.changed_facilities > int(limits.max_k):
        reasons.append("adaptive_k_or_hard_k_exceeded")
    if not prediction.uncertainty_pass:
        reasons.append("uncertainty_rejected")
    if not prediction.ood_pass:
        reasons.append("ood_rejected")
    if not prediction.safety_classifier_pass:
        reasons.append("safety_classifier_rejected")
    action_cost = (
        float(changed_facility_penalty) * max(0, int(prediction.changed_facilities))
        + float(variation_penalty) * max(0.0, float(prediction.total_abs_variation))
        + float(reversal_penalty) * max(0, int(prediction.reversal_count))
    )
    benefit = max(0.0, references.internal_tfv - prediction.tfv_ucb_m3) + max(0.0, references.internal_peak - prediction.peak_ucb)
    ratio = math.inf if action_cost <= 1.0e-12 and benefit > 0.0 else (benefit / action_cost if action_cost > 0.0 else 0.0)
    if benefit < float(minimum_material_benefit):
        reasons.append("material_benefit_below_minimum")
    if ratio < float(minimum_benefit_cost_ratio):
        reasons.append("benefit_cost_ratio_below_minimum")
    return CandidateGateResult(
        accepted=not reasons,
        reasons=tuple(reasons),
        action_cost=float(action_cost),
        benefit=float(benefit),
        benefit_cost_ratio=float(ratio),
        pfv_cap_m3=float(pfv_cap),
        tfv_cap_m3=float(tfv_cap),
        peak_cap=float(peak_cap),
    )


def adaptive_k(
    *,
    phase: HydraulicPhase,
    pfv_headroom_fraction: float,
    uncertainty_score: float,
    allowed_values: Iterable[int] = (0, 2, 4, 6, 8),
) -> int:
    values = sorted({max(0, int(v)) for v in allowed_values}) or [0]
    if phase in {HydraulicPhase.PRE_RAIN, HydraulicPhase.RECOVERED}:
        target = 0
    elif pfv_headroom_fraction <= 0.05 or uncertainty_score >= 0.80:
        target = 0
    elif phase == HydraulicPhase.RISING:
        target = 2
    elif phase == HydraulicPhase.PEAK and pfv_headroom_fraction >= 0.50 and uncertainty_score <= 0.25:
        target = 6
    else:
        target = 4
    return max(v for v in values if v <= target) if any(v <= target for v in values) else min(values)


def enforce_final_readback(
    *,
    requested: Sequence[float],
    projected: Sequence[float],
    readback: Sequence[float],
    anchor: Sequence[float],
    actuator_ids: Sequence[str],
    binary_pump_ids: Iterable[str],
    max_k: int,
    tolerance: float,
    deadband: float,
) -> dict[str, object]:
    req = np.asarray(requested, dtype=float).reshape(-1)
    proj = np.asarray(projected, dtype=float).reshape(-1)
    rb = np.asarray(readback, dtype=float).reshape(-1)
    anc = np.asarray(anchor, dtype=float).reshape(-1)
    n = min(len(actuator_ids), req.size, proj.size, rb.size, anc.size)
    binary = {str(x) for x in binary_pump_ids}
    mismatches: list[str] = []
    binary_violations: list[str] = []
    changed: list[str] = []
    for i, aid in enumerate(actuator_ids[:n]):
        if not np.isfinite(rb[i]) or abs(rb[i] - proj[i]) > float(tolerance):
            mismatches.append(str(aid))
        if str(aid) in binary and min(abs(rb[i]), abs(rb[i] - 1.0)) > float(tolerance):
            binary_violations.append(str(aid))
        if abs(rb[i] - anc[i]) > float(deadband):
            changed.append(str(aid))
    reasons: list[str] = []
    if mismatches:
        reasons.append("write_readback_mismatch")
    if binary_violations:
        reasons.append("binary_readback_violation")
    if len(changed) > int(max_k):
        reasons.append("final_readback_k_exceeded")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "mismatch_facility_ids": mismatches,
        "binary_violation_facility_ids": binary_violations,
        "readback_deviation_facility_ids": changed,
        "readback_k": len(changed),
    }
