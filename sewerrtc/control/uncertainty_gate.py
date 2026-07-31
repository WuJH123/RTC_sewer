from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UncertaintyGateDecision:
    pass_gate: bool
    reason: str


def evaluate_uncertainty_gate(
    delta_pfv_p50: float,
    delta_tfv_p90: float,
    delta_peak_p90: float,
    uncertainty_score: float,
    event_risk_class: str,
    min_pfv_gain: float,
    epsilon_tfv: float,
    epsilon_peak: float,
    max_uncertainty: float,
) -> UncertaintyGateDecision:
    reasons = []
    if str(event_risk_class) == "low_risk_event" and abs(float(delta_pfv_p50)) < abs(float(min_pfv_gain)) * 2.0:
        reasons.append("low_risk_requires_strong_gain")
    if float(delta_pfv_p50) >= -abs(float(min_pfv_gain)):
        reasons.append("pfv_gain_insufficient")
    if float(delta_tfv_p90) > float(epsilon_tfv):
        reasons.append("tfv_p90_guard_failed")
    if float(delta_peak_p90) > float(epsilon_peak):
        reasons.append("peak_p90_guard_failed")
    if float(uncertainty_score) > float(max_uncertainty):
        reasons.append("uncertainty_too_high")
    return UncertaintyGateDecision(not reasons, ";".join(reasons) if reasons else "pass")
