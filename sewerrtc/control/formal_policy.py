from __future__ import annotations

from .policy_base import InterventionDecision


def scenario_aware_intervention_decision(
    event_risk_class: str,
    current_priority_risk_class: str,
    predicted_pfv_gain: float,
    predicted_pfv_gain_abs: float,
    min_pfv_gain_low: float = 0.20,
    min_pfv_gain_medium: float = 0.05,
    min_pfv_gain_high: float = 0.02,
    require_predicted_pfv_improvement: bool = True,
) -> InterventionDecision:
    cls = str(event_risk_class or "unknown")
    gain = float(predicted_pfv_gain or 0.0)
    gain_abs = float(predicted_pfv_gain_abs or 0.0)
    if require_predicted_pfv_improvement and gain <= 0 and gain_abs <= 0:
        return InterventionDecision(False, "no_predicted_pfv_improvement", cls, current_priority_risk_class, gain, gain_abs)
    if cls == "low_risk_event":
        min_gain = float(min_pfv_gain_low)
    elif cls == "high_risk_event":
        min_gain = float(min_pfv_gain_high)
    else:
        min_gain = float(min_pfv_gain_medium)
    if gain < min_gain and gain_abs < min_gain:
        return InterventionDecision(False, f"predicted_gain_below_{min_gain}", cls, current_priority_risk_class, gain, gain_abs)
    return InterventionDecision(True, "scenario_aware_gate_pass", cls, current_priority_risk_class, gain, gain_abs)
