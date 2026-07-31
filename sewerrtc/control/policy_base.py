from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InterventionDecision:
    allowed: bool
    reason: str
    event_risk_class: str = "unknown"
    current_priority_risk_class: str = "unknown"
    predicted_pfv_gain: float = 0.0
    predicted_pfv_gain_abs: float = 0.0


def risk_class_from_pfv(pfv: float, low_threshold: float = 1000.0, high_threshold: float = 50000.0) -> str:
    val = float(pfv or 0.0)
    if val >= float(high_threshold):
        return "high_risk_event"
    if val <= float(low_threshold):
        return "low_risk_event"
    return "medium_risk_event"
