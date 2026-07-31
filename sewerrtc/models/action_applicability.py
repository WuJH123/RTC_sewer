from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ApplicabilityResult:
    applicability_prob: float
    expected_benefit: float
    risk_penalty: float
    uncertainty_penalty: float
    recommended_phase: str
    disallow_reason: str

    @property
    def action_value(self) -> float:
        return float(self.expected_benefit - self.risk_penalty - self.uncertainty_penalty)


def heuristic_action_applicability(features: dict) -> ApplicabilityResult:
    phase = str(features.get("rain_phase", features.get("phase", "unknown")))
    priority_risk = float(features.get("priority_risk_score", features.get("priority_depth_max", 0.0)) or 0.0)
    downstream_margin = float(features.get("downstream_capacity_margin", 1.0) or 0.0)
    storage_available = float(features.get("upstream_storage_available", 1.0) or 0.0)
    uncertainty = float(features.get("uncertainty_score", 0.0) or 0.0)
    benefit = max(0.0, priority_risk) * (0.5 + 0.5 * max(0.0, storage_available))
    risk_penalty = max(0.0, -downstream_margin) + (0.5 if phase == "recession" and priority_risk < 0.2 else 0.0)
    prob = max(0.0, min(1.0, 0.35 + benefit - risk_penalty - uncertainty))
    reason = "" if prob >= 0.5 else "low_applicability_or_high_transfer_risk"
    return ApplicabilityResult(prob, benefit, risk_penalty, uncertainty, phase, reason)
