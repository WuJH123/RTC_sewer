from __future__ import annotations

from .formal_policy import scenario_aware_intervention_decision


def observation_driven_gate(observation: dict, prediction: dict, config: dict | None = None):
    cfg = (config or {}).get("intervention_policy", config or {})
    return scenario_aware_intervention_decision(
        event_risk_class=str(observation.get("event_risk_class", "unknown")),
        current_priority_risk_class=str(observation.get("current_priority_risk_class", "unknown")),
        predicted_pfv_gain=float(prediction.get("predicted_pfv_gain", 0.0) or 0.0),
        predicted_pfv_gain_abs=float(prediction.get("predicted_pfv_gain_abs", 0.0) or 0.0),
        min_pfv_gain_low=float(cfg.get("min_pfv_gain_low_risk", 0.20)),
        min_pfv_gain_medium=float(cfg.get("min_pfv_gain_medium_risk", 0.05)),
        min_pfv_gain_high=float(cfg.get("min_pfv_gain_high_risk", 0.02)),
        require_predicted_pfv_improvement=bool(cfg.get("require_predicted_pfv_improvement", True)),
    )
