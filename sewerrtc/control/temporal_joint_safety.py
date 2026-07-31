from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class JointSafetyConfig:
    pfv_abs_margin_m3: float = 100.0
    pfv_rel_margin: float = 0.005
    event_pfv_budget_enabled: bool = False
    event_pfv_abs_margin_m3: float = 100.0
    event_pfv_rel_margin: float = 0.005
    peak_margin: float = 0.0
    uncertainty_z: float = 1.645
    min_tfv_lcb_reduction: float = 0.0
    min_pfv_noninferiority_probability: float = 0.0
    min_tfv_improvement_probability: float = 0.0
    min_peak_safe_probability: float = 0.0
    use_classifier_thresholds: bool = True
    tfv_reduction_weight: float = 1.0
    peak_reduction_weight: float = 0.0
    action_l1_penalty: float = 0.0
    simultaneous_action_penalty: float = 0.0
    pump_switch_penalty: float = 0.0
    engineering_template_bonus: float = 0.0


@dataclass(frozen=True)
class JointCandidatePrediction:
    label: str
    delta_pfv: float
    delta_tfv: float
    delta_peak: float
    sigma_pfv: float
    sigma_tfv: float
    sigma_peak: float
    simultaneous_actuators: int
    action_l1: float
    pump_switches: int
    pfv_noninferiority_probability: float = 1.0
    tfv_improvement_probability: float = 1.0
    peak_safe_probability: float = 1.0
    pfv_classifier_threshold: float = 0.0
    tfv_classifier_threshold: float = 0.0
    peak_classifier_threshold: float = 0.0


def select_lexicographic_candidate(
    predictions: Iterable[JointCandidatePrediction],
    *,
    reference_pfv: float,
    config: JointSafetyConfig,
) -> tuple[JointCandidatePrediction, dict[str, dict[str, float | bool | str]]]:
    """Apply PFV and peak safety before ranking reliable TFV repair.

    Effects use ``candidate - online No-control reference`` semantics. The
    uncertainty bounds are deliberately computed on the paired effect, not on
    two independently inflated absolute predictions.
    """

    rows = list(predictions)
    if not rows:
        raise ValueError("at least one candidate prediction is required")
    reference = next((row for row in rows if row.label == "reference"), rows[0])
    pfv_margin = max(
        float(config.pfv_abs_margin_m3),
        float(config.pfv_rel_margin) * max(0.0, float(reference_pfv)),
    )
    z = max(0.0, float(config.uncertainty_z))
    accepted: list[tuple[JointCandidatePrediction, float]] = []
    audit: dict[str, dict[str, float | bool | str]] = {}
    for row in rows:
        pfv_ucb = float(row.delta_pfv) + z * max(0.0, float(row.sigma_pfv))
        peak_ucb = float(row.delta_peak) + z * max(0.0, float(row.sigma_peak))
        tfv_reduction_lcb = -float(row.delta_tfv) - z * max(0.0, float(row.sigma_tfv))
        reason = "accepted"
        if row.label != reference.label and pfv_ucb > pfv_margin:
            reason = "pfv_noninferiority"
        elif (
            row.label != reference.label
            and bool(config.use_classifier_thresholds)
            and float(row.pfv_noninferiority_probability) < max(
                float(config.min_pfv_noninferiority_probability), float(row.pfv_classifier_threshold)
            )
        ):
            reason = "pfv_classifier"
        elif row.label != reference.label and peak_ucb > float(config.peak_margin):
            reason = "peak_safety"
        elif (
            row.label != reference.label
            and bool(config.use_classifier_thresholds)
            and float(row.peak_safe_probability) < max(
                float(config.min_peak_safe_probability), float(row.peak_classifier_threshold)
            )
        ):
            reason = "peak_classifier"
        elif (
            row.label != reference.label
            and bool(config.use_classifier_thresholds)
            and float(row.tfv_improvement_probability) < max(
                float(config.min_tfv_improvement_probability), float(row.tfv_classifier_threshold)
            )
        ):
            reason = "tfv_classifier"
        elif row.label != reference.label and tfv_reduction_lcb < float(config.min_tfv_lcb_reduction):
            reason = "insufficient_tfv_lcb"
        else:
            accepted.append((row, tfv_reduction_lcb))
        audit[row.label] = {
            "accepted": reason == "accepted",
            "rejection_reason": reason,
            "pfv_effect_ucb": pfv_ucb,
            "pfv_margin": pfv_margin,
            "peak_effect_ucb": peak_ucb,
            "peak_margin": float(config.peak_margin),
            "tfv_reduction_lcb": tfv_reduction_lcb,
            "pfv_noninferiority_probability": float(row.pfv_noninferiority_probability),
            "minimum_pfv_noninferiority_probability": float(config.min_pfv_noninferiority_probability),
            "pfv_classifier_threshold": float(row.pfv_classifier_threshold),
            "tfv_improvement_probability": float(row.tfv_improvement_probability),
            "minimum_tfv_improvement_probability": float(config.min_tfv_improvement_probability),
            "tfv_classifier_threshold": float(row.tfv_classifier_threshold),
            "peak_safe_probability": float(row.peak_safe_probability),
            "minimum_peak_safe_probability": float(config.min_peak_safe_probability),
            "peak_classifier_threshold": float(row.peak_classifier_threshold),
        }

    active = [(row, benefit) for row, benefit in accepted if row.label != reference.label]
    if not active:
        return reference, audit
    def deployment_score(row: JointCandidatePrediction, tfv_reduction_lcb: float) -> float:
        # Positive is better. TFV is the primary repair target; peak reduction
        # is an explicit secondary benefit when configured. Complexity
        # penalties keep engineering actions executable without suppressing all
        # multi-actuator behavior.
        peak_reduction = -float(row.delta_peak)
        return (
            float(config.tfv_reduction_weight) * float(tfv_reduction_lcb)
            + float(config.peak_reduction_weight) * peak_reduction
            + (float(config.engineering_template_bonus) if str(row.label).startswith("engineered_") else 0.0)
            - float(config.action_l1_penalty) * float(row.action_l1)
            - float(config.simultaneous_action_penalty) * float(row.simultaneous_actuators)
            - float(config.pump_switch_penalty) * float(row.pump_switches)
        )

    best = max(
        active,
        key=lambda item: (
            deployment_score(item[0], item[1]),
            -int(item[0].simultaneous_actuators),
            -float(item[0].action_l1),
            -int(item[0].pump_switches),
            str(item[0].label),
        ),
    )[0]
    return best, audit
