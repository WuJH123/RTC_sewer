from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyThresholds:
    pfv_min_improve: float = 1.0
    tfv_guard: float = 0.005
    peak_guard: float = 0.010


def is_safe_delta(delta_pfv: float, delta_tfv: float, delta_peak: float, thresholds: SafetyThresholds) -> bool:
    return delta_pfv < -thresholds.pfv_min_improve and delta_tfv <= thresholds.tfv_guard and delta_peak <= thresholds.peak_guard

