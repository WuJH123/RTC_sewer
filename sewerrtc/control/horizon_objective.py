from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HorizonCandidateScore:
    score: float
    gate_pass: bool
    tfv_guard: float
    peak_guard: float
    tfv_violation: float
    peak_violation: float
    pfv_term: float
    smooth_term: float
    penalty_term: float


@dataclass(frozen=True)
class HorizonSequenceScore:
    score: float
    gate_pass: bool
    pfv_total: float
    reference_pfv_total: float
    tfv_total: float
    reference_tfv_total: float
    peak_tfv_rate: float
    reference_peak_tfv_rate: float
    pfv_violation: float
    tfv_violation: float
    peak_violation: float
    smooth_term: float
    penalty_term: float


def _guard_value(reference: float, pct: float) -> float:
    ref = float(reference or 0.0)
    if ref > 1.1:
        return float(pct) * ref
    return 0.0


def score_horizon_candidate(
    *,
    delta_pfv: float,
    delta_tfv: float,
    delta_peak: float,
    action_change_penalty: float,
    baseline_tfv: float,
    baseline_peak: float,
    tfv_guard_pct: float = 0.005,
    peak_guard_pct: float = 0.010,
    smooth_weight: float = 0.05,
    violation_penalty: float = 1.0e6,
) -> HorizonCandidateScore:
    """Score one discrete horizon candidate under PFV-first safety constraints.

    Negative ``delta_pfv`` is good. TFV and peak deltas may only exceed their
    baseline-relative guards by paying a large penalty, so an unsafe large PFV
    gain cannot dominate a smaller safe improvement.
    """
    tfv_guard = _guard_value(baseline_tfv, tfv_guard_pct)
    peak_guard = _guard_value(baseline_peak, peak_guard_pct)
    tfv_violation = max(0.0, float(delta_tfv) - tfv_guard)
    peak_violation = max(0.0, float(delta_peak) - peak_guard)
    pfv_term = float(delta_pfv)
    smooth_term = float(smooth_weight) * max(0.0, float(action_change_penalty))
    penalty_term = float(violation_penalty) * (tfv_violation + peak_violation)
    return HorizonCandidateScore(
        score=pfv_term + smooth_term + penalty_term,
        gate_pass=bool(tfv_violation <= 1e-12 and peak_violation <= 1e-12),
        tfv_guard=float(tfv_guard),
        peak_guard=float(peak_guard),
        tfv_violation=float(tfv_violation),
        peak_violation=float(peak_violation),
        pfv_term=float(pfv_term),
        smooth_term=float(smooth_term),
        penalty_term=float(penalty_term),
    )


def _as_float_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.asarray([0.0], dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def score_horizon_sequence(
    *,
    pfv,
    tfv,
    peak_tfv_rate,
    action_change,
    reference_pfv=None,
    reference_tfv=None,
    reference_peak=None,
    smooth_weight: float = 0.05,
    violation_penalty: float = 1.0e6,
    pfv_tolerance: float = 0.0,
    pfv_required_improvement: float = 0.0,
    tfv_tolerance: float = 0.0,
    peak_tolerance: float = 0.0,
) -> HorizonSequenceScore:
    """Score a rolling-horizon action sequence under PFV-first constraints.

    ``pfv`` and ``tfv`` are per-step horizon volumes or rates already expressed
    in a consistent unit by the predictor. The objective minimizes total PFV,
    while TFV and peak TFV rate are hard constraints represented by a large
    penalty. This function is independent of native SWMM rules and can be used
    with any city-specific graph/surrogate predictor.
    """
    pfv_arr = _as_float_array(pfv)
    tfv_arr = _as_float_array(tfv)
    peak_arr = _as_float_array(peak_tfv_rate)
    ref_pfv_arr = _as_float_array(reference_pfv) if reference_pfv is not None else np.asarray([], dtype=float)
    ref_tfv_arr = _as_float_array(reference_tfv)
    ref_peak_arr = _as_float_array(reference_peak)
    action_arr = _as_float_array(action_change)

    pfv_total = float(np.sum(np.maximum(0.0, pfv_arr)))
    reference_pfv_total = float(np.sum(np.maximum(0.0, ref_pfv_arr))) if ref_pfv_arr.size else float("inf")
    tfv_total = float(np.sum(np.maximum(0.0, tfv_arr)))
    reference_tfv_total = float(np.sum(np.maximum(0.0, ref_tfv_arr)))
    peak = float(np.max(np.maximum(0.0, peak_arr)))
    reference_peak_value = float(np.max(np.maximum(0.0, ref_peak_arr)))
    pfv_violation = (
        0.0
        if not np.isfinite(reference_pfv_total)
        else max(0.0, pfv_total - reference_pfv_total + float(pfv_required_improvement) - float(pfv_tolerance))
    )
    tfv_violation = max(0.0, tfv_total - reference_tfv_total - float(tfv_tolerance))
    peak_violation = max(0.0, peak - reference_peak_value - float(peak_tolerance))
    smooth_term = float(smooth_weight) * float(np.sum(np.abs(action_arr)))
    penalty_term = float(violation_penalty) * (pfv_violation + tfv_violation + peak_violation)
    return HorizonSequenceScore(
        score=float(pfv_total + smooth_term + penalty_term),
        gate_pass=bool(pfv_violation <= 1e-12 and tfv_violation <= 1e-12 and peak_violation <= 1e-12),
        pfv_total=pfv_total,
        reference_pfv_total=float(reference_pfv_total),
        tfv_total=tfv_total,
        reference_tfv_total=reference_tfv_total,
        peak_tfv_rate=peak,
        reference_peak_tfv_rate=reference_peak_value,
        pfv_violation=float(pfv_violation),
        tfv_violation=float(tfv_violation),
        peak_violation=float(peak_violation),
        smooth_term=float(smooth_term),
        penalty_term=float(penalty_term),
    )


def score_horizon_system_repair_sequence(
    *,
    pfv,
    tfv,
    peak_tfv_rate,
    action_change,
    reference_pfv,
    reference_tfv,
    reference_peak,
    smooth_weight: float = 0.05,
    violation_penalty: float = 1.0e6,
    pfv_tolerance: float = 0.0,
    tfv_required_improvement: float = 0.0,
    tfv_tolerance: float = 0.0,
    peak_tolerance: float = 0.0,
    peak_weight: float = 1.0,
    pfv_weight: float = 1.0,
    tfv_hard_constraint: bool = True,
) -> HorizonSequenceScore:
    """Score no-control-preserving system-risk repair candidates.

    This objective is deliberately different from PFV-first control. It accepts
    a candidate only when priority-zone PFV is non-inferior to the no-control
    reference, TFV improves by at least the requested margin, and peak flooding
    remains within tolerance. Among accepted candidates it minimizes TFV, peak
    risk, and action movement.
    """
    pfv_arr = _as_float_array(pfv)
    tfv_arr = _as_float_array(tfv)
    peak_arr = _as_float_array(peak_tfv_rate)
    ref_pfv_arr = _as_float_array(reference_pfv)
    ref_tfv_arr = _as_float_array(reference_tfv)
    ref_peak_arr = _as_float_array(reference_peak)
    action_arr = _as_float_array(action_change)

    pfv_total = float(np.sum(np.maximum(0.0, pfv_arr)))
    reference_pfv_total = float(np.sum(np.maximum(0.0, ref_pfv_arr)))
    tfv_total = float(np.sum(np.maximum(0.0, tfv_arr)))
    reference_tfv_total = float(np.sum(np.maximum(0.0, ref_tfv_arr)))
    peak = float(np.max(np.maximum(0.0, peak_arr)))
    reference_peak_value = float(np.max(np.maximum(0.0, ref_peak_arr)))

    pfv_violation = max(0.0, pfv_total - reference_pfv_total - float(pfv_tolerance))
    tfv_violation = max(
        0.0,
        tfv_total - reference_tfv_total + float(tfv_required_improvement) - float(tfv_tolerance),
    )
    peak_violation = max(0.0, peak - reference_peak_value - float(peak_tolerance))
    smooth_term = float(smooth_weight) * float(np.sum(np.abs(action_arr)))
    hard_tfv_violation = tfv_violation if bool(tfv_hard_constraint) else 0.0
    penalty_term = float(violation_penalty) * (pfv_violation + hard_tfv_violation + peak_violation)
    objective = (
        float(pfv_weight) * pfv_total
        + tfv_total
        + float(peak_weight) * peak
        + smooth_term
        + penalty_term
    )
    return HorizonSequenceScore(
        score=float(objective),
        gate_pass=bool(
            pfv_violation <= 1e-12
            and peak_violation <= 1e-12
            and (tfv_violation <= 1e-12 if bool(tfv_hard_constraint) else True)
        ),
        pfv_total=pfv_total,
        reference_pfv_total=reference_pfv_total,
        tfv_total=tfv_total,
        reference_tfv_total=reference_tfv_total,
        peak_tfv_rate=peak,
        reference_peak_tfv_rate=reference_peak_value,
        pfv_violation=float(pfv_violation),
        tfv_violation=float(tfv_violation),
        peak_violation=float(peak_violation),
        smooth_term=float(smooth_term),
        penalty_term=float(penalty_term),
    )
