from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _boundary_profile(value: float, phase: str) -> list[float]:
    mask = [0, 1, 1, 1, 1, 0] if phase == "peak" else [1, 1, 1, 1, 1, 0]
    return [float(value) * step for step in mask]


def build_boundary_v4_specifications(phase: str) -> list[dict]:
    """Return sparse stress candidates intended to cross the PFV safety boundary.

    These candidates are offline rejection examples. They remain within the
    deployed eight-actuator simultaneous-action limit and preserve temporal
    pump semantics; they are not automatically eligible for online execution.
    """
    if phase not in {"peak", "recession"}:
        raise ValueError("phase must be peak or recession")
    priority_regulators = [
        "ADD424.1", "ADD424.2", "ADD424.3", "cc006.1",
        "dwxh.2", "Zhongyi-2.2", "RTC_IN_02", "jichangheTank.2",
    ]
    outlet_regulators = [
        "RTC_OUT_01", "RTC_OUT_02", "RTC_OUT_03", "cc006.1",
        "dwxh.2", "ADD424.1", "ADD424.2", "ADD424.3",
    ]
    pump_profile = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0] if phase == "peak" else [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    pump_storage = [
        "ADD301.2", "ADD301.3", "RTC_OUT_01", "RTC_OUT_02",
        "RTC_OUT_03", "cc006.1", "dwxh.2", "ADD424.1",
    ]
    common = {
        "family": "pfv_unsafe_boundary_v4",
        "kind": "strong_counterfactual",
        "horizon_steps": 6,
        "sequence_semantics": "relative_to_same_state_no_control_reference",
        "intended_evidence_role": "offline_safety_rejection_only",
    }
    return [
        {
            **common,
            "mode": "eight_priority_regulators_close",
            "actuators": priority_regulators,
            "signed_profiles": {item: _boundary_profile(-1.0, phase) for item in priority_regulators},
        },
        {
            **common,
            "mode": "storage_outlet_downstream_close",
            "actuators": outlet_regulators,
            "signed_profiles": {item: _boundary_profile(-1.0, phase) for item in outlet_regulators},
        },
        {
            **common,
            "mode": "pump_storage_peak_stress",
            "actuators": pump_storage,
            "signed_profiles": {
                item: _boundary_profile(-1.0, phase)
                for item in pump_storage
                if item not in {"ADD301.2", "ADD301.3"}
            },
            "target_profiles": {"ADD301.2": pump_profile, "ADD301.3": pump_profile},
        },
    ]


def build_boundary_v5_specifications(phase: str) -> list[dict]:
    """Build label-enrichment stress cases that are never online candidates."""
    if phase not in {"peak", "recession"}:
        raise ValueError("phase must be peak or recession")
    priority_regulators = [
        "ADD424.1", "ADD424.2", "ADD424.3", "cc006.1",
        "dwxh.2", "Zhongyi-2.2", "RTC_IN_02", "jichangheTank.2",
    ]
    outlet_regulators = [
        "RTC_OUT_01", "RTC_OUT_02", "RTC_OUT_03", "cc006.1",
        "dwxh.2", "ADD424.1", "ADD424.2", "ADD424.3",
    ]
    pump_storage = [
        "ADD301.2", "ADD301.3", "RTC_OUT_01", "RTC_OUT_02",
        "RTC_OUT_03", "cc006.1", "dwxh.2", "ADD424.1",
    ]

    def profile(magnitude: float, *, hold: bool) -> list[float]:
        if phase == "peak":
            mask = [0, 1, 1, 1, 1, 1 if hold else 0]
        else:
            mask = [1, 1, 1, 1, 1, 1 if hold else 0]
        return [-float(magnitude) * step for step in mask]

    def continuous_spec(
        *,
        mode: str,
        actuators: list[str],
        magnitude: float,
        hold: bool,
    ) -> dict:
        return {
            "family": "pfv_safety_boundary_v5",
            "kind": "strong_counterfactual",
            "mode": mode,
            "actuators": actuators,
            "signed_profiles": {item: profile(magnitude, hold=hold) for item in actuators},
            "horizon_steps": 6,
            "sequence_semantics": "relative_to_same_state_no_control_reference",
            "intended_evidence_role": "offline_safety_rejection_only",
            "online_candidate_eligible": False,
            "stress_magnitude": float(magnitude),
            "stress_profile": "hold_through_horizon" if hold else "restore_last_step",
        }

    pump_profile = (
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        if phase == "peak"
        else [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    )
    pump_continuous = [item for item in pump_storage if item not in {"ADD301.2", "ADD301.3"}]
    pump_specification = {
        "family": "pfv_safety_boundary_v5",
        "kind": "strong_counterfactual",
        "mode": "pump_storage_stress_0p85",
        "actuators": pump_storage,
        "signed_profiles": {item: profile(0.85, hold=False) for item in pump_continuous},
        "target_profiles": {"ADD301.2": pump_profile, "ADD301.3": pump_profile},
        "horizon_steps": 6,
        "sequence_semantics": "relative_to_same_state_no_control_reference",
        "intended_evidence_role": "offline_safety_rejection_only",
        "online_candidate_eligible": False,
        "stress_magnitude": 0.85,
        "stress_profile": "restore_last_step",
    }
    return [
        continuous_spec(
            mode="priority_regulators_stress_0p85",
            actuators=priority_regulators,
            magnitude=0.85,
            hold=False,
        ),
        continuous_spec(
            mode="outlet_regulators_stress_0p85",
            actuators=outlet_regulators,
            magnitude=0.85,
            hold=False,
        ),
        pump_specification,
        continuous_spec(
            mode="priority_regulators_close_hold",
            actuators=priority_regulators,
            magnitude=1.0,
            hold=True,
        ),
        continuous_spec(
            mode="outlet_regulators_close_hold",
            actuators=outlet_regulators,
            magnitude=1.0,
            hold=True,
        ),
    ]


def _validated_pair(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference_array = np.asarray(reference, dtype=np.float32)
    candidate_array = np.asarray(candidate, dtype=np.float32)
    if reference_array.ndim != 2 or candidate_array.shape != reference_array.shape:
        raise ValueError("reference and candidate must share [H,A] shape")
    if not np.isfinite(reference_array).all() or not np.isfinite(candidate_array).all():
        raise ValueError("action sequences must be finite")
    return reference_array, candidate_array


def scale_residual_candidate(reference: np.ndarray, candidate: np.ndarray, *, scale: float) -> np.ndarray:
    """Scale a realized candidate-reference residual without changing its timing."""
    reference_array, candidate_array = _validated_pair(reference, candidate)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    return np.clip(reference_array + float(scale) * (candidate_array - reference_array), 0.0, 1.0).astype(np.float32)


def combine_residual_candidates(
    reference: np.ndarray,
    candidates: Iterable[np.ndarray],
    *,
    max_changed_actuators: int,
    atol: float = 1.0e-6,
) -> np.ndarray:
    """Combine sparse candidate residuals while preserving actuator and time axes."""
    reference_array = np.asarray(reference, dtype=np.float32)
    if reference_array.ndim != 2:
        raise ValueError("reference must have [H,A] shape")
    residual = np.zeros_like(reference_array)
    count = 0
    for candidate in candidates:
        _, candidate_array = _validated_pair(reference_array, candidate)
        residual += candidate_array - reference_array
        count += 1
    if count == 0:
        raise ValueError("at least one candidate is required")
    combined = np.clip(reference_array + residual, 0.0, 1.0).astype(np.float32)
    changed = np.flatnonzero(np.any(np.abs(combined - reference_array) > float(atol), axis=0))
    if len(changed) > int(max_changed_actuators):
        raise ValueError(
            f"combined sequence changes {len(changed)} actuators, exceeding max_changed_actuators={max_changed_actuators}"
        )
    if len(changed) == 0:
        raise ValueError("combined candidate is a no-op after clipping")
    return combined
