from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def action_window(
    detail: pd.DataFrame,
    *,
    action_ids: list[str],
    start_min: float,
    horizon_steps: int,
) -> np.ndarray:
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(float)
    start = int(np.searchsorted(elapsed, float(start_min), side="left"))
    window = detail.iloc[start : start + int(horizon_steps)]
    if len(window) != int(horizon_steps):
        raise ValueError(f"reference horizon is incomplete: {len(window)}/{horizon_steps}")
    columns = [f"a:{actuator_id}" for actuator_id in action_ids]
    missing = [column for column in columns if column not in window]
    if missing:
        raise KeyError(f"reference detail lacks action columns: {missing[:3]}")
    return window[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)


def materialize_candidate(
    reference_action_seq: np.ndarray,
    *,
    action_ids: list[str],
    specification: dict[str, Any],
) -> np.ndarray:
    reference = np.asarray(reference_action_seq, dtype=np.float32)
    candidate = reference.copy()
    horizon, action_count = reference.shape
    if action_count != len(action_ids):
        raise ValueError("canonical action axis does not match action_ids")
    index = {actuator_id: position for position, actuator_id in enumerate(action_ids)}
    delta_profiles = specification.get("signed_profiles", {}) or {}
    if "signed_profile" in specification:
        actuators = list(specification.get("actuators", []))
        if len(actuators) != 1:
            raise ValueError("signed_profile requires exactly one actuator")
        delta_profiles = {str(actuators[0]): specification["signed_profile"]}
    target_profiles = specification.get("target_profiles", {}) or {}
    if "target_profile" in specification:
        actuators = list(specification.get("actuators", []))
        if len(actuators) != 1:
            raise ValueError("target_profile requires exactly one actuator")
        target_profiles = {str(actuators[0]): specification["target_profile"]}
    if not delta_profiles and not target_profiles:
        raise ValueError("candidate requires signed or absolute target profiles")
    for actuator_id, values in delta_profiles.items():
        if actuator_id not in index:
            raise KeyError(f"unknown canonical actuator: {actuator_id}")
        profile = np.asarray(values, dtype=np.float32)
        if profile.shape != (horizon,):
            raise ValueError(f"{actuator_id} delta profile must have {horizon} values")
        position = index[actuator_id]
        candidate[:, position] = np.clip(reference[:, position] + profile, 0.0, 1.0)
    for actuator_id, values in target_profiles.items():
        if actuator_id not in index:
            raise KeyError(f"unknown canonical actuator: {actuator_id}")
        profile = np.asarray(values, dtype=np.float32)
        if profile.shape != (horizon,):
            raise ValueError(f"{actuator_id} target profile must have {horizon} values")
        candidate[:, index[actuator_id]] = np.clip(profile, 0.0, 1.0)
    return candidate


def sequence_diagnostics(
    candidate_action_seq: np.ndarray,
    reference_action_seq: np.ndarray,
    *,
    action_ids: list[str],
    binary_pump_ids: set[str] | None = None,
    minimum_effective_delta: float = 0.02,
) -> dict[str, Any]:
    candidate = np.asarray(candidate_action_seq, dtype=np.float32)
    reference = np.asarray(reference_action_seq, dtype=np.float32)
    if candidate.shape != reference.shape or candidate.ndim != 2:
        return {"valid": False, "reason": "shape_mismatch"}
    residual = candidate - reference
    changed = np.abs(residual) > 1.0e-7
    changed_positions = np.flatnonzero(changed.any(axis=0))
    changed_ids = [action_ids[position] for position in changed_positions]
    linf = float(np.abs(residual).max(initial=0.0))
    binary_pumps = set(binary_pump_ids or set())
    for actuator_id in changed_ids:
        if actuator_id not in binary_pumps:
            continue
        values = candidate[:, action_ids.index(actuator_id)]
        if not np.all(np.isin(values, np.asarray([0.0, 1.0], dtype=np.float32))):
            return {"valid": False, "reason": f"fractional_binary_pump:{actuator_id}"}
    valid = bool(changed.any()) and linf >= float(minimum_effective_delta)
    reason = "ok" if valid else ("noop_after_clipping" if not changed.any() else "below_minimum_effective_delta")
    actual_by_actuator = {
        action_ids[position]: residual[:, position].astype(float).tolist()
        for position in changed_positions
    }
    return {
        "valid": valid,
        "reason": reason,
        "is_noop": not bool(changed.any()),
        "changed_actuator_count": int(len(changed_positions)),
        "changed_time_step_count": int(changed.any(axis=1).sum()),
        "max_simultaneous_changes": int(changed.sum(axis=1).max(initial=0)),
        "action_l1_difference": float(np.abs(residual).sum()),
        "action_linf_difference": linf,
        "changed_actuator_ids": changed_ids,
        "actual_delta_after_clipping": actual_by_actuator,
    }


def parse_specification(value: str | dict[str, Any]) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else dict(value)


def event_pattern(event_id: str) -> str:
    parts = str(event_id).split("_", 2)
    return parts[2] if len(parts) == 3 else "unknown"


def event_return_period(event_id: str) -> str:
    return str(event_id).split("_", 1)[0]
