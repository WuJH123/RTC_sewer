from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def summarize_effect_coverage(
    *,
    event_ids: np.ndarray,
    splits: np.ndarray,
    phases: np.ndarray,
    candidate_action_seq: np.ndarray,
    reference_action_seq: np.ndarray,
    action_ids: Sequence[str],
    atol: float = 1.0e-7,
) -> pd.DataFrame:
    candidate = np.asarray(candidate_action_seq, dtype=np.float32)
    reference = np.asarray(reference_action_seq, dtype=np.float32)
    if candidate.shape != reference.shape or candidate.ndim != 3:
        raise ValueError("candidate/reference actions must share [N,H,A]")
    if candidate.shape[2] != len(action_ids):
        raise ValueError("action_ids do not match the action tensor")
    if not (len(event_ids) == len(splits) == len(phases) == len(candidate)):
        raise ValueError("metadata rows do not align with action tensors")

    rows: list[dict[str, object]] = []
    residual = candidate - reference
    for row_index in range(len(candidate)):
        for action_index, actuator_id in enumerate(action_ids):
            values = residual[row_index, :, action_index]
            directions = []
            if bool(np.any(values > float(atol))):
                directions.append("increase")
            if bool(np.any(values < -float(atol))):
                directions.append("decrease")
            for direction in directions:
                rows.append({
                    "actuator_id": str(actuator_id),
                    "direction": direction,
                    "phase": str(phases[row_index]),
                    "split": str(splits[row_index]),
                    "event_id": str(event_ids[row_index]),
                    "action_linf": float(np.abs(values).max(initial=0.0)),
                })
    columns = ["actuator_id", "direction", "phase", "split", "rows", "independent_events", "action_linf_max"]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["actuator_id", "direction", "phase", "split"], as_index=False)
        .agg(
            rows=("event_id", "size"),
            independent_events=("event_id", "nunique"),
            action_linf_max=("action_linf", "max"),
        )
        .sort_values(["actuator_id", "direction", "phase", "split"])
        .reset_index(drop=True)
    )


def build_coverage_gaps(
    coverage: pd.DataFrame | Iterable[dict[str, object]],
    *,
    action_ids: Sequence[str],
    phases: Sequence[str] = ("rising", "peak", "recession"),
    min_train_events: int = 3,
    min_validation_events: int = 2,
) -> list[dict[str, object]]:
    frame = coverage.copy() if isinstance(coverage, pd.DataFrame) else pd.DataFrame(list(coverage))
    existing: dict[tuple[str, str, str, str], int] = {}
    for row in frame.to_dict("records"):
        key = (str(row["actuator_id"]), str(row["direction"]), str(row["phase"]), str(row["split"]))
        existing[key] = int(row.get("independent_events", 0))
    gaps: list[dict[str, object]] = []
    for actuator_id in action_ids:
        for direction in ("decrease", "increase"):
            for phase in phases:
                for split, target in (("train", int(min_train_events)), ("validation", int(min_validation_events))):
                    current = existing.get((str(actuator_id), direction, str(phase), split), 0)
                    gaps.append({
                        "actuator_id": str(actuator_id),
                        "direction": direction,
                        "phase": str(phase),
                        "split": split,
                        "current_independent_events": current,
                        "target_independent_events": target,
                        "missing_events": max(0, target - current),
                    })
    return gaps


def phase_delta_profile(
    direction: str,
    magnitude: float,
    phase: str,
    *,
    horizon_steps: int = 6,
) -> np.ndarray:
    if direction not in {"increase", "decrease"}:
        raise ValueError("direction must be increase or decrease")
    if int(horizon_steps) < 3:
        raise ValueError("horizon_steps must be at least 3")
    sign = 1.0 if direction == "increase" else -1.0
    value = sign * abs(float(magnitude))
    profile = np.zeros(int(horizon_steps), dtype=np.float32)
    phase = str(phase).lower()
    if phase == "rising":
        profile[1 : max(2, int(horizon_steps) - 2)] = value
    elif phase == "peak":
        profile[: max(2, int(horizon_steps) - 1)] = value
    elif phase == "recession":
        profile[: max(2, int(horizon_steps) // 2)] = value
    else:
        raise ValueError("phase must be rising, peak, or recession")
    return profile


def build_phase_profile_library(
    direction: str,
    *,
    magnitudes: Sequence[float],
    phase: str,
    horizon_steps: int = 6,
) -> list[dict[str, object]]:
    """Build bounded, interpretable temporal contrasts for paired experiments."""
    profiles: list[dict[str, object]] = []
    for magnitude in sorted({abs(float(value)) for value in magnitudes if abs(float(value)) > 0.0}):
        phase_hold = phase_delta_profile(
            direction,
            magnitude,
            phase,
            horizon_steps=horizon_steps,
        )
        delayed = np.zeros(int(horizon_steps), dtype=np.float32)
        value = (1.0 if direction == "increase" else -1.0) * magnitude
        phase_key = str(phase).lower()
        if phase_key == "rising":
            delayed[2 : max(3, int(horizon_steps) - 1)] = value
        elif phase_key == "peak":
            delayed[1 : max(3, int(horizon_steps) - 1)] = value
        elif phase_key == "recession":
            delayed[1 : max(3, int(horizon_steps) // 2 + 1)] = value
        else:
            raise ValueError("phase must be rising, peak, or recession")
        profiles.extend([
            {
                "magnitude": magnitude,
                "variant": "phase_hold",
                "profile": phase_hold,
            },
            {
                "magnitude": magnitude,
                "variant": "delayed_restore",
                "profile": delayed,
            },
        ])
    return profiles
