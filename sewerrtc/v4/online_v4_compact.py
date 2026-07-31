"""Leakage-free online adapter for the frozen V4.1 compact model."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .train_v4_loader import (
    ALLOWED_ACTION_SCALAR_COLUMNS,
    ALLOWED_STATE_FEATURE_COLUMNS,
    PROCESS_RESIDUAL_COLUMNS,
    TrainingData,
    build_feature_matrix,
)


def _schedule_json(value: np.ndarray) -> str:
    array = np.asarray(value, dtype=float)
    if array.shape != (12, 36):
        raise ValueError(f"online schedule must be 12x36, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("online schedule contains non-finite values")
    return json.dumps(array.tolist(), separators=(",", ":"))


def build_online_feature_frame(
    *,
    event_id: str,
    checkpoint_id: str,
    state: dict[str, Any],
    actual_schedule: np.ndarray,
    requested_schedule: np.ndarray,
    anchor_schedule: np.ndarray,
    strict_state: bool = False,
) -> pd.DataFrame:
    """Create exactly the sanctioned V4.1 feature row.

    ``state`` may contain only current/history-derived hydraulic values and
    rainfall forecasts.  Unknown fields are ignored; labels and realised
    future hydraulics never enter this frame.
    """
    row: dict[str, Any] = {
        "event_id": str(event_id),
        "checkpoint_id": str(checkpoint_id),
    }
    missing = [name for name in ALLOWED_STATE_FEATURE_COLUMNS if name not in state]
    if strict_state and missing:
        raise ValueError(
            "online state is incomplete; refusing zero-filled telemetry: "
            + ", ".join(missing)
        )
    for name in ALLOWED_STATE_FEATURE_COLUMNS:
        value = float(state.get(name, 0.0))
        if not np.isfinite(value):
            raise ValueError(f"online state feature is non-finite: {name}")
        row[name] = value
    for name in ALLOWED_ACTION_SCALAR_COLUMNS:
        row[name] = float(state.get(name, 0.0))
    row["requested_schedule_json"] = _schedule_json(requested_schedule)
    row["projected_schedule_json"] = _schedule_json(actual_schedule)
    row["anchor_schedule_json"] = _schedule_json(anchor_schedule)
    return pd.DataFrame([row])


def _online_data(frame: pd.DataFrame) -> TrainingData:
    features, names = build_feature_matrix(frame)
    n = len(frame)
    return TrainingData(
        features=features,
        feature_names=names,
        continuous={},
        classification={},
        residuals=np.zeros((n, len(PROCESS_RESIDUAL_COLUMNS), 12)),
        residual_channels=list(PROCESS_RESIDUAL_COLUMNS),
        ranking={},
        full_event_enabled=False,
        full_event_mask=np.zeros(n, dtype=bool),
        split=np.asarray(["online"] * n, dtype=object),
        state_key=(frame["event_id"].astype(str) + "::" + frame["checkpoint_id"].astype(str)).to_numpy(),
        event_id=frame["event_id"].astype(str).to_numpy(),
        hard_negative_type=np.asarray([""] * n),
        n_samples=n,
    )


class CompactOnlinePredictor:
    """Small adapter from Candidate telemetry to ``CompactHeadSpecificModel``."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.model = pickle.loads(self.model_path.read_bytes())
        self.feature_names = list(getattr(self.model, "feature_names_", []))
        if not self.feature_names:
            raise ValueError("compact model lacks frozen feature names")

    def predict(self, frame: pd.DataFrame) -> dict[str, Any]:
        data = _online_data(frame)
        if list(data.feature_names) != self.feature_names:
            raise ValueError("online feature contract differs from frozen compact model")
        return self.model.predict(data, np.arange(data.n_samples, dtype=int))


def require_predicted_reference_forecasts(context: dict[str, Any]) -> None:
    """Reject realised SWMM horizons disguised as online reference forecasts."""
    forecasts = context.get("predicted_reference_forecasts")
    if not isinstance(forecasts, dict):
        raise ValueError(
            "V4.1 online control requires predicted_reference_forecasts; "
            "authoritative future SWMM references are prohibited"
        )
    required = {"no_control_pfv", "dynamic_internal_tfv", "dynamic_internal_peak"}
    missing = required - set(forecasts)
    if missing:
        raise ValueError(
            "predicted reference forecasts missing: " + ", ".join(sorted(missing))
        )
    if bool(context.get("reference_forecasts_from_authoritative_swmm", False)):
        raise ValueError("future authoritative SWMM reference forecasts are prohibited online")
