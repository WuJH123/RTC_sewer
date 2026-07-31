import json

import numpy as np
import pytest

from sewerrtc.v4.online_v4_compact import (
    build_online_feature_frame,
    require_predicted_reference_forecasts,
)
from sewerrtc.v4.train_v4_loader import build_feature_matrix


def test_online_adapter_builds_the_frozen_570_feature_contract() -> None:
    schedule = np.zeros((12, 36), dtype=float)
    frame = build_online_feature_frame(
        event_id="development_event",
        checkpoint_id="60",
        state={"elapsed_min": 60.0, "opportunity_score": 0.4},
        actual_schedule=schedule,
        requested_schedule=schedule,
        anchor_schedule=schedule,
    )
    features, names = build_feature_matrix(frame)
    assert features.shape == (1, 570)
    assert len(names) == 570
    assert "delta_pfv_h120_vs_no_control" not in frame.columns
    assert json.loads(frame.loc[0, "projected_schedule_json"]) == schedule.tolist()


def test_online_adapter_refuses_incomplete_telemetry_and_future_swmm_references() -> None:
    schedule = np.zeros((12, 36), dtype=float)
    with pytest.raises(ValueError, match="incomplete"):
        build_online_feature_frame(
            event_id="e", checkpoint_id="1", state={}, actual_schedule=schedule,
            requested_schedule=schedule, anchor_schedule=schedule, strict_state=True,
        )
    with pytest.raises(ValueError, match="requires predicted_reference_forecasts"):
        require_predicted_reference_forecasts({})
    with pytest.raises(ValueError, match="prohibited"):
        require_predicted_reference_forecasts({
            "predicted_reference_forecasts": {
                "no_control_pfv": [0.0], "dynamic_internal_tfv": [0.0],
                "dynamic_internal_peak": [0.0],
            },
            "reference_forecasts_from_authoritative_swmm": True,
        })
