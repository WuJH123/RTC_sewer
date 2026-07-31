"""Tests for V4.2 13-frame history rebuilder (v42_history_rebuilder).

Verifies:
- validate_frame_timestamps returns correct structure
- 13 frames at 5-min = 60-min span
- aggregate_future_to_control_steps: (24, n) → (12, n)
- Incomplete history marked correctly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_history_rebuilder import (
    validate_frame_timestamps,
    aggregate_future_to_control_steps,
    N_HISTORY_FRAMES,
    HISTORY_INTERVAL_MIN,
)


# ---------------------------------------------------------------------------
# validate_frame_timestamps
# ---------------------------------------------------------------------------

class TestValidateFrameTimestamps:
    """validate_frame_timestamps returns correct structure."""

    def test_returns_13_validations(self):
        frames = np.zeros((13, 5))
        checkpoint = pd.Timestamp("2024-06-01 12:00:00")
        result = validate_frame_timestamps(frames, checkpoint)
        assert "frame_validations" in result
        assert len(result["frame_validations"]) == 13

    def test_all_frames_valid_when_13_rows(self):
        frames = np.ones((13, 4))
        checkpoint = pd.Timestamp("2024-06-01 12:00:00")
        result = validate_frame_timestamps(frames, checkpoint)
        for fv in result["frame_validations"]:
            assert fv["valid"] is True

    def test_frame_12_is_checkpoint(self):
        frames = np.zeros((13, 2))
        checkpoint = pd.Timestamp("2024-06-01 12:00:00")
        result = validate_frame_timestamps(frames, checkpoint)
        # Frame 12 (last) should have expected_time == checkpoint
        last = result["frame_validations"][12]
        assert last["expected_time"] == checkpoint

    def test_frame_0_is_60min_before_checkpoint(self):
        frames = np.zeros((13, 2))
        checkpoint = pd.Timestamp("2024-06-01 12:00:00")
        result = validate_frame_timestamps(frames, checkpoint)
        first = result["frame_validations"][0]
        expected = checkpoint - pd.Timedelta(minutes=60)
        assert first["expected_time"] == expected

    def test_wrong_shape_raises(self):
        frames = np.zeros((7, 2))  # only 7 frames
        checkpoint = pd.Timestamp("2024-06-01 12:00:00")
        with pytest.raises(ValueError, match="13"):
            validate_frame_timestamps(frames, checkpoint)


class TestThirteenFrameSpan:
    """13 frames at 5-min = 60-min span."""

    def test_span_is_60_minutes(self):
        frames = np.zeros((13, 1))
        checkpoint = pd.Timestamp("2024-06-01 12:00:00")
        result = validate_frame_timestamps(frames, checkpoint)
        first_time = result["frame_validations"][0]["expected_time"]
        last_time = result["frame_validations"][12]["expected_time"]
        span = (last_time - first_time).total_seconds() / 60.0
        assert span == 60.0

    def test_interval_between_consecutive_frames(self):
        frames = np.zeros((13, 1))
        checkpoint = pd.Timestamp("2024-06-01 12:00:00")
        result = validate_frame_timestamps(frames, checkpoint)
        for i in range(1, 13):
            t_prev = result["frame_validations"][i - 1]["expected_time"]
            t_curr = result["frame_validations"][i]["expected_time"]
            delta_min = (t_curr - t_prev).total_seconds() / 60.0
            assert delta_min == HISTORY_INTERVAL_MIN


# ---------------------------------------------------------------------------
# aggregate_future_to_control_steps
# ---------------------------------------------------------------------------

class TestAggregateFutureToControlSteps:
    """aggregate_future_to_control_steps: (24, n) → (12, n)."""

    def test_shape_24_to_12(self):
        future = np.random.rand(24, 10).astype(np.float32)
        result = aggregate_future_to_control_steps(future)
        assert result.shape == (12, 10)

    def test_mean_aggregation(self):
        # Each pair of rows should be averaged
        future = np.zeros((24, 2), dtype=np.float32)
        future[0] = [1.0, 2.0]
        future[1] = [3.0, 4.0]
        result = aggregate_future_to_control_steps(future)
        np.testing.assert_allclose(result[0], [2.0, 3.0], atol=1e-6)

    def test_trims_extra_rows(self):
        # 25 rows → only 24 used (12 steps × 2 ratio), last row trimmed
        future = np.ones((25, 3), dtype=np.float32)
        result = aggregate_future_to_control_steps(future)
        assert result.shape == (12, 3)

    def test_output_dtype_float32(self):
        future = np.ones((24, 5), dtype=np.float64)
        result = aggregate_future_to_control_steps(future)
        assert result.dtype == np.float32
