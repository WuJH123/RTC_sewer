"""V4.2 evaluation availability — state machine thresholds, locked minimum."""
from __future__ import annotations

import pytest

from sewerrtc.v4.v42_fresh_eval import (
    TARGET_CALIBRATION,
    TARGET_FULL,
    TARGET_LOCKED,
    _determine_status,
)


class TestEvaluationAvailability:
    def test_ready_full_threshold(self):
        """≥16 → ready_full with cal=4, locked=8, accrual=4."""
        s = _determine_status(16)
        assert s["status"] == "ready_full"
        assert s["calibration"] == TARGET_CALIBRATION
        assert s["locked"] == TARGET_LOCKED
        assert s["accrual"] == TARGET_FULL - TARGET_CALIBRATION - TARGET_LOCKED

    def test_ready_without_accrual_threshold(self):
        """12-15 → ready_without_accrual with cal=4, locked=8, accrual=0."""
        for count in (12, 13, 14, 15):
            s = _determine_status(count)
            assert s["status"] == "ready_without_accrual"
            assert s["accrual"] == 0
            assert s["locked"] == TARGET_LOCKED

    def test_prelocked_only_threshold(self):
        """8-11 → prelocked_only, no fresh locked authorized."""
        for count in (8, 9, 10, 11):
            s = _determine_status(count)
            assert s["status"] == "prelocked_only"
            assert s["authorizes_fresh_locked"] is False

    def test_insufficient_below_8(self):
        """<8 → insufficient_fresh_events."""
        for count in (0, 1, 5, 7):
            s = _determine_status(count)
            assert s["status"] == "insufficient_fresh_events"
            assert s["locked"] == 0

    def test_locked_minimum_not_reduced(self):
        """When ready_full, locked must be at least 8."""
        s = _determine_status(20)
        assert s["locked"] >= 8

    def test_boundary_16_is_ready_full(self):
        s = _determine_status(16)
        assert s["status"] == "ready_full"
        assert s["authorizes_fresh_locked"] is True

    def test_boundary_12_is_ready_without_accrual(self):
        s = _determine_status(12)
        assert s["status"] == "ready_without_accrual"
