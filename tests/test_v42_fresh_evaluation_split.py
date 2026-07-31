"""V4.2 fresh evaluation split — pool filtering, no KPI reading, SHA uniqueness."""
from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.v42_fresh_eval import (
    _determine_status,
    _is_consumed,
    _is_contract_compatible,
)


class TestFreshPoolFiltering:
    def test_consumed_events_excluded(self):
        row = pd.Series({
            "assigned_split": "train1600",
            "used_in_pilot": False,
            "used_in_p3": False,
            "used_in_train1600": True,
        })
        assert _is_consumed(row) is True

    def test_unused_events_not_consumed(self):
        row = pd.Series({
            "assigned_split": "",
            "used_in_pilot": False,
            "used_in_p3": False,
        })
        assert _is_consumed(row) is False

    def test_challenge_events_consumed(self):
        row = pd.Series({"assigned_split": "challenge", "used_in_challenge": True})
        assert _is_consumed(row) is True

    def test_contract_compatible_default(self):
        row = pd.Series({"event_id": "e1"})
        assert _is_contract_compatible(row) is True

    def test_contract_incompatible_flagged(self):
        row = pd.Series({"contract_compatible": False})
        assert _is_contract_compatible(row) is False


class TestRainfallSHAUniqueness:
    def test_fresh_pool_drops_sha_duplicates(self):
        """After filtering, fresh pool should have unique rainfall_sha256."""
        df = pd.DataFrame({
            "event_id": ["e1", "e2", "e3"],
            "rainfall_sha256": ["sha_A", "sha_A", "sha_B"],
            "assigned_split": ["", "", ""],
        })
        fresh = df[~df.apply(_is_consumed, axis=1)].copy()
        fresh = fresh.drop_duplicates(subset="rainfall_sha256", keep="first")
        assert fresh["rainfall_sha256"].is_unique
        assert len(fresh) == 2


class TestStatusMachine:
    def test_ready_full_at_16(self):
        s = _determine_status(16)
        assert s["status"] == "ready_full"
        assert s["authorizes_fresh_locked"] is True

    def test_ready_without_accrual_at_12(self):
        s = _determine_status(12)
        assert s["status"] == "ready_without_accrual"
        assert s["locked"] == 8

    def test_prelocked_only_at_8(self):
        s = _determine_status(8)
        assert s["status"] == "prelocked_only"
        assert s["authorizes_fresh_locked"] is False

    def test_insufficient_below_8(self):
        s = _determine_status(7)
        assert s["status"] == "insufficient_fresh_events"
        assert s["locked"] == 0
