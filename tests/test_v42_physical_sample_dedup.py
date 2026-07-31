"""Tests for V4.2 physical sample deduplication (v42_sample_lineage).

Verifies:
- Physical dedup uses multi-dimensional signature (not just SHA)
- Same rainfall + checkpoint + actions = duplicate
- Different actions = not duplicate
"""

from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.v42_sample_lineage import audit_physical_deduplication


def _make_row(idx, rf="rf_a", ts="ts_1", ps="ps_1", act="act_1",
              ct="ct_1", nc="nc_1", di="di_1", hold="hold_1",
              kpi="(1,1,1)", lineage="round0:case_0", rnd="round0"):
    return {
        "sample_idx": idx,
        "event_id": f"E{idx}",
        "checkpoint_id": f"C{idx}",
        "state_key": f"S{idx}",
        "source_round": rnd,
        "original_case_id": f"case_{idx}",
        "rainfall_fingerprint": rf,
        "checkpoint_timestamp": ts,
        "prefix_state_hash": ps,
        "actual_schedule_sha": act,
        "candidate_trajectory_sha": ct,
        "ref_nc_trajectory_sha": nc,
        "ref_di_trajectory_sha": di,
        "ref_hold_trajectory_sha": hold,
        "kpi_label_tuple": kpi,
        "derived_manifest_ids": lineage,
    }


class TestMultiDimensionalSignature:
    """Physical dedup uses multi-dimensional signature, not just SHA."""

    def test_all_same_means_duplicate(self):
        """Identical multi-dim signature → duplicate."""
        df = pd.DataFrame([_make_row(0), _make_row(1)])
        result = audit_physical_deduplication(df)
        assert result["duplicate_group_count"] == 1

    def test_different_rainfall_not_duplicate(self):
        """Different rainfall fingerprint → not duplicate."""
        df = pd.DataFrame([
            _make_row(0, rf="rf_a"),
            _make_row(1, rf="rf_b"),
        ])
        result = audit_physical_deduplication(df)
        assert result["duplicate_group_count"] == 0

    def test_different_actions_not_duplicate(self):
        """Different action schedule → not duplicate."""
        df = pd.DataFrame([
            _make_row(0, act="act_1"),
            _make_row(1, act="act_2"),
        ])
        result = audit_physical_deduplication(df)
        assert result["duplicate_group_count"] == 0

    def test_different_checkpoint_not_duplicate(self):
        """Different checkpoint timestamp → not duplicate."""
        df = pd.DataFrame([
            _make_row(0, ts="ts_1"),
            _make_row(1, ts="ts_2"),
        ])
        result = audit_physical_deduplication(df)
        assert result["duplicate_group_count"] == 0


class TestSameRainfallCheckpointActions:
    """Same rainfall + checkpoint + actions = duplicate."""

    def test_same_core_dims_are_duplicate(self):
        """Same rainfall, checkpoint, actions, trajectories, kpi → duplicate."""
        df = pd.DataFrame([
            _make_row(0, rf="rf_x", ts="ts_y", act="act_z"),
            _make_row(1, rf="rf_x", ts="ts_y", act="act_z"),
        ])
        result = audit_physical_deduplication(df)
        assert result["duplicate_group_count"] == 1
        assert result["duplicate_sample_count"] == 1


class TestDifferentActionsNotDuplicate:
    """Different actions = not duplicate."""

    def test_only_action_differs(self):
        """Only action schedule differs → not duplicate."""
        df = pd.DataFrame([
            _make_row(0, act="act_A", ct="ct_1", nc="nc_1"),
            _make_row(1, act="act_B", ct="ct_1", nc="nc_1"),
        ])
        result = audit_physical_deduplication(df)
        assert result["duplicate_group_count"] == 0
