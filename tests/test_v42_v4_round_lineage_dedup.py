"""Tests for V4.2 sample lineage and round dedup (v42_sample_lineage).

Verifies:
- Lineage records source_round for each sample
- Dedup identifies duplicate groups
- Canonical selection picks one per group
"""

from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.v42_sample_lineage import (
    audit_physical_deduplication,
    _DEDUP_DIMS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_lineage_df() -> pd.DataFrame:
    """Minimal synthetic lineage DataFrame with known duplicates."""
    rows = [
        {
            "sample_idx": 0,
            "event_id": "E1",
            "checkpoint_id": "C1",
            "state_key": "S1",
            "source_round": "round0",
            "original_case_id": "case_0",
            "rainfall_fingerprint": "rf_a",
            "checkpoint_timestamp": "ts_1",
            "prefix_state_hash": "ps_1",
            "actual_schedule_sha": "act_1",
            "candidate_trajectory_sha": "ct_1",
            "ref_nc_trajectory_sha": "nc_1",
            "ref_di_trajectory_sha": "di_1",
            "ref_hold_trajectory_sha": "hold_1",
            "kpi_label_tuple": "(1,1,1)",
            "derived_manifest_ids": "round0:case_0",
        },
        {
            "sample_idx": 1,
            "event_id": "E1",
            "checkpoint_id": "C1",
            "state_key": "S1",
            "source_round": "round1",
            "original_case_id": "case_1",
            "rainfall_fingerprint": "rf_a",
            "checkpoint_timestamp": "ts_1",
            "prefix_state_hash": "ps_1",
            "actual_schedule_sha": "act_1",
            "candidate_trajectory_sha": "ct_1",
            "ref_nc_trajectory_sha": "nc_1",
            "ref_di_trajectory_sha": "di_1",
            "ref_hold_trajectory_sha": "hold_1",
            "kpi_label_tuple": "(1,1,1)",
            "derived_manifest_ids": "round0:case_0",
        },
        {
            "sample_idx": 2,
            "event_id": "E2",
            "checkpoint_id": "C2",
            "state_key": "S2",
            "source_round": "round0",
            "original_case_id": "case_2",
            "rainfall_fingerprint": "rf_b",
            "checkpoint_timestamp": "ts_2",
            "prefix_state_hash": "ps_2",
            "actual_schedule_sha": "act_2",
            "candidate_trajectory_sha": "ct_2",
            "ref_nc_trajectory_sha": "nc_2",
            "ref_di_trajectory_sha": "di_2",
            "ref_hold_trajectory_sha": "hold_2",
            "kpi_label_tuple": "(0,1,0)",
            "derived_manifest_ids": "round0:case_2",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLineageSourceRound:
    """Lineage records source_round for each sample."""

    def test_source_round_column_present(self, sample_lineage_df):
        assert "source_round" in sample_lineage_df.columns

    def test_source_round_values(self, sample_lineage_df):
        rounds = set(sample_lineage_df["source_round"].tolist())
        assert rounds == {"round0", "round1"}


class TestDedupIdentifiesGroups:
    """Dedup identifies duplicate groups."""

    def test_duplicate_detected(self, sample_lineage_df):
        result = audit_physical_deduplication(sample_lineage_df)
        # Samples 0 and 1 have identical dedup keys → 1 duplicate group
        assert result["duplicate_group_count"] == 1
        assert result["duplicate_sample_count"] == 1

    def test_no_duplicate_when_all_unique(self):
        rows = [
            {
                "sample_idx": i,
                "event_id": f"E{i}",
                "checkpoint_id": f"C{i}",
                "state_key": f"S{i}",
                "source_round": "round0",
                "original_case_id": f"case_{i}",
                "rainfall_fingerprint": f"rf_{i}",
                "checkpoint_timestamp": f"ts_{i}",
                "prefix_state_hash": f"ps_{i}",
                "actual_schedule_sha": f"act_{i}",
                "candidate_trajectory_sha": f"ct_{i}",
                "ref_nc_trajectory_sha": f"nc_{i}",
                "ref_di_trajectory_sha": f"di_{i}",
                "ref_hold_trajectory_sha": f"hold_{i}",
                "kpi_label_tuple": f"({i},0,0)",
                "derived_manifest_ids": f"round0:case_{i}",
            }
            for i in range(3)
        ]
        df = pd.DataFrame(rows)
        result = audit_physical_deduplication(df)
        assert result["duplicate_group_count"] == 0


class TestCanonicalSelection:
    """Canonical selection picks one per group."""

    def test_canonical_picks_earliest_round(self, sample_lineage_df):
        result = audit_physical_deduplication(sample_lineage_df)
        groups = result["duplicate_groups"]
        assert len(groups) == 1
        # round0 (priority 0) should be canonical over round1 (priority 1)
        canonical_idx = groups[0]["canonical_sample_idx"]
        assert canonical_idx == 0  # sample_idx 0 is round0

    def test_dedup_dimensions_constant(self):
        assert len(_DEDUP_DIMS) == 10  # 10-dimensional signature
