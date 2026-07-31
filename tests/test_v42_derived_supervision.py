"""V4.2 derived supervision — one-step, multi-horizon, pairwise, CR pairs."""
from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.v42_derived_supervision import (
    BRANCH_ROLES,
    MULTI_HORIZONS,
    N_HORIZON_STEPS,
    derive_candidate_reference_pairs,
    derive_multi_horizon_targets,
    derive_one_step_transitions,
    derive_pairwise_ranking,
)


def _trajectory_df(n_states: int = 2, n_candidates_per_state: int = 5) -> pd.DataFrame:
    """Build synthetic trajectory data with KPI columns."""
    rows = []
    for s in range(n_states):
        for c in range(n_candidates_per_state):
            row = {
                "event_id": f"evt_{s}",
                "state_key": f"evt_{s}::cp_{s}",
                "sample_id": f"sample_{s}_{c}",
                "split_group": "candidate",
                "anchor_type": "candidate",
            }
            # Add KPI columns for each branch and horizon
            for branch in BRANCH_ROLES:
                for h_name in MULTI_HORIZONS:
                    row[f"{branch}_PFV_{h_name}"] = float(s * 100 + c * 10 + int(h_name[1:]))
                    row[f"{branch}_TFV_{h_name}"] = float(s * 50 + c * 5)
                    row[f"{branch}_peak_TFV_rate_{h_name}"] = float(c)
            rows.append(row)
        # Add reference rows for each state
        for ref_type, alias in [("no_control", "no_control"),
                                 ("dynamic_internal_rules", "internal_rules"),
                                 ("hold_previous", "hold")]:
            row = {
                "event_id": f"evt_{s}",
                "state_key": f"evt_{s}::cp_{s}",
                "sample_id": f"ref_{s}_{ref_type}",
                "split_group": ref_type,
                "anchor_type": alias,
            }
            for branch in BRANCH_ROLES:
                for h_name in MULTI_HORIZONS:
                    row[f"{branch}_PFV_{h_name}"] = float(s * 100 + 999)
                    row[f"{branch}_TFV_{h_name}"] = float(s * 50 + 99)
                    row[f"{branch}_peak_TFV_rate_{h_name}"] = float(99)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One-step transitions
# ---------------------------------------------------------------------------

class TestOneStepTransitions:
    def test_count_matches_trajectories_x_steps_x_branches(self):
        df = _trajectory_df(n_states=2, n_candidates_per_state=3)
        result = derive_one_step_transitions(df)
        expected = len(df) * N_HORIZON_STEPS * len(BRANCH_ROLES)
        assert len(result) == expected

    def test_time_indices_cover_0_to_11(self):
        df = _trajectory_df(n_states=1, n_candidates_per_state=1)
        result = derive_one_step_transitions(df)
        assert set(result["time_index"].unique()) == set(range(N_HORIZON_STEPS))

    def test_all_branches_present(self):
        df = _trajectory_df(n_states=1, n_candidates_per_state=1)
        result = derive_one_step_transitions(df)
        assert set(result["split_group"].unique()) == set(BRANCH_ROLES)


# ---------------------------------------------------------------------------
# Multi-horizon targets
# ---------------------------------------------------------------------------

class TestMultiHorizonTargets:
    def test_five_horizons_per_trajectory(self):
        df = _trajectory_df(n_states=1, n_candidates_per_state=2)
        result = derive_multi_horizon_targets(df)
        assert len(result) == len(df) * len(MULTI_HORIZONS)

    def test_horizon_names(self):
        df = _trajectory_df(n_states=1, n_candidates_per_state=1)
        result = derive_multi_horizon_targets(df)
        assert set(result["horizon"].unique()) == set(MULTI_HORIZONS.keys())


# ---------------------------------------------------------------------------
# Pairwise ranking
# ---------------------------------------------------------------------------

class TestPairwiseRanking:
    def test_c52_pairs_per_state(self):
        df = _trajectory_df(n_states=1, n_candidates_per_state=5)
        result = derive_pairwise_ranking(df)
        # C(5,2) = 10 pairs per state
        assert len(result) == 10

    def test_no_cross_state_pairs(self):
        df = _trajectory_df(n_states=3, n_candidates_per_state=5)
        result = derive_pairwise_ranking(df)
        # Each pair has a single state_key (not NaN)
        assert result["state_key"].notna().all()
        # 3 states × 10 pairs = 30
        assert len(result) == 30

    def test_single_candidate_no_pairs(self):
        df = _trajectory_df(n_states=1, n_candidates_per_state=1)
        result = derive_pairwise_ranking(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Candidate-Reference pairs
# ---------------------------------------------------------------------------

class TestCandidateReferencePairs:
    def test_pairs_with_nc_di_hold(self):
        df = _trajectory_df(n_states=2, n_candidates_per_state=5)
        result = derive_candidate_reference_pairs(df)
        # 2 states × 5 candidates × 3 ref types = 30
        assert len(result) == 30

    def test_reference_types_valid(self):
        df = _trajectory_df(n_states=1, n_candidates_per_state=3)
        result = derive_candidate_reference_pairs(df)
        valid = {"no_control", "dynamic_internal_rules", "hold_previous"}
        assert set(result["reference_type"].unique()).issubset(valid)
