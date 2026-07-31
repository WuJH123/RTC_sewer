"""V4.2 pairwise ranking — no cross-state, dead-zone, softplus direction."""
from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.v42_derived_supervision import (
    BRANCH_ROLES,
    MULTI_HORIZONS,
    derive_pairwise_ranking,
)


def _pool_df(n_states: int = 3, n_cands: int = 5) -> pd.DataFrame:
    rows = []
    for s in range(n_states):
        for c in range(n_cands):
            pfv_val = float(s * 1000 + c * 100)
            rows.append({
                "event_id": f"evt_{s}",
                "checkpoint_id": f"cp_{s}",
                "state_key": f"evt_{s}::cp_{s}",
                "sample_id": f"cand_{s}_{c}",
                "anchor_type": "candidate",
                "candidate_PFV_H120": pfv_val,
                "candidate_TFV_H120": float(c * 10),
                "candidate_peak_TFV_rate_H120": float(c),
            })
    return pd.DataFrame(rows)


class TestPairwiseRanking:
    def test_no_cross_state_pairs(self):
        df = _pool_df(n_states=3, n_cands=5)
        result = derive_pairwise_ranking(df)
        # Every pair has exactly one state_key
        assert result["state_key"].notna().all()
        # Pairs per state = C(5,2) = 10
        for sk in df["state_key"].unique():
            n_pairs = (result["state_key"] == sk).sum()
            assert n_pairs == 10

    def test_dead_zone_filter(self):
        """Candidates with very similar PFV are flagged as hard-negative."""
        df = _pool_df(n_states=1, n_cands=5)
        # Make two candidates have very close PFV (< 100 threshold)
        df.loc[0, "candidate_PFV_H120"] = 500.0
        df.loc[1, "candidate_PFV_H120"] = 550.0
        result = derive_pairwise_ranking(df)
        # At least one pair should be hard-negative
        assert result["hard_negative_relation"].any()

    def test_softplus_direction(self):
        """pfv_first_lexicographic: 1 when A has lower PFV (better)."""
        df = _pool_df(n_states=1, n_cands=2)
        df.loc[0, "candidate_PFV_H120"] = 100.0  # better
        df.loc[1, "candidate_PFV_H120"] = 500.0  # worse
        result = derive_pairwise_ranking(df)
        assert len(result) == 1
        assert result.iloc[0]["pfv_first_lexicographic"] == 1

    def test_total_pairs_count(self):
        df = _pool_df(n_states=4, n_cands=5)
        result = derive_pairwise_ranking(df)
        assert len(result) == 4 * 10  # 4 states × C(5,2)
