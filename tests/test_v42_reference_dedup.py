"""V4.2 reference dedup — event_id+checkpoint_id+reference_type uniqueness."""
from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.v42_unified_pool import dedup_references


def _ref_df(n_candidates: int = 5, n_refs_per_type: int = 1) -> pd.DataFrame:
    """Build a synthetic manifest with 5 candidates + NC/DI/Hold references."""
    rows = []
    # 5 candidates for state e1::c1
    for i in range(n_candidates):
        rows.append({
            "event_id": "e1", "checkpoint_id": "c1",
            "anchor_type": "candidate",
            "reference_type": "candidate",
            "contract_sha": f"cand_sha_{i}",
            "state_key": "e1::c1",
        })
    # References: NC, DI, Hold — each potentially duplicated
    for ref_type, aliases in [
        ("NC", ["no_control"] * n_refs_per_type),
        ("DI", ["internal_rules"] * n_refs_per_type),
        ("Hold", ["hold_previous"] * n_refs_per_type),
    ]:
        for j, alias in enumerate(aliases):
            rows.append({
                "event_id": "e1", "checkpoint_id": "c1",
                "anchor_type": alias,
                "reference_type": ref_type,
                "contract_sha": f"ref_{ref_type}_{j}",
                "state_key": "e1::c1",
            })
    return pd.DataFrame(rows)


class TestReferenceDedup:
    def test_five_candidates_share_one_ref(self):
        """5 candidates + 1 NC + 1 DI + 1 Hold = 8 rows; dedup keeps all unique."""
        df = _ref_df(n_candidates=5, n_refs_per_type=1)
        result = dedup_references(df)
        # All rows have unique (event_id, checkpoint_id, reference_type, contract_sha)
        assert len(result) == len(df)

    def test_duplicate_refs_collapsed(self):
        """Duplicate NC references at same checkpoint collapse to 1."""
        df = _ref_df(n_candidates=5, n_refs_per_type=1)
        # Add a duplicate NC row with same contract_sha
        dup = {
            "event_id": "e1", "checkpoint_id": "c1",
            "anchor_type": "no_control",
            "reference_type": "NC",
            "contract_sha": "ref_NC_0",  # same as existing
            "state_key": "e1::c1",
        }
        df = pd.concat([df, pd.DataFrame([dup])], ignore_index=True)
        result = dedup_references(df)
        # The duplicate should be removed
        assert len(result) == len(df) - 1

    def test_dedup_per_state(self):
        """Dedup is per (event_id, checkpoint_id) — different states keep separate refs."""
        rows = []
        for state in ("e1::c1", "e1::c2"):
            rows.append({
                "event_id": "e1", "checkpoint_id": state.split("::")[1],
                "anchor_type": "no_control",
                "reference_type": "NC",
                "contract_sha": "same_sha",
                "state_key": state,
            })
        df = pd.DataFrame(rows)
        result = dedup_references(df)
        # Different checkpoint_id → different dedup key → both kept
        assert len(result) == 2
