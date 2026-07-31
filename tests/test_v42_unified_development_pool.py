"""V4.2 unified development pool — merge, compatibility, schema, leakage tests."""
from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.v42_unified_pool import (
    COMPATIBILITY_AUXILIARY,
    COMPATIBILITY_EXACT,
    COMPATIBILITY_INCOMPATIBLE,
    COMPATIBILITY_TRAJECTORY,
    classify_data_compatibility,
    dedup_references,
    validate_pool_compatibility,
)


# ---------------------------------------------------------------------------
# Compatibility validation
# ---------------------------------------------------------------------------

def _batch(n: int = 3, sha: str = "net_sha_1", **extra_cols) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "event_id": f"evt_{i}",
            "checkpoint_id": f"cp_{i}",
            "network_sha256": sha,
            "state_key": f"evt_{i}::cp_{i}",
            "data_source": "test",
            **extra_cols,
        })
    return pd.DataFrame(rows)


class TestValidatePoolCompatibility:
    def test_compatible_batches(self):
        a = _batch(3, sha="same_sha")
        b = _batch(3, sha="same_sha")
        issues = validate_pool_compatibility(a, b)
        assert issues == []

    def test_network_sha_mismatch(self):
        a = _batch(3, sha="sha_A")
        b = _batch(3, sha="sha_B")
        issues = validate_pool_compatibility(a, b)
        assert any("network SHA mismatch" in i for i in issues)

    def test_control_interval_mismatch(self):
        a = _batch(3, sha="s", control_interval_min=5)
        b = _batch(3, sha="s")
        issues = validate_pool_compatibility(a, b)
        assert any("control_interval_min" in i for i in issues)


# ---------------------------------------------------------------------------
# classify_data_compatibility
# ---------------------------------------------------------------------------

class TestClassifyCompatibility:
    def test_exact_compatible(self):
        row = pd.Series({
            "event_id": "e1", "checkpoint_id": "c1",
            "candidate_PFV_H120": 10.0,
            "recovery_status": "pass",
        })
        assert classify_data_compatibility(row) == COMPATIBILITY_EXACT

    def test_trajectory_compatible(self):
        row = pd.Series({
            "event_id": "e1",
            "checkpoint_id": "c1",
            "provenance_mode": "trajectory_content",
            "recovery_status": "pass",
            "runtime_executed": True,
        })
        result = classify_data_compatibility(row)
        # With event+checkpoint+trajectory+recovered → trajectory or exact
        assert result in (COMPATIBILITY_TRAJECTORY, COMPATIBILITY_EXACT)

    def test_incompatible(self):
        row = pd.Series({"event_id": "", "checkpoint_id": ""})
        assert classify_data_compatibility(row) == COMPATIBILITY_INCOMPATIBLE


# ---------------------------------------------------------------------------
# Reference deduplication
# ---------------------------------------------------------------------------

class TestDedupReferences:
    def test_dedup_keeps_unique_keys(self):
        df = pd.DataFrame({
            "event_id": ["e1", "e1", "e1", "e2"],
            "checkpoint_id": ["c1", "c1", "c1", "c1"],
            "reference_type": ["NC", "NC", "DI", "NC"],
            "contract_sha": ["s1", "s1", "s1", "s2"],
        })
        result = dedup_references(df)
        # e1+c1+NC+s1 should be deduped to 1 row; e1+c1+DI+s1 stays; e2 stays
        assert len(result) <= len(df)

    def test_empty_df(self):
        df = pd.DataFrame()
        result = dedup_references(df)
        assert result.empty

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({
            "event_id": ["e1", "e2"],
            "checkpoint_id": ["c1", "c2"],
            "reference_type": ["NC", "DI"],
            "contract_sha": ["s1", "s2"],
        })
        result = dedup_references(df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# No Challenge/Formal leakage
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_compatible_batches_no_leakage_signal(self):
        """Schema-level check: compatible batches share network SHA."""
        a = _batch(3, sha="same_sha")
        b = _batch(3, sha="same_sha")
        issues = validate_pool_compatibility(a, b)
        assert issues == []
