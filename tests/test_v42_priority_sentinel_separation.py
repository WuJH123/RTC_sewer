"""V4.2 Priority Contract — PFV core 8 and sentinel 2 are completely disjoint."""
from __future__ import annotations

import pytest

from sewerrtc.v4.v42_priority_contract import (
    PFV_CORE_8_IDS,
    DEPTH_SENTINEL_2_IDS,
    audit_contract,
)


class TestPFVCore8Count:
    def test_pfv_core_has_exactly_8_nodes(self):
        assert len(PFV_CORE_8_IDS) == 8


class TestSentinel2Count:
    def test_sentinel_has_exactly_2_nodes(self):
        assert len(DEPTH_SENTINEL_2_IDS) == 2


class TestDisjointSets:
    def test_pfv_core_and_sentinel_are_disjoint(self):
        overlap = set(PFV_CORE_8_IDS) & set(DEPTH_SENTINEL_2_IDS)
        assert overlap == set(), f"Unexpected overlap: {overlap}"


class TestAuditContract:
    def test_audit_returns_pass(self):
        report = audit_contract()
        assert report["status"] == "PASS"


class TestKnownNodeIDs:
    EXPECTED_PVF_CORE = {
        "MSLBZW001", "HS1316314", "YS2530050", "HS2529198",
        "MH0200773", "HS1330349", "HS2529139", "HS2529052",
    }
    EXPECTED_SENTINEL = {"MH0200770", "HS1355904"}

    def test_pfv_core_ids_match_expected(self):
        assert set(PFV_CORE_8_IDS) == self.EXPECTED_PVF_CORE

    def test_sentinel_ids_match_expected(self):
        assert set(DEPTH_SENTINEL_2_IDS) == self.EXPECTED_SENTINEL
