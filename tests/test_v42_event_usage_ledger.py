"""V4.2 event usage ledger — build, audit, dedup, role conflict tests."""
from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.v42_event_ledger import (
    LEDGER_COLUMNS,
    _determine_current_role,
    _determine_eligibility,
    _parse_duration,
    _parse_return_period,
    _parse_rainfall_family,
    audit_v42_event_usage_ledger,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic ledger builder
# ---------------------------------------------------------------------------

def _make_row(**overrides: object) -> dict:
    base = {c: False for c in LEDGER_COLUMNS}
    base.update({
        "event_id": "T10_D240_chicago",
        "rainfall_sha256": "sha_aaa",
        "rainfall_family": "chicago",
        "return_period": 10,
        "duration": 240,
        "source_inventory": "ledger",
        "contract_compatible": True,
        "current_role": "available",
        "eligible_for_v42_development": False,
        "eligible_for_v42_fresh_evaluation": True,
        "exclusion_reason": "",
    })
    base.update(overrides)
    return base


def _clean_ledger(n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(_make_row(
            event_id=f"T10_D240_event{i}",
            rainfall_sha256=f"sha_{i:03d}",
        ))
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

class TestParsingHelpers:
    def test_parse_return_period(self):
        assert _parse_return_period("T100_D240_chicago") == 100
        assert _parse_return_period("T5_D60_block") == 5
        assert _parse_return_period("bad_id") is None

    def test_parse_duration(self):
        assert _parse_duration("T100_D240_chicago") == 240
        assert _parse_duration("T10_D60_block") == 60
        assert _parse_duration("no_dur") is None

    def test_parse_rainfall_family(self):
        assert _parse_rainfall_family("T100_D240_chicago_late") == "chicago_late"
        assert _parse_rainfall_family("T100_D240_block") == "block"
        assert _parse_rainfall_family("T100_D240_double_peak") == "double_peak"


# ---------------------------------------------------------------------------
# Role determination
# ---------------------------------------------------------------------------

class TestRoleDetermination:
    def test_sealed_when_challenge(self):
        role = _determine_current_role(
            used_in_train1600_train=False, used_in_v40_calibration=False,
            used_in_v40_locked=False, used_in_v41_calibration=False,
            used_in_v41_locked=False, used_in_pilot=False,
            used_in_challenge=True, used_in_formal=False,
            v41_assigned_split="",
        )
        assert role == "sealed"

    def test_consumed_development_when_v41_cal(self):
        role = _determine_current_role(
            used_in_train1600_train=False, used_in_v40_calibration=False,
            used_in_v40_locked=False, used_in_v41_calibration=True,
            used_in_v41_locked=False, used_in_pilot=False,
            used_in_challenge=False, used_in_formal=False,
            v41_assigned_split="",
        )
        assert role == "consumed_development"

    def test_available_when_never_used(self):
        role = _determine_current_role(
            used_in_train1600_train=False, used_in_v40_calibration=False,
            used_in_v40_locked=False, used_in_v41_calibration=False,
            used_in_v41_locked=False, used_in_pilot=False,
            used_in_challenge=False, used_in_formal=False,
            v41_assigned_split="",
        )
        assert role == "available"


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

class TestEligibility:
    def test_fresh_when_never_used_and_compatible(self):
        dev, fresh, reason = _determine_eligibility(
            used_in_train1600_train=False, used_in_v40_calibration=False,
            used_in_v40_locked=False, used_in_v41_calibration=False,
            used_in_v41_locked=False, used_in_challenge=False,
            used_in_formal=False, contract_compatible=True, current_role="available",
        )
        assert not dev
        assert fresh
        assert reason == ""

    def test_not_fresh_when_sealed(self):
        dev, fresh, _ = _determine_eligibility(
            used_in_train1600_train=False, used_in_v40_calibration=False,
            used_in_v40_locked=False, used_in_v41_calibration=False,
            used_in_v41_locked=False, used_in_challenge=True,
            used_in_formal=False, contract_compatible=True, current_role="sealed",
        )
        assert not dev
        assert not fresh


# ---------------------------------------------------------------------------
# Audit — 7 checks
# ---------------------------------------------------------------------------

class TestAudit:
    def test_clean_ledger_passes_all(self):
        df = _clean_ledger(5)
        result = audit_v42_event_usage_ledger(df)
        assert result["status"] == "pass"
        assert result["exit_code"] == 0
        assert result["summary"]["n_failed"] == 0

    def test_duplicate_event_ids_fail(self):
        df = _clean_ledger(3)
        df.loc[1, "event_id"] = df.loc[0, "event_id"]  # duplicate
        result = audit_v42_event_usage_ledger(df)
        assert not result["checks"]["no_duplicate_event_ids"]["pass"]

    def test_locked_marked_fresh_fails(self):
        df = _clean_ledger(3)
        df.loc[0, "used_in_v41_locked"] = True
        df.loc[0, "eligible_for_v42_fresh_evaluation"] = True
        result = audit_v42_event_usage_ledger(df)
        assert not result["checks"]["locked_not_fresh"]["pass"]

    def test_cross_role_sha_conflict(self):
        df = _clean_ledger(3)
        # Two events share same SHA but different roles
        df.loc[0, "rainfall_sha256"] = "sha_shared"
        df.loc[0, "current_role"] = "available"
        df.loc[1, "rainfall_sha256"] = "sha_shared"
        df.loc[1, "current_role"] = "consumed_development"
        result = audit_v42_event_usage_ledger(df)
        assert not result["checks"]["no_cross_role_sha_conflicts"]["pass"]

    def test_challenge_formal_not_excluded_fails(self):
        df = _clean_ledger(3)
        df.loc[0, "used_in_challenge"] = True
        df.loc[0, "eligible_for_v42_development"] = True
        result = audit_v42_event_usage_ledger(df)
        assert not result["checks"]["challenge_formal_excluded"]["pass"]
