"""Regression tests for Section I Peak Boundary role freeze (oracle_revealed)."""

import importlib.util
from pathlib import Path

import pytest

from sewerrtc.v4.event_splits import (
    build_event_usage_ledger,
    validate_ledger,
)

_HELPER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "project6_runs"
    / "freeze_peak_boundary_role.py"
)
_spec = importlib.util.spec_from_file_location(
    "freeze_peak_boundary_role", _HELPER
)
freeze_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(freeze_mod)
freeze_peak_boundary_role = freeze_mod.freeze_peak_boundary_role


def _base_ledger():
    import pandas as pd

    catalog = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3", "e4"],
            "rainfall_sha256": ["a", "b", "c", "d"],
            "event_tier": ["standard_4plus"] * 4,
        }
    )
    # e1/e2 are development (scanned); e3/e4 remain formal-eligible.
    ledger = build_event_usage_ledger(
        catalog, scanned_event_ids={"e1", "e2"}
    )
    # ClassifyExistingGate5R stamps the tuned Peak Boundary events.
    tuned = ledger["event_id"].isin(["e1", "e2"])
    ledger.loc[tuned, "used_peak_boundary"] = True
    ledger.loc[tuned, "policy_tuned_on_event"] = True
    return ledger


def test_freeze_sets_oracle_revealed_on_tuned_events():
    ledger = _base_ledger()
    assert not ledger.set_index("event_id").loc["e1", "oracle_revealed"]
    frozen = freeze_peak_boundary_role(ledger)
    idx = frozen.set_index("event_id")
    for event in ("e1", "e2"):
        assert bool(idx.loc[event, "oracle_revealed"]) is True
        assert bool(idx.loc[event, "used_peak_boundary"]) is True
        assert bool(idx.loc[event, "policy_tuned_on_event"]) is True
        assert bool(idx.loc[event, "formal_eligible"]) is False


def test_freeze_leaves_non_tuned_events_untouched():
    ledger = _base_ledger()
    frozen = freeze_peak_boundary_role(ledger)
    idx = frozen.set_index("event_id")
    # Formal-eligible events must keep oracle_revealed False and stay eligible.
    for event in ("e3", "e4"):
        assert bool(idx.loc[event, "oracle_revealed"]) is False
        assert bool(idx.loc[event, "formal_eligible"]) is True


def test_freeze_keeps_ledger_valid_and_is_idempotent():
    ledger = _base_ledger()
    once = freeze_peak_boundary_role(ledger)
    twice = freeze_peak_boundary_role(once)
    validate_ledger(twice)
    # Idempotent: a second pass changes nothing.
    assert once.equals(twice)


def test_freeze_never_makes_tuned_events_formal_eligible():
    ledger = _base_ledger()
    frozen = freeze_peak_boundary_role(ledger)
    tuned = (
        frozen["used_peak_boundary"].astype(bool)
        & frozen["policy_tuned_on_event"].astype(bool)
    )
    assert not frozen.loc[tuned, "formal_eligible"].astype(bool).any()


def test_freeze_requires_expected_columns():
    import pandas as pd

    with pytest.raises(ValueError):
        freeze_peak_boundary_role(pd.DataFrame({"event_id": ["e1"]}))
