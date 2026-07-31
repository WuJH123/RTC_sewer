import pandas as pd
import pytest

from sewerrtc.v4.event_splits import (
    LEDGER_COLUMNS,
    assign_split,
    build_event_usage_ledger,
    validate_ledger,
)


def make_tier_catalog(num_events: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(num_events)],
            "rainfall_sha256": [f"r{i}" for i in range(num_events)],
            "event_tier": "standard_4plus",
        }
    )


def test_ledger_has_one_row_per_event_and_scanned_events_lose_formal() -> None:
    ledger = build_event_usage_ledger(
        make_tier_catalog(4), scanned_event_ids={"e0", "e1"}
    )

    assert list(ledger.columns) == list(LEDGER_COLUMNS)
    assert len(ledger) == 4
    assert not ledger["event_id"].duplicated().any()
    scanned = ledger[ledger["opportunity_scanned"]]
    assert set(scanned["event_id"]) == {"e0", "e1"}
    assert not scanned["formal_eligible"].any()
    assert (
        scanned["exclusion_reason"]
        .eq("opportunity_scanned_development_only")
        .all()
    )
    unscanned = ledger[~ledger["opportunity_scanned"]]
    assert unscanned["formal_eligible"].all()
    validate_ledger(ledger)


def test_ledger_rejects_duplicate_rainfall_sha() -> None:
    catalog = make_tier_catalog(3)
    catalog.loc[2, "rainfall_sha256"] = catalog.loc[1, "rainfall_sha256"]
    with pytest.raises(ValueError, match="duplicate rainfall_sha256"):
        build_event_usage_ledger(catalog, scanned_event_ids=set())


def test_assign_split_freezes_events_and_rejects_cross_split_moves() -> None:
    ledger = build_event_usage_ledger(
        make_tier_catalog(4), scanned_event_ids={"e0", "e1", "e2", "e3"}
    )
    ledger = assign_split(ledger, ["e0", "e1"], "pilot", assignment_run_uuid="u1")
    assert ledger.set_index("event_id").loc["e0", "used_pilot"]
    # Idempotent re-assignment to the same split is allowed.
    ledger = assign_split(ledger, ["e0"], "pilot", assignment_run_uuid="u2")

    with pytest.raises(ValueError, match="already frozen"):
        assign_split(ledger, ["e0"], "train", assignment_run_uuid="u3")


def test_formal_split_rejects_opportunity_scanned_events() -> None:
    ledger = build_event_usage_ledger(
        make_tier_catalog(2), scanned_event_ids={"e0"}
    )
    with pytest.raises(ValueError, match="formal"):
        assign_split(ledger, ["e0"], "formal", assignment_run_uuid="u1")
