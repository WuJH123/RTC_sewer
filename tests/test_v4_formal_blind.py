import pandas as pd

from sewerrtc.v4.evaluation import audit_formal_blind_inventory
from sewerrtc.v4.event_splits import (
    build_event_usage_ledger,
    select_formal_blind_candidates,
)


def test_formal_blind_requires_24_new_unique_unrevealed_rainfall_events() -> None:
    inventory = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(24)],
            "rainfall_sha256": [f"r{i}" for i in range(24)],
            "historically_used": False,
            "revealed": False,
        }
    )

    assert audit_formal_blind_inventory(inventory)["status"] == "pass"
    inventory.loc[0, "revealed"] = True
    assert audit_formal_blind_inventory(inventory)["status"] == "blocked"


def test_formal_blind_never_uses_opportunity_scanned_events() -> None:
    catalog = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(6)],
            "rainfall_sha256": [f"r{i}" for i in range(6)],
            "event_tier": "standard_4plus",
        }
    )
    # e0..e3 were part of the 244-event opportunity scan and are therefore
    # development-only; only never-scanned events may reach Formal Blind.
    ledger = build_event_usage_ledger(
        catalog, scanned_event_ids={"e0", "e1", "e2", "e3"}
    )

    candidates = select_formal_blind_candidates(ledger)

    assert set(candidates["event_id"]) == {"e4", "e5"}
    assert not candidates["opportunity_scanned"].any()
    assert not candidates["oracle_revealed"].any()
    assert not candidates["policy_tuned_on_event"].any()
    assert candidates["formal_eligible"].all()

    # Any later usage flag also disqualifies an unscanned event.
    ledger.loc[ledger["event_id"] == "e4", "used_pilot"] = True
    assert set(select_formal_blind_candidates(ledger)["event_id"]) == {"e5"}
