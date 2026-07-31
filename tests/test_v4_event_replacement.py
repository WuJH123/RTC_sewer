from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.training_plan import (
    build_state_budget_ledger,
    plan_event_replacement,
)


def make_catalog() -> pd.DataFrame:
    rows = []
    for event, split in (("e1", "train"), ("e2", "val")):
        for checkpoint in range(5):
            rows.append(
                {
                    "event_id": event,
                    "checkpoint_id": f"{event}_c{checkpoint}",
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def make_reserve(event: str = "r1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [event] * 5,
            "checkpoint_id": [f"{event}_c{i}" for i in range(5)],
            "split": ["reserve"] * 5,
        }
    )


def shortfall_ledger(event: str, checkpoint: str) -> pd.DataFrame:
    catalog = make_catalog()
    accepted = pd.DataFrame(
        {
            "event_id": [event],
            "checkpoint_id": [checkpoint],
            "actual_schedule_sha256": ["only-one"],
        }
    )
    attempted = pd.DataFrame(
        {
            "event_id": [event] * 10,
            "checkpoint_id": [checkpoint] * 10,
            "actual_schedule_sha256": [f"x{i}" for i in range(10)],
        }
    )
    return build_state_budget_ledger(catalog, accepted, attempted)


def test_incomplete_event_is_replaced_wholesale() -> None:
    ledger = shortfall_ledger("e1", "e1_c0")
    result = plan_event_replacement(ledger, make_catalog(), make_reserve())

    assert result["event_shortfalls"] == ["e1"]
    assert len(result["replacements"]) == 1
    replacement = result["replacements"][0]
    assert replacement["whole_event"] is True
    assert replacement["replacement_event"] == "r1"
    # All five checkpoints move together, never a single checkpoint.
    assert len(replacement["replacement_rows"]) == 5
    assert result["partial_event_in_main_table"] is False


def test_replacement_keeps_original_split() -> None:
    ledger = shortfall_ledger("e2", "e2_c3")
    result = plan_event_replacement(ledger, make_catalog(), make_reserve())

    replacement = result["replacements"][0]
    assert replacement["split"] == "val"
    assert set(replacement["replacement_rows"]["split"]) == {"val"}


def test_shortfall_event_becomes_auxiliary_only() -> None:
    ledger = shortfall_ledger("e1", "e1_c0")
    result = plan_event_replacement(ledger, make_catalog(), make_reserve())

    assert result["auxiliary_events"] == ["e1"]


def test_unresolved_when_reserve_pool_is_empty() -> None:
    ledger = shortfall_ledger("e1", "e1_c0")
    empty_reserve = make_reserve().head(0)
    result = plan_event_replacement(ledger, make_catalog(), empty_reserve)

    assert result["unresolved"] == ["e1"]
    assert result["replacements"] == []


def test_reserve_event_must_carry_exactly_5_checkpoints() -> None:
    ledger = shortfall_ledger("e1", "e1_c0")
    broken_reserve = make_reserve().head(4)

    with pytest.raises(ValueError):
        plan_event_replacement(ledger, make_catalog(), broken_reserve)


def test_no_shortfall_means_no_replacement() -> None:
    catalog = make_catalog()
    accepted = pd.concat(
        [
            pd.DataFrame(
                {
                    "event_id": [row["event_id"]] * 5,
                    "checkpoint_id": [row["checkpoint_id"]] * 5,
                    "actual_schedule_sha256": [
                        f"{row['checkpoint_id']}-s{i}" for i in range(5)
                    ],
                }
            )
            for _, row in catalog.iterrows()
        ]
    )
    ledger = build_state_budget_ledger(catalog, accepted, accepted)
    result = plan_event_replacement(ledger, catalog, make_reserve())

    assert result["event_shortfalls"] == []
    assert result["replacements"] == []
    assert result["unresolved"] == []
