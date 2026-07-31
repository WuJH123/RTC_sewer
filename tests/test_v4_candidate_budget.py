from __future__ import annotations

import pandas as pd

from sewerrtc.v4.pilot import (
    PILOT_BUDGET_DEFAULTS,
    audit_pilot_state_progress,
    plan_pilot_reserve,
)
from sewerrtc.v4.training_plan import (
    TRAIN_BUDGET_DEFAULTS,
    build_state_budget_ledger,
)


def make_checkpoints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["e1", "e1", "e2"],
            "checkpoint_id": ["c1", "c2", "c3"],
            "checkpoint_role": ["responsive", "low_opportunity", "responsive"],
        }
    )


def accepted_rows(event: str, checkpoint: str, count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [event] * count,
            "checkpoint_id": [checkpoint] * count,
            "actual_schedule_sha256": [
                f"{event}-{checkpoint}-sha{i}" for i in range(count)
            ],
        }
    )


def test_pilot_budget_defaults_match_contract() -> None:
    assert PILOT_BUDGET_DEFAULTS["primary_candidates_per_state"] == 10
    assert PILOT_BUDGET_DEFAULTS["reserve_candidates_per_state"] == 5
    assert PILOT_BUDGET_DEFAULTS["min_accepted_informative_total"] == 300
    assert PILOT_BUDGET_DEFAULTS["min_accepted_per_responsive_state"] == 6
    assert PILOT_BUDGET_DEFAULTS["max_candidate_budget_per_state"] == 15


def test_pilot_primary_and_reserve_queues_are_separate() -> None:
    # e1/c1 responsive with 4 accepted after 10 primary attempts: reserve
    # eligible.  e1/c2 low-opportunity short state: never reserve eligible.
    accepted = accepted_rows("e1", "c1", 4)
    attempted = pd.concat(
        [
            accepted_rows("e1", "c1", 10),
            accepted_rows("e1", "c2", 10),
            accepted_rows("e2", "c3", 10),
        ]
    )
    progress = audit_pilot_state_progress(
        accepted, attempted, make_checkpoints()
    )
    by_state = progress.set_index(["event_id", "checkpoint_id"])

    assert bool(by_state.loc[("e1", "c1"), "reserve_eligible"])
    assert not bool(by_state.loc[("e1", "c2"), "reserve_eligible"])
    assert int(by_state.loc[("e1", "c1"), "budget_remaining"]) == 5


def test_pilot_reserve_never_copies_accepted_actual() -> None:
    accepted = accepted_rows("e1", "c1", 4)
    attempted = accepted_rows("e1", "c1", 10)
    progress = audit_pilot_state_progress(
        accepted, attempted, make_checkpoints()
    )
    reserve_catalog = pd.DataFrame(
        {
            "event_id": ["e1"] * 3,
            "checkpoint_id": ["c1"] * 3,
            "candidate_id": ["r1", "r2", "r3"],
            "family": ["famA", "famB", "famA"],
            "projected_schedule_sha256": [
                "e1-c1-sha0",  # duplicates an accepted actual: must be dropped
                "fresh-1",
                "fresh-2",
            ],
        }
    )
    reserve = plan_pilot_reserve(progress, reserve_catalog, accepted)

    assert len(reserve) == 2
    assert "e1-c1-sha0" not in set(reserve["projected_schedule_sha256"])
    assert set(reserve["queue"]) == {"reserve"}
    assert reserve["case_id"].str.contains("__reserve__").all()


def test_pilot_reserve_stops_at_max_budget() -> None:
    accepted = accepted_rows("e1", "c1", 1)
    attempted = accepted_rows("e1", "c1", 14)  # only 1 budget slot left
    progress = audit_pilot_state_progress(
        accepted, attempted, make_checkpoints()
    )
    reserve_catalog = pd.DataFrame(
        {
            "event_id": ["e1"] * 5,
            "checkpoint_id": ["c1"] * 5,
            "candidate_id": [f"r{i}" for i in range(5)],
            "family": ["famA"] * 5,
            "projected_schedule_sha256": [f"fresh-{i}" for i in range(5)],
        }
    )
    reserve = plan_pilot_reserve(progress, reserve_catalog, accepted)

    assert len(reserve) == 1


def test_pilot_state_shortfall_when_budget_exhausted_and_short() -> None:
    accepted = accepted_rows("e1", "c1", 2)
    attempted = accepted_rows("e1", "c1", 15)
    progress = audit_pilot_state_progress(
        accepted, attempted, make_checkpoints()
    )
    row = progress.set_index(["event_id", "checkpoint_id"]).loc[("e1", "c1")]

    assert bool(row["state_shortfall"])
    assert bool(row["budget_exhausted"])
    assert not bool(row["reserve_eligible"])


def test_train_target_5_and_budget_10_are_separate() -> None:
    assert TRAIN_BUDGET_DEFAULTS["target_accepted_per_state"] == 5
    assert TRAIN_BUDGET_DEFAULTS["initial_candidate_budget_per_state"] == 6
    assert TRAIN_BUDGET_DEFAULTS["maximum_candidate_budget_per_state"] == 10
    assert TRAIN_BUDGET_DEFAULTS["target_accepted_total"] == 1600
    assert TRAIN_BUDGET_DEFAULTS["allow_duplicate_replacement"] is False

    catalog = pd.DataFrame(
        {"event_id": ["e1"], "checkpoint_id": ["c1"], "split": ["train"]}
    )
    # 3 accepted after 8 attempts: target not met, budget not exhausted.
    ledger = build_state_budget_ledger(
        catalog,
        accepted_rows("e1", "c1", 3),
        accepted_rows("e1", "c1", 8),
    )
    row = ledger.iloc[0]

    assert int(row["target_accepted"]) == 5
    assert int(row["accepted_actual_unique"]) == 3
    assert int(row["budget_remaining"]) == 2
    assert not bool(row["target_met"])
    assert not bool(row["budget_exhausted"])


def test_train_accepted_counts_actual_unique_only() -> None:
    catalog = pd.DataFrame(
        {"event_id": ["e1"], "checkpoint_id": ["c1"], "split": ["train"]}
    )
    accepted = pd.DataFrame(
        {
            "event_id": ["e1"] * 3,
            "checkpoint_id": ["c1"] * 3,
            # Two rows share one actual SHA: they count as one sample.
            "actual_schedule_sha256": ["same", "same", "other"],
        }
    )
    ledger = build_state_budget_ledger(catalog, accepted, accepted)

    assert int(ledger.iloc[0]["accepted_actual_unique"]) == 2
