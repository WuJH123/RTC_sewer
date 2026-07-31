from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.active_learning import filter_selectable_candidates
from sewerrtc.v4.training_plan import (
    REPLENISH_ORDER,
    build_state_budget_ledger,
    plan_state_replenishment,
)


def make_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "checkpoint_id": ["c1", "c2"],
            "split": ["train", "train"],
        }
    )


def accepted_rows(event: str, checkpoint: str, shas: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [event] * len(shas),
            "checkpoint_id": [checkpoint] * len(shas),
            "actual_schedule_sha256": shas,
        }
    )


def make_pool(event: str, checkpoint: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [event] * 3,
            "checkpoint_id": [checkpoint] * 3,
            "candidate_id": ["p1", "p2", "p3"],
            "candidate_family": ["famA", "famB", "famC"],
            "replenish_source": [
                "boundary_uncertainty_coverage_gap",
                "state_reserve_candidate",
                "new_candidate_family",
            ],
            "projected_schedule_sha256": ["sha-p1", "sha-p2", "sha-p3"],
        }
    )


def test_replenishment_never_copies_accepted_actual() -> None:
    accepted = accepted_rows("e1", "c1", ["sha-p2", "old-1"])
    attempted = accepted_rows("e1", "c1", ["x"] * 6)
    ledger = build_state_budget_ledger(make_catalog(), accepted, attempted)
    plan = plan_state_replenishment(ledger, make_pool("e1", "c1"), accepted)

    # sha-p2 duplicates an accepted actual: it must never be planned again.
    assert "sha-p2" not in set(plan["projected_schedule_sha256"])
    assert set(plan["queue"]) == {"replenish"}


def test_replenishment_follows_legal_source_order() -> None:
    accepted = accepted_rows("e1", "c1", ["old-1", "old-2", "old-3"])
    attempted = accepted_rows("e1", "c1", ["x"] * 6)
    ledger = build_state_budget_ledger(make_catalog(), accepted, attempted)
    plan = plan_state_replenishment(ledger, make_pool("e1", "c1"), accepted)

    # Need 2 more: state reserve first, then new family, never the
    # boundary candidate before those two.
    assert plan["replenish_source"].tolist() == [
        "state_reserve_candidate",
        "new_candidate_family",
    ]
    assert list(REPLENISH_ORDER) == [
        "state_reserve_candidate",
        "new_candidate_family",
        "boundary_uncertainty_coverage_gap",
    ]


def test_replenishment_rejects_illegal_source() -> None:
    accepted = accepted_rows("e1", "c1", ["old-1"])
    attempted = accepted_rows("e1", "c1", ["x"] * 6)
    ledger = build_state_budget_ledger(make_catalog(), accepted, attempted)
    pool = make_pool("e1", "c1")
    pool.loc[pool.index[0], "replenish_source"] = "borrow_from_other_split"

    with pytest.raises(ValueError):
        plan_state_replenishment(ledger, pool, accepted)


def test_exhausted_state_is_marked_and_never_replenished() -> None:
    accepted = accepted_rows("e1", "c1", ["old-1", "old-2"])
    attempted = accepted_rows("e1", "c1", ["x"] * 10)
    ledger = build_state_budget_ledger(make_catalog(), accepted, attempted)
    row = ledger.set_index(["event_id", "checkpoint_id"]).loc[("e1", "c1")]

    assert bool(row["budget_exhausted"])
    assert bool(row["state_shortfall"])

    plan = plan_state_replenishment(ledger, make_pool("e1", "c1"), accepted)
    assert len(plan) == 0


def test_satisfied_state_is_filtered_from_active_learning() -> None:
    accepted = accepted_rows("e1", "c1", [f"s{i}" for i in range(5)])
    attempted = accepted_rows("e1", "c1", ["x"] * 5)
    ledger = build_state_budget_ledger(make_catalog(), accepted, attempted)
    candidates = pd.DataFrame(
        {
            "case_id": ["k1", "k2"],
            "event_id": ["e1", "e2"],
            "checkpoint_id": ["c1", "c2"],
            "projected_schedule_sha256": ["n1", "n2"],
        }
    )
    selectable = filter_selectable_candidates(candidates, ledger)

    # e1/c1 reached 5 accepted: never resampled.  e2/c2 stays selectable.
    assert selectable["case_id"].tolist() == ["k2"]


def test_filter_drops_duplicates_completed_and_stale_rows() -> None:
    ledger = build_state_budget_ledger(
        make_catalog(), pd.DataFrame(), pd.DataFrame()
    )
    accepted = accepted_rows("e2", "c2", ["dup-sha"])
    candidates = pd.DataFrame(
        {
            "case_id": ["a", "b", "c", "d"],
            "event_id": ["e2"] * 4,
            "checkpoint_id": ["c2"] * 4,
            "projected_schedule_sha256": ["dup-sha", "n1", "n2", "n3"],
            "plan_sha256": ["cur", "cur", "cur", "stale"],
        }
    )
    selectable = filter_selectable_candidates(
        candidates,
        ledger,
        accepted_shas=accepted,
        completed_case_ids={"b"},
        current_plan_sha="cur",
    )

    # a = duplicate actual, b = already completed, d = stale plan SHA.
    assert selectable["case_id"].tolist() == ["c"]
