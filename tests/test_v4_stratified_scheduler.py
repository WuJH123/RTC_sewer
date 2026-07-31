from __future__ import annotations

import pandas as pd

from sewerrtc.v4.runtime import select_pending, stratified_order


def make_pilot_like_plan() -> pd.DataFrame:
    # 8 events x 5 checkpoints x 10 candidates in raw CSV order: all rows of
    # one event first, exactly the layout the scheduler must break up.
    rows = []
    for event in range(8):
        for checkpoint in range(5):
            for candidate in range(10):
                rows.append(
                    {
                        "case_id": f"e{event}_c{checkpoint}_k{candidate}",
                        "event_id": f"e{event}",
                        "checkpoint_id": f"e{event}_c{checkpoint}",
                        "candidate_family": f"fam{candidate % 3}",
                        "K": (candidate % 4) + 1,
                        "candidate_priority": candidate,
                    }
                )
    return pd.DataFrame(rows)


def test_first_16_cases_cover_many_events_and_states() -> None:
    ordered = stratified_order(make_pilot_like_plan())
    first16 = ordered.head(16)

    assert first16["event_id"].nunique() == 8
    assert first16[["event_id", "checkpoint_id"]].drop_duplicates().shape[0] == 16


def test_first_40_cases_cover_all_40_pilot_states() -> None:
    ordered = stratified_order(make_pilot_like_plan())
    first40 = ordered.head(40)

    states = first40[["event_id", "checkpoint_id"]].drop_duplicates()
    assert len(states) == 40
    assert first40["event_id"].nunique() == 8
    assert first40["candidate_family"].nunique() >= 2


def test_never_runs_one_event_back_to_back_in_csv_order() -> None:
    ordered = stratified_order(make_pilot_like_plan())
    first_events = ordered["event_id"].head(8).tolist()

    assert len(set(first_events)) == 8


def test_ordering_is_deterministic() -> None:
    plan = make_pilot_like_plan()
    once = stratified_order(plan)["case_id"].tolist()
    twice = stratified_order(plan.sample(frac=1.0, random_state=7))[
        "case_id"
    ].tolist()

    assert once == stratified_order(plan)["case_id"].tolist()
    # Shuffled input still yields a per-prefix event spread even if
    # tie-breaking differs; the first eight cases cover eight events.
    assert len(set(twice[:8])) == 8


def test_select_pending_stratifies_after_filter_before_limit() -> None:
    plan = make_pilot_like_plan()
    # Complete every case of e0 so pending must interleave the rest.
    completed = plan[plan["event_id"] == "e0"]["case_id"].tolist()
    pending = select_pending(plan, completed, limit=7)

    assert len(pending) == 7
    assert pending["event_id"].nunique() == 7
    assert "e0" not in set(pending["event_id"])


def test_plan_without_scheduler_columns_passes_through() -> None:
    plan = pd.DataFrame({"case_id": ["a", "b"]})
    ordered = stratified_order(plan)

    assert ordered["case_id"].tolist() == ["a", "b"]
