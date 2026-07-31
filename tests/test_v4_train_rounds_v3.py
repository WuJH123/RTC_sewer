"""Round 0-2 rotation: 3 x 400, each state 2 extras, 5 after three rounds."""
from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.train1600_v3 import (
    ROUND_TARGETS_V3,
    _interleave_round_plan,
    assign_primary_candidates_to_rounds_v3,
    audit_round_dataset_v3,
    build_per_state_progress_v3,
    select_round_candidates_v3,
    verify_round_rotation_v3,
)
from train_v3_helpers import make_plan_chain


@pytest.fixture(scope="module")
def chain() -> dict:
    return make_plan_chain()


def test_rotation_240_states_3x400_2_extras_5_total(chain: dict) -> None:
    rotation = chain["rotation"]

    assert len(rotation) == 240
    report = verify_round_rotation_v3(rotation, expected_states=240)
    assert report["status"] == "pass"
    assert report["round_totals"] == {0: 400, 1: 400, 2: 400}
    assert report["grand_total"] == 1200
    assert ROUND_TARGETS_V3 == (400, 400, 400)
    # Every state has exactly one rest round (1) and two extra rounds (2).
    extras = sum(
        (rotation[f"round{r}_target"] == 2).astype(int) for r in (0, 1, 2)
    )
    assert extras.eq(2).all()
    per_state = (
        rotation["round0_target"]
        + rotation["round1_target"]
        + rotation["round2_target"]
    )
    assert per_state.eq(5).all()


def test_primary_assignment_matches_rotation_targets(chain: dict) -> None:
    assigned = assign_primary_candidates_to_rounds_v3(
        chain["role_plan"], chain["rotation"]
    )

    assert len(assigned) == 1200
    assert assigned["round"].value_counts().to_dict() == {0: 400, 1: 400, 2: 400}
    got = (
        assigned.groupby(["event_id", "checkpoint_id", "round"])
        .size()
        .unstack(fill_value=0)
    )
    want = chain["rotation"].set_index(["event_id", "checkpoint_id"])[
        ["round0_target", "round1_target", "round2_target"]
    ]
    want.columns = [0, 1, 2]
    assert got.sort_index().equals(want.sort_index())


def test_round0_selection_400_train_only_and_no_case_reuse(chain: dict) -> None:
    progress = build_per_state_progress_v3(chain["role_plan"])

    selected = select_round_candidates_v3(
        chain["role_plan"], chain["rotation"], 0, progress, set()
    )
    assert len(selected) == 400
    assert (selected["split"].astype(str) == "train").all()
    assert not selected["case_id"].duplicated().any()

    # Used case ids are never selected again: nothing is copied to fill.
    used = set(selected["case_id"].astype(str))
    round1 = select_round_candidates_v3(
        chain["role_plan"], chain["rotation"], 1, progress, used
    )
    assert len(round1) == 400
    assert not set(round1["case_id"].astype(str)) & used


def test_states_at_target_are_skipped_accepted_vs_budget(chain: dict) -> None:
    role_plan = chain["role_plan"]
    first = role_plan[role_plan["split"] == "train"].iloc[0]
    key_event, key_cp = str(first["event_id"]), str(first["checkpoint_id"])
    # Five accepted actual-unique rows: the state reached its target even
    # though budget (10 planned candidates) remains.
    accepted = pd.DataFrame(
        {
            "event_id": [key_event] * 5,
            "checkpoint_id": [key_cp] * 5,
            "actual_schedule_sha256": [f"sha{i}" for i in range(5)],
        }
    )
    progress = build_per_state_progress_v3(role_plan, accepted)
    row = progress[
        (progress["event_id"] == key_event)
        & (progress["checkpoint_id"] == key_cp)
    ].iloc[0]
    assert int(row["accepted"]) == 5
    assert int(row["remaining_to_target"]) == 0
    assert int(row["maximum_budget"]) == 10

    selected = select_round_candidates_v3(
        role_plan, chain["rotation"], 0, progress, set()
    )
    mask = (selected["event_id"] == key_event) & (
        selected["checkpoint_id"] == key_cp
    )
    assert not mask.any()


def test_accepted_counts_actual_unique_only(chain: dict) -> None:
    role_plan = chain["role_plan"]
    first = role_plan[role_plan["split"] == "train"].iloc[0]
    # Duplicate actual schedules collapse to one accepted sample.
    accepted = pd.DataFrame(
        {
            "event_id": [first["event_id"]] * 3,
            "checkpoint_id": [first["checkpoint_id"]] * 3,
            "actual_schedule_sha256": ["same", "same", "other"],
        }
    )
    progress = build_per_state_progress_v3(role_plan, accepted)
    row = progress[
        (progress["event_id"] == first["event_id"])
        & (progress["checkpoint_id"] == first["checkpoint_id"])
    ].iloc[0]
    assert int(row["accepted"]) == 2


def test_round_audit_hard_conditions() -> None:
    samples = pd.DataFrame(
        {
            "event_id": ["e1", "e1", "e2"],
            "checkpoint_id": ["c1", "c2", "c1"],
            "candidate_family": ["a", "b", "a"],
            "delta_pfv_h120_vs_no_control": [0.0, -1.0, 2.0],
            "same_state_verified": [True, True, True],
        }
    )
    accounting = {
        "accounting_closed": True,
        "accepted": 3,
        "actual_duplicates": 0,
    }
    audit = audit_round_dataset_v3(
        samples,
        accounting,
        stage="AuditTrainRound0V3",
        accepted_target=3,
        hard_columns=("same_state_verified",),
        reference_cache_sha256="ref",
        expected_reference_cache_sha256="ref",
    )
    assert audit["status"] == "pass"
    assert audit["informational"]["continuous_deltas_not_all_constant"]
    assert audit["informational"]["per_state_joint_candidate_not_required"]

    # A stale reference cache SHA is a hard block.
    stale = audit_round_dataset_v3(
        samples,
        accounting,
        stage="AuditTrainRound0V3",
        accepted_target=3,
        hard_columns=("same_state_verified",),
        reference_cache_sha256="tampered",
        expected_reference_cache_sha256="ref",
    )
    assert stale["status"] == "blocked"
    assert not stale["hard_checks"]["reference_cache_sha_consistent"]

    # Actual duplicates are a hard block, never silently accepted.
    dup = audit_round_dataset_v3(
        samples,
        {**accounting, "actual_duplicates": 1},
        stage="AuditTrainRound0V3",
        accepted_target=3,
        hard_columns=("same_state_verified",),
    )
    assert dup["status"] == "blocked"
    assert not dup["hard_checks"]["actual_duplicates_zero"]


def test_round_audit_blocks_forbidden_sampled_only_label() -> None:
    # A sampled-only Train1600 state relabelled as a P3-Exact verdict is a
    # hard block; recording no joint among 5 candidates is not fallback-only.
    samples = pd.DataFrame(
        {
            "event_id": ["e1", "e1", "e2"],
            "checkpoint_id": ["c1", "c2", "c1"],
            "candidate_family": ["a", "b", "a"],
            "delta_pfv_h120_vs_no_control": [0.0, -1.0, 2.0],
            "same_state_verified": [True, True, True],
            "state_feasibility_label_source": [
                "sampled_only",
                "fallback_only_under_budget",
                "sampled_only",
            ],
            "state_feasibility_label_validity": ["sampled_only"] * 3,
        }
    )
    accounting = {
        "accounting_closed": True,
        "accepted": 3,
        "actual_duplicates": 0,
    }
    audit = audit_round_dataset_v3(
        samples,
        accounting,
        stage="AuditTrainRound0V3",
        accepted_target=3,
        hard_columns=("same_state_verified",),
    )
    assert audit["status"] == "blocked"
    assert not audit["hard_checks"]["no_forbidden_sampled_only_label"]


def test_interleave_round_plan_spreads_families_without_changing_set() -> None:
    # A family-clustered selection (all family A ranked ahead of B and C) must
    # be reordered so any short completed prefix spans multiple families, while
    # the selected set stays byte-identical (no add/drop/copy of candidates).
    clustered = pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(9)],
            "candidate_family": ["A"] * 5 + ["B"] * 3 + ["C"],
            "event_id": [f"e{i}" for i in range(9)],
        }
    )
    ordered = _interleave_round_plan(clustered)
    # Same multiset of rows: nothing copied, nothing dropped.
    assert sorted(ordered["case_id"]) == sorted(clustered["case_id"])
    assert len(ordered) == len(clustered)
    # The first three rows already span all three families (round-robin).
    assert ordered["candidate_family"].head(3).nunique() == 3
    # Within a family, the incoming (AL-rank) order is preserved.
    a_order = ordered[ordered["candidate_family"] == "A"]["case_id"].tolist()
    assert a_order == ["c0", "c1", "c2", "c3", "c4"]
