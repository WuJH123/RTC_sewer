from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.training_plan import (
    build_round_rotation,
    verify_round_rotation,
)


def make_train_catalog() -> pd.DataFrame:
    # 64 events x 5 checkpoints = 320 formal states, split 48/8/8.
    rows = []
    for event in range(64):
        split = "train" if event < 48 else ("val" if event < 56 else "test")
        for checkpoint in range(5):
            rows.append(
                {
                    "event_id": f"e{event:02d}",
                    "checkpoint_id": f"e{event:02d}_c{checkpoint}",
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def test_four_rounds_of_400_give_each_state_exactly_5() -> None:
    state_targets, extra_rotation = build_round_rotation(make_train_catalog())

    per_round = state_targets.groupby("round")["round_target"].sum()
    assert per_round.eq(400).all()

    per_state = state_targets.groupby(
        ["event_id", "checkpoint_id"]
    )["round_target"].sum()
    assert per_state.eq(5).all()
    assert per_state.size == 320

    assert int(state_targets["basic_target"].sum()) == 4 * 320
    assert int(state_targets["extra_target"].sum()) == 4 * 80
    assert len(extra_rotation) == 320


def test_extra_80_sets_are_disjoint_across_rounds() -> None:
    _, extra_rotation = build_round_rotation(make_train_catalog())

    assert extra_rotation.groupby("round").size().eq(80).all()
    assert not extra_rotation.duplicated(
        ["event_id", "checkpoint_id"]
    ).any()
    # Union of the four extra sets is exactly the 320 states.
    assert (
        extra_rotation[["event_id", "checkpoint_id"]]
        .drop_duplicates()
        .shape[0]
        == 320
    )


def test_rotation_is_deterministic_and_event_balanced() -> None:
    catalog = make_train_catalog()
    once, extra_once = build_round_rotation(catalog)
    again, extra_again = build_round_rotation(
        catalog.sample(frac=1.0, random_state=3)
    )

    pd.testing.assert_frame_equal(
        once.reset_index(drop=True), again.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        extra_once.reset_index(drop=True),
        extra_again.reset_index(drop=True),
    )
    # Dealt round-robin over events: no event contributes more than 2 extras
    # to one round (64 events x 5 states -> 80 extras per round).
    per_event_round = extra_once.groupby(["round", "event_id"]).size()
    assert per_event_round.le(2).all()


def test_rotation_keeps_split_labels() -> None:
    state_targets, extra_rotation = build_round_rotation(make_train_catalog())

    assert set(state_targets["split"]) == {"train", "val", "test"}
    assert set(extra_rotation["split"]) <= {"train", "val", "test"}


def test_verify_round_rotation_proof_passes() -> None:
    state_targets, _ = build_round_rotation(make_train_catalog())
    proof = verify_round_rotation(state_targets)

    assert proof["status"] == "pass"
    assert all(proof["checks"].values())


def test_verify_round_rotation_catches_tampering() -> None:
    state_targets, _ = build_round_rotation(make_train_catalog())
    tampered = state_targets.copy()
    tampered.loc[tampered.index[0], "round_target"] = 3
    proof = verify_round_rotation(tampered)

    assert proof["status"] == "blocked"


def test_rotation_rejects_wrong_state_count() -> None:
    catalog = make_train_catalog().head(300)
    with pytest.raises(ValueError):
        build_round_rotation(catalog)
