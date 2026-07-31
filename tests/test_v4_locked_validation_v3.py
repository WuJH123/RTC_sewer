"""Locked Validation 200: frozen before run, never read by Active Learning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from sewerrtc.v4.pipeline import (
    PREREQUISITES,
    RUN_STAGE_PLANS,
    STAGE_ARTIFACTS,
)
from sewerrtc.v4.pipeline_train_v3 import (
    LOCKED_FROZEN_REL,
    PLAN_FREEZE_REL,
    _plan_frozen_segment_handler,
)
from sewerrtc.v4.runtime import RuntimeOptions
from sewerrtc.v4.train1600_v3 import (
    assert_train_split_only,
    build_per_state_progress_v3,
    rank_remaining_candidates_v3,
)
from train_v3_helpers import make_plan_chain


def test_locked_plan_is_200_rows_8_events_5_states_5_primary() -> None:
    chain = make_plan_chain()
    role_plan = chain["role_plan"]

    locked = role_plan[
        (role_plan["split"] == "locked_validation")
        & (role_plan["plan_tier"] == "primary")
    ]
    assert len(locked) == 200
    assert locked["event_id"].nunique() == 8
    per_state = locked.groupby(["event_id", "checkpoint_id"]).size()
    assert len(per_state) == 40
    assert per_state.eq(5).all()


def test_locked_run_wiring_requires_the_frozen_plan_stage() -> None:
    assert (
        STAGE_ARTIFACTS["PlanLockedValidation200V3"]
        == RUN_STAGE_PLANS["RunLockedValidation200V3"]
        == "train1600_v3/locked_validation/plan.csv"
    )
    assert "PlanLockedValidation200V3" in (
        PREREQUISITES["RunLockedValidation200V3"]
    )


def test_active_learner_fails_closed_on_locked_or_calibration_rows() -> None:
    for split in ("locked_validation", "calibration"):
        frame = pd.DataFrame({"split": ["train", split]})
        with pytest.raises(ValueError, match="only read the Train split"):
            assert_train_split_only(frame)
    # A frame without split provenance is equally unreadable.
    with pytest.raises(ValueError, match="no split column"):
        assert_train_split_only(pd.DataFrame({"event_id": ["e"]}))


def test_ranking_never_emits_calibration_or_locked_candidates() -> None:
    chain = make_plan_chain()
    role_plan = chain["role_plan"]
    progress = build_per_state_progress_v3(role_plan)
    accepted_train = pd.DataFrame(
        {
            "split": ["train"],
            "event_id": [role_plan.iloc[0]["event_id"]],
            "checkpoint_id": [role_plan.iloc[0]["checkpoint_id"]],
            "candidate_family": ["a"],
        }
    )

    ranked = rank_remaining_candidates_v3(role_plan, accepted_train, progress)

    assert len(ranked)
    assert set(ranked["split"].astype(str).unique()) == {"train"}
    # Locked accepted rows poison the learner input and must raise.
    poisoned = pd.concat(
        [accepted_train, accepted_train.assign(split="locked_validation")],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="only read the Train split"):
        rank_remaining_candidates_v3(role_plan, poisoned, progress)


def test_locked_plan_publication_is_immune_to_post_freeze_edits(
    tmp_path: Path,
) -> None:
    plan = pd.DataFrame(
        {
            "case_id": [f"lock{i}" for i in range(200)],
            "sample_id": [f"lock{i}" for i in range(200)],
            "split": "locked_validation",
        }
    )
    frozen = tmp_path / LOCKED_FROZEN_REL
    frozen.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(frozen, index=False)
    sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
    freeze = tmp_path / PLAN_FREEZE_REL
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text(
        json.dumps({"locked_validation_plan_sha256": sha}), encoding="utf-8"
    )
    handler = _plan_frozen_segment_handler(
        tmp_path,
        tmp_path,
        {},
        stage="PlanLockedValidation200V3",
        seg="locked_validation",
        frozen_rel=LOCKED_FROZEN_REL,
        sha_key="locked_validation_plan_sha256",
    )

    result = handler(RuntimeOptions(stage="PlanLockedValidation200V3"))
    assert result.status == "pass"
    assert result.evidence["rows"] == 200

    # Any post-hoc edit (e.g. reacting to Calibration results) breaks the
    # frozen SHA and blocks republication.
    frozen.write_text(
        frozen.read_text(encoding="utf-8").replace("lock0,", "lockX,"),
        encoding="utf-8",
    )
    rerun = handler(RuntimeOptions(stage="PlanLockedValidation200V3"))
    assert rerun.status == "blocked"
    assert rerun.evidence["reason"] == "frozen_plan_sha_mismatch"
