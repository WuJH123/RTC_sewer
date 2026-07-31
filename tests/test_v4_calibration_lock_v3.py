"""Calibration 200: plan frozen by PlanTrain1600V3 before any SWMM run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from sewerrtc.v4.pipeline import (
    PREREQUISITES,
    RUN_STAGE_PLANS,
    STAGE_ARTIFACTS,
)
from sewerrtc.v4.pipeline_train_v3 import (
    CAL_FROZEN_REL,
    PLAN_FREEZE_REL,
    _plan_frozen_segment_handler,
)
from sewerrtc.v4.runtime import RuntimeOptions
from train_v3_helpers import make_plan_chain


def test_calibration_plan_is_200_rows_8_events_5_states_5_primary() -> None:
    chain = make_plan_chain()
    role_plan = chain["role_plan"]

    calibration = role_plan[
        (role_plan["split"] == "calibration")
        & (role_plan["plan_tier"] == "primary")
    ]
    assert len(calibration) == 200
    assert calibration["event_id"].nunique() == 8
    per_state = calibration.groupby(["event_id", "checkpoint_id"]).size()
    assert len(per_state) == 40
    assert per_state.eq(5).all()


def test_calibration_run_plan_is_the_published_frozen_artifact() -> None:
    # The Plan stage's artifact IS the Run stage's plan: publishing requires
    # the freeze-time SHA, so the run can only ever see the frozen plan.
    assert (
        STAGE_ARTIFACTS["PlanCalibration200V3"]
        == RUN_STAGE_PLANS["RunCalibration200V3"]
        == "train1600_v3/calibration/plan.csv"
    )
    assert "PlanCalibration200V3" in PREREQUISITES["RunCalibration200V3"]


def _write_frozen_calibration(root: Path, rows: int) -> str:
    plan = pd.DataFrame(
        {
            "case_id": [f"cal{i}" for i in range(rows)],
            "sample_id": [f"cal{i}" for i in range(rows)],
            "split": "calibration",
        }
    )
    frozen = root / CAL_FROZEN_REL
    frozen.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(frozen, index=False)
    return hashlib.sha256(frozen.read_bytes()).hexdigest()


def _write_freeze(root: Path, sha: str) -> None:
    freeze = root / PLAN_FREEZE_REL
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text(
        json.dumps({"calibration_plan_sha256": sha}), encoding="utf-8"
    )


def test_frozen_calibration_plan_publishes_only_with_matching_sha(
    tmp_path: Path,
) -> None:
    sha = _write_frozen_calibration(tmp_path, 200)
    _write_freeze(tmp_path, sha)
    handler = _plan_frozen_segment_handler(
        tmp_path,
        tmp_path,
        {},
        stage="PlanCalibration200V3",
        seg="calibration",
        frozen_rel=CAL_FROZEN_REL,
        sha_key="calibration_plan_sha256",
    )

    result = handler(RuntimeOptions(stage="PlanCalibration200V3"))

    assert result.status == "pass"
    published = tmp_path / STAGE_ARTIFACTS["PlanCalibration200V3"]
    assert published.exists()
    assert len(pd.read_csv(published)) == 200
    check = json.loads(
        (published.parent / "plan_freeze_check.json").read_text(
            encoding="utf-8"
        )
    )
    assert check["frozen_before_any_corresponding_swmm"] is True
    assert check["frozen_sha256"] == sha


def test_tampered_calibration_plan_is_blocked_not_republished(
    tmp_path: Path,
) -> None:
    sha = _write_frozen_calibration(tmp_path, 200)
    _write_freeze(tmp_path, sha)
    # Post-freeze tampering (any edit after PlanTrain1600V3 froze the SHA).
    frozen = tmp_path / CAL_FROZEN_REL
    frozen.write_text(
        frozen.read_text(encoding="utf-8") + "calX,calX,calibration\n",
        encoding="utf-8",
    )
    handler = _plan_frozen_segment_handler(
        tmp_path,
        tmp_path,
        {},
        stage="PlanCalibration200V3",
        seg="calibration",
        frozen_rel=CAL_FROZEN_REL,
        sha_key="calibration_plan_sha256",
    )

    result = handler(RuntimeOptions(stage="PlanCalibration200V3"))

    assert result.status == "blocked"
    assert result.evidence["reason"] == "frozen_plan_sha_mismatch"
    assert not (
        tmp_path / STAGE_ARTIFACTS["PlanCalibration200V3"]
    ).exists()


def test_wrong_row_count_never_passes_even_with_valid_sha(
    tmp_path: Path,
) -> None:
    sha = _write_frozen_calibration(tmp_path, 40)
    _write_freeze(tmp_path, sha)
    handler = _plan_frozen_segment_handler(
        tmp_path,
        tmp_path,
        {},
        stage="PlanCalibration200V3",
        seg="calibration",
        frozen_rel=CAL_FROZEN_REL,
        sha_key="calibration_plan_sha256",
    )

    result = handler(RuntimeOptions(stage="PlanCalibration200V3"))

    assert result.status == "blocked"
    assert result.evidence["expected_rows"] == 200
