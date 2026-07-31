"""Tests for the Layer-3 pilot branch plan (400 candidates -> 1600 branches)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from sewerrtc.v4.pilot_candidates import (
    PILOT_BRANCH_ROLES,
    build_pilot_branch_plan,
)

N_EVENTS = 8
N_CHECKPOINTS = 5
N_PER_STATE = 10
WIDTH = 3


def _candidate_plan() -> pd.DataFrame:
    """Synthetic 40-state x 10-sample candidate plan (lightweight)."""
    projected = [[0.5] * WIDTH for _ in range(12)]
    anchor = [[0.0] * WIDTH for _ in range(12)]
    rows = []
    for event_index in range(N_EVENTS):
        event_id = f"e{event_index}"
        for checkpoint_index in range(N_CHECKPOINTS):
            checkpoint_id = f"{event_id}_c{checkpoint_index}"
            for sample_index in range(N_PER_STATE):
                sample_id = f"{checkpoint_id}__s{sample_index}"
                sample_projected = [row[:] for row in projected]
                sample_projected[0][0] = 0.01 * (sample_index + 1)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "event_id": event_id,
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_state_sha256": f"state_{checkpoint_id}",
                        "rainfall_sha256": f"rain_{event_id}",
                        "network_sha256": "netsha",
                        "runner_kwargs": json.dumps(
                            {"inp_path": "network.inp"}
                        ),
                        "projected_schedule_json": json.dumps(
                            sample_projected
                        ),
                        "anchor_schedule_json": json.dumps(anchor),
                        "projected_schedule_path": (
                            f"schedules/{sample_id}.json"
                        ),
                        "split": "pilot_train"
                        if event_index < 5
                        else "pilot_calibration",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def branch_plan() -> pd.DataFrame:
    return build_pilot_branch_plan(
        _candidate_plan(), contract_sha256="contract-sha"
    )


def test_branch_plan_expands_400_candidates_into_1600_rows(
    branch_plan: pd.DataFrame,
) -> None:
    assert len(branch_plan) == 1600
    assert branch_plan["branch_id"].is_unique
    assert branch_plan["sample_id"].nunique() == 400
    roles_per_sample = branch_plan.groupby("sample_id")["branch_role"].apply(
        set
    )
    assert roles_per_sample.eq(set(PILOT_BRANCH_ROLES)).all()


def test_only_candidate_branches_are_counted_as_samples(
    branch_plan: pd.DataFrame,
) -> None:
    counted = branch_plan[branch_plan["counted_as_sample"].astype(bool)]
    assert len(counted) == 400
    assert set(counted["branch_role"]) == {"candidate"}
    references = branch_plan[~branch_plan["counted_as_sample"].astype(bool)]
    assert set(references["branch_role"]) == {
        "no_control",
        "dynamic_internal_rules",
        "hold_previous",
    }
    assert references["reference_cache_key"].astype(str).str.len().gt(0).all()
    assert counted["reference_cache_key"].astype(str).eq("").all()


def test_reference_cache_keys_are_shared_per_state_and_total_120(
    branch_plan: pd.DataFrame,
) -> None:
    references = branch_plan[branch_plan["branch_role"] != "candidate"]
    # Every state's 10 samples share one key per reference branch.
    per_state = references.groupby(
        ["event_id", "checkpoint_id", "branch_role"]
    )["reference_cache_key"].nunique()
    assert per_state.eq(1).all()
    # 40 states x 3 reference branches = 120 unique keys overall.
    assert references["reference_cache_key"].nunique() == 120


def test_branch_kwargs_encode_control_modes_and_actions(
    branch_plan: pd.DataFrame,
) -> None:
    sample_rows = branch_plan[branch_plan["sample_id"] == "e0_c0__s0"]
    by_role = {
        str(row["branch_role"]): json.loads(str(row["runner_kwargs"]))
        for _, row in sample_rows.iterrows()
    }
    assert (
        by_role["dynamic_internal_rules"]["post_control_mode"]
        == "native_rules"
    )
    for role in ("candidate", "no_control", "hold_previous"):
        assert by_role[role]["post_control_mode"] == "external_override"
    # No-control opens everything: 12 steps of all-1.0 actions.
    assert by_role["no_control"]["post_action"] == [
        [1.0] * WIDTH for _ in range(12)
    ]
    # Candidate carries its own projected schedule (first cell perturbed).
    assert by_role["candidate"]["post_action"][0][0] == pytest.approx(0.01)
    # Hold-previous replays the anchor schedule.
    assert by_role["hold_previous"]["post_action"] == [
        [0.0] * WIDTH for _ in range(12)
    ]


def test_branch_plan_rejects_incomplete_candidate_plan() -> None:
    bad = _candidate_plan().drop(columns=["anchor_schedule_json"])
    with pytest.raises(ValueError, match="candidate plan missing"):
        build_pilot_branch_plan(bad)
