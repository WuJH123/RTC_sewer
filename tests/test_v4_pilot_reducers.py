"""Tests for the Pilot400 dedicated reducers (spec section VII)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.partial_audit import HARD_AUTHENTICITY_COLUMNS
from sewerrtc.v4.pilot_reducers import (
    TEMPORAL_RESIDUAL_SIGNALS,
    build_pilot_partial_bundle,
)

FACILITIES = ["ADD301.2", "ADD301.3", "add350.1"]
PRIORITY = ["NODEP"]
CHECKPOINT = 60.0
MARGIN = {"pfv_m3": 0.0, "tfv_m3": 0.0, "peak_m3s": 0.0}
DEAD_ZONE = {"pfv_m3": 0.0, "tfv_m3": 0.0, "peak_m3s": 0.0}


def _write_branch_detail(
    directory: Path,
    post_flood_rate: float,
    actual_value: float,
    spike_rate: float | None = None,
) -> Path:
    # Same-state prefix; branches diverge only after the checkpoint.
    elapsed = np.arange(0.0, 185.0, 5.0)
    pre = elapsed <= CHECKPOINT
    rate = np.where(pre, 0.01, post_flood_rate)
    if spike_rate is not None:
        rate = rate.copy()
        rate[np.argmin(np.abs(elapsed - (CHECKPOINT + 30.0)))] = spike_rate
    frame = {"elapsed_min": elapsed}
    frame["flood:NODEP"] = rate
    frame["tfv_rate_m3s"] = rate
    for facility in FACILITIES:
        frame[f"flow:{facility}"] = np.where(pre, 0.5, 0.5 + post_flood_rate)
        frame[f"storage_volume:{facility}"] = np.where(pre, 10.0, 12.0)
        setting = np.where(pre, 0.0, actual_value)
        frame[f"requested_setting:{facility}"] = setting
        frame[f"target_setting:{facility}"] = setting
        frame[f"actual_setting:{facility}"] = setting
        frame[f"readback_setting:{facility}"] = setting
        frame[f"a:{facility}"] = setting
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "detail.csv"
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def _completion_row(
    root: Path,
    sample_id: str,
    branch: str,
    post_flood: float,
    actual: float,
    *,
    status: str = "pass",
    spike_rate: float | None = None,
) -> dict:
    case_id = f"{sample_id}__{branch}"
    detail_path = _write_branch_detail(
        root / case_id, post_flood, actual, spike_rate=spike_rate
    )
    return {
        "case_id": case_id,
        "sample_id": sample_id,
        "branch": branch,
        "status": status,
        "detail_path": str(detail_path),
        "rainfall_sha256": "rainsha",
        "input_sha": "insha",
        "runner_kwargs": json.dumps({"inp_path": "x.inp"}),
        "result": {
            "hotstart_used": False,
            "use_hotstart_call_count": 0,
            "save_hotstart_call_count": 0,
        },
    }


def _candidate_plan_row(
    sample_id: str, event_id: str = "e0", checkpoint_id: str = "e0_c0"
) -> dict:
    return {
        "sample_id": sample_id,
        "event_id": event_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_min": CHECKPOINT,
        "k_target": 2,
        "rainfall_sha256": "rainsha",
        "binary_semantics_ok": True,
        "rate_limit_ok": True,
        "dwell_ok": True,
        "interlock_ok": True,
        "projected_schedule_json": json.dumps([[1.0, 1.0, 1.0]]),
        "anchor_schedule_json": json.dumps([[0.0, 0.0, 0.0]]),
    }


def _branch_plan_rows(sample_id: str) -> list[dict]:
    return [
        {
            "sample_id": sample_id,
            "case_id": f"{sample_id}__{branch}",
            "branch_role": branch,
        }
        for branch in (
            "candidate",
            "no_control",
            "dynamic_internal_rules",
            "hold_previous",
        )
    ]


def _four_branch_completions(
    root: Path, sample_id: str, *, actual: float = 1.0
) -> list[dict]:
    return [
        _completion_row(root, sample_id, "candidate", 0.02, actual),
        _completion_row(root, sample_id, "no_control", 0.05, 0.0),
        _completion_row(root, sample_id, "dynamic_internal_rules", 0.03, 0.0),
        _completion_row(root, sample_id, "hold_previous", 0.01, 0.0),
    ]


def _bundle(candidate_plan, branch_plan, completions) -> dict:
    return build_pilot_partial_bundle(
        pd.DataFrame(candidate_plan),
        pd.DataFrame(branch_plan),
        pd.DataFrame(completions),
        priority_nodes=PRIORITY,
        facility_ids=FACILITIES,
        scientific_margin=MARGIN,
        dead_zone=DEAD_ZONE,
    )


def test_complete_sample_is_accepted_with_correct_branch_references(
    tmp_path: Path,
) -> None:
    bundle = _bundle(
        [_candidate_plan_row("s1")],
        _branch_plan_rows("s1"),
        _four_branch_completions(tmp_path, "s1"),
    )

    assert bundle["completed_total"] == 1
    assert len(bundle["pending"]) == 0
    manifest = bundle["sample_manifest"]
    assert len(manifest) == 1
    row = manifest.iloc[0]
    for column in HARD_AUTHENTICITY_COLUMNS:
        assert bool(row[column]) is True, column
    # PFV delta is against no_control (0.05) while TFV delta is against
    # dynamic_internal_rules (0.03): both negative, PFV gap exactly 3x the
    # TFV gap for these constant rates.  Any NC/DI mix-up breaks the ratio.
    delta_pfv = float(row["delta_pfv_h120_vs_no_control"])
    delta_tfv = float(row["delta_tfv_h120_vs_dynamic_internal"])
    assert delta_pfv < 0.0
    assert delta_tfv < 0.0
    assert delta_pfv / delta_tfv == pytest.approx(3.0, rel=1e-6)
    assert float(
        row["delta_peak_h120_vs_dynamic_internal"]
    ) == pytest.approx(-0.01, abs=1e-9)
    # Twelve-step temporal residuals are attached for every signal.
    for signal in TEMPORAL_RESIDUAL_SIGNALS:
        assert len(json.loads(str(row[signal]))) == 12
    assert bool(row["full_event_eligible"]) is False


def test_peak_uses_each_branchs_own_maximum(tmp_path: Path) -> None:
    # Candidate has a lower mean but a higher spike than dynamic internal:
    # the peak delta must compare per-branch maxima, not means.
    completions = [
        _completion_row(
            tmp_path, "s1", "candidate", 0.02, 1.0, spike_rate=0.10
        ),
        _completion_row(tmp_path, "s1", "no_control", 0.05, 0.0),
        _completion_row(
            tmp_path, "s1", "dynamic_internal_rules", 0.03, 0.0
        ),
        _completion_row(tmp_path, "s1", "hold_previous", 0.01, 0.0),
    ]
    bundle = _bundle(
        [_candidate_plan_row("s1")], _branch_plan_rows("s1"), completions
    )

    manifest = bundle["sample_manifest"]
    assert len(manifest) == 1
    assert float(
        manifest.iloc[0]["delta_peak_h120_vs_dynamic_internal"]
    ) == pytest.approx(0.10 - 0.03, abs=1e-9)


def test_missing_branch_is_pending_never_missing(tmp_path: Path) -> None:
    completions = _four_branch_completions(tmp_path, "s1")[:3]
    bundle = _bundle(
        [_candidate_plan_row("s1")], _branch_plan_rows("s1"), completions
    )

    assert bundle["completed_total"] == 0
    assert bundle["pending"]["sample_id"].tolist() == ["s1"]
    assert len(bundle["missing_confirmed"]) == 0
    assert len(bundle["rejected"]) == 0


def test_failed_branch_is_hard_rejection_not_pending(tmp_path: Path) -> None:
    completions = _four_branch_completions(tmp_path, "s1")
    completions[1]["status"] = "failed"
    bundle = _bundle(
        [_candidate_plan_row("s1")], _branch_plan_rows("s1"), completions
    )

    assert bundle["completed_total"] == 1
    assert len(bundle["pending"]) == 0
    assert len(bundle["sample_manifest"]) == 0
    assert bundle["rejected"].iloc[0]["rejection_reason"] == (
        "four_branch_incomplete_or_failed"
    )
    assert bundle["hard_violation_total"] == 1


def test_duplicate_actual_schedules_in_same_state_are_deduplicated(
    tmp_path: Path,
) -> None:
    candidate_plan = [
        _candidate_plan_row("s1"),
        _candidate_plan_row("s2"),
    ]
    branch_plan = _branch_plan_rows("s1") + _branch_plan_rows("s2")
    completions = _four_branch_completions(
        tmp_path, "s1"
    ) + _four_branch_completions(tmp_path, "s2")
    bundle = _bundle(candidate_plan, branch_plan, completions)

    assert bundle["completed_total"] == 2
    assert len(bundle["sample_manifest"]) == 1
    duplicates = bundle["actual_duplicates"]
    assert len(duplicates) == 1
    assert duplicates.iloc[0]["rejection_reason"] == (
        "duplicate_actual_schedule"
    )


def test_output_isolated_true_for_isolated_pilot_detail_paths(
    tmp_path: Path,
) -> None:
    # Pilot semantics: candidate detail sits in the sample's own run
    # directory (path contains the sample id) while each reference detail
    # sits in the shared cache directory named after its branch role.
    bundle = _bundle(
        [_candidate_plan_row("s1")],
        _branch_plan_rows("s1"),
        _four_branch_completions(tmp_path, "s1"),
    )

    manifest = bundle["sample_manifest"]
    assert len(manifest) == 1
    assert bool(manifest.iloc[0]["output_isolated"]) is True


def test_output_isolated_false_when_branches_share_a_detail_path(
    tmp_path: Path,
) -> None:
    completions = _four_branch_completions(tmp_path, "s1")
    # hold_previous illegally reuses the no_control detail file: the path
    # exists but lacks the branch name and collapses the four-path set.
    completions[3]["detail_path"] = completions[1]["detail_path"]
    bundle = _bundle(
        [_candidate_plan_row("s1")], _branch_plan_rows("s1"), completions
    )

    manifest = bundle["sample_manifest"]
    assert len(manifest) == 1
    assert bool(manifest.iloc[0]["output_isolated"]) is False
