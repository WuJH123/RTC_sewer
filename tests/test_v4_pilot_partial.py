"""Tests for the Pilot partial wiring inside the V4 pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sewerrtc.v4 import pipeline
from sewerrtc.v4.runtime import atomic_write_json


FACILITIES = ["ADD301.2", "ADD301.3", "add350.1"]
CHECKPOINT = 60.0
BRANCHES = (
    "candidate",
    "no_control",
    "dynamic_internal_rules",
    "hold_previous",
)
MARGIN = {"pfv_m3": 0.0, "tfv_m3": 0.0, "peak_m3s": 0.0}
DEAD_ZONE = {"pfv_m3": 0.0, "tfv_m3": 0.0, "peak_m3s": 0.0}


def _write_branch_detail(
    directory: Path, post_flood_rate: float, actual_value: float
) -> Path:
    elapsed = np.arange(0.0, 185.0, 5.0)
    pre = elapsed <= CHECKPOINT
    frame = {"elapsed_min": elapsed}
    frame["flood:NODEP"] = np.where(pre, 0.01, post_flood_rate)
    frame["tfv_rate_m3s"] = np.where(pre, 0.01, post_flood_rate)
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


def _candidate_plan_row(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "case_id": f"{sample_id}__candidate",
        "event_id": "e0",
        "checkpoint_id": "e0_c0",
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


def _sample_completion(run_root: Path, sample_id: str) -> dict:
    rates = {
        "candidate": (0.02, 1.0),
        "no_control": (0.05, 0.0),
        "dynamic_internal_rules": (0.03, 0.0),
        "hold_previous": (0.01, 0.0),
    }
    branches = {}
    for branch, (rate, actual) in rates.items():
        detail = _write_branch_detail(
            run_root / sample_id / branch, rate, actual
        )
        branches[branch] = {
            "status": "pass",
            "detail_path": str(detail),
            "result": {
                "hotstart_used": False,
                "use_hotstart_call_count": 0,
                "save_hotstart_call_count": 0,
            },
            "runner_kwargs": json.dumps({"inp_path": "x.inp"}),
        }
    return {
        "case_id": sample_id,
        "sample_id": sample_id,
        "status": "pass",
        "input_sha": "insha",
        "rainfall_sha256": "rainsha",
        "branches": branches,
    }


def _prepare(tmp_path: Path, *, completed_samples: list[str]) -> tuple:
    project_root = tmp_path / "project"
    output_root = tmp_path / "out"
    planning = output_root / "pilot" / "planning"
    run_root = output_root / "pilot" / "runs"
    planning.mkdir(parents=True)
    run_root.mkdir(parents=True)
    project_root.mkdir(parents=True)
    (project_root / "prio.txt").write_text("NODEP\n", encoding="utf-8")
    (project_root / "ids.txt").write_text(
        "\n".join(FACILITIES) + "\n", encoding="utf-8"
    )

    sample_ids = ["s1", "s2"]
    plan = pd.DataFrame([_candidate_plan_row(s) for s in sample_ids])
    plan.to_csv(planning / "pilot_candidate_plan.csv", index=False)
    branch_plan = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "case_id": f"{sample_id}__{branch}",
                "branch_role": branch,
            }
            for sample_id in sample_ids
            for branch in BRANCHES
        ]
    )
    branch_plan.to_csv(planning / "pilot_branch_plan.csv", index=False)

    for sample_id in completed_samples:
        completion = _sample_completion(run_root, sample_id)
        atomic_write_json(
            run_root / sample_id / "completion.json", completion
        )

    config = {
        "project": {"priority_nodes": "prio.txt", "canonical_ids": "ids.txt"},
        "thresholds": {"scientific_margin": MARGIN, "dead_zone": DEAD_ZONE},
    }
    return project_root, output_root, config


def test_run_pilot400_groups_by_sample_id() -> None:
    # One plan row = one four-branch sample task; the runtime limit and
    # resume bookkeeping must count samples, not branches.
    assert pipeline.RUN_STAGE_GROUP_KEYS["RunPilot400"] == "sample_id"


def test_pilot_partial_bundle_reduces_completed_sample(
    tmp_path: Path,
) -> None:
    project_root, output_root, config = _prepare(
        tmp_path, completed_samples=["s1"]
    )

    plan, bundle = pipeline._pilot_partial_bundle(
        project_root, output_root, config
    )

    assert len(plan) == 2
    assert bundle["completed_total"] == 1
    assert len(bundle["sample_manifest"]) == 1
    assert bundle["sample_manifest"].iloc[0]["sample_id"] == "s1"
    # The unstarted sample is pending -- never missing at partial time.
    assert bundle["pending"]["sample_id"].tolist() == ["s2"]
    assert len(bundle["missing_confirmed"]) == 0


def test_pilot_partial_with_no_completions_keeps_everything_pending(
    tmp_path: Path,
) -> None:
    project_root, output_root, config = _prepare(
        tmp_path, completed_samples=[]
    )

    _plan, bundle = pipeline._pilot_partial_bundle(
        project_root, output_root, config
    )

    assert bundle["completed_total"] == 0
    assert len(bundle["sample_manifest"]) == 0
    assert sorted(bundle["pending"]["sample_id"]) == ["s1", "s2"]
    assert len(bundle["missing_confirmed"]) == 0
    assert len(bundle["rejected"]) == 0
