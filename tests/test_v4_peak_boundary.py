import json
from pathlib import Path

import numpy as np
import pandas as pd

from sewerrtc.v4.peak_boundary import (
    audit_peak_boundary,
    build_peak_partial_bundle,
    peak_constraint_binding_audit,
)
from sewerrtc.v4.partial_audit import HARD_AUTHENTICITY_COLUMNS


FACILITIES = ["ADD301.2", "ADD301.3", "add350.1"]
PRIORITY = ["NODEP"]
CHECKPOINT = 60.0
MARGIN = {"pfv_m3": 0.0, "tfv_m3": 0.0, "peak_m3s": 0.0}
DEAD_ZONE = {"pfv_m3": 0.0, "tfv_m3": 0.0, "peak_m3s": 0.0}


def _write_branch_detail(
    directory: Path, post_flood_rate: float, actual_value: float
) -> Path:
    # Pre-checkpoint hydraulics are identical across branches (same-state);
    # branches diverge only after the checkpoint.
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


def _completion_row(
    root: Path, sample_id: str, branch: str, post_flood: float, actual: float
) -> dict:
    case_id = f"{sample_id}__{branch}"
    detail_path = _write_branch_detail(root / case_id, post_flood, actual)
    return {
        "case_id": case_id,
        "sample_id": sample_id,
        "branch": branch,
        "status": "pass",
        "detail_path": str(detail_path),
        "checkpoint_min": CHECKPOINT,
        "event_id": "e0",
        "checkpoint_id": "c0",
        "k_target": 2,
        "family": "synchronized_pump_starts",
        "opportunity_class": "peak",
        "phase": "peak",
        "rainfall_sha256": "rainsha",
        "binary_semantics_ok": True,
        "rate_limit_ok": True,
        "dwell_ok": True,
        "interlock_ok": True,
        "projected_schedule_json": json.dumps([[1.0, 1.0, 1.0]]),
        "anchor_schedule_json": json.dumps([[0.0, 0.0, 0.0]]),
        "input_sha": "insha",
        "runner_kwargs": json.dumps({"inp_path": "x.inp"}),
        "result": {
            "hotstart_used": False,
            "use_hotstart_call_count": 0,
            "save_hotstart_call_count": 0,
        },
    }


def _four_branch_sample(root: Path, sample_id: str) -> list[dict]:
    return [
        _completion_row(root, sample_id, "candidate", 0.02, 1.0),
        _completion_row(root, sample_id, "no_control", 0.05, 0.0),
        _completion_row(root, sample_id, "dynamic_internal_rules", 0.03, 0.0),
        _completion_row(root, sample_id, "hold_previous", 0.01, 0.0),
    ]


def test_partial_bundle_reduces_complete_sample_with_hard_authenticity(
    tmp_path: Path,
) -> None:
    rows = _four_branch_sample(tmp_path, "s1")
    completions = pd.DataFrame(rows)
    plan = completions[["sample_id", "case_id", "branch"]].copy()

    bundle = build_peak_partial_bundle(
        plan,
        completions,
        priority_nodes=PRIORITY,
        facility_ids=FACILITIES,
        scientific_margin=MARGIN,
        dead_zone=DEAD_ZONE,
    )

    assert bundle["completed_total"] == 1
    assert len(bundle["pending"]) == 0
    assert len(bundle["rejected"]) == 0
    manifest = bundle["sample_manifest"]
    assert len(manifest) == 1
    row = manifest.iloc[0]
    for column in HARD_AUTHENTICITY_COLUMNS:
        assert bool(row[column]) is True, column
    assert bool(row["is_noop"]) is False
    assert row["actual_action_distance"] > 0.0
    assert bool(row["output_isolated"]) is True
    assert bool(row["state_hash_match"]) is True
    assert bool(row["readback_ok"]) is True
    assert row["branch_role"] == "candidate"


def test_partial_bundle_marks_sample_missing_a_branch_as_pending(
    tmp_path: Path,
) -> None:
    rows = _four_branch_sample(tmp_path, "s1")
    completions = pd.DataFrame(rows[:3])  # drop hold_previous
    plan = pd.DataFrame(rows)[["sample_id", "case_id", "branch"]]

    bundle = build_peak_partial_bundle(
        plan,
        completions,
        priority_nodes=PRIORITY,
        facility_ids=FACILITIES,
        scientific_margin=MARGIN,
        dead_zone=DEAD_ZONE,
    )

    assert bundle["completed_total"] == 0
    assert len(bundle["sample_manifest"]) == 0
    assert bundle["pending"]["sample_id"].tolist() == ["s1"]


def test_partial_bundle_rejects_noop_actual_schedule(tmp_path: Path) -> None:
    # Candidate whose actual schedule equals the anchor is a no-op and must not
    # be accepted as an informative sample.
    rows = [
        _completion_row(tmp_path, "s1", "candidate", 0.02, 0.0),
        _completion_row(tmp_path, "s1", "no_control", 0.05, 0.0),
        _completion_row(tmp_path, "s1", "dynamic_internal_rules", 0.03, 0.0),
        _completion_row(tmp_path, "s1", "hold_previous", 0.01, 0.0),
    ]
    completions = pd.DataFrame(rows)
    plan = completions[["sample_id", "case_id", "branch"]].copy()

    bundle = build_peak_partial_bundle(
        plan,
        completions,
        priority_nodes=PRIORITY,
        facility_ids=FACILITIES,
        scientific_margin=MARGIN,
        dead_zone=DEAD_ZONE,
    )

    assert bundle["completed_total"] == 1
    assert len(bundle["sample_manifest"]) == 0
    assert len(bundle["rejected"]) == 1
    assert (
        bundle["rejected"].iloc[0]["rejection_reason"] == "no_op_not_accepted"
    )


def test_peak_partial_bundle_handler_core_resolves_run_plan(
    tmp_path: Path,
) -> None:
    # Guards the handler wiring: the Peak partial builders must resolve the
    # RunPeakBoundary plan/runs directly (via _run_stage_sources), never through
    # the partial-name mapping, which previously raised KeyError('RunPeakBoundary').
    from sewerrtc.v4 import pipeline

    project_root = tmp_path / "project"
    output_root = tmp_path / "out"
    run_root = output_root / "peak_boundary" / "runs"
    run_root.mkdir(parents=True)
    project_root.mkdir(parents=True)
    (project_root / "prio.txt").write_text("NODEP\n", encoding="utf-8")
    (project_root / "ids.txt").write_text(
        "\n".join(FACILITIES) + "\n", encoding="utf-8"
    )

    rows = _four_branch_sample(run_root, "s1")
    for row in rows:
        case_dir = run_root / row["case_id"]
        (case_dir / "completion.json").write_text(
            json.dumps(row), encoding="utf-8"
        )
    plan = pd.DataFrame(rows)[["sample_id", "case_id", "branch"]]
    plan_path = output_root / "peak_boundary" / "peak_boundary_plan.csv"
    plan.to_csv(plan_path, index=False)

    config = {
        "project": {"priority_nodes": "prio.txt", "canonical_ids": "ids.txt"},
        "thresholds": {"scientific_margin": MARGIN, "dead_zone": DEAD_ZONE},
    }

    _plan, bundle = pipeline._peak_partial_bundle(
        project_root, output_root, config
    )

    assert bundle["completed_total"] == 1
    assert len(bundle["sample_manifest"]) == 1
    assert len(bundle["pending"]) == 0


def test_peak_boundary_requires_cross_event_and_pfv_safe_hard_negatives() -> None:
    rows = []
    for event_index in range(3):
        for checkpoint_index in range(2):
            for sample_index in range(5):
                rows.append(
                    {
                        "event_id": f"e{event_index}",
                        "checkpoint_id": f"c{checkpoint_index}",
                        "actual_schedule_sha256": f"{event_index}-{checkpoint_index}-{sample_index}",
                        "peak_noninferior": False,
                        "pfv_safe": sample_index < 2,
                        "family": "synchronized_pump_starts"
                        if sample_index % 2
                        else "simultaneous_orifice_opening",
                    }
                )
    audit = audit_peak_boundary(pd.DataFrame(rows))

    assert audit["status"] == "pass"
    assert audit["peak_degraded"] == 30
    assert audit["pfv_safe_peak_hard_negative"] >= 10


def test_absent_peak_degradation_produces_binding_audit_not_constraint_removal() -> None:
    samples = pd.DataFrame(
        {
            "event_id": ["e"],
            "checkpoint_id": ["c"],
            "actual_schedule_sha256": ["a"],
            "peak_noninferior": [True],
            "pfv_safe": [True],
            "family": ["synchronized_pump_starts"],
        }
    )
    audit = peak_constraint_binding_audit(samples)

    assert audit["peak_degraded"] == 0
    assert audit["remove_peak_constraint"] is False
