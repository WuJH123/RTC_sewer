from __future__ import annotations

import numpy as np
import pandas as pd


def test_v8_case_budget_allocates_training_calibration_and_locked_validation():
    from sewerrtc.experiments.tier2_residual_v8 import allocate_v8_case_budget

    budget = allocate_v8_case_budget(
        total_cases=1100,
        train_boundary_cases=320,
        calibration_cases=180,
        validation_cases=300,
    )

    assert budget["fit_deployment"] == 300
    assert budget["fit_boundary"] == 320
    assert budget["calibration_boundary"] == 180
    assert budget["locked_validation_boundary"] == 300
    assert sum(budget.values()) == 1100


def test_v8_event_roles_are_fresh_and_cover_all_requested_roles():
    from sewerrtc.experiments.tier2_residual_v8 import select_v8_event_roles

    rainfall = pd.DataFrame(
        [
            {
                "event_id": f"T{return_period}_D{duration}_{pattern}",
                "rain_id": f"T{return_period}",
                "duration_min": duration,
                "pattern": pattern,
            }
            for return_period in (5, 10, 20, 30, 50, 75, 100)
            for duration in (75, 150, 240, 300)
            for pattern in ("chicago_early", "chicago_center", "chicago_late", "block", "double_peak")
        ]
    )

    roles = select_v8_event_roles(
        rainfall,
        excluded_events={"T5_D75_chicago_early", "T100_D300_double_peak"},
        fit_events=14,
        calibration_events=6,
        validation_events=8,
        seed=20260715,
    )

    assert roles.groupby("role")["event_id"].nunique().to_dict() == {
        "calibration": 6,
        "fit": 14,
        "locked_validation": 8,
    }
    assert set(roles["event_id"]).isdisjoint({"T5_D75_chicago_early", "T100_D300_double_peak"})
    assert not roles["event_id"].duplicated().any()
    assert {"T5", "T10", "T20", "T30", "T50", "T75", "T100"}.issubset(set(roles["rain_id"].astype(str)))


def test_v8_boundary_spec_library_contains_online_and_offline_roles():
    from sewerrtc.experiments.tier2_residual_v8 import build_v8_boundary_specifications

    specs = build_v8_boundary_specifications(
        phase="recession",
        action_ids=[
            "ADD424.1", "ADD424.2", "ADD424.3", "cc006.1", "dwxh.2", "Zhongyi-2.2",
            "RTC_IN_02", "RTC_OUT_02", "RTC_OUT_01", "RTC_OUT_03", "ADD301.2", "ADD301.3",
        ],
    )

    roles = {spec["intended_evidence_role"] for spec in specs}
    assert {"deployment_boundary", "offline_safety_rejection_only"}.issubset(roles)
    assert any(spec.get("online_candidate_eligible") is True for spec in specs)
    assert any(spec.get("online_candidate_eligible") is False for spec in specs)
    assert all(spec["sequence_semantics"] == "relative_to_same_state_no_control_reference" for spec in specs)


def test_v8_preflight_summary_requires_locked_validation_unsafe_capacity():
    from sewerrtc.experiments.tier2_residual_v8 import summarize_v8_manifest_preflight

    manifest = pd.DataFrame(
        [
            {
                "branch": "B",
                "event_id": f"E{index % 8}",
                "event_role": "locked_validation",
                "split": "validation",
                "intended_evidence_role": "offline_safety_rejection_only",
                "is_noop": False,
                "materialized_candidate_action_sequence": np.zeros((6, 36)).tolist(),
            }
            for index in range(34)
        ]
        + [
            {
                "branch": "B",
                "event_id": f"F{index % 14}",
                "event_role": "fit",
                "split": "train",
                "intended_evidence_role": "deployment_boundary",
                "is_noop": False,
                "materialized_candidate_action_sequence": np.ones((6, 36)).tolist(),
            }
            for index in range(250)
        ]
    )

    summary = summarize_v8_manifest_preflight(
        manifest,
        target_cases=284,
        min_locked_validation_cases=30,
        min_locked_validation_events=6,
    )

    assert summary["passed"] is True
    assert summary["locked_validation_cases"] == 34
    assert summary["locked_validation_event_coverage"] == 8
    assert summary["planned_case_count"] == 284


def test_v8_preflight_rejects_validation_under_support():
    from sewerrtc.experiments.tier2_residual_v8 import summarize_v8_manifest_preflight

    manifest = pd.DataFrame(
        [
            {
                "branch": "B",
                "event_id": "E1",
                "event_role": "locked_validation",
                "split": "validation",
                "intended_evidence_role": "offline_safety_rejection_only",
                "is_noop": False,
                "materialized_candidate_action_sequence": np.zeros((6, 36)).tolist(),
            }
            for _ in range(8)
        ]
    )

    summary = summarize_v8_manifest_preflight(
        manifest,
        target_cases=8,
        min_locked_validation_cases=30,
        min_locked_validation_events=6,
    )

    assert summary["passed"] is False
    assert summary["checks"]["locked_validation_case_support"] is False
    assert summary["checks"]["locked_validation_event_support"] is False
