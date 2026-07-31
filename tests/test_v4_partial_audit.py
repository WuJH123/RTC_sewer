from __future__ import annotations

import json

import pandas as pd

from sewerrtc.v4.partial_audit import (
    HARD_AUTHENTICITY_COLUMNS,
    audit_partial_quality,
    build_partial_bundle,
    classify_partial_cases,
    partial_accounting,
)


def make_plan(case_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": case_ids,
            "event_id": [f"e{i % 4}" for i in range(len(case_ids))],
            "checkpoint_id": [f"c{i % 8}" for i in range(len(case_ids))],
        }
    )


def make_completion(case_id: str, index: int = 0, **overrides) -> dict:
    row = {
        "case_id": case_id,
        "status": "pass",
        "branch_role": "candidate",
        "event_id": f"e{index % 4}",
        "checkpoint_id": f"c{index % 8}",
        "checkpoint_role": "responsive",
        "candidate_family": f"fam{index % 3}",
        "K": 2,
        "is_noop": False,
        "requested_schedule_sha256": f"req-{case_id}",
        "actual_schedule_sha256": f"act-{case_id}",
        "actual_action_distance": 1.0 + index,
        "local_response_magnitude": 0.5,
        "delta_pfv_h120_vs_no_control": -1.0 * index,
        "delta_tfv_h120_vs_dynamic_internal": 2.0 * index,
        "delta_peak_h120_vs_dynamic_internal": 0.1 * index,
        "pfv_safe": True,
        "tfv_improved": index % 2 == 0,
        "peak_noninferior": True,
        "neutral": False,
        "joint_noninferior": index % 2 == 0,
        "materially_beneficial": index % 3 == 0,
        "output_isolated": True,
    }
    row.update({column: True for column in HARD_AUTHENTICITY_COLUMNS})
    row.update(overrides)
    return row


def test_partial_never_counts_pending_as_missing() -> None:
    plan = make_plan(["a", "b", "c", "d"])
    completions = pd.DataFrame([make_completion("a")])

    parts = classify_partial_cases(plan, completions)

    assert set(parts["pending"]["case_id"]) == {"b", "c", "d"}
    assert len(parts["missing_confirmed"]) == 0
    assert len(parts["failed"]) == 0


def test_partial_accounting_always_reports_scope_incomplete() -> None:
    plan = make_plan(["a", "b"])
    completions = pd.DataFrame([make_completion("a")])
    bundle = build_partial_bundle(plan, completions)

    accounting = partial_accounting(
        bundle,
        planned_scope_total=len(plan),
        run_uuid="u",
        input_sha="i",
        config_sha="c",
        code_sha="k",
    )

    assert accounting["scope_complete"] is False
    assert accounting["partial_only"] is True
    assert accounting["pending"] == 1
    assert accounting["accepted"] == 1
    assert accounting["missing_confirmed"] == 0


def test_completed_hard_failure_stops_scale_up() -> None:
    plan = make_plan(["a", "b"])
    completions = pd.DataFrame(
        [
            make_completion("a", 0),
            make_completion("b", 1, same_state_ok=False),
        ]
    )
    bundle = build_partial_bundle(plan, completions)
    quality = audit_partial_quality(bundle)

    assert bundle["hard_violation_total"] == 1
    assert quality["status"] == "scientific_fail"
    assert quality["stop_scale_up"] is True


def test_missing_hard_evidence_column_fails_closed() -> None:
    plan = make_plan(["a"])
    row = make_completion("a")
    row.pop("kpi_recompute_ok")
    bundle = build_partial_bundle(plan, pd.DataFrame([row]))

    assert bundle["hard_violation_total"] == 1
    assert len(bundle["sample_manifest"]) == 0


def test_noop_and_actual_duplicates_are_never_accepted() -> None:
    plan = make_plan(["a", "b", "c"])
    completions = pd.DataFrame(
        [
            make_completion("a", 0),
            make_completion("b", 0, actual_schedule_sha256="act-a"),
            make_completion("c", 1, is_noop=True),
        ]
    )
    bundle = build_partial_bundle(plan, completions)

    assert len(bundle["sample_manifest"]) == 1
    assert bundle["actual_duplicates"]["case_id"].tolist() == ["b"]
    reasons = set(bundle["rejected"]["rejection_reason"])
    assert "no_op_not_accepted" in reasons


def test_all_constant_labels_stop_immediately() -> None:
    plan = make_plan(["a", "b", "c"])
    completions = pd.DataFrame(
        [
            make_completion(
                case_id,
                idx,
                delta_pfv_h120_vs_no_control=1.0,
                delta_tfv_h120_vs_dynamic_internal=2.0,
                delta_peak_h120_vs_dynamic_internal=0.5,
            )
            for idx, case_id in enumerate(["a", "b", "c"])
        ]
    )
    bundle = build_partial_bundle(plan, completions)
    quality = audit_partial_quality(bundle)

    assert quality["label_boundaries"]["all_continuous_labels_constant"]
    assert quality["stop_scale_up"] is True


def test_partial_json_payloads_use_native_bool() -> None:
    plan = make_plan(["a", "b"])
    completions = pd.DataFrame([make_completion("a")])
    bundle = build_partial_bundle(plan, completions)
    accounting = partial_accounting(
        bundle,
        planned_scope_total=2,
        run_uuid="u",
        input_sha="i",
        config_sha="c",
        code_sha="k",
    )
    quality = audit_partial_quality(bundle)

    # json.dumps with allow_nan=False rejects numpy types and NaN.
    json.dumps(accounting, allow_nan=False)
    json.dumps(quality, allow_nan=False)
    assert isinstance(accounting["scope_complete"], bool)
    assert isinstance(quality["stop_scale_up"], bool)
    assert isinstance(
        quality["authenticity"]["all_completed_cases_authentic"], bool
    )


def test_partial_gate_pass_is_never_a_full_gate_pass() -> None:
    plan = make_plan(["a", "b"])
    completions = pd.DataFrame([make_completion("a")])
    quality = audit_partial_quality(build_partial_bundle(plan, completions))

    assert quality["partial_only"] is True
    assert quality["full_gate_pass"] is False
