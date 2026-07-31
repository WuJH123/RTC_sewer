from __future__ import annotations

import pytest

from sewerrtc.data.coverage_contract import classify_coverage, is_noop, validate_candidate_manifest_row, validate_coverage_cell
from sewerrtc.data.forecast_contract import load_forecast_contract, validate_forecast_record


def test_coverage_cell_requires_decision_relevance() -> None:
    cell = {
        "event_id": "E1",
        "storm_family": "T10_center",
        "split": "design",
        "checkpoint_id": "c001",
        "phase": "rising",
        "state_risk_cluster": "pfv_active",
        "anchor_type": "internal",
        "facility_or_hydraulic_group": "core26",
        "direction": "restrict",
        "magnitude": "0.10",
        "duration_steps": 3,
        "concurrency": "group",
        "interaction_type": "legacy_group",
        "unique_event_support": 1,
        "feasibility_status": "insufficient_support",
        "outcome_class": "unknown",
        "decision_relevance": "candidate_support",
    }
    assert validate_coverage_cell(cell) == []
    del cell["decision_relevance"]
    assert "missing:decision_relevance" in validate_coverage_cell(cell)


def test_candidate_manifest_schema_includes_action_references() -> None:
    row = {
        "case_id": "case_001",
        "event_id": "E1",
        "storm_family": "T10_center",
        "split": "design",
        "checkpoint_id": "c001",
        "phase": "peak",
        "forecast_scenario_id": "operational_nominal",
        "state_clone_hash": "abc",
        "coverage_cell_id": "cell_001",
        "anchor_type": "internal",
        "hydraulic_group_id": "core26",
        "facility_ids": "ADD424.4",
        "direction": "restrict",
        "magnitude": "0.10",
        "duration_steps": 3,
        "concurrency": "single",
        "interaction_type": "single_facility",
        "override_count": 1,
        "binary_switch_count": 0,
        "requested_action_ref": "requested.npy",
        "projected_action_ref": "projected.npy",
        "expected_actual_action_ref": "actual.npy",
        "dwell_precheck": "pass",
        "interlock_precheck": "pass",
        "rate_limit_precheck": "pass",
        "support_status": "in_support",
        "ood_status": "accepted",
        "sampling_reason": "fill_coverage_gap",
        "branch_definitions": "candidate-vs-internal",
        "continuation_policy_id": "first30_fixed_anchor",
        "tail_policy_id": "tail_until_recovery_or_timeout",
        "pre_run_status": "scheduled",
    }
    assert validate_candidate_manifest_row(row) == []
    del row["expected_actual_action_ref"]
    assert "missing:expected_actual_action_ref" in validate_candidate_manifest_row(row)


def test_partial_coverage_uses_insufficient_support_and_noop_uses_linf() -> None:
    rows = classify_coverage({"per_facility": 2}, {"per_facility": 60})
    assert rows[0]["status"] == "insufficient_support"
    assert is_noop(9.0e-7)
    assert not is_noop(2.0e-6)


def test_forecast_contract_rejects_truth_available_to_controller() -> None:
    contract = load_forecast_contract("docs/contracts/forecast_contract.json")
    record = {
        "forecast_issue_time": "2026-01-01T00:00:00+08:00",
        "forecast_valid_time": "2026-01-01T00:10:00+08:00",
        "forecast_source": "operational_forecast",
        "forecast_version": "v3",
        "forecast_horizon": 120,
        "forecast_scenario_id": "nominal",
        "truth_available_to_controller": False,
        "timezone": "Asia/Shanghai",
        "rainfall_units": "mm_per_hour",
        "spatial_mode": "single_gage_design_storm",
        "update_interval_min": 10,
        "maximum_forecast_age_min": 10,
        "minimum_required_horizon_min": 120,
        "insufficient_horizon_handling": "fail_or_safe_fallback_only",
        "missing_forecast_fallback": "selected_safe_fallback_without_learned_candidate",
        "operational_forecast_generator": "project6_design_storm_operational_surrogate",
        "operational_forecast_version": "v3",
    }
    assert validate_forecast_record(record, contract) == []
    record["truth_available_to_controller"] = True
    assert "truth_available_to_controller_must_be_false" in validate_forecast_record(record, contract)
