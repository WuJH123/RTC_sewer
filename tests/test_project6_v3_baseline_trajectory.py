from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "sewerrtc" / "simulation" / "baseline_trajectory.py"
WRITER = ROOT / "sewerrtc" / "simulation" / "trajectory_writer.py"
GEN_SCRIPT = ROOT / "scripts" / "160_generate_baseline_trajectories.py"
CONTRACT = ROOT / "docs" / "contracts" / "baseline_trajectory_plan_contract.json"


def test_baseline_plan_has_three_required_policies() -> None:
    text = BASELINE.read_text(encoding="utf-8")
    for token in ["no_control", "internal_rules", "executable_passive"]:
        assert token in text


def test_trajectory_schema_separates_truth_from_controller_visible_state() -> None:
    text = WRITER.read_text(encoding="utf-8")
    assert "truth_fields" in text
    assert "controller_visible_fields" in text
    assert "truth_to_controller_forbidden" in text


def test_generation_stage_calls_real_swmm_runner_not_static_scaffold() -> None:
    text = GEN_SCRIPT.read_text(encoding="utf-8")
    assert "run_swmm_trajectory" in text
    assert "build_event_inp_from_plan" in text
    assert "baseline_swmm_generation_not_run_by_codex" not in text
    assert "trajectory_quality_report.json" in text


def test_baseline_plan_contract_contains_required_columns() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    for token in [
        "plan_schema_version",
        "trajectory_id",
        "canonical_event_id",
        "split",
        "policy_mode",
        "rainfall_path",
        "network_path",
        "event_catalog_sha256",
        "prompt2_import_lock_sha256",
        "native_rule_audit_sha256",
        "fallback_selection_contract_sha256",
    ]:
        assert token in text


def test_baseline_planner_uses_event_catalog_split_manifest_and_hashes() -> None:
    text = BASELINE.read_text(encoding="utf-8")
    for token in [
        "event_split_manifest.csv",
        "event_split_leakage_audit.csv",
        "duplicate_event_policy",
        "missing_three_policies",
        "rainfall_path_missing",
        "gat_independent_holdout",
        "calibration_or_formal_or_holdout",
    ]:
        assert token in text


def test_generate_rejects_old_simplified_plan_schema() -> None:
    text = GEN_SCRIPT.read_text(encoding="utf-8")
    assert "validate_frozen_baseline_plan" in text
    assert "old_or_invalid_baseline_plan_schema" in BASELINE.read_text(encoding="utf-8")
