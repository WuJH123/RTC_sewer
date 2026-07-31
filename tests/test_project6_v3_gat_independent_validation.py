from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ELIGIBILITY = ROOT / "sewerrtc" / "state" / "gat_holdout_eligibility.py"
CATALOG = ROOT / "sewerrtc" / "state" / "gat_independent_validation.py"
TRAJECTORY = ROOT / "sewerrtc" / "state" / "gat_holdout_trajectory.py"
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
ROBUSTNESS_SCRIPT = ROOT / "scripts" / "143_run_sr0p15_robustness_audit.py"
READINESS = ROOT / "sewerrtc" / "state" / "prompt2_gat_readiness.py"
GEN_SCRIPT = ROOT / "scripts" / "152_generate_gat_independent_holdout_trajectories.py"


def test_holdout_eligibility_excludes_exact_rainfall_trajectory_and_family_overlap() -> None:
    text = ELIGIBILITY.read_text(encoding="utf-8")
    for token in [
        "exact_event_overlap",
        "rainfall_hash_overlap",
        "trajectory_hash_overlap",
        "storm_family_overlap",
        "rainfall_near_duplicate",
        "intensity_scale",
        "time_shift",
        "sr0p15_sensor_available",
        "requires_new_trajectory",
        "independent_rainfall_without_complete_truth",
    ]:
        assert token in text


def test_independent_catalog_preserves_contaminated_result_and_writes_gap_plan() -> None:
    text = CATALOG.read_text(encoding="utf-8")
    for token in [
        "failed_due_to_exact_training_event_leakage",
        "pending_independent_holdout_validation",
        "pending_new_independent_holdout",
        "gat_independent_validation_gap_report.json",
        "gat_independent_trajectory_plan.csv",
        "cache_path_missing",
        "gat_contaminated_event_manifest.csv",
        "gat_model_selection_event_manifest.csv",
    ]:
        assert token in text


def test_runner_requires_locked_independent_manifest_for_formal_robustness() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "[switch]$BuildGATIndependentValidationCatalog" in text
    assert "[switch]$GenerateGATIndependentHoldoutTrajectories" in text
    assert "[switch]$LockGATIndependentValidationManifest" in text
    assert "[string]$ValidationManifest" in text
    assert "independent_validation_manifest_required" in text
    assert "gat_independent_validation_lock_missing" in text
    assert "gat\\independent_holdout\\sr0p15" in text


def test_holdout_generation_builds_sr0p15_cache_and_manifest_from_plan() -> None:
    text = TRAJECTORY.read_text(encoding="utf-8")
    script = GEN_SCRIPT.read_text(encoding="utf-8")
    for token in [
        "gat_independent_holdout_sr0p15_cache.npz",
        "gat_independent_validation_manifest.csv",
        "run_swmm_trajectory",
        "mutate_inp_for_event",
        "full_node_truth_available",
        "sr0p15_sensor_available",
        "has_60min_history",
        "event_ids",
        "sources",
    ]:
        assert token in text
    assert "--max-events" in script
    assert "--policies" in script
    assert "--tail-min" in script
    assert "--time-stride" in script


def test_runner_exposes_holdout_generation_stage_between_catalog_and_lock() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "Run-GenerateGATIndependentHoldoutTrajectories" in text
    assert "scripts\\152_generate_gat_independent_holdout_trajectories.py" in text
    assert "gat_independent_trajectory_plan.csv" in text
    assert "gat_independent_holdout_sr0p15_cache.npz" in text
    assert "[int]$MaxHoldoutEvents" in text
    assert "[string]$HoldoutPolicies" in text


def test_robustness_audit_no_longer_defaults_to_contaminated_cache_for_formal_path() -> None:
    text = ROBUSTNESS_SCRIPT.read_text(encoding="utf-8")
    assert "--validation-manifest" in text
    assert "No usable NPZ validation cache was found in the independent manifest" in text
    assert "gat_sr0p15_independent_robustness_gate.json" in text


def test_prompt2_readiness_uses_independent_gate_not_diagnostic_gate() -> None:
    text = READINESS.read_text(encoding="utf-8")
    assert "gat_sr0p15_independent_robustness_gate.json" in text
    assert "independent_robustness_gate_used" in text
    assert "diagnostic_contaminated_gate_failed_or_ignored" in text
    assert "gat_independent_validation_lock.json" in text
