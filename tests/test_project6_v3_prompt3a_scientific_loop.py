from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_INPUT = ROOT / "sewerrtc" / "state" / "state_input_manifest.py"
RUNTIME_STATE = ROOT / "sewerrtc" / "state" / "runtime_state_features.py"
BASELINE = ROOT / "scripts" / "160_generate_baseline_trajectories.py"
CHECKPOINT = ROOT / "scripts" / "161_build_checkpoint_catalog_v3.py"
CLONE = ROOT / "sewerrtc" / "state" / "state_clone_equivalence.py"
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
TRUTH = ROOT / "sewerrtc" / "status" / "current_truth.py"


def test_state_input_uses_trajectory_event_policy_key_and_processes_skipped_existing_details() -> None:
    text = STATE_INPUT.read_text(encoding="utf-8")
    assert "trajectory_key" in text
    assert "trajectory_id" in text
    assert "policy_id" in text
    assert 'row.get("status") == "skipped_existing" and row.get("detail_file", "").strip()' in text


def test_runtime_state_materializes_feature_indexes_for_actual_tensor_dimensions() -> None:
    text = RUNTIME_STATE.read_text(encoding="utf-8")
    for token in [
        "node_feature_index.json",
        "facility_feature_index.json",
        "storage_feature_index.json",
        "feature_materialization_audit.csv",
        "node_feature_name_count_mismatch",
        "facility_feature_name_count_mismatch",
        "storage_feature_name_count_mismatch",
    ]:
        assert token in text


def test_baseline_generation_uses_five_min_visible_state_and_writes_recovery_and_checkpoints() -> None:
    text = BASELINE.read_text(encoding="utf-8")
    assert "control_step_sec=300" in text
    assert "visible_state_step_sec" in text
    assert "rtc_decision_interval_sec" in text
    assert "baseline_recovery_audit.csv" in text
    assert "baseline_checkpoint_audit.csv" in text


def test_checkpoint_catalog_requires_real_hotstart_and_controller_memory() -> None:
    text = CHECKPOINT.read_text(encoding="utf-8")
    assert "baseline_checkpoint_audit.csv" in text
    assert "hotstart_path" in text
    assert "controller_memory_path" in text
    assert "eligible_for_state_clone" in text
    assert "missing_hotstart_or_controller_memory_or_temporal_support" in text


def test_state_clone_gate_cannot_pass_when_equivalence_is_not_run() -> None:
    text = CLONE.read_text(encoding="utf-8")
    assert '"hotstart_equivalence_status": "not_run"' in text
    assert '"controller_memory_restore_status": "not_run"' in text
    assert 'row.get("status") == "pass"' in text
    assert "runtime_executed and all_pass and memory_pass and timeline_pass and noise_measured and full_support" in text
    assert '"formal_same_state_unlock_allowed": gate_pass' in text


def test_runner_exposes_prompt3a_small_loop_stages() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    for stage in [
        "PrepareStateCloneCheckpoints",
        "EstimateStateCloneNumericalNoise",
        "RunStateCloneEquivalence",
        "EvaluateStateCloneGate",
        "EvaluatePrompt3ARuntimeGate",
    ]:
        assert stage in text


def test_runtime_gate_reads_recovery_and_state_clone_gate() -> None:
    text = TRUTH.read_text(encoding="utf-8")
    assert "baseline_recovery_audit.csv" in text
    assert "recovery_contract_complete" in text
    assert "state_clone_gate.json" in text
    assert "controller_memory_restore_status" in text
