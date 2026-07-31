from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS = ROOT / "sewerrtc" / "state" / "gat_robustness.py"
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"


def test_sr0p15_robustness_outputs_all_required_artifacts() -> None:
    text = ROBUSTNESS.read_text(encoding="utf-8")
    for name in [
        "gat_sr0p15_validation_dataset_manifest.json",
        "gat_sr0p15_validation_sample_inventory.csv",
        "gat_sr0p15_validation_event_support.csv",
        "gat_sr0p15_validation_provenance_audit.csv",
        "gat_sr0p15_validation_leakage_audit.csv",
        "gat_sr0p15_rainfall_near_duplicate_audit.csv",
        "gat_sr0p15_split_membership_audit.csv",
        "gat_sr0p15_temporal_dependence_audit.csv",
        "gat_sr0p15_priority_leaveout_audit.csv",
        "gat_sr0p15_sentinel_leaveout_audit.csv",
        "gat_sr0p15_highwater_phase_audit.csv",
        "gat_sr0p15_sensor_failure_contract.json",
        "gat_sr0p15_sensor_failure_completion_matrix.csv",
        "gat_sr0p15_sensor_failure_audit.csv",
        "gat_sr0p15_sensor_failure_summary.csv",
        "gat_sr0p15_latency_contract.json",
        "gat_sr0p15_latency_repeatability_audit.csv",
        "gat_sr0p15_latency_summary.json",
        "gat_sr0p15_robustness_gate.json",
    ]:
        assert name in text


def test_robustness_gate_requires_priority_sentinel_highwater_sensor_failure_and_latency() -> None:
    text = ROBUSTNESS.read_text(encoding="utf-8")
    for token in [
        "priority_leaveout_complete",
        "sentinel_leaveout_complete",
        "highwater_phase_complete",
        "sensor_failure_execution_complete",
        "latency_measurement_complete",
        "no_training_event_leakage",
    ]:
        assert token in text
    assert '"round0_unlock_allowed": False' in text


def test_robustness_stage_depends_on_primary_lock() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert '"RunGATRobustnessAudit"' in text
    assert 'Assert-UpstreamCompletion -Stage "RunGATRobustnessAudit" -UpstreamStage "SelectPrimaryGAT"' in text
    assert "scripts\\143_run_sr0p15_robustness_audit.py" in text


def test_robustness_separates_max_samples_from_batch_size_and_has_memory_plan() -> None:
    text = ROBUSTNESS.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "143_run_sr0p15_robustness_audit.py").read_text(encoding="utf-8")
    assert "gat_robustness_memory_plan.json" in text
    assert "_effective_batch_size" in text
    assert "batch_size=effective_batch" in text
    assert "out_of_memory" in text
    assert "--batch-size" in script
    assert "default=8" in script
    assert "--max-memory-gb" in script
    assert "[int]$BatchSize = 8" in runner
    assert "--batch-size" in runner
    assert "--max-memory-gb" in runner


def test_robustness_uses_inference_mode_and_does_not_default_to_full_sample_batch() -> None:
    text = ROBUSTNESS.read_text(encoding="utf-8")
    assert "with torch.inference_mode()" in text
    assert "for start in range(0, len(state), max(1, int(batch_size)))" in text
    assert "model(x, m, r, ns, ei)" in text


def test_robustness_gate_uses_four_state_checks() -> None:
    text = ROBUSTNESS.read_text(encoding="utf-8")
    assert 'status not in {"pass", "fail", "incomplete", "not_applicable"}' in text
    assert "passed_checks" in text
    assert "failed_checks" in text
    assert "incomplete_checks" in text
    assert "allowed_to_enter_prompt3a" in text
