from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "sewerrtc" / "state" / "runtime_state_features.py"
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"


def test_runtime_state_manifest_requires_explicit_inputs_and_seven_frames() -> None:
    text = STATE.read_text(encoding="utf-8")
    for column in [
        "state_history_path",
        "facility_history_path",
        "frame_count",
        "history_window_min",
        "contains_future_data",
        "missing_flow_encoded_as_zero",
    ]:
        assert column in text
    assert "frame_count != 7" in text
    assert "history_window_less_than_60min" in text
    assert "future_data_present" in text


def test_runtime_state_shapes_and_missing_flow_policy_are_enforced() -> None:
    text = STATE.read_text(encoding="utf-8")
    assert "state_history_shape_not_samples_7_N_F" in text
    assert "facility_history_shape_not_samples_7_36_F" in text
    assert "missing_flow_encoded_as_zero" in text
    assert '"unlocks_round0": False' in text


def test_build_state_features_requires_manifest_and_primary_gat_lock() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "StateInputManifest_required" in text
    assert '$requiresGatGate = ($StateValidationMode -eq "gat_independent_node_only" -or $StateValidationMode -eq "project4_node_only")' in text
    assert 'Assert-UpstreamCompletion -Stage "BuildStateFeatures" -UpstreamStage "SelectPrimaryGAT"' in text
    assert 'Assert-UpstreamCompletion -Stage "BuildStateFeatures" -UpstreamStage "EvaluateGATRobustnessGate"' in text
    assert "sr0p15_robustness_gate_not_pass" in text
    assert "gat_sr0p15_independent_robustness_gate.json" in text
    assert "StateInputManifest_placeholder_path" in text
    assert "scripts\\144_build_runtime_state_features.py" in text


def test_runtime_state_manifest_distinguishes_node_validation_from_full_project6_state() -> None:
    text = STATE.read_text(encoding="utf-8")
    assert "gat_node_state_validation_eligible" in text
    assert "full_project6_augmented_state_eligible" in text
    assert "node_state_validation_only_not_full_project6_augmented_state" in text
    assert "full_project6_augmented_state_not_eligible" in text
    assert "gat_independent_node_only" in text
    assert "diagnostic_contaminated_node_validation_complete" in text
