from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "sewerrtc" / "state" / "prompt2_completion_gate.py"
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"


def test_prompt2_gate_has_required_blocked_states() -> None:
    text = GATE.read_text(encoding="utf-8")
    for status in [
        "blocked_pending_manual_selection_lock",
        "blocked_pending_sr0p15_robustness",
        "blocked_pending_runtime_state_validation",
    ]:
        assert status in text
    assert '"round0_unlock_allowed": False' in text


def test_prompt2_gate_requires_robustness_and_runtime_state_before_pass() -> None:
    text = GATE.read_text(encoding="utf-8")
    for token in [
        "sr0p15_robustness_gate_passed",
        "runtime_state_shape_causality_missingness_passed",
        "priority_leaveout_completed",
        "sentinel_leaveout_completed",
        "sensor_failure_completed",
        "validation_no_leakage_or_independent",
    ]:
        assert token in text


def test_runner_exposes_evaluate_prompt2_completion_without_formal_unlock() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "[switch]$EvaluatePrompt2Completion" in text
    assert "scripts\\145_evaluate_prompt2_completion.py" in text
    assert "FormalBlind" in text
