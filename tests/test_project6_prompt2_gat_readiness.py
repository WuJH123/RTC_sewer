from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
READINESS = ROOT / "sewerrtc" / "state" / "prompt2_gat_readiness.py"
GATE_SCRIPT = ROOT / "scripts" / "147_evaluate_gat_robustness_gate.py"


def test_runner_has_read_only_gat_gate_and_prompt2_readiness_stages() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "[switch]$EvaluateGATRobustnessGate" in text
    assert "[switch]$EvaluatePrompt2GATReadiness" in text
    assert "scripts\\147_evaluate_gat_robustness_gate.py" in text
    assert "scripts\\148_evaluate_prompt2_gat_readiness.py" in text


def test_gat_gate_script_reads_reports_not_model() -> None:
    text = GATE_SCRIPT.read_text(encoding="utf-8")
    assert "evaluate_gat_robustness_gate" in text
    assert "torch" not in text
    assert "run_sr0p15_robustness_audit" not in text


def test_prompt2_gat_readiness_does_not_require_full_project6_state() -> None:
    text = READINESS.read_text(encoding="utf-8")
    assert '"full_project6_augmented_state_complete": False' in text
    assert "allowed_to_enter_prompt3a" in text
    assert "node_level_7frame_validation_complete" in text
