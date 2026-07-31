from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
IMPORT_SCRIPT = ROOT / "scripts" / "153_import_prompt2_artifacts.py"
IMPORT_MODULE = ROOT / "sewerrtc" / "contracts" / "prompt3a.py"


def test_prompt3a_import_stage_and_contract_outputs_are_wired() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "[switch]$ImportPrompt2Artifacts" in text
    assert "scripts\\153_import_prompt2_artifacts.py" in text
    assert "project6_prompt2_import_contract.json" in text
    assert "prompt2_import_manifest.json" in text
    assert "prompt3a_entry_gate.json" in text


def test_prompt2_import_requires_readiness_and_independent_gate() -> None:
    text = IMPORT_MODULE.read_text(encoding="utf-8")
    for token in [
        "project6_prompt2_gat_readiness_gate.json",
        "allowed_to_enter_prompt3a",
        "gat_sr0p15_independent_robustness_gate.json",
        "augmented_state_shape_audit.json",
        "augmented_state_causality_audit.csv",
        "round0_unlock_allowed",
    ]:
        assert token in text


def test_import_script_has_cli() -> None:
    text = IMPORT_SCRIPT.read_text(encoding="utf-8")
    assert "argparse.ArgumentParser" in text
    assert "--config" in text

