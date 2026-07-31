from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "sewerrtc" / "state" / "state_input_manifest.py"
SCRIPT = ROOT / "scripts" / "146_build_state_input_manifest.py"
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"


def test_state_input_manifest_has_two_explicit_source_modes() -> None:
    text = MODULE.read_text(encoding="utf-8")
    assert "project4_gat_validation" in text
    assert "project4_diagnostic_contaminated" in text
    assert "gat_independent_holdout" in text
    assert "project6_retrofit_baseline" in text
    assert "unsupported_source_mode" in text


def test_project4_manifest_cannot_claim_full_project6_augmented_state() -> None:
    text = MODULE.read_text(encoding="utf-8")
    assert '"gat_node_state_validation_eligible": "true"' in text
    assert '"full_project6_augmented_state_eligible": "false"' in text
    assert "project4_cache_lacks_project6_facility_storage_pump_ttl_fallback_fields" in text


def test_independent_holdout_manifest_requires_explicit_validation_manifest() -> None:
    text = MODULE.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "gat_independent_holdout_manifest_required" in text
    assert "--validation-manifest" in script
    assert "--validation-manifest" in runner


def test_project6_retrofit_mode_requires_trajectory_root_and_blocks_without_it() -> None:
    text = MODULE.read_text(encoding="utf-8")
    assert "project6_retrofit_trajectory_root_missing" in text
    assert "baseline_trajectory_manifest_missing" in text
    assert "project6_full_baseline_state_ready" in text
    assert "full_project6_augmented_state_eligible\": \"true\"" in text


def test_runner_exposes_build_state_input_manifest_stage() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "[switch]$BuildStateInputManifest" in text
    assert "Run-BuildStateInputManifest" in text
    assert "scripts\\146_build_state_input_manifest.py" in text
    assert "[string]$SourceMode" in text
    assert "[string]$TrajectoryRoot" in text


def test_manifest_script_has_required_cli() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--source-mode" in text
    assert "--trajectory-root" in text
    assert "--max-samples" in text
