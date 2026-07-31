from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAIN = ROOT / "sewerrtc" / "data" / "rainfall_asset_index.py"
EVENT = ROOT / "sewerrtc" / "data" / "event_catalog.py"
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"


def test_rainfall_asset_index_scans_only_fixed_allowed_dirs() -> None:
    text = RAIN.read_text(encoding="utf-8")
    for token in [
        "Project4",
        "Project5",
        "Project6",
        "rainfall_asset_inventory.csv",
        "rainfall_asset_resolution_audit.csv",
        "fixed_allowed_rainfall_directories",
    ]:
        assert token in text
    assert "PROJECT_ROOTS" not in text


def test_event_catalog_resolves_rainfall_by_canonical_id_and_keeps_unresolved_out() -> None:
    text = EVENT.read_text(encoding="utf-8")
    for token in [
        "load_selected_rainfall_assets",
        "resolved_exact_basename",
        "rainfall_resolution_status",
        "unresolved_rainfall_events.csv",
        "round0_eligible",
    ]:
        assert token in text


def test_runner_has_rainfall_asset_stage_before_event_catalog() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "[switch]$BuildRainfallAssetIndex" in text
    assert "scripts\\167_build_rainfall_asset_index.py" in text
    assert 'Assert-UpstreamCompletion -Stage "BuildEventCatalog" -UpstreamStage "BuildRainfallAssetIndex"' in text


def test_baseline_planner_does_not_turn_blank_rainfall_path_into_dot() -> None:
    baseline = (ROOT / "sewerrtc" / "simulation" / "baseline_trajectory.py").read_text(encoding="utf-8")
    assert "rainfall_path_missing" in baseline
    assert 'Path(str(row.get("rainfall_path", "")))' not in baseline
