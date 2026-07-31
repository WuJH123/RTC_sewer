from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_wuhan_network_contains_no_dwf_section_or_patterns() -> None:
    text = (
        PROJECT_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"
    ).read_text(encoding="utf-8", errors="ignore")

    assert "[DWF]" not in text.upper()
    assert "Hourly_HXH" not in text
    assert "Hourly_JCH" not in text
    assert "A-380L_Day" not in text


def test_gate5r_points_to_verified_no_dwf_network_and_preserves_original() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "wuhan_project6_v4_gate5r.yaml").read_text(
            encoding="utf-8"
        )
    )
    contract = json.loads(
        (
            PROJECT_ROOT
            / "docs"
            / "contracts"
            / "PROJECT6_V4_NETWORK_NO_BASE_INFLOW_PROVENANCE.json"
        ).read_text(encoding="utf-8")
    )

    active = PROJECT_ROOT / config["project"]["network"]
    original = Path(contract["preserved_original"])
    assert active.resolve() == Path(contract["active_network"]).resolve()
    assert hashlib.sha256(active.read_bytes()).hexdigest() == contract[
        "active_network_sha256"
    ]
    assert original.exists()
    assert hashlib.sha256(original.read_bytes()).hexdigest() == contract[
        "preserved_original_sha256"
    ]


def test_active_project_has_no_dwf_processing_modules() -> None:
    removed_paths = [
        "sewerrtc/simulation/dry_weather_runner.py",
        "sewerrtc/hydraulics/baseline_relative_recovery.py",
        "scripts/235_static_inp_recovery_audit.py",
        "scripts/236_audit_v4_dwf_units.py",
        "scripts/236_run_dry_weather_baseline.py",
        "scripts/237b_compute_recession_evidence.py",
        "scripts/237_run_or_ingest_v4_three_day_recession.py",
        "scripts/238_finalize_v4_gate3_dual_scope.py",
        "scripts/238_recovery_prescreen_runner.py",
        "scripts/239_inp_recovery_inventory_v2.py",
        "scripts/240_run_v4_dry_weather_baseline.py",
        "scripts/241_event_prescreen_v2.py",
        "scripts/241_finalize_v4_gate35_evidence.py",
        "scripts/_check_spinup.py",
        "scripts/_check_spinup2.py",
        "scripts/_final_report.py",
    ]

    assert not [
        relative
        for relative in removed_paths
        if (PROJECT_ROOT / relative).exists()
    ]
