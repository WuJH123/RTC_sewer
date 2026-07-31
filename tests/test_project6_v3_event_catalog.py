from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVENT_MODULE = ROOT / "sewerrtc" / "data" / "event_catalog.py"
SPLIT_MODULE = ROOT / "sewerrtc" / "data" / "split_contract.py"


def test_event_catalog_excludes_gat_holdout_from_round0() -> None:
    text = EVENT_MODULE.read_text(encoding="utf-8")
    assert "gat_independent_holdout" in text
    assert "round0_eligible" in text
    assert "action_effect_fit_eligible" in text
    assert 'event_id not in holdout_events and rainfall_resolution_status != "unresolved"' in text


def test_split_contract_detects_event_and_storm_family_leakage() -> None:
    text = SPLIT_MODULE.read_text(encoding="utf-8")
    assert "audit_split_leakage" in text
    assert "storm_family" in text
    assert "formal_blind" in text
