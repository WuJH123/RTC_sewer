from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS = ROOT / "sewerrtc" / "state" / "gat_robustness.py"
GATE = ROOT / "sewerrtc" / "state" / "gat_robustness_gate.py"


def test_validation_provenance_writes_sample_inventory_and_split_audits() -> None:
    text = ROBUSTNESS.read_text(encoding="utf-8")
    for token in [
        "gat_sr0p15_validation_sample_inventory.csv",
        "source_row",
        "time_index",
        "training_membership",
        "model_selection_membership",
        "gat_sr0p15_split_membership_audit.csv",
        "gat_sr0p15_rainfall_near_duplicate_audit.csv",
    ]:
        assert token in text


def test_missing_event_identity_is_incomplete_not_fail() -> None:
    text = ROBUSTNESS.read_text(encoding="utf-8")
    assert 'provenance_status = "incomplete"' in text
    assert "cache does not expose recoverable event identity/split membership" in text
    assert "independent validation split not proven" in text


def test_gate_evaluator_separates_incomplete_from_fail() -> None:
    text = GATE.read_text(encoding="utf-8")
    assert "return \"fail\"" in text
    assert "return \"incomplete\"" in text
    assert "training-event leakage is not ruled out" in text
