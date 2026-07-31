from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "sewerrtc" / "data" / "coverage_database.py"
SCRIPT = ROOT / "scripts" / "162_build_coverage_contract_v3.py"


def test_coverage_cell_schema_contains_prompt3a_dimensions() -> None:
    text = COVERAGE.read_text(encoding="utf-8")
    for token in ["checkpoint_id", "phase", "state_cluster", "anchor_type", "facility_group", "direction", "magnitude", "concurrency", "interaction_type", "decision_relevance"]:
        assert token in text


def test_coverage_contract_marks_maximum_not_minimum() -> None:
    assert "maximum_is_not_minimum" in SCRIPT.read_text(encoding="utf-8")

