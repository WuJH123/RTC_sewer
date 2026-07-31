from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sentinel_contract_is_not_silently_resolved() -> None:
    path = ROOT / "docs" / "contracts" / "sentinel_nodes_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sentinel_contract_status"] == "human_resolution_required"
    assert "FatalAudit" in payload["blocking_rule"]
    for node in payload["nodes"]:
        assert node["safety_threshold"] is None
        assert node["threshold_status"] == "uncalibrated"


def test_sentinel_frozen_file_contains_candidate_nodes_without_fake_hashes() -> None:
    path = ROOT / "outputs" / "project6_pfvfirst_dualfallback_10min_v3" / "contracts" / "sentinel_nodes_frozen.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    assert {row["node_id"] for row in rows} == {"MH0200770", "HS1355904"}
    assert all(row["provenance_status"] == "human_resolution_required" for row in rows)


def test_pump_semantics_are_unresolved_and_not_residual_candidates() -> None:
    path = ROOT / "outputs" / "project6_pfvfirst_dualfallback_10min_v3" / "contracts" / "pump_semantics_audit.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    assert {row["facility_id"] for row in rows} == {"add350.1", "ADD301.2", "ADD301.3"}
    for row in rows:
        assert row["semantics_status"] == "unresolved"
        assert row["residual_override_allowed"] == "false"
        assert row["binary_or_continuous"] == "unresolved"


def test_facility_semantics_contract_blocks_unresolved_pumps() -> None:
    path = ROOT / "docs" / "contracts" / "facility_semantics_contract.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pump_semantics_rule"]["unresolved_pump_residual_override_allowed"] is False
    assert "AuditFallbacks" in payload["blocking_rule"]
