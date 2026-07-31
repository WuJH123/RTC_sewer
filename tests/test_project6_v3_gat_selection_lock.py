from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
DECISION = ROOT / "docs" / "contracts" / "gat_primary_selection_decision.json"
SELECTION_SCRIPT = ROOT / "scripts" / "142_select_primary_gat.py"


def test_primary_selection_decision_is_user_confirmed_sr0p15() -> None:
    text = DECISION.read_text(encoding="utf-8")
    assert '"registry_name": "sr0p15"' in text
    assert '"expected_sensor_count": 134' in text
    assert '"automatic_selection": false' in text
    assert '"round0_unlock_allowed": false' in text


def test_select_primary_gat_runner_requires_acknowledgement_and_sr0p15() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "[switch]$SelectPrimaryGAT" in text
    assert "[switch]$AcknowledgeSelection" in text
    assert 'GATRegistryName_required' in text
    assert '--acknowledge-selection' in text
    assert "scripts\\142_select_primary_gat.py" in text


def test_select_primary_gat_script_rejects_non_sr0p15_and_writes_no_round0_unlock() -> None:
    text = SELECTION_SCRIPT.read_text(encoding="utf-8")
    assert "write_primary_gat_lock" in text
    helper = (ROOT / "sewerrtc" / "state" / "gat_selection.py").read_text(encoding="utf-8")
    assert 'registry_name_must_be_sr0p15' in helper
    assert 'missing_acknowledgement' in helper
    assert '"round0_unlock_allowed": False' in helper
    assert '"robustness_status": "pending"' in helper


def test_selection_lock_validates_required_hash_and_reports() -> None:
    helper = (ROOT / "sewerrtc" / "state" / "gat_selection.py").read_text(encoding="utf-8")
    assert "11f40e6a36016202139e604f04c7d888b5ec3805511c46172ad968a7c20d0e20" in helper
    for token in [
        "gat_compatibility_report.json",
        "gat_checkpoint_hashes.csv",
        "gat_strict_load_audit.csv",
        "gat_node_mapping.csv",
        "gat_sensor_mapping.csv",
        "gat_graph_signature_audit.csv",
    ]:
        assert token in helper


def test_selection_lock_hash_fields_have_distinct_sources() -> None:
    helper = (ROOT / "sewerrtc" / "state" / "gat_selection.py").read_text(encoding="utf-8")
    assert "_hash_evidence" in helper
    assert '"state_dict_signature": _hash_evidence(registry_row.get("state_dict_key_signature"), registry_path, "state_dict_key_signature")' in helper
    assert '"edge_set_hash": _hash_evidence(edge_set_hash, graph_path, "directed_edge_list_hash")' in helper
    assert "state_dict_signature\": candidate.get(\"graph_signature\")" not in helper
