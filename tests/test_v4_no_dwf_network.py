from pathlib import Path

from sewerrtc.v4.contracts import audit_network


ROOT = Path(__file__).resolve().parents[1]


def test_formal_network_is_no_dwf_and_matches_frozen_hashes() -> None:
    evidence = audit_network(ROOT / "data/wuhan_v8_storage_retrofit.inp")

    assert evidence["status"] == "pass"
    assert evidence["network_variant"] == "rainfall_only_no_dwf"
    assert evidence["active_dwf_flow_rows"] == 0
    assert len(evidence["network_sha256"]) == 64
    assert len(evidence["physical_network_sha256"]) == 64


def test_active_dwf_fails_closed_without_modifying_inp(tmp_path: Path) -> None:
    inp = tmp_path / "with_dwf.inp"
    original = "[TITLE]\nX\n[DWF]\nJ1 FLOW 1.0\n"
    inp.write_text(original, encoding="utf-8")

    evidence = audit_network(inp)

    assert evidence["status"] == "blocked"
    assert evidence["active_dwf_flow_rows"] == 1
    assert inp.read_text(encoding="utf-8") == original
