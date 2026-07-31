from __future__ import annotations

from pathlib import Path

from sewerrtc.data import round0_prompt2 as p2


def test_prompt2_entry_blocks_when_same_state_gate_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(p2, "PROMPT2_DIR", tmp_path / "prompt2")
    monkeypatch.setattr(p2, "GATES_DIR", tmp_path / "gates")

    code, outputs = p2.audit_prompt2_entry(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"))

    assert code == 3
    assert outputs["gate"].exists()


def test_prompt2_entry_requires_hotstart_not_used_for_candidate_labels(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(p2, "PROMPT2_DIR", tmp_path / "prompt2")
    monkeypatch.setattr(p2, "GATES_DIR", tmp_path / "gates")
    (tmp_path / "gates").mkdir(parents=True)
    p2.write_json(tmp_path / "gates" / "project6_prompt2_gat_readiness_gate.json", {"status": "pass", "allowed_to_enter_prompt3a": True})
    p2.write_json(tmp_path / "gates" / "project6_prompt2_gat_readiness_gate.json", {"status": "pass", "allowed_to_enter_prompt3a": True})
    (tmp_path / "gat").mkdir()
    p2.write_json(tmp_path / "gat" / "gat_primary_selection_lock.json", {"status": "pass"})
    (tmp_path / "state_clone").mkdir()
    p2.write_json(tmp_path / "state_clone" / "same_state_branch_gate.json", {"status": "pass", "selected_same_state_method": "deterministic_prefix_replay"})
    p2.write_json(tmp_path / "state_clone" / "same_state_replay_report.json", {"passed_checkpoint_count": 18, "formal_same_state_unlock_allowed": True})
    (tmp_path / "hotstart").mkdir()
    p2.write_json(tmp_path / "hotstart" / "hotstart_acceleration_readiness_gate.json", {"hotstart_acceleration_allowed": True})

    code, _ = p2.audit_prompt2_entry(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"))

    assert code == 3
