from __future__ import annotations

import json
from pathlib import Path

from sewerrtc.state.same_state_replay import evaluate_same_state_branch_gate, write_object_order_audit


def _minimal_inp(path: Path, nodes: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "[OPTIONS]",
                "FLOW_UNITS CFS",
                "[JUNCTIONS]",
                *[f"{node} 0 1 0 0 0" for node in nodes],
                "[CONDUITS]",
                "C1 A B 1 0.01 0 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_object_order_same_count_different_order_fails(tmp_path: Path) -> None:
    source = tmp_path / "source.inp"
    clone = tmp_path / "clone.inp"
    _minimal_inp(source, ["A", "B"])
    _minimal_inp(clone, ["B", "A"])

    _, report_path, ok = write_object_order_audit(tmp_path, source, clone)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert ok is False
    assert report["hotstart_eligible"] is False


def test_same_state_gate_allows_replay_when_hotstart_failed(tmp_path: Path) -> None:
    (tmp_path / "continuous_replay_determinism_report.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    (tmp_path / "same_state_replay_report.json").write_text(
        json.dumps({"status": "pass", "formal_same_state_unlock_allowed": True}),
        encoding="utf-8",
    )
    (tmp_path / "state_clone_report.json").write_text(json.dumps({"status": "failed_gate"}), encoding="utf-8")

    code, outputs = evaluate_same_state_branch_gate(tmp_path)

    assert code == 0
    gate = json.loads(outputs["gate"].read_text(encoding="utf-8"))
    assert gate["selected_same_state_method"] == "deterministic_prefix_replay"
    assert gate["hotstart_acceleration_allowed"] is False
    assert gate["formal_same_state_unlock_allowed"] is True


def test_same_state_gate_blocks_without_continuous_replay_determinism(tmp_path: Path) -> None:
    (tmp_path / "continuous_replay_determinism_report.json").write_text(json.dumps({"status": "failed_gate"}), encoding="utf-8")
    (tmp_path / "same_state_replay_report.json").write_text(
        json.dumps({"status": "pass", "formal_same_state_unlock_allowed": True}),
        encoding="utf-8",
    )

    code, outputs = evaluate_same_state_branch_gate(tmp_path)

    assert code == 3
    gate = json.loads(outputs["gate"].read_text(encoding="utf-8"))
    assert gate["formal_same_state_unlock_allowed"] is False
    assert "continuous_replay_determinism_not_pass" in gate["blocking_reasons"]

