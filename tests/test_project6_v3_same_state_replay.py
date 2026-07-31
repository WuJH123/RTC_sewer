from __future__ import annotations

import csv
from pathlib import Path

from sewerrtc.state.same_state_replay import write_control_aligned_checkpoint_audit


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_75_and_135_min_are_not_round0_control_aligned(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "state_clone_checkpoint_readiness.csv",
        [
            {"checkpoint_id": "c60", "checkpoint_elapsed_min": 60.0},
            {"checkpoint_id": "c75", "checkpoint_elapsed_min": 75.0},
            {"checkpoint_id": "c135", "checkpoint_elapsed_min": 135.0},
        ],
    )

    path = write_control_aligned_checkpoint_audit(tmp_path)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig")))

    by_id = {row["checkpoint_id"]: row for row in rows}
    assert by_id["c60"]["round0_candidate_eligible"] == "true"
    assert by_id["c75"]["state_clone_diagnostic_eligible"] == "true"
    assert by_id["c75"]["round0_candidate_eligible"] == "false"
    assert by_id["c135"]["round0_candidate_eligible"] == "false"


def test_same_state_replay_module_declares_deterministic_prefix_method() -> None:
    text = Path("sewerrtc/state/same_state_replay.py").read_text(encoding="utf-8")
    assert "deterministic_prefix_replay" in text
    assert "continuous_replay_determinism_report.json" in text
    assert "same_state_branch_gate.json" in text
    assert "hotstart_acceleration_allowed" in text

