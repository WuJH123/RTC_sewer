from __future__ import annotations

from pathlib import Path

from sewerrtc.data import round0_prompt2 as p2


def _candidate(event: str, checkpoint: str, elapsed: int, phase: str = "rising") -> dict[str, str]:
    return {
        "checkpoint_id": checkpoint,
        "trajectory_id": f"{event}_internal_rules",
        "event_id": event,
        "storm_family_id": f"family_{event}",
        "policy_id": "internal_rules",
        "split": "development_fit",
        "elapsed_min": str(elapsed),
        "event_time": str(elapsed),
        "phase": phase,
        "history_available_min": "60",
        "future_available_min": "120",
        "full_recovery_contract_available": "true",
        "round0_candidate_eligible": "true",
    }


def test_checkpoint_support_gate_blocks_six_checkpoints(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "PROMPT2_CHECKPOINT_DIR", tmp_path / "prompt2_checkpoints")
    p2.write_csv(tmp_path / "prompt2_checkpoints" / "prompt2_selected_control_checkpoints.csv", [_candidate("e1", f"cp{i}", 60 + 10 * i) for i in range(6)])
    p2.write_json(tmp_path / "prompt2_checkpoints" / "prompt2_control_checkpoint_support_audit.json", {"status": "blocked", "unique_fit_events": 1, "selected_checkpoint_count": 6, "blocking_reasons": ["unique_fit_events_below_30", "control_checkpoints_below_120"]})

    code, outputs = p2.evaluate_prompt2_checkpoint_support_gate(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"))
    gate = p2.read_json(outputs["gate"])

    assert code == 3
    assert gate["status"] == "blocked"


def test_select_control_checkpoints_excludes_non_10_min_and_insufficient_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "PROMPT2_CHECKPOINT_DIR", tmp_path / "prompt2_checkpoints")
    p2.write_csv(
        tmp_path / "prompt2_checkpoints" / "prompt2_control_checkpoint_candidates.csv",
        [
            _candidate("e1", "cp60", 60),
            _candidate("e1", "cp75", 75),
            {**_candidate("e1", "cp80_no_history", 80), "history_available_min": "50"},
            {**_candidate("e1", "cp90_no_future", 90), "future_available_min": "100"},
        ],
    )

    code, outputs = p2.select_prompt2_control_checkpoints(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"), target_checkpoints=1, max_per_event=6)
    rows = p2.read_csv(outputs["selected"])

    assert code == 3
    assert [r["checkpoint_id"] for r in rows] == ["cp60"]


def test_checkpoint_support_gate_passes_with_120_checkpoints_and_30_events(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "PROMPT2_CHECKPOINT_DIR", tmp_path / "prompt2_checkpoints")
    phases = (
        ["pre_rise_or_early_rising"] * 12
        + ["rising"] * 30
        + ["near_peak"] * 24
        + ["peak"] * 24
        + ["recession"] * 24
        + ["recovery_or_release"] * 12
    )
    rows = [_candidate(f"e{i % 30}", f"cp{i}", 60 + 10 * (i % 6), phase=phase) for i, phase in enumerate(phases)]
    p2.write_csv(tmp_path / "prompt2_checkpoints" / "prompt2_selected_control_checkpoints.csv", rows)

    code, outputs = p2.audit_prompt2_control_checkpoint_support(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"))
    report = p2.read_json(outputs["audit"])

    assert code == 0
    assert report["unique_fit_events"] == 30
    assert report["selected_checkpoint_count"] >= 120
