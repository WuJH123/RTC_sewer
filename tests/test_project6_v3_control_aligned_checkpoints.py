from __future__ import annotations

from pathlib import Path

from sewerrtc.data import round0_prompt2 as p2


def test_non_10_min_checkpoint_is_not_round0_candidate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(p2, "CONTROL_DIR", tmp_path / "control_checkpoints")
    source_dir = tmp_path / "checkpoint_catalog"
    source_dir.mkdir()
    p2.write_csv(
        source_dir / "checkpoint_catalog.csv",
        [
            {"checkpoint_id": "cp75", "event_id": "e1", "storm_family_id": "s1", "checkpoint_elapsed_min": 75, "split": "action_effect_fit", "eligible_for_effect_training": "true"},
            {"checkpoint_id": "cp80", "event_id": "e1", "storm_family_id": "s1", "checkpoint_elapsed_min": 80, "split": "action_effect_fit", "eligible_for_effect_training": "true"},
        ],
    )

    code, outputs = p2.build_control_aligned_checkpoint_catalog(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"))
    rows = p2.read_csv(outputs["catalog"])

    assert code == 0
    assert next(r for r in rows if r["checkpoint_id"] == "cp75")["round0_candidate_eligible"] == "false"
    assert next(r for r in rows if r["checkpoint_id"] == "cp80")["round0_candidate_eligible"] == "true"


def test_event_time_is_used_when_elapsed_columns_are_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(p2, "CONTROL_DIR", tmp_path / "control_checkpoints")
    source_dir = tmp_path / "checkpoint_catalog"
    source_dir.mkdir()
    p2.write_csv(
        source_dir / "checkpoint_catalog.csv",
        [
            {
                "checkpoint_id": "cp60",
                "event_id": "e1",
                "storm_family_id": "s1",
                "event_time": "60.0",
                "split": "action_effect_fit",
                "history_60min_available": "true",
                "future_120min_available": "true",
                "eligible_for_effect_training": "false",
            }
        ],
    )

    code, outputs = p2.build_control_aligned_checkpoint_catalog(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"))
    rows = p2.read_csv(outputs["catalog"])

    assert code == 0
    assert rows[0]["elapsed_min"] == "60.0"
    assert rows[0]["round0_candidate_eligible"] == "true"


def test_gat_holdout_calibration_and_formal_splits_are_excluded(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(p2, "CONTROL_DIR", tmp_path / "control_checkpoints")
    source_dir = tmp_path / "checkpoint_catalog"
    source_dir.mkdir()
    p2.write_csv(
        source_dir / "checkpoint_catalog.csv",
        [
            {"checkpoint_id": "holdout", "event_id": "e1", "checkpoint_elapsed_min": 80, "split": "gat_independent_holdout", "eligible_for_effect_training": "true"},
            {"checkpoint_id": "formal", "event_id": "e2", "checkpoint_elapsed_min": 80, "split": "formal_blind", "eligible_for_effect_training": "true"},
        ],
    )

    _, outputs = p2.build_control_aligned_checkpoint_catalog(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"))
    rows = p2.read_csv(outputs["catalog"])

    assert all(r["round0_candidate_eligible"] == "false" for r in rows)
