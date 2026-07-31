from __future__ import annotations

from pathlib import Path

from sewerrtc.data import round0_prompt2 as p2


def _event(event_id: str, family: str, split: str = "development_fit", holdout: str = "false") -> dict[str, str]:
    return {
        "event_id": event_id,
        "canonical_event_id": event_id,
        "storm_family_id": family,
        "split": split,
        "round0_eligible": "true",
        "gat_independent_holdout": holdout,
        "calibration_eligible": "false",
        "formal_eligible": "false",
        "rainfall_path": __file__,
        "rainfall_file_sha256": "r" + event_id,
        "rainfall_series_sha256": "s" + event_id,
        "start_time": "2026-01-01 00:00",
        "end_time": "2026-01-01 12:00",
    }


def test_fit_event_expansion_blocks_when_only_two_fit_events(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "EVENT_CATALOG_DIR", tmp_path / "event_catalog")
    monkeypatch.setattr(p2, "PROMPT2_EXPANSION_DIR", tmp_path / "prompt2_expansion")
    p2.write_csv(tmp_path / "event_catalog" / "event_catalog.csv", [_event("T3_D75_chicago_early", "T3_early"), _event("T5_D75_chicago_center", "T5_center")])
    p2.write_csv(tmp_path / "event_catalog" / "event_split_manifest.csv", [{"event_id": "T3_D75_chicago_early", "split": "development_fit", "round0_eligible": "true"}, {"event_id": "T5_D75_chicago_center", "split": "development_fit", "round0_eligible": "true"}])

    code, outputs = p2.plan_prompt2_fit_event_expansion(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"), target_fit_events=36)
    report = p2.read_json(outputs["report"])

    assert code == 3
    assert report["selected_event_count"] == 2
    assert "unique_fit_events_below_30" in report["blocking_reasons"]


def test_fit_event_expansion_excludes_gat_holdout_calibration_and_formal(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "EVENT_CATALOG_DIR", tmp_path / "event_catalog")
    monkeypatch.setattr(p2, "PROMPT2_EXPANSION_DIR", tmp_path / "prompt2_expansion")
    rows = [_event(f"T{i}_D75_chicago_center", f"family{i}") for i in range(32)]
    rows.append(_event("holdout", "h", "development_fit", "true"))
    rows.append(_event("cal", "c", "calibration"))
    rows.append(_event("formal", "f", "formal_blind"))
    p2.write_csv(tmp_path / "event_catalog" / "event_catalog.csv", rows)
    p2.write_csv(tmp_path / "event_catalog" / "event_split_manifest.csv", [{"event_id": r["event_id"], "split": r["split"], "round0_eligible": "true"} for r in rows])

    code, outputs = p2.plan_prompt2_fit_event_expansion(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"), target_fit_events=36)
    selected = p2.read_csv(outputs["plan"])

    assert code == 0
    assert {r["event_id"] for r in selected}.isdisjoint({"holdout", "cal", "formal"})


def test_audit_fit_event_expansion_rejects_single_event_dominance(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "PROMPT2_EXPANSION_DIR", tmp_path / "prompt2_expansion")
    p2.write_csv(
        tmp_path / "prompt2_expansion" / "prompt2_fit_event_expansion_plan.csv",
        [{"event_id": f"e{i}", "storm_family_id": "dominant", "split": "development_fit", "selection_status": "selected"} for i in range(30)],
    )

    code, outputs = p2.audit_prompt2_fit_event_expansion(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"))
    report = p2.read_json(outputs["audit"])

    assert code == 5
    assert "storm_family_dominance_above_20pct" in report["blocking_reasons"]
