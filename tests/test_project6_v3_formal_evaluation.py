from __future__ import annotations

from pathlib import Path

from sewerrtc.prompt3 import action_effect_mpc as p3


def _write_detail(path: Path, policy_id: str = "internal_rules") -> Path:
    p3.write_csv(
        path,
        [
            {
                "event_id": "e1",
                "policy_id": policy_id,
                "elapsed_min": "0",
                "datetime": "2026-01-01 00:00:00",
                "flood:MH0200770": "0.0",
                "a:ADD301.2": "0",
                "setting:ADD301.2": "0",
                "a:add350.1": "0.35",
                "setting:add350.1": "0.35",
            },
            {
                "event_id": "e1",
                "policy_id": policy_id,
                "elapsed_min": "10",
                "datetime": "2026-01-01 00:10:00",
                "flood:MH0200770": "0.1",
                "a:ADD301.2": "1",
                "setting:ADD301.2": "1",
                "a:add350.1": "0.50",
                "setting:add350.1": "0.50",
            },
        ],
    )
    return path


def _formal_result_row(tmp_path: Path, event_id: str, policy_id: str) -> dict[str, str]:
    detail = _write_detail(tmp_path / f"{event_id}_{policy_id}_detail.csv", policy_id)
    return {
        "event_id": event_id,
        "policy_id": policy_id,
        "split": "formal_blind",
        "initial_state_sha256": "shared-initial",
        "rainfall_sha256": "rain",
        "PFV_m3": "10",
        "TFV_m3": "20",
        "peak_TFV_rate": "1.5",
        "priority_flood_duration_min": "5",
        "recovery_time_min": "180",
        "recovery_censored": "false",
        "action_changes": "3" if policy_id == p3.PROPOSED_POLICY_ID else "0",
        "pump_starts": "0",
        "pump_stops": "0",
        "variable_speed_setting_changes": "1" if policy_id == p3.PROPOSED_POLICY_ID else "0",
        "engineering_violations": "0",
        "hydraulic_evidence_source": "authoritative_swmm",
        "runtime_executed": "true",
        "detail_file": str(detail),
        "detail_sha256": p3.sha256_file(detail),
        "rows": "2",
        "status": "pass",
    }


def test_formal_table_exporter_uses_fixture_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "EVALUATION_DIR", tmp_path / "formal_evaluation")
    p3.EVALUATION_DIR.mkdir(parents=True)
    rows = []
    for event_id in ["formal_fixture_1", "formal_fixture_2"]:
        for policy in p3.EVALUATION_POLICIES:
            rows.append(_formal_result_row(tmp_path, event_id, policy))
    p3.write_csv(p3.EVALUATION_DIR / "formal_event_policy_results.csv", rows)
    p3.write_json(p3.EVALUATION_DIR / "formal_performance_gate.json", {"status": "pass"})

    code, outputs = p3.export_formal_paper_tables("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")

    assert code == 0
    assert outputs["mean_csv"].exists()
    assert outputs["median_md"].exists()
    mean_rows = p3.read_csv(outputs["mean_csv"])
    assert {row["Metric"] for row in mean_rows} >= {"PFV_m3", "TFV_m3", "peak_TFV_rate"}


def test_formal_comparison_blocks_without_formal_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "EVALUATION_DIR", tmp_path / "formal_evaluation")
    p3.EVALUATION_DIR.mkdir(parents=True)

    code, outputs = p3.build_formal_paired_comparison("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")

    assert code == 3
    assert p3.read_json(outputs["report"])["status"] == "blocked"


def test_formal_split_gate_rejects_synthetic_rows_without_detail_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "EVALUATION_DIR", tmp_path / "formal_evaluation")
    p3.EVALUATION_DIR.mkdir(parents=True)
    rows = [
        {
            "event_id": "fake_event",
            "policy_id": policy,
            "split": "formal_blind",
            "initial_state_sha256": "same",
            "PFV_m3": "10",
            "TFV_m3": "20",
            "peak_TFV_rate": "1",
            "engineering_violations": "0",
            "hydraulic_evidence_source": "authoritative_swmm",
            "runtime_executed": "true",
            "status": "pass",
        }
        for policy in p3.EVALUATION_POLICIES
    ]
    p3.write_csv(p3.EVALUATION_DIR / "formal_event_policy_results.csv", rows)

    code, outputs = p3._evaluate_split_gate(
        "configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml",
        "formal_blind",
        "formal_event_policy_results.csv",
        "formal_gate.json",
    )

    assert code == 5
    gate = p3.read_json(outputs["gate"])
    assert gate["status"] == "failed_gate"
    assert any("missing_detail_file" in failure for failure in gate["failures"])


def test_formal_runner_invokes_real_closed_loop_and_normalizes_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "EVALUATION_DIR", tmp_path / "formal_evaluation")
    p3.EVALUATION_DIR.mkdir(parents=True)
    rain_path = tmp_path / "rain.csv"
    rain_path.write_text("elapsed_min,intensity_mm_h\n0,0\n5,1\n", encoding="utf-8")
    p3.write_json(p3.EVALUATION_DIR / "evaluation_event_split_audit.json", {"status": "pass"})
    p3.write_csv(
        p3.EVALUATION_DIR / "evaluation_event_splits.csv",
        [{"event_id": "e1", "split": "calibration_a", "rainfall_path": str(rain_path), "rainfall_sha256": p3.sha256_file(rain_path)}],
    )

    closed_loop = tmp_path / "closed_loop"
    closed_loop.mkdir()
    detail_paths = {}
    for policy in ["internal_rules", "no_control", "passive_anchor", p3.PROPOSED_POLICY_ID]:
        detail_paths[policy] = _write_detail(closed_loop / f"{policy}_detail.csv", policy)
    p3.write_csv(
        closed_loop / "baseline_results.csv",
        [
            {"event_id": "e1", "policy_id": policy, "PFV": "10", "TFV": "20", "peak_TFV_rate": "1", "priority_flood_duration_min": "5", "action_changes": "0", "detail_file": str(detail_paths[policy]), "rows": "2", "duration_min": "60"}
            for policy in ["internal_rules", "no_control", "passive_anchor"]
        ],
    )
    p3.write_csv(
        closed_loop / "proposed_results.csv",
        [{"event_id": "e1", "policy_id": "proposed_pfv_first_mpc", "PFV": "8", "TFV": "19", "peak_TFV_rate": "0.9", "priority_flood_duration_min": "3", "action_changes": "2", "detail_file": str(detail_paths[p3.PROPOSED_POLICY_ID]), "history_file": "", "rows": "2", "duration_min": "60", "wall_time_sec": "1.0"}],
    )
    p3.write_json(closed_loop / "closed_loop_report.json", {"events": 1, "proposed_controller": p3.PROPOSED_POLICY_ID})

    calls = []

    def fake_invoke(config, split, events, max_events, workers, resume):
        calls.append((split, events, max_events, workers, resume))
        return 0, closed_loop, {"command": ["python", "scripts/08_run_closed_loop.py"], "returncode": 0}

    monkeypatch.setattr(p3, "_invoke_closed_loop_authoritative_swmm", fake_invoke)

    code, outputs = p3.calibration_a("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", max_events=1, workers=16, resume=True)

    assert code == 0
    assert calls == [("calibration_a", ["e1"], 1, 16, True)]
    manifest = p3.read_json(outputs["report"])
    assert manifest["hydraulic_evidence_source"] == "authoritative_swmm"
    assert manifest["closed_loop_mode"] == "closed_loop_authoritative_swmm"
    assert manifest["uses_lookup_table_substitute"] is False
    rows = p3.read_csv(outputs["event_policy_results"])
    assert {row["policy_id"] for row in rows} == set(p3.EVALUATION_POLICIES)
    assert all(Path(row["detail_file"]).exists() for row in rows)


def test_formal_config_uses_only_project6_v3_controller() -> None:
    text = Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml").read_text(encoding="utf-8")
    assert "proposed_controller: proposed_pfvfirst_dualfallback_v3" in text
    assert "proposed_controller: native_shield" not in text


def test_formal_authoritative_invocation_uses_v3_controller(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "EVALUATION_DIR", tmp_path / "formal_evaluation")
    p3.EVALUATION_DIR.mkdir(parents=True)
    rain_path = tmp_path / "rain.csv"
    rain_path.write_text("elapsed_min,intensity_mm_h\n0,0\n5,1\n", encoding="utf-8")
    p3.write_csv(
        p3.EVALUATION_DIR / "evaluation_event_splits.csv",
        [{"event_id": "e1", "split": "calibration_a", "rainfall_path": str(rain_path), "rainfall_sha256": p3.sha256_file(rain_path)}],
    )
    calls = []

    class Result:
        returncode = 1
        stdout = ""
        stderr = "stopped before SWMM for command inspection"

    def fake_sync_rainfall(runner_config, split, events):
        path = tmp_path / "rainfall_event_table.csv"
        path.write_text("event_id,rainfall_csv,duration_min,simulation_duration_min\n", encoding="utf-8")
        return path

    def fake_sync_inputs(runner_config):
        return {}

    def fake_run(cmd, cwd, capture_output, text):
        del cwd, capture_output, text
        calls.append(cmd)
        return Result()

    monkeypatch.setattr(p3, "_runner_config_for_authoritative_swmm", lambda config: Path("configs/wuhan_project6_v8_storage.yaml"))
    monkeypatch.setattr(p3, "_sync_formal_rainfall_table_for_closed_loop", fake_sync_rainfall)
    monkeypatch.setattr(p3, "_sync_formal_closed_loop_legacy_inputs", fake_sync_inputs)
    monkeypatch.setattr(p3.subprocess, "run", fake_run)

    config_path = tmp_path / "formal_workers.yaml"
    config_path.write_text(
        "formal_evaluation:\n  proposed_workers: 2\n",
        encoding="utf-8",
    )

    code, _, invocation = p3._invoke_closed_loop_authoritative_swmm(
        config_path,
        "calibration_a",
        ["e1"],
        max_events=1,
        workers=16,
        resume=True,
    )

    assert code == 4
    command = " ".join(calls[0])
    assert "--proposed-controller proposed_pfvfirst_dualfallback_v3" in command
    assert "--action-effect-model" in command
    assert "--proposed-controller native_shield" not in command
    assert "--workers 16" in command
    assert "--proposed-workers 2" in command
    assert invocation["workers_requested"] == 16
    assert invocation["proposed_workers"] == 2
