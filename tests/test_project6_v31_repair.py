from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np

from sewerrtc.prompt3 import action_effect_v31 as v31
from sewerrtc.prompt3.action_effect_mpc import _write_formal_timeseries_parquet_streaming
from sewerrtc.simulation.pyswmm_runner import _checkpoint_file_stem


def test_v31_candidate_gate_rejects_realized_labels() -> None:
    decision = v31.candidate_execution_decision_v31(
        {
            "ucb_delta_PFV_vs_internal": -1.0,
            "ucb_delta_TFV_vs_internal": -1.0,
            "ucb_delta_peak_vs_internal": -1.0,
            "ucb_delta_PFV_vs_selected_fallback": -1.0,
            "ucb_delta_TFV_vs_selected_fallback": -1.0,
            "realized_delta_PFV": -999.0,
        },
        {"id": "internal_rules", "frozen": True},
        {"execution_gate": {}},
    )

    assert decision["decision"] == "fallback"
    assert decision["reason"] == "realized_label_field_present"


def test_v31_candidate_gate_rejects_pfv_or_tfv_ucb_failure() -> None:
    cfg = {"execution_gate": {"pfv_ucb_margin_m3": 0.0, "tfv_ucb_margin_m3": 0.0, "peak_ucb_margin": 0.0, "pfv_vs_fallback_margin_m3": 0.0, "tfv_vs_fallback_margin_m3": 0.0}}

    pfv_bad = v31.candidate_execution_decision_v31(
        {
            "ucb_delta_PFV_vs_internal": 1.0,
            "ucb_delta_TFV_vs_internal": -1.0,
            "ucb_delta_peak_vs_internal": -1.0,
            "ucb_delta_PFV_vs_selected_fallback": -1.0,
            "ucb_delta_TFV_vs_selected_fallback": -1.0,
            "safety_pass": True,
            "engineering_pass": True,
            "backup_reachable": True,
        },
        {"frozen": True},
        cfg,
    )
    tfv_bad = v31.candidate_execution_decision_v31(
        {
            "ucb_delta_PFV_vs_internal": -1.0,
            "ucb_delta_TFV_vs_internal": 1.0,
            "ucb_delta_peak_vs_internal": -1.0,
            "ucb_delta_PFV_vs_selected_fallback": -1.0,
            "ucb_delta_TFV_vs_selected_fallback": -1.0,
            "safety_pass": True,
            "engineering_pass": True,
            "backup_reachable": True,
        },
        {"frozen": True},
        cfg,
    )

    assert pfv_bad["decision"] == "fallback"
    assert "pfv_internal" in pfv_bad["failed_checks"]
    assert tfv_bad["decision"] == "fallback"
    assert "tfv_internal" in tfv_bad["failed_checks"]


def test_v31_fallback_must_be_frozen_before_candidate_execution() -> None:
    decision = v31.candidate_execution_decision_v31(
        {
            "ucb_delta_PFV_vs_internal": -1.0,
            "ucb_delta_TFV_vs_internal": -1.0,
            "ucb_delta_peak_vs_internal": -1.0,
            "ucb_delta_PFV_vs_selected_fallback": -1.0,
            "ucb_delta_TFV_vs_selected_fallback": -1.0,
            "safety_pass": True,
            "engineering_pass": True,
            "backup_reachable": True,
        },
        {"frozen": False},
        {"execution_gate": {}},
    )

    assert decision["decision"] == "fallback"
    assert "fallback_frozen" in decision["failed_checks"]


def test_v31_action_smoothing_deadband_binary_and_variable_speed_semantics() -> None:
    smoothed = v31.smooth_action_v31(
        {"ADD301.2": 0.0, "ADD301.3": 1.0, "add350.1": 0.5, "OR1": 0.5},
        {"ADD301.2": 0.5, "ADD301.3": 0.0, "add350.1": 0.7, "OR1": 0.51},
        {"add350_bounds_verified": False},
        {"action_smoothing": {"setting_deadband": 0.02, "continuous_min_hold_steps": 2}},
    )

    assert smoothed["executed"]["ADD301.2"] == 0.0
    assert smoothed["reasons"]["ADD301.2"] == "binary_intermediate_rejected"
    assert smoothed["executed"]["ADD301.3"] == 0.0
    assert smoothed["executed"]["add350.1"] == 0.5
    assert smoothed["reasons"]["add350.1"] == "variable_speed_bounds_unverified"
    assert smoothed["executed"]["OR1"] == 0.5
    assert smoothed["reasons"]["OR1"] == "deadband_hold_previous"


def test_v31_old_formal_events_cannot_enter_formal_v31(tmp_path, monkeypatch) -> None:
    old = tmp_path / "old" / "formal_evaluation"
    old.mkdir(parents=True)
    v31.write_csv(
        old / "evaluation_event_splits.csv",
        [
            {
                "event_id": "old_formal_1",
                "split": "formal_blind",
                "rainfall_series_sha256": "oldsha",
                "used_for_round0_1_2": "false",
                "used_for_model_training": "false",
            },
            {
                "event_id": "fresh_1",
                "split": "development",
                "rainfall_series_sha256": "freshsha",
                "used_for_round0_1_2": "false",
                "used_for_model_training": "false",
            },
        ],
    )
    cfg = tmp_path / "v31.yaml"
    cfg.write_text(
        "project:\n  output_root: outputs/v31\n"
        "v31:\n  old_formal_root: old/formal_evaluation\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(v31, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v31, "V3_ROOT", tmp_path / "old")

    code, outputs = v31.build_evaluation_splits_v31(cfg)

    assert code == 3
    exclusions = v31.read_csv(outputs["exclusions"])
    old_rows = [row for row in exclusions if row["event_id"] == "old_formal_1"]
    assert old_rows
    assert str(old_rows[0]["eligible_for_formal_v31"]).lower() == "false"
    assert str(old_rows[0]["used_by_round3_hard_negative"]).lower() == "true"


def test_v31_policy_lock_blocks_before_model_and_split_gates(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "v31.yaml"
    cfg.write_text("project:\n  output_root: outputs/v31\n", encoding="utf-8")
    monkeypatch.setattr(v31, "PROJECT_ROOT", tmp_path)

    code, outputs = v31.policy_lock_v31(cfg)

    assert code == 3
    lock = v31.read_json(outputs["lock"])
    assert lock["status"] == "blocked"
    assert lock["formal_v31_allowed"] is False


def test_runner_registers_v31_stages() -> None:
    runner = Path("scripts/project6_runs/RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1").read_text(encoding="utf-8")
    for stage in [
        "DiagnoseFormalFailuresV31",
        "PlanRound3HardNegativesV31",
        "GenerateRound3HardNegativesV31",
        "TrainActionEffectV31",
        "RunClosedLoopDevV31",
        "BuildEvaluationRainfallAssetsV31",
        "BuildEvaluationSplitsV31",
        "CalibrationAV31",
        "LockedValidationBV31",
        "PolicyLockV31",
        "FormalBlindV31",
        "BuildFormalComparisonV31",
        "EvaluateFormalPerformanceV31",
        "ExportFormalTablesV31",
    ]:
        assert f'"{stage}"' in runner
    assert "scripts\\202_prompt3_v31.py" in runner
    assert "project6_pfvfirst_dualfallback_10min_v3_1" in runner


def test_v31_round3_plan_includes_reserve(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "v31.yaml"
    cfg.write_text("project:\n  output_root: outputs/v31\n", encoding="utf-8")
    monkeypatch.setattr(v31, "PROJECT_ROOT", tmp_path)
    diag = tmp_path / "outputs" / "v31" / "diagnostics"
    v31.write_csv(
        diag / "v3_formal_failure_decisions.csv",
        [
            {
                "event_id": f"E{i}",
                "elapsed_min": 60 + i * 10,
                "phase": "rising",
                "failure_types": "candidate_dominated_by_internal",
                "active_facility_ids": "ADD301.2;W1",
            }
            for i in range(90)
        ],
    )
    v31.write_csv(diag / "v3_formal_failure_events.csv", [{"event_id": f"E{i}"} for i in range(90)])

    code, outputs = v31.plan_round3_hard_negatives_v31(cfg, target_samples=600, seed=1)
    rows = v31.read_csv(outputs["plan"])
    report = v31.read_json(outputs["report"])

    assert code == 0
    assert len(rows) == 720
    assert report["target_effective_samples"] == 600
    assert report["reserve_samples"] == 120


def test_v31_round3_resume_converts_pending_without_duplicating_existing(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "v31.yaml"
    cfg.write_text(
        "project:\n  output_root: outputs/v31\n"
        "v31:\n  old_formal_root: old/formal_evaluation\n"
        "round3:\n  target_effective_samples: 3\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(v31, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v31, "V3_ROOT", tmp_path / "old")
    root = tmp_path / "outputs" / "v31"
    diag = root / "diagnostics"
    formal = tmp_path / "old" / "formal_evaluation"
    v31.write_csv(
        diag / "v3_formal_failure_events.csv",
        [
            {
                "event_id": "E1",
                "delta_PFV_vs_internal": "10",
                "delta_TFV_vs_internal": "20",
                "delta_peak_vs_internal": "-1",
                "delta_priority_duration_vs_internal": "0",
                "delta_recovery_vs_internal": "0",
                "proposed_PFV_m3": "110",
                "proposed_TFV_m3": "220",
                "proposed_peak_TFV_rate": "9",
                "internal_PFV_m3": "100",
                "internal_TFV_m3": "200",
                "internal_peak_TFV_rate": "10",
                "delta_PFV_vs_passive": "5",
                "delta_TFV_vs_passive": "5",
                "delta_peak_vs_passive": "0",
            }
        ],
    )
    v31.write_csv(
        root / "round3" / "round3_hard_negative_plan.csv",
        [
            {"round3_candidate_id": "r0", "source_old_formal_event_id": "E1", "checkpoint_elapsed_min": "60", "phase": "peak", "failure_types": "", "variant_type": "actual_executed_candidate", "active_facility_ids": "A;B"},
            {"round3_candidate_id": "r1", "source_old_formal_event_id": "E1", "checkpoint_elapsed_min": "70", "phase": "peak", "failure_types": "", "variant_type": "internal", "active_facility_ids": ""},
            {"round3_candidate_id": "r2", "source_old_formal_event_id": "E1", "checkpoint_elapsed_min": "80", "phase": "peak", "failure_types": "", "variant_type": "passive", "active_facility_ids": ""},
            {"round3_candidate_id": "r3", "source_old_formal_event_id": "E1", "checkpoint_elapsed_min": "90", "phase": "peak", "failure_types": "", "variant_type": "half_magnitude", "active_facility_ids": "A;B"},
        ],
    )
    v31.write_csv(
        formal / "formal_event_policy_results.csv",
        [
            {"event_id": "E1", "policy_id": "internal_rules", "PFV_m3": "100", "TFV_m3": "200", "peak_TFV_rate": "10", "priority_flood_duration_min": "5", "recovery_time_min": "180", "initial_state_sha256": "sha-initial"},
            {"event_id": "E1", "policy_id": "passive_anchor", "PFV_m3": "105", "TFV_m3": "210", "peak_TFV_rate": "9", "priority_flood_duration_min": "4", "recovery_time_min": "170", "initial_state_sha256": "sha-initial"},
            {"event_id": "E1", "policy_id": v31.PROPOSED_POLICY_ID, "PFV_m3": "110", "TFV_m3": "220", "peak_TFV_rate": "8", "priority_flood_duration_min": "3", "recovery_time_min": "160", "initial_state_sha256": "sha-initial"},
        ],
    )
    v31.write_csv(root / "round3" / "round3_generation_manifest.csv", [{"sample_id": "r0", "round": "round3_v31", "event_id": "E1", "runtime_executed": "true", "true_future_in_model_input": "false", **{label: "0" for label in v31.LABELS_V31}}])

    code, outputs = v31.generate_round3_hard_negatives_v31(cfg, max_samples=2, smoke=False, resume=True)
    rows = v31.read_csv(outputs["manifest"])
    pending = v31.read_csv(outputs["pending"])

    assert code == 0
    assert [row["sample_id"] for row in rows].count("r0") == 1
    assert {"r1", "r2"}.issubset({row["sample_id"] for row in rows})
    assert any(row["round3_candidate_id"] == "r3" for row in pending)
    assert "r3" not in {row["sample_id"] for row in rows}
    assert all(str(row.get("runtime_executed", "")).lower() == "true" for row in rows)


def test_v31_generated_rainfall_assets_feed_independent_split(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "v31.yaml"
    cfg.write_text(
        "project:\n  output_root: outputs/v31\n"
        "v31:\n  old_formal_root: old/formal_evaluation\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(v31, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v31, "V3_ROOT", tmp_path / "old")
    old = tmp_path / "old" / "formal_evaluation"
    v31.write_csv(
        old / "evaluation_event_splits.csv",
        [{"event_id": "old_formal_1", "split": "formal_blind", "rainfall_series_sha256": "oldsha", "storm_family_id": "old_family"}],
    )
    v31.write_csv(tmp_path / "old" / "rainfall_assets" / "rainfall_asset_inventory.csv", [])

    asset_code, _ = v31.build_evaluation_rainfall_assets_v31(cfg)
    split_code, split_outputs = v31.build_evaluation_splits_v31(cfg)
    audit_code, audit_outputs = v31.audit_evaluation_splits_v31(cfg)
    audit = v31.read_json(audit_outputs["audit"])

    assert asset_code == 0
    assert split_code == 0
    assert audit_code == 0
    assert audit["split_counts"]["formal_blind_v31"] == 36
    split_rows = v31.read_csv(split_outputs["splits"])
    assert all(row["event_id"] != "old_formal_1" for row in split_rows)


def test_v31_split_audit_rejects_missing_empty_and_duplicate_rainfall_hash(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "v31.yaml"
    cfg.write_text("project:\n  output_root: outputs/v31\nv31:\n  old_formal_root: old/formal_evaluation\n", encoding="utf-8")
    monkeypatch.setattr(v31, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v31, "V3_ROOT", tmp_path / "old")
    old = tmp_path / "old" / "formal_evaluation"
    v31.write_csv(old / "evaluation_event_splits.csv", [])
    rain = tmp_path / "rain.csv"
    v31.write_csv(rain, [{"elapsed_min": 0, "intensity_mm_h": 1.0}])
    base = {
        "event_id": "E",
        "canonical_event_id": "E",
        "storm_family_id": "F",
        "split": "calibration_a_v31",
        "rainfall_path": str(rain),
        "rainfall_sha256": v31._file_hash(rain),
        "rainfall_file_sha256": v31._file_hash(rain),
        "source_project": "test",
        "eligible_for_formal_v31": "true",
        "formal_v31_role": "calibration_a_v31",
    }
    out = tmp_path / "outputs" / "v31" / "formal_evaluation"
    v31.write_csv(out / "evaluation_event_splits_v31.csv", [{**base, "rainfall_series_sha256": "", "rainfall_series_hash": ""}])
    code, outputs = v31.audit_evaluation_splits_v31(cfg)
    audit = v31.read_json(outputs["audit"])
    assert code == 3
    assert any(f["reason"] == "rainfall_series_sha256_empty" for f in audit["schema_failures"])

    rows = []
    for i, split in enumerate(["calibration_a_v31"] * 12 + ["locked_validation_b_v31"] * 12 + ["formal_blind_v31"] * 36):
        rows.append({**base, "event_id": f"E{i}", "canonical_event_id": f"E{i}", "split": split, "formal_v31_role": split, "rainfall_series_sha256": "dup", "rainfall_series_hash": "dup"})
    v31.write_csv(out / "evaluation_event_splits_v31.csv", rows)
    code, outputs = v31.audit_evaluation_splits_v31(cfg)
    audit = v31.read_json(outputs["audit"])
    assert code == 5
    assert audit["rainfall_series_duplicate_count"] == 59


def test_v31_split_audit_rejects_builder_runner_field_mismatch(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "v31.yaml"
    cfg.write_text("project:\n  output_root: outputs/v31\nv31:\n  old_formal_root: old/formal_evaluation\n", encoding="utf-8")
    monkeypatch.setattr(v31, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v31, "V3_ROOT", tmp_path / "old")
    v31.write_csv(tmp_path / "old" / "formal_evaluation" / "evaluation_event_splits.csv", [])
    rain = tmp_path / "rain.csv"
    v31.write_csv(rain, [{"elapsed_min": 0, "intensity_mm_per_hr": 1.0}])
    rows = []
    for i, split in enumerate(["calibration_a_v31"] * 12 + ["locked_validation_b_v31"] * 12 + ["formal_blind_v31"] * 36):
        sha = f"sha{i}"
        rows.append({
            "event_id": f"E{i}",
            "canonical_event_id": f"E{i}",
            "storm_family_id": f"F{i}",
            "split": split,
            "rainfall_path": str(rain),
            "rainfall_sha256": v31._file_hash(rain),
            "rainfall_file_sha256": v31._file_hash(rain),
            "rainfall_series_sha256": sha,
            "rainfall_series_hash": sha,
            "source_project": "test",
            "eligible_for_formal_v31": "true",
            "formal_v31_role": split,
        })
    v31.write_csv(tmp_path / "outputs" / "v31" / "formal_evaluation" / "evaluation_event_splits_v31.csv", rows)
    code, outputs = v31.audit_evaluation_splits_v31(cfg)
    audit = v31.read_json(outputs["audit"])
    assert code == 3
    assert any("rainfall_csv_missing_columns" in f["reason"] for f in audit["schema_failures"])


def test_v31_policy_lock_requires_calibration_and_locked_validation(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "v31.yaml"
    cfg.write_text("project:\n  output_root: outputs/v31\n", encoding="utf-8")
    monkeypatch.setattr(v31, "PROJECT_ROOT", tmp_path)
    root = tmp_path / "outputs" / "v31"
    v31.write_json(root / "formal_evaluation" / "evaluation_event_split_audit_v31.json", {"status": "pass"})
    v31.write_json(root / "action_effect_models" / "model_gate_v31.json", {"status": "pass"})
    v31.write_json(root / "formal_evaluation" / "calibration_a_v31_run_manifest.json", {"status": "blocked", "runtime_executed": False})
    v31.write_json(root / "formal_evaluation" / "locked_validation_b_v31_run_manifest.json", {"status": "pass", "runtime_executed": True})

    code, outputs = v31.policy_lock_v31(cfg)
    lock = v31.read_json(outputs["lock"])
    assert code == 3
    assert lock["status"] == "blocked"
    assert "calibration_a_v31_not_pass" in lock["blocking_reasons"]
    assert lock["formal_v31_allowed"] is False


def test_v31_validation_and_formal_block_without_prerequisites(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "v31.yaml"
    cfg.write_text("project:\n  output_root: outputs/v31\nv31:\n  old_formal_root: old/formal_evaluation\n", encoding="utf-8")
    monkeypatch.setattr(v31, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v31, "V3_ROOT", tmp_path / "old")
    root = tmp_path / "outputs" / "v31"
    v31.write_json(root / "formal_evaluation" / "evaluation_event_split_audit_v31.json", {"status": "pass"})
    v31.write_json(root / "action_effect_models" / "model_gate_v31.json", {"status": "pass"})

    validation_code, validation_outputs = v31.locked_validation_b_v31(cfg, max_events=1, contract_dry_run=True)
    formal_code, formal_outputs = v31.formal_blind_v31(cfg, max_events=1, contract_dry_run=True)

    assert validation_code == 3
    assert "calibration_a_v31_not_pass" in v31.read_json(validation_outputs["report"])["blocking_reasons"]
    assert formal_code == 3
    assert "policy_lock_v31_missing_or_not_pass" in v31.read_json(formal_outputs["report"])["blocking_reasons"]


def test_v31_training_exports_runtime_npz_for_authoritative_loop(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "v31.yaml"
    cfg.write_text("project:\n  output_root: outputs/v31\n", encoding="utf-8")
    monkeypatch.setattr(v31, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v31, "V3_ROOT", tmp_path / "old")
    row = {
        "sample_id": "s1",
        "k_value": "3",
        "concurrency": "3",
        "action_direction": "increase",
        "action_magnitude": "medium",
        "phase": "rising",
        "delta_PFV_vs_internal": "-1.0",
        "delta_TFV_vs_internal": "-2.0",
        "delta_peak_vs_internal": "-0.5",
        "delta_PFV_vs_selected_fallback": "-1.5",
        "delta_TFV_vs_selected_fallback": "-2.5",
        "delta_peak_vs_selected_fallback": "-0.6",
        "priority_duration_delta": "-10",
        "recovery_delta": "-15",
    }
    v31.write_csv(tmp_path / "outputs" / "v31" / "round3_dataset" / "round3_dataset_smoke_manifest.csv", [row])

    code, outputs = v31.train_action_effect_v31(cfg, smoke=True, ensemble_size=2)
    report = v31.read_json(outputs["report"])

    assert code == 0
    runtime_path = Path(report["runtime_model_path"])
    assert runtime_path.exists()
    data = np.load(runtime_path, allow_pickle=False)
    assert data["weights"].shape == (2, 9, 3)
    assert data["feature_mean"].shape == (2, 8)
    assert data["labels"].tolist() == list(v31.v3.LABELS)


def test_v31_calibration_blocks_stale_model_gate_without_runtime_npz(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "v31.yaml"
    cfg.write_text("project:\n  output_root: outputs/v31\nv31:\n  old_formal_root: old/formal_evaluation\n", encoding="utf-8")
    monkeypatch.setattr(v31, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v31, "V3_ROOT", tmp_path / "old")
    root = tmp_path / "outputs" / "v31"
    v31.write_json(root / "formal_evaluation" / "evaluation_event_split_audit_v31.json", {"status": "pass"})
    v31.write_json(root / "action_effect_models" / "model_gate_v31.json", {"status": "pass", "model_sha256": "old"})

    code, outputs = v31.calibration_a_v31(cfg, max_events=1, contract_dry_run=True)
    report = v31.read_json(outputs["report"])

    assert code == 3
    assert "model_gate_v31_stale_missing_runtime_npz" in report["blocking_reasons"]


def test_long_v31_checkpoint_ids_use_short_file_stems() -> None:
    checkpoint_id = "V31_RP5_D1H_P20_v31_independent_gamma_000__internal_rules__near_peak__0060m"
    stem = _checkpoint_file_stem(checkpoint_id)
    assert len(stem) < len(checkpoint_id)
    assert len(stem) <= 20
    assert stem.startswith("cp_")
    assert stem == _checkpoint_file_stem(checkpoint_id)


def test_formal_timeseries_streaming_parquet_handles_wide_detail_files(tmp_path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("elapsed_min,node_a\n0,1.0\n5,2.0\n", encoding="utf-8")
    second.write_text("elapsed_min,node_b\n0,3.0\n", encoding="utf-8")
    out = tmp_path / "formal_timeseries.parquet"

    status, count = _write_formal_timeseries_parquet_streaming(
        [
            {"split": "formal_blind_v31", "policy_id": "internal_rules", "detail_file": str(first)},
            {"split": "formal_blind_v31", "policy_id": "proposed_pfvfirst_dualfallback_v3", "detail_file": str(second)},
        ],
        out,
        chunksize=1,
    )

    assert status == "written"
    assert count == 3
    frame = np.asarray(__import__("pandas").read_parquet(out)["formal_policy_id"])
    assert set(frame.tolist()) == {"internal_rules", "proposed_pfvfirst_dualfallback_v3"}


def test_formal_timeseries_streaming_parquet_avoids_fragmented_dataframe_warning(tmp_path) -> None:
    first = tmp_path / "wide_a.csv"
    second = tmp_path / "wide_b.csv"
    first.write_text(",".join(["elapsed_min", *[f"a{i}" for i in range(160)]]) + "\n" + ",".join(["0", *["1"] * 160]) + "\n", encoding="utf-8")
    second.write_text(",".join(["elapsed_min", *[f"b{i}" for i in range(160)]]) + "\n" + ",".join(["0", *["2"] * 160]) + "\n", encoding="utf-8")
    out = tmp_path / "formal_timeseries.parquet"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        status, count = _write_formal_timeseries_parquet_streaming(
            [
                {"split": "calibration_a_v32", "policy_id": "internal_rules", "detail_file": str(first)},
                {"split": "calibration_a_v32", "policy_id": "proposed_pfvfirst_dualfallback_v3", "detail_file": str(second)},
            ],
            out,
            chunksize=1,
        )

    assert status == "written"
    assert count == 2
    assert not [w for w in caught if "highly fragmented" in str(w.message)]
