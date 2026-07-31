from __future__ import annotations

import csv
import json
from pathlib import Path

from sewerrtc.prompt3 import action_effect_v32 as v32


def test_pfv_budget_spends_one_step_ucb_and_never_recharges() -> None:
    budget = v32.EventPfvRiskBudgetV32(initial_budget=10.0, reserve_margin=2.0)
    first = budget.evaluate(-50.0)
    second = budget.evaluate(3.0)
    third = budget.evaluate(6.0)
    assert first["risk_cost"] == 0.0
    assert first["remaining_budget_after"] == 8.0
    assert second["remaining_budget_after"] == 5.0
    assert third["allowed"] is False
    assert third["remaining_budget_after"] == 5.0


def test_adaptive_k_only_uses_allowed_values_and_fallback_conditions() -> None:
    assert v32.adaptive_k_v32({"ood_score": 0.9, "uncertainty": 0.1, "sensor_quality": 1.0}) == 0
    for features in [
        {"predicted_tfv_benefit": 10, "joint_support": 1.0, "sensor_quality": 1.0},
        {"predicted_tfv_benefit": 80, "joint_support": 0.5, "sensor_quality": 1.0},
        {"predicted_tfv_benefit": 200, "joint_support": 0.7, "priority_storage_risk": 0.6, "sensor_quality": 1.0},
        {"predicted_tfv_benefit": 300, "joint_support": 0.9, "priority_storage_risk": 0.9, "sensor_quality": 1.0, "extreme_condition": "true"},
    ]:
        assert v32.adaptive_k_v32(features) in {0, 2, 4, 6, 8}


def test_applicability_gate_blocks_ood_before_candidate_generation() -> None:
    gate = v32.applicability_gate_v32({"ood_score": 0.8, "uncertainty": 0.1, "sensor_quality": 1.0, "preferred_fallback": "passive_anchor"})
    assert gate["domain_status"] == "out_of_domain"
    assert gate["k_limit"] == 0
    assert gate["fallback"] == "passive_anchor"


def test_action_cost_cannot_offset_pfv_safety_failure() -> None:
    decision = v32.action_cost_gate_v32(
        {"pfv_h30_pass": False, "conservative_tfv_benefit": 10000, "conservative_peak_benefit": 10000},
        {"changed_facility_count": 1, "total_variation": 1},
        {"minimum_material_benefit": 1},
    )
    assert decision["decision"] == "fallback"
    assert decision["reason"] == "pfv_h30_pass"


def test_action_cost_holds_when_benefit_is_not_material() -> None:
    prediction = {
        "pfv_h30_pass": True,
        "pfv_h60_pass": True,
        "pfv_h120_pass": True,
        "pfv_recovery_pass": True,
        "tfv_ucb_pass": True,
        "peak_ucb_pass": True,
        "conservative_tfv_benefit": 20,
        "conservative_peak_benefit": 0,
    }
    decision = v32.action_cost_gate_v32(prediction, {"changed_facility_count": 3, "direction_reversals": 2}, {"minimum_material_benefit": 25})
    assert decision["decision"] == "hold_or_fallback"


def test_v32_split_audit_blocks_missing_formal_rainfall_assets(tmp_path: Path) -> None:
    root = tmp_path / "out"
    split_dir = root / "formal_evaluation"
    split_dir.mkdir(parents=True)
    rows = []
    for i in range(60):
        split = "calibration_a_v32" if i < 12 else "locked_validation_b_v32" if i < 24 else "formal_blind_v32"
        rows.append(
            {
                "event_id": f"e{i}",
                "split": split,
                "rainfall_path": "" if i == 0 else str(tmp_path / f"r{i}.csv"),
                "rainfall_series_sha256": f"h{i}",
                "used_by_v31_formal": "false",
            }
        )
    with (split_dir / "evaluation_event_splits_v32.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"project:\n  output_root: {root.as_posix()}\n", encoding="utf-8")
    code, outputs = v32.audit_evaluation_splits_v32(cfg)
    assert code == 3
    audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
    assert audit["status"] == "blocked"
    assert audit["missing_rainfall_asset_events"] == ["e0"]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_v32_formal_comparison_reads_v32_manifest_and_extra_baselines(tmp_path: Path) -> None:
    root = tmp_path / "out"
    fe = root / "formal_evaluation"
    fe.mkdir(parents=True)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"project:\n  output_root: {root.as_posix()}\n", encoding="utf-8")
    (fe / "policy_lock_v32.json").write_text(json.dumps({"status": "pass", "formal_v32_allowed": True}), encoding="utf-8")
    (fe / "policy_lock_audit_v32.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    (fe / "formal_blind_v32_run_manifest.json").write_text(json.dumps({"status": "pass", "runtime_executed": True, "hydraulic_evidence_source": "authoritative_swmm"}), encoding="utf-8")
    (fe / "formal_blind_v32_extra_baseline_run_manifest.json").write_text(json.dumps({"status": "pass", "runtime_executed": True}), encoding="utf-8")
    core_rows = []
    extra_rows = []
    for event in ["e1", "e2"]:
        for policy, pfv in [("proposed_pfvfirst_dualfallback_v3", 10.0), ("internal_rules", 11.0), ("no_control", 13.0), ("passive_anchor", 12.0)]:
            core_rows.append({"event_id": event, "policy_id": policy, "PFV_m3": pfv, "TFV_m3": pfv * 10, "peak_TFV_rate": pfv / 2, "priority_flood_duration_min": 1, "recovery_time_min": 2, "action_changes": 3, "pump_starts": 0, "pump_stops": 0})
        for policy, pfv in [("auto_rbc", 12.5), ("efd_storage_priority", 12.2)]:
            extra_rows.append({"event_id": event, "policy_id": policy, "PFV_m3": pfv, "TFV_m3": pfv * 10, "peak_TFV_rate": pfv / 2, "priority_flood_duration_min": 1, "recovery_time_min": 2, "action_changes": 3, "pump_starts": 0, "pump_stops": 0})
    _write_csv(fe / "formal_blind_v32_event_policy_results.csv", core_rows)
    _write_csv(fe / "formal_blind_v32_extra_baseline_event_policy_results.csv", extra_rows)
    code, outputs = v32.build_formal_comparison_v32(cfg)
    assert code == 0
    report = json.loads(outputs["report"].read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    text = outputs["comparison"].read_text(encoding="utf-8")
    assert "auto_rbc" in text
    assert "efd_storage_priority" in text


def test_v32_formal_comparison_blocks_until_auto_rbc_efd_are_run(tmp_path: Path) -> None:
    root = tmp_path / "out"
    fe = root / "formal_evaluation"
    fe.mkdir(parents=True)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"project:\n  output_root: {root.as_posix()}\n", encoding="utf-8")
    (fe / "policy_lock_v32.json").write_text(json.dumps({"status": "pass", "formal_v32_allowed": True}), encoding="utf-8")
    (fe / "policy_lock_audit_v32.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    (fe / "formal_blind_v32_run_manifest.json").write_text(json.dumps({"status": "pass", "runtime_executed": True, "hydraulic_evidence_source": "authoritative_swmm"}), encoding="utf-8")
    _write_csv(fe / "formal_blind_v32_event_policy_results.csv", [{"event_id": "e1", "policy_id": "proposed_pfvfirst_dualfallback_v3", "PFV_m3": 1, "TFV_m3": 1, "peak_TFV_rate": 1}])
    code, outputs = v32.build_formal_comparison_v32(cfg)
    assert code == 3
    report = json.loads(outputs["report"].read_text(encoding="utf-8"))
    assert "formal_blind_v32_auto_rbc_efd_not_run" in report["blocking_reasons"]
