from __future__ import annotations

import csv
import json
from pathlib import Path

from sewerrtc.prompt3 import action_effect_v33 as v33


def test_v33_three_metric_gate_blocks_tfv_or_peak_regression() -> None:
    pred = {
        "ucb_delta_PFV_vs_internal": -1,
        "ucb_delta_TFV_vs_internal": 1,
        "ucb_delta_peak_vs_internal": -1,
        "ucb_delta_PFV_vs_fallback": -1,
        "ucb_delta_TFV_vs_fallback": -1,
        "ucb_delta_peak_vs_fallback": -1,
    }
    gate = v33.three_metric_gate_v33(pred)
    assert gate["pass"] is False
    assert gate["first_failure"] == "tfv_internal"


def test_v33_pfv_budget_one_step_no_recharge_and_event_reset() -> None:
    b1 = v33.EventPfvBudgetV33(10, 2)
    assert b1.evaluate(-100)["budget_after"] == 8
    assert b1.evaluate(3)["budget_after"] == 5
    assert b1.evaluate(6)["allowed"] is False
    b2 = v33.EventPfvBudgetV33(10, 2)
    assert b2.evaluate(3)["budget_before"] == 8


def test_v33_action_cost_is_hard_gate_after_safety() -> None:
    pred = {
        "ucb_delta_PFV_vs_internal": -1,
        "ucb_delta_TFV_vs_internal": -1,
        "ucb_delta_peak_vs_internal": -1,
        "ucb_delta_PFV_vs_fallback": -1,
        "ucb_delta_TFV_vs_fallback": -1,
        "ucb_delta_peak_vs_fallback": -1,
    }
    decision = v33.action_cost_hard_gate_v33(pred, {"executed_changed_facilities": 8, "reversals": 2}, {"minimum_material_benefit": 25, "minimum_benefit_cost_ratio": 1.5})
    assert decision["decision"] == "hold_previous_readback"


def test_v33_adaptive_k_allowed_values_and_pre_score_fallback() -> None:
    assert v33.adaptive_k_v33({"ood_score": 0.9, "uncertainty": 0.1, "sensor_quality": 1.0}) == 0
    for features in [
        {"conservative_tfv_benefit": 30, "joint_support": 0.2},
        {"conservative_tfv_benefit": 120, "joint_support": 0.7},
        {"conservative_tfv_benefit": 300, "joint_support": 0.9, "priority_storage_risk": 0.9, "extreme_condition": "true"},
    ]:
        assert v33.adaptive_k_v33(features) in {0, 2, 4, 6, 8}


def test_v33_risk_transfer_labels() -> None:
    labels = v33.risk_transfer_labels({"PFV_m3": 9, "TFV_m3": 20, "peak_TFV_rate": 4, "internal_PFV_m3": 10, "internal_TFV_m3": 18, "internal_peak_TFV_rate": 3, "action_changes": 200})
    assert labels["PFV_good_TFV_bad"] == "true"
    assert labels["PFV_good_peak_bad"] == "true"
    assert labels["non_priority_risk_transfer"] == "true"
    assert labels["low_benefit_high_action"] == "true"


def test_v33_split_audit_blocks_v31_v32_formal_overlap(tmp_path: Path) -> None:
    root = tmp_path / "out"
    fe = root / "formal_evaluation"
    fe.mkdir(parents=True)
    rows = []
    for i in range(60):
        split = "calibration_a_v33" if i < 12 else "locked_validation_b_v33" if i < 24 else "formal_blind_v33"
        rows.append({"event_id": f"e{i}", "split": split, "rainfall_path": str(tmp_path / f"r{i}.csv"), "rainfall_series_sha256": f"h{i}", "used_by_v31_formal": "true" if i == 0 else "false", "used_by_v32_formal": "false", "used_by_round5": "false"})
    with (fe / "evaluation_event_splits_v33.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"project:\n  output_root: {root.as_posix()}\n", encoding="utf-8")
    code, outputs = v33.audit_evaluation_splits_v33(cfg)
    assert code == 3
    audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
    assert audit["leakage_overlap_events"] == ["e0"]


def _write_v33_formal_prereqs(tmp_path: Path) -> Path:
    root = tmp_path / "out"
    fe = root / "formal_evaluation"
    models = root / "action_effect_models"
    fe.mkdir(parents=True)
    models.mkdir(parents=True)
    rows = []
    for i in range(60):
        if i < 12:
            split = "calibration_a_v33"
        elif i < 24:
            split = "locked_validation_b_v33"
        else:
            split = "formal_blind_v33"
        rain = tmp_path / f"rain_{i}.csv"
        rain.write_text("ts,val\n0,0\n", encoding="utf-8")
        rows.append(
            {
                "event_id": f"e{i}",
                "split": split,
                "rainfall_path": str(rain),
                "rainfall_series_sha256": f"h{i}",
                "used_by_v31_formal": "false",
                "used_by_v32_formal": "false",
                "used_by_round5": "false",
            }
        )
    with (fe / "evaluation_event_splits_v33.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (fe / "evaluation_event_split_audit_v33.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    (models / "model_gate_v33.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"project:\n  output_root: {root.as_posix()}\n", encoding="utf-8")
    return cfg


def test_v33_calibration_contract_dry_run_replaces_old_placeholder(tmp_path: Path) -> None:
    cfg = _write_v33_formal_prereqs(tmp_path)
    code, outputs = v33.calibration_a_v33(cfg, max_events=1, workers=1, resume=False, contract_dry_run=True)
    assert code == 0
    report = json.loads(outputs["report"].read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["contract_dry_run"] is True
    assert "not_executed_by_codex" not in outputs["report"].read_text(encoding="utf-8")


def test_v33_dependency_chain_blocks_validation_policy_lock_and_formal(tmp_path: Path) -> None:
    cfg = _write_v33_formal_prereqs(tmp_path)
    code_lv, outputs_lv = v33.locked_validation_b_v33(cfg, max_events=1, workers=1, resume=False, contract_dry_run=True)
    assert code_lv == 3
    blocked_lv = json.loads(outputs_lv["report"].read_text(encoding="utf-8"))
    assert "calibration_a_v33_not_pass" in blocked_lv["blocking_reasons"]

    code_lock, outputs_lock = v33.policy_lock_v33(cfg)
    assert code_lock == 3
    lock = json.loads(outputs_lock["lock"].read_text(encoding="utf-8"))
    assert "calibration_a_v33_not_pass" in lock["blocking_reasons"]
    assert lock["formal_v33_allowed"] is False

    code_formal, outputs_formal = v33.formal_blind_v33(cfg, max_events=1, workers=1, resume=False, contract_dry_run=True)
    assert code_formal == 3
    blocked_formal = json.loads(outputs_formal["report"].read_text(encoding="utf-8"))
    assert "locked_validation_b_v33_not_pass" in blocked_formal["blocking_reasons"] or "policy_lock_v33_not_pass" in blocked_formal["blocking_reasons"]


def test_v33_extra_baseline_run_tag_stays_within_windows_path_budget() -> None:
    event_id = "V31_RP10_D2H_P65_v31_front_back_split_086"
    root = Path(r"E:\RTC_sewer\Project6\outputs\closed_loop_paired_no_controls\formal")
    recovery = (
        root
        / v33.EXTRA_BASELINE_RUN_TAG_V33
        / "baselines"
        / "recovery"
        / event_id
        / f"{event_id}__efd_storage_priority__recovery.json"
    )
    assert len(str(recovery)) <= 240
