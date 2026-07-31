from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v7_uses_explicit_sr0p20_gat_path_and_relaxed_effect_gate():
    from sewerrtc.io.project_paths import load_config, resolve_gat_model_path

    cfg = load_config(ROOT / "configs" / "wuhan_project6_36_hierarchical_residual_v7.yaml")
    gat = resolve_gat_model_path(cfg)
    validation = cfg["controller"]["temporal_joint"]["training_validation"]

    assert "research_sensor_sweep" in str(gat).replace("\\", "/")
    assert "sr0p20" in str(gat).replace("\\", "/")
    assert validation["min_tfv_direction_accuracy"] == 0.60
    assert validation["min_peak_direction_accuracy"] == 0.70
    assert validation["min_pfv_unsafe_recall"] == 0.80
    assert validation["min_peak_unsafe_recall"] == 0.80
    assert validation["max_pfv_false_safe_rate"] == 0.10


def test_formal_no_control_gate_uses_primary_and_stress_contract():
    from sewerrtc.io.project_paths import load_config

    gate = _load_script("75_no_control_repair_gate.py")
    cfg = load_config(ROOT / "configs" / "wuhan_project6_36_hierarchical_residual_v7.yaml")
    rows = []
    primary_periods = ["T5", "T10", "T20", "T30", "T50"]
    for i in range(20):
        event_id = f"{primary_periods[i % len(primary_periods)]}_D150_chicago_center_{i}"
        rows.extend(
            [
                {"event_id": event_id, "policy_id": "no_control", "PFV": 1000.0, "TFV": 10000.0, "peak_TFV_rate": 100.0, "event_risk_class": "medium_risk_event"},
                {"event_id": event_id, "policy_id": "proposed_gat_mpc", "PFV": 1010.0, "TFV": 9000.0, "peak_TFV_rate": 99.0, "event_risk_class": "medium_risk_event"},
            ]
        )
    for i, period in enumerate(["T75", "T100", "T75", "T100"]):
        event_id = f"{period}_D240_block_{i}"
        rows.extend(
            [
                {"event_id": event_id, "policy_id": "no_control", "PFV": 2000.0, "TFV": 50000.0, "peak_TFV_rate": 300.0, "event_risk_class": "high_risk_event"},
                {"event_id": event_id, "policy_id": "proposed_gat_mpc", "PFV": 2010.0, "TFV": 51000.0, "peak_TFV_rate": 299.0, "event_risk_class": "high_risk_event"},
            ]
        )

    report = gate.evaluate_repair_gate(pd.DataFrame(rows), cfg)

    assert report["passed"] is True
    assert report["gate_profile"] == "formal"
    assert report["primary_baseline_comparisons"][0]["paired_events"] == 20
    assert report["stress_baseline_comparisons"][0]["paired_events"] == 4
    assert {row["pfv_rel_margin"] for row in report["pfv_sensitivity_vs_no_control"]} == {0.005, 0.01, 0.02}


def test_smoke_functionality_gate_requires_temporal_and_simultaneous_actions(tmp_path: Path):
    from sewerrtc.evaluation.smoke_functionality_gate import evaluate_smoke_functionality

    proposed = tmp_path / "proposed"
    proposed.mkdir()
    control = pd.DataFrame(
        {
            "actuator_id": ["A", "B", "ADD301.2"],
            "link_type": ["orifice", "weir", "pump"],
            "asset_role": ["regulator", "regulator", "pump"],
        }
    )
    seq = "[[0.9, 0.8, 1.0], [0.9, 0.8, 1.0], [1.0, 0.8, 1.0]]"
    first = "[0.9, 0.8, 1.0]"
    for event_id in ("T5_D75_chicago_center", "T20_D150_block", "T100_D240_chicago_late"):
        pd.DataFrame(
            {
                "event_id": [event_id],
                "selected_action_sequence": [seq],
                "executed_first_action": [first],
                "simultaneous_actuator_count": [2],
            }
        ).to_csv(proposed / f"{event_id}__controller_history.csv", index=False)

    report = evaluate_smoke_functionality(
        run_dir=tmp_path,
        control_table=control,
        required_return_period_groups={"light_or_t10": ["T5", "T10"], "medium": ["T20"], "severe": ["T100"]},
        binary_pump_ids=["ADD301.2"],
    )

    assert report["passed"] is True
    assert report["action_written_rows"] == 3
    assert report["temporal_action_rows"] == 3
    assert report["simultaneous_action_rows"] == 3
    assert report["fractional_binary_pump_rows"] == 0


def test_mpc_readiness_accepts_hierarchical_core26_residual10_mode(tmp_path: Path):
    from sewerrtc.data.three_step_research_builders import evaluate_mpc_readiness

    legacy_model = tmp_path / "legacy.pt"
    legacy_model.write_bytes(b"pt")
    report = tmp_path / "report.json"
    report.write_text(
        """
        {
          "model": "residual.pt",
          "validation_gate_passed": false,
          "validation_gate_failures": [{"check": "TFV_direction", "reason": "threshold_not_met"}],
          "rolling_horizon_smoke_eligibility": {"passed": 0}
        }
        """,
        encoding="utf-8",
    )
    result = evaluate_mpc_readiness(
        config={
            "project_root": str(tmp_path),
            "controller": {
                "mode": "hierarchical_core26_residual10",
                "reference_policy_for_constraints": "online_predicted_default",
                "temporal_joint": {
                    "model_path": "residual.pt",
                    "legacy_groups": [{"label": "core"}],
                    "safety": {"pfv_abs_margin_m3": 100.0},
                    "hierarchical": {
                        "enabled": True,
                        "legacy_model_path": str(legacy_model),
                        "residual_actuator_ids": [f"R{i:02d}" for i in range(10)],
                    },
                },
            },
        },
        model_report_path=report,
    )

    assert "controller_mode_not_temporal_joint_36" not in result["blocking_reasons"]
    assert result["passed"] is True
    assert result["deployment_mode"] == "tier1_only"
    assert result["tier2_residual_allowed"] is False
