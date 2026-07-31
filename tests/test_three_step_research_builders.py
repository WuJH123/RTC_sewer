from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sewerrtc.io.project_paths import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_sensor_sweep_config_inherits_base_from_nested_output_dir(tmp_path: Path) -> None:
    out_config_dir = tmp_path / "nested" / "sensor_configs"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "100_write_sensor_sweep_configs.py"),
            "--base-config",
            "configs/wuhan_project6_36_temporal_joint.yaml",
            "--ratios",
            "0.10",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--out-config-dir",
            str(out_config_dir),
            "--out-root",
            str(tmp_path / "outputs"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    cfg = load_config(out_config_dir / "wuhan_sr0p10.yaml")
    assert cfg["experiment"]["sensor_ratio"] == 0.10
    assert Path(cfg["network"]["inp"]).name == "wuhan_v8_storage_retrofit.inp"


def _detail(path, event_id: str, policy_id: str, nodes: list[str], actions: list[str], rows: int = 8) -> None:
    data = {
        "event_id": [event_id] * rows,
        "policy_id": [policy_id] * rows,
        "elapsed_min": [5 * i for i in range(rows)],
        "rainfall_mm_h": [float(i % 3) for i in range(rows)],
    }
    for n, node in enumerate(nodes):
        data[f"h:{node}"] = [float(n + i) for i in range(rows)]
        pattern = [0.0, float(n), float(n + 1), 0.0, 0.0, 1.0, 0.0, 0.0]
        data[f"flood:{node}"] = [pattern[i % len(pattern)] for i in range(rows)]
    for a, aid in enumerate(actions):
        data[f"a:{aid}"] = [1.0 - 0.01 * ((a + i) % 3) for i in range(rows)]
    pd.DataFrame(data).to_csv(path, index=False)


def test_mixed_gat_cache_uses_manifest_order_and_writes_required_keys(tmp_path):
    from sewerrtc.data.three_step_research_builders import build_mixed_gat_cache

    nodes = ["N1", "N2"]
    actions = ["A1", "A2"]
    detail = tmp_path / "E1__no_control_detail.csv"
    _detail(detail, "E1", "no_control", nodes, actions)
    manifest = pd.DataFrame({
        "detail_file": [str(detail)],
        "gat_use": [True],
        "event_id": ["E1"],
        "policy_id": ["no_control"],
    })
    manifest_path = tmp_path / "gat_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    out_npz = tmp_path / "cache" / "transition_cache.npz"

    report = build_mixed_gat_cache(
        manifest_path=manifest_path,
        out_npz=out_npz,
        base_node_cols=[f"h:{n}" for n in nodes],
        time_stride=2,
    )

    assert report["samples"] == 4
    with np.load(out_npz, allow_pickle=True) as data:
        assert data["state"].shape == (4, 2)
        assert data["rain"].shape == (4, 1)
        assert data["node_cols"].tolist() == ["h:N1", "h:N2"]
        assert set(data["policy_ids"].astype(str)) == {"no_control"}


def test_temporal_action_pretrain_dataset_defaults_to_lightweight_targets(tmp_path):
    from sewerrtc.data.three_step_research_builders import build_temporal_action_pretrain_dataset

    nodes = ["N1", "N2"]
    actions = [f"A{i:02d}" for i in range(36)]
    detail = tmp_path / "E1__legacy_detail.csv"
    _detail(detail, "E1", "legacy", nodes, actions, rows=9)
    manifest = pd.DataFrame({
        "detail_file": [str(detail)],
        "action_learning_use": [True],
        "event_id": ["E1"],
        "policy_id": ["legacy"],
        "effect_label_role": ["observational_dynamics_pretraining"],
    })
    manifest_path = tmp_path / "action_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    out_npz = tmp_path / "action_pretrain.npz"

    report = build_temporal_action_pretrain_dataset(
        manifest_path=manifest_path,
        out_npz=out_npz,
        base_node_cols=[f"h:{n}" for n in nodes],
        canonical_action_ids=actions,
        priority_nodes=["N1"],
        local_node_cols=["h:N1"],
        horizon_steps=6,
        time_stride=2,
    )

    assert report["samples"] == 2
    assert report["target_mode"] == "risk_local"
    with np.load(out_npz, allow_pickle=True) as data:
        assert data["state"].shape == (2, 2)
        assert data["candidate_action_seq"].shape == (2, 6, 36)
        assert data["rain_seq"].shape == (2, 6, 1)
        assert "target_state_seq" not in data.files
        assert data["risk_rate_seq"].shape == (2, 6, 3)
        np.testing.assert_allclose(
            data["risk_rate_seq"][:, :, 2],
            np.maximum.accumulate(data["risk_rate_seq"][:, :, 1], axis=1),
        )
        assert data["local_state_seq"].shape == (2, 6, 1)
        assert data["action_ids"].tolist() == actions
    assert report["risk_label_channels"][2] == "running_peak_TFV_rate"


def test_temporal_action_pretrain_dataset_can_opt_in_full_state_targets(tmp_path):
    from sewerrtc.data.three_step_research_builders import build_temporal_action_pretrain_dataset

    nodes = ["N1", "N2"]
    actions = [f"A{i:02d}" for i in range(36)]
    detail = tmp_path / "E1__legacy_detail.csv"
    _detail(detail, "E1", "legacy", nodes, actions, rows=8)
    manifest = pd.DataFrame({
        "detail_file": [str(detail)],
        "action_learning_use": [True],
        "event_id": ["E1"],
        "policy_id": ["legacy"],
        "effect_label_role": ["observational_dynamics_pretraining"],
    })
    manifest_path = tmp_path / "action_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    out_npz = tmp_path / "action_pretrain_full.npz"

    report = build_temporal_action_pretrain_dataset(
        manifest_path=manifest_path,
        out_npz=out_npz,
        base_node_cols=[f"h:{n}" for n in nodes],
        canonical_action_ids=actions,
        horizon_steps=6,
        target_mode="full_state",
    )

    assert report["target_mode"] == "full_state"
    with np.load(out_npz, allow_pickle=True) as data:
        assert data["target_state_seq"].shape == (2, 6, 2)
        assert data["risk_rate_seq"].shape == (2, 6, 3)


def test_temporal_action_pretrain_dataset_can_stream_to_npz_shards(tmp_path):
    from sewerrtc.data.three_step_research_builders import build_temporal_action_pretrain_dataset

    nodes = ["N1", "N2"]
    actions = [f"A{i:02d}" for i in range(36)]
    detail = tmp_path / "E1__legacy_detail.csv"
    _detail(detail, "E1", "legacy", nodes, actions, rows=9)
    manifest = pd.DataFrame({
        "detail_file": [str(detail)],
        "action_learning_use": [True],
        "event_id": ["E1"],
        "policy_id": ["legacy"],
        "effect_label_role": ["observational_dynamics_pretraining"],
    })
    manifest_path = tmp_path / "action_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    out_npz = tmp_path / "action_pretrain_sharded.npz"

    report = build_temporal_action_pretrain_dataset(
        manifest_path=manifest_path,
        out_npz=out_npz,
        base_node_cols=[f"h:{n}" for n in nodes],
        canonical_action_ids=actions,
        priority_nodes=["N1"],
        local_node_cols=["h:N1"],
        horizon_steps=6,
        chunk_size_samples=2,
    )

    assert report["samples"] == 3
    assert report["storage_mode"] == "sharded_npz"
    assert report["shard_count"] == 2
    with np.load(out_npz, allow_pickle=True) as index:
        shard_files = [Path(value) for value in index["shard_files"].tolist()]
        assert index["action_ids"].tolist() == actions
        assert int(index["sample_count"][0]) == 3
    assert all(path.exists() for path in shard_files)
    with np.load(shard_files[0], allow_pickle=True) as first:
        assert first["state"].shape == (2, 2)
        assert first["candidate_action_seq"].shape == (2, 6, 36)
        assert first["local_state_seq"].shape == (2, 6, 1)
    with np.load(shard_files[1], allow_pickle=True) as second:
        assert second["state"].shape == (1, 2)
        assert second["event_ids"].tolist() == ["E1"]


def test_mpc_preflight_blocks_smoke_when_validation_gate_false(tmp_path):
    from sewerrtc.data.three_step_research_builders import evaluate_mpc_readiness

    report = tmp_path / "train_report.json"
    report.write_text(json.dumps({"validation_gate_passed": False, "validation_gate_failures": [{"check": "TFV_direction"}]}))

    result = evaluate_mpc_readiness(
        config={"controller": {"mode": "temporal_joint_36", "temporal_joint": {"safety": {"pfv_abs_margin_m3": 100.0}}}},
        model_report_path=report,
    )

    assert not result["passed"]
    assert "model_validation_gate_false" in result["blocking_reasons"]
    assert result["closed_loop_allowed"] is False


def test_mpc_preflight_requires_explicit_effect_direction_gate(tmp_path):
    from sewerrtc.data.three_step_research_builders import evaluate_mpc_readiness

    report = tmp_path / "train_report.json"
    report.write_text(json.dumps({"validation_gate_passed": True}))
    config = {
        "controller": {
            "mode": "temporal_joint_36",
            "reference_policy_for_constraints": "online_predicted_default",
            "temporal_joint": {"safety": {"pfv_abs_margin_m3": 100.0}},
        }
    }

    missing = evaluate_mpc_readiness(config=config, model_report_path=report)
    assert not missing["passed"]
    assert "missing_effect_direction_gate" in missing["blocking_reasons"]

    report.write_text(json.dumps({
        "validation_gate_passed": True,
        "rolling_horizon_smoke_eligibility": {
            "passed": True,
            "required_checks": {
                "PFV_noninferiority": True,
                "TFV_improvement_direction": True,
                "peak_safety_direction": True,
            },
        },
    }))
    ready = evaluate_mpc_readiness(config=config, model_report_path=report)
    assert ready["passed"]


def test_hierarchical_preflight_allows_tier1_but_blocks_unvalidated_residual(tmp_path):
    from sewerrtc.data.three_step_research_builders import evaluate_mpc_readiness

    legacy_model = tmp_path / "legacy.pt"
    legacy_model.write_bytes(b"legacy")
    residual_model = tmp_path / "residual.pt"
    residual_model.write_bytes(b"residual")
    report = tmp_path / "train_report.json"
    report.write_text(json.dumps({
        "model": str(residual_model),
        "validation_gate_passed": False,
        "validation_gate_failures": [{"check": "TFV_direction"}],
        "rolling_horizon_smoke_eligibility": {"passed": False},
    }))
    config = {
        "controller": {
            "mode": "temporal_joint_36",
            "reference_policy_for_constraints": "online_predicted_default",
            "temporal_joint": {
                "model_path": str(residual_model),
                "safety": {"pfv_abs_margin_m3": 100.0},
                "legacy_groups": [["A", "B"]],
                "hierarchical": {
                    "enabled": True,
                    "legacy_model_path": str(legacy_model),
                    "require_residual_validation": True,
                    "residual_actuator_ids": ["C"],
                },
            },
        }
    }

    result = evaluate_mpc_readiness(config=config, model_report_path=report)

    assert result["passed"]
    assert result["closed_loop_allowed"]
    assert result["deployment_mode"] == "tier1_only"
    assert result["tier2_residual_allowed"] is False
    assert "model_validation_gate_false" in result["residual_blocking_reasons"]


def test_gate_comparison_identifies_26_partial_success_and_36_failure():
    from sewerrtc.data.three_step_research_builders import summarize_gate_comparison

    gate26 = {
        "passed": False,
        "reasons": ["PFV mean increase vs no_control 0.712% > 0.500%"],
        "baseline_comparisons": [
            {"baseline_policy": "no_control", "PFV_mean_reduction_pct": -0.7, "TFV_mean_reduction_pct": 9.8, "peak_mean_reduction_pct": 24.0, "PFV_worse_frac_noninferiority": 0.3},
            {"baseline_policy": "internal_rules", "PFV_mean_reduction_pct": 37.0, "TFV_mean_reduction_pct": 12.0, "peak_mean_reduction_pct": 26.0},
        ],
    }
    gate36 = {
        "passed": False,
        "reasons": ["TFV mean reduction vs no_control -666.471% < 3.000%"],
        "baseline_comparisons": [
            {"baseline_policy": "no_control", "PFV_mean_reduction_pct": -88.0, "TFV_mean_reduction_pct": -666.0, "peak_mean_reduction_pct": -211.0, "PFV_worse_frac_noninferiority": 1.0},
        ],
    }

    summary = summarize_gate_comparison(gate26, gate36)

    assert summary["v8_26"]["interpretation"] == "system_repair_success_but_strict_pfv_noninferiority_failed"
    assert summary["temporal_36"]["interpretation"] == "systemic_failure_vs_no_control"
