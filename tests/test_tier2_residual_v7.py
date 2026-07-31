from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sewerrtc.data.effect_dataset_merge import merge_effect_payloads
from sewerrtc.experiments.tier2_residual_v7 import (
    build_residual_specifications,
    freeze_dataset_manifest,
    select_fresh_event_roles,
    select_deployment_tier1_bases,
    select_safe_tier1_bases,
)
from sewerrtc.models.raw_joint_training import gate_aligned_selection_score
from sewerrtc.models.raw_joint_action_surrogate import RawJointActionSurrogate
from sewerrtc.experiments.targeted_joint_pairs import materialize_candidate


def _dataset(path: Path, event_ids: list[str], splits: list[str]) -> None:
    rows = len(event_ids)
    payload = {
        "event_ids": np.asarray(event_ids),
        "pair_ids": np.asarray([f"pair-{event_id}-{index}" for index, event_id in enumerate(event_ids)]),
        "state": np.zeros((rows, 2), dtype=np.float32),
        "candidate_action_seq": np.zeros((rows, 6, 36), dtype=np.float32),
        "reference_action_seq": np.zeros((rows, 6, 36), dtype=np.float32),
        "rain_seq": np.zeros((rows, 6, 1), dtype=np.float32),
        "reference_risk_rate_seq": np.zeros((rows, 6, 3), dtype=np.float32),
        "delta_risk_rate_seq": np.zeros((rows, 6, 3), dtype=np.float32),
        "priority_depth_seq": np.zeros((rows, 6), dtype=np.float32),
        "storage_level_seq": np.zeros((rows, 6), dtype=np.float32),
        "target_state_seq": np.zeros((rows, 6, 2), dtype=np.float32),
        "split": np.asarray(splits),
        "candidate_kind": np.asarray(["legacy"] * rows),
        "candidate_family": np.asarray(["legacy"] * rows),
        "phase": np.asarray(["peak"] * rows),
        "checkpoint_id": np.asarray([f"checkpoint-{index}" for index in range(rows)]),
        "source_dataset": np.asarray(["test"] * rows),
        "node_ids": np.asarray(["N1", "N2"]),
        "action_ids": np.asarray([f"A{index}" for index in range(36)]),
        "label_semantics": np.asarray("same_state_candidate_minus_no_control"),
        "horizon_steps": np.asarray(6),
        "risk_label_channels": np.asarray(["PFV_rate", "TFV_rate", "running_peak_TFV_rate"]),
        "peak_label_definition": np.asarray("test"),
    }
    np.savez_compressed(path, **payload)


def test_freeze_manifest_records_immutable_1451_row_contract(tmp_path: Path):
    dataset = tmp_path / "base.npz"
    _dataset(dataset, ["E1"] * 1451, ["train"] * 1451)

    manifest = freeze_dataset_manifest(dataset, intended_rows=1451)

    assert manifest["frozen"] is True
    assert manifest["rows"] == 1451
    assert manifest["sha256"]
    assert manifest["intended_use"] == "v7_development_warm_start_only"


def test_fresh_event_roles_are_disjoint_from_base_and_locked_validation_is_new():
    rainfall = pd.DataFrame(
        [
            {"event_id": f"T{rp}_D{duration}_{pattern}", "rain_id": f"T{rp}", "duration_min": duration, "pattern": pattern}
            for rp in (20, 30, 50, 75, 100)
            for duration in (75, 150, 240)
            for pattern in ("chicago_center", "chicago_late", "block", "double_peak")
        ]
    )
    base_events = {"T20_D75_chicago_center", "T50_D150_block"}

    roles = select_fresh_event_roles(
        rainfall,
        excluded_events=base_events,
        fit_events=8,
        calibration_events=4,
        validation_events=6,
        seed=20260714,
    )

    assert len(roles) == 18
    assert set(roles["event_id"]).isdisjoint(base_events)
    assert roles.groupby("role")["event_id"].nunique().to_dict() == {
        "calibration": 4,
        "fit": 8,
        "locked_validation": 6,
    }
    assert not roles["event_id"].duplicated().any()


def test_merge_relabels_frozen_base_as_train_and_uses_only_locked_validation(tmp_path: Path):
    base_path = tmp_path / "base.npz"
    supplement_path = tmp_path / "supplement.npz"
    _dataset(base_path, ["OLD_TRAIN", "OLD_VALID"], ["train", "validation"])
    _dataset(
        supplement_path,
        ["NEW_FIT", "NEW_CAL", "NEW_LOCKED"],
        ["train", "train", "validation"],
    )
    base = np.load(base_path, allow_pickle=True)
    supplement = np.load(supplement_path, allow_pickle=True)

    payload, report = merge_effect_payloads(
        base,
        supplement,
        base_split_policy="all_train",
        locked_validation_events={"NEW_LOCKED"},
    )

    by_event = dict(zip(payload["event_ids"].astype(str), payload["split"].astype(str)))
    assert by_event == {
        "OLD_TRAIN": "train",
        "OLD_VALID": "train",
        "NEW_FIT": "train",
        "NEW_CAL": "train",
        "NEW_LOCKED": "validation",
    }
    assert report["locked_validation_events"] == ["NEW_LOCKED"]


def test_safe_tier1_selection_is_lexicographic_pfv_peak_then_tfv():
    frame = pd.DataFrame(
        [
            {"event_id": "E1", "phase": "peak", "candidate_id": "unsafe_pfv", "delta_PFV": 101.0, "delta_TFV": -900.0, "delta_peak": -2.0},
            {"event_id": "E1", "phase": "peak", "candidate_id": "unsafe_peak", "delta_PFV": 0.0, "delta_TFV": -800.0, "delta_peak": 0.1},
            {"event_id": "E1", "phase": "peak", "candidate_id": "safe_small", "delta_PFV": 20.0, "delta_TFV": -100.0, "delta_peak": 0.0},
            {"event_id": "E1", "phase": "peak", "candidate_id": "safe_best", "delta_PFV": 50.0, "delta_TFV": -500.0, "delta_peak": -0.1},
        ]
    )

    selected = select_safe_tier1_bases(frame, pfv_abs_margin_m3=100.0, pfv_rel_margin=0.02)

    assert selected.iloc[0]["candidate_id"] == "safe_best"


def test_locked_validation_base_selection_does_not_use_validation_outcomes():
    rows = []
    for mode, train_tfv in (("mode_a", -500.0), ("mode_b", -100.0)):
        rows.append({
            "event_id": "FIT", "phase": "peak", "role": "fit", "tier1_mode": mode,
            "reference_PFV": 1000.0, "delta_PFV": 0.0, "delta_TFV": train_tfv, "delta_peak": -1.0,
        })
        rows.append({
            "event_id": "LOCKED", "phase": "peak", "role": "locked_validation", "tier1_mode": mode,
            "reference_PFV": 1000.0, "delta_PFV": 9999.0 if mode == "mode_a" else -9999.0,
            "delta_TFV": 9999.0 if mode == "mode_a" else -9999.0, "delta_peak": 9999.0 if mode == "mode_a" else -9999.0,
        })

    selected, policy = select_deployment_tier1_bases(
        pd.DataFrame(rows), pfv_abs_margin_m3=100.0, pfv_rel_margin=0.02
    )

    locked = selected[selected["event_id"].eq("LOCKED")]
    assert policy["peak"] == "mode_a"
    assert locked.iloc[0]["tier1_mode"] == "mode_a"
    assert locked.iloc[0]["selection_basis"] == "fit_only_phase_policy"


def test_residual_specs_match_online_tier2_family_and_budget():
    action_ids = [
        "RTC_IN_01", "RTC_OUT_01", "RTC_IN_02", "RTC_OUT_02", "RTC_IN_03", "RTC_OUT_03",
        "HS2512760.1", "gbz1.8", "ADD301.2", "ADD301.3",
    ] + [f"LEGACY_{index}" for index in range(26)]
    reference = np.ones((6, 36), dtype=np.float32)
    reference[:, action_ids.index("ADD301.2")] = 0.0
    reference[:, action_ids.index("ADD301.3")] = 1.0
    base_profiles = {"LEGACY_0": [-0.05, -0.05, -0.05, 0.0, 0.0, 0.0]}

    specs = build_residual_specifications(
        action_ids=action_ids,
        no_control_reference=reference,
        tier1_signed_profiles=base_profiles,
        phase="peak",
        magnitude=0.10,
    )

    assert len(specs) == 20
    assert all(spec["kind"] == "legacy_plus_new_residual" for spec in specs)
    assert all(spec["online_candidate_eligible"] is True for spec in specs)
    assert any("target_profiles" in spec for spec in specs)
    assert any(len(spec.get("actuators", [])) >= 2 for spec in specs)
    tier1 = materialize_candidate(
        reference,
        action_ids=action_ids,
        specification={"signed_profiles": base_profiles},
    )
    assert all(
        np.any(
            np.abs(
                materialize_candidate(reference, action_ids=action_ids, specification=spec) - tier1
            ) > 1.0e-7
        )
        for spec in specs
    )


def test_gate_aligned_selection_uses_weakest_deployment_metric():
    strong_average_weak_peak = {
        "tfv_direction": 0.95,
        "peak_direction": 0.60,
        "tfv_balanced_accuracy": 0.95,
        "peak_balanced_accuracy": 0.60,
        "peak_unsafe_recall": 0.60,
    }
    balanced = {
        "tfv_direction": 0.82,
        "peak_direction": 0.86,
        "tfv_balanced_accuracy": 0.82,
        "peak_balanced_accuracy": 0.86,
        "peak_unsafe_recall": 0.91,
    }
    thresholds = {
        "tfv_direction": 0.80,
        "peak_direction": 0.85,
        "tfv_balanced_accuracy": 0.80,
        "peak_balanced_accuracy": 0.85,
        "peak_unsafe_recall": 0.90,
    }

    assert gate_aligned_selection_score(balanced, thresholds) > gate_aligned_selection_score(
        strong_average_weak_peak, thresholds
    )


def test_gate_aligned_selection_ignores_disabled_thresholds():
    metrics = {
        "tfv_direction": 0.61,
        "peak_direction": 0.71,
        "tfv_balanced_accuracy": 0.0,
        "peak_balanced_accuracy": 0.0,
        "peak_unsafe_recall": 0.81,
    }
    thresholds = {
        "tfv_direction": 0.60,
        "peak_direction": 0.70,
        "tfv_balanced_accuracy": 0.0,
        "peak_balanced_accuracy": 0.0,
        "peak_unsafe_recall": 0.80,
    }

    assert gate_aligned_selection_score(metrics, thresholds) > 1.0


def test_peak_direction_sampling_weight_prioritizes_non_deadband_peak_rows():
    import importlib.util

    script = Path("scripts/93_train_raw_joint_action_surrogate_v3.py")
    spec = importlib.util.spec_from_file_location("raw_joint_v3", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weights = np.ones(4, dtype=float) / 4.0
    targets = np.asarray(
        [
            [0.0, 0.0, 0.00],
            [0.0, 0.0, 0.05],
            [0.0, 0.0, 0.20],
            [0.0, 0.0, -0.30],
        ],
        dtype=float,
    )

    out = module._apply_peak_direction_sampling_weight(
        weights,
        targets,
        peak_tolerance=0.1,
        peak_multiplier=3.0,
    )

    assert out[2] > out[0]
    assert out[3] > out[1]
    assert np.isclose(out.sum(), 1.0)


def test_direction_v6_outputs_direction_logits_and_preserves_zero_action_semantics():
    model = RawJointActionSurrogate(
        n_nodes=4,
        n_actions=3,
        node_static_dim=2,
        actuator_feature_dim=2,
        horizon_steps=6,
        hidden_dim=16,
        heads=2,
        architecture_version="causal_phase_direction_v6",
    )
    candidate = torch.zeros((2, 6, 3), dtype=torch.float32)
    reference = torch.zeros((2, 6, 3), dtype=torch.float32)
    candidate[1, 2:, 1] = 0.2
    output = model(
        state=torch.zeros((2, 4), dtype=torch.float32),
        candidate_action_seq=candidate,
        reference_action_seq=reference,
        rain_seq=torch.zeros((2, 6, 1), dtype=torch.float32),
        actuator_mask=torch.ones((2, 3), dtype=torch.float32),
        actuator_features=torch.zeros((3, 2), dtype=torch.float32),
        node_static=torch.zeros((4, 2), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        action_node_map=torch.ones((3, 4), dtype=torch.float32) / 4.0,
        priority_indices=torch.tensor([0, 1], dtype=torch.long),
        storage_indices=torch.tensor([2], dtype=torch.long),
        phase_index=torch.tensor([2, 2], dtype=torch.long),
    )

    assert output["direction_classification_logits"].shape == (2, 3)
    assert torch.allclose(output["direction_classification_logits"][0], torch.zeros(3), atol=1.0e-6)


def test_v7_runtime_config_points_to_new_model_and_locked_report():
    import yaml

    path = Path("configs/wuhan_project6_36_hierarchical_residual_v7.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    temporal = config["controller"]["temporal_joint"]
    hierarchical = temporal["hierarchical"]

    assert "tier2_residual_v7" in temporal["model_path"]
    assert "tier2_residual_v7" in hierarchical["residual_validation_report"]
    assert hierarchical["require_residual_validation"] is True
