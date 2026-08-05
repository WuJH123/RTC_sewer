from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from sewerrtc.v4.models_v42.hydraulic_trajectory_losses import (
    HydraulicLossWeights,
    HydraulicTrajectoryLoss,
)


def test_step2_dynamic_internal_input_is_causal(tmp_path) -> None:
    from scripts.train_v42_step2_formal_f2 import _add_causal_dynamic_internal_input

    detail = tmp_path / "dynamic_internal.csv"
    columns = ["elapsed_min", *[f"setting:{i}" for i in range(36)]]
    rows = [[120.0, *([0.25] * 36)], [130.0, *([0.75] * 36)]]
    pd.DataFrame(rows, columns=columns).to_csv(detail, index=False)
    future = np.repeat(np.asarray([[0.75] * 36], dtype=np.float32), 12, axis=0)
    frame = pd.DataFrame(
        {
            "source_detail_path_dynamic_internal": [str(detail)],
            "checkpoint_min": [120.0],
            "action_dynamic_internal_readback": [json.dumps(future.tolist())],
        }
    )
    repaired = _add_causal_dynamic_internal_input(frame, tmp_path)
    actual = np.asarray(
        json.loads(repaired.iloc[0]["action_dynamic_internal_input_readback"]),
        dtype=np.float32,
    )
    expected = np.repeat(np.asarray([[0.25] * 36], dtype=np.float32), 12, axis=0)
    np.testing.assert_array_equal(actual, expected)
    assert repaired.iloc[0]["dynamic_internal_action_input_contract"] == (
        "causal_current_native_rule_readback_persistence"
    )


def test_priority_flooding_weight_targets_selected_nodes() -> None:
    branches = {}
    target = {}
    for name in HydraulicTrajectoryLoss.BRANCHES:
        pred_flood = torch.zeros(1, 1, 2)
        if name == "candidate":
            pred_flood[0, 0, 0] = 1.0
        branches[name] = {
            "node_depth": torch.zeros(1, 1, 2),
            "node_flooding_rate": pred_flood,
        }
        target[f"trajectory_depth_{name}"] = torch.zeros(1, 1, 2)
        target[f"trajectory_flood_{name}"] = torch.zeros(1, 1, 2)
    pred = {
        "branches": branches,
        "pfv_delta": torch.zeros(1),
        "tfv_delta": torch.zeros(1),
        "peak_delta": torch.zeros(1),
    }
    loss = HydraulicTrajectoryLoss(
        HydraulicLossWeights(
            depth=0.0,
            node_flooding=0.0,
            kpi_consistency=0.0,
            priority_flooding=1.0,
        ),
        require_storage_targets=False,
        require_facility_flow_targets=False,
        require_outfall_flow_targets=False,
        priority_node_indices=torch.tensor([0]),
    )
    assert float(loss.total(loss(pred, target))) > 0.0


def test_action_effect_loss_is_zero_for_matching_candidate_reference_deltas() -> None:
    shape = (1, 2, 3)
    pred = {"branches": {}}
    target = {}
    for branch in HydraulicTrajectoryLoss.BRANCHES:
        value = torch.full(shape, float(len(branch)))
        pred["branches"][branch] = {
            "node_depth": value,
            "node_flooding_rate": value,
        }
        target[f"trajectory_depth_{branch}"] = value.clone()
        target[f"trajectory_flood_{branch}"] = value.clone()
    pred.update(
        {"pfv_delta": torch.zeros(1), "tfv_delta": torch.zeros(1), "peak_delta": torch.zeros(1)}
    )
    target.update(
        {"pfv_delta": torch.zeros(1), "tfv_delta": torch.zeros(1), "peak_delta": torch.zeros(1)}
    )
    loss_fn = HydraulicTrajectoryLoss(
        HydraulicLossWeights(action_effect=1.0),
        require_storage_targets=False,
        require_facility_flow_targets=False,
        require_outfall_flow_targets=False,
    )
    losses = loss_fn(pred, target)
    assert float(losses["action_effect_trajectory"]) == 0.0


def test_action_effect_loss_is_scale_invariant_for_small_flooding_rates() -> None:
    small_target = torch.tensor([[[1.0e-3, 2.0e-3]]])
    small_pred = small_target * 2.0
    large_target = small_target * 1000.0
    large_pred = large_target * 2.0
    small = HydraulicTrajectoryLoss._normalized_effect_smooth_l1(
        small_pred, small_target
    )
    large = HydraulicTrajectoryLoss._normalized_effect_smooth_l1(
        large_pred, large_target
    )
    assert torch.isfinite(small)
    assert torch.isfinite(large)
    assert torch.allclose(small, large, rtol=1.0e-5, atol=1.0e-6)


def test_tfv_direction_loss_penalizes_wrong_improvement_direction() -> None:
    branches = {}
    target = {}
    for name in HydraulicTrajectoryLoss.BRANCHES:
        value = torch.zeros(2, 1, 2)
        branches[name] = {"node_depth": value, "node_flooding_rate": value}
        target[f"trajectory_depth_{name}"] = value.clone()
        target[f"trajectory_flood_{name}"] = value.clone()
    target.update(
        {
            "pfv_delta": torch.zeros(2),
            "tfv_delta": torch.tensor([-100.0, 100.0]),
            "peak_delta": torch.zeros(2),
        }
    )
    wrong = {
        "branches": branches,
        "pfv_delta": torch.zeros(2),
        "tfv_delta": torch.tensor([100.0, 100.0]),
        "peak_delta": torch.zeros(2),
    }
    right = dict(wrong)
    right["tfv_delta"] = torch.tensor([-100.0, 100.0])
    loss_fn = HydraulicTrajectoryLoss(
        HydraulicLossWeights(tfv_direction=1.0),
        require_storage_targets=False,
        require_facility_flow_targets=False,
        require_outfall_flow_targets=False,
    )
    assert float(loss_fn(wrong, target)["tfv_direction"]) > float(
        loss_fn(right, target)["tfv_direction"]
    )


def test_tfv_ranking_loss_prefers_true_within_state_order() -> None:
    branches = {}
    target = {}
    for name in HydraulicTrajectoryLoss.BRANCHES:
        value = torch.zeros(3, 1, 2)
        branches[name] = {"node_depth": value, "node_flooding_rate": value}
        target[f"trajectory_depth_{name}"] = value.clone()
        target[f"trajectory_flood_{name}"] = value.clone()
    target.update(
        {
            "pfv_delta": torch.zeros(3),
            "tfv_delta": torch.tensor([-100.0, 0.0, 100.0]),
            "peak_delta": torch.zeros(3),
            "_state_index": torch.zeros(3, dtype=torch.long),
        }
    )
    wrong = {
        "branches": branches,
        "pfv_delta": torch.zeros(3),
        "tfv_delta": torch.tensor([100.0, 0.0, -100.0]),
        "peak_delta": torch.zeros(3),
    }
    right = dict(wrong)
    right["tfv_delta"] = torch.tensor([-100.0, 0.0, 100.0])
    loss_fn = HydraulicTrajectoryLoss(
        HydraulicLossWeights(tfv_ranking=1.0),
        require_storage_targets=False,
        require_facility_flow_targets=False,
        require_outfall_flow_targets=False,
    )
    assert float(loss_fn(wrong, target)["tfv_ranking"]) > float(
        loss_fn(right, target)["tfv_ranking"]
    )


def test_pfv_ranking_loss_prefers_true_within_state_order() -> None:
    branches = {}
    target = {}
    for name in HydraulicTrajectoryLoss.BRANCHES:
        value = torch.zeros(3, 1, 2)
        branches[name] = {"node_depth": value, "node_flooding_rate": value}
        target[f"trajectory_depth_{name}"] = value.clone()
        target[f"trajectory_flood_{name}"] = value.clone()
    target.update(
        {
            "pfv_delta": torch.tensor([-100.0, 0.0, 100.0]),
            "tfv_delta": torch.zeros(3),
            "peak_delta": torch.zeros(3),
            "_state_index": torch.zeros(3, dtype=torch.long),
        }
    )
    wrong = {
        "branches": branches,
        "pfv_delta": torch.tensor([100.0, 0.0, -100.0]),
        "tfv_delta": torch.zeros(3),
        "peak_delta": torch.zeros(3),
    }
    right = dict(wrong)
    right["pfv_delta"] = target["pfv_delta"]
    loss_fn = HydraulicTrajectoryLoss(
        HydraulicLossWeights(pfv_ranking=1.0),
        require_storage_targets=False,
        require_facility_flow_targets=False,
        require_outfall_flow_targets=False,
    )
    assert float(loss_fn(wrong, target)["pfv_ranking"]) > float(
        loss_fn(right, target)["pfv_ranking"]
    )
