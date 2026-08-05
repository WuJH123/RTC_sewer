from __future__ import annotations

import torch

from sewerrtc.v4.models_v42.hydraulic_trajectory_losses import (
    HydraulicTrajectoryLoss,
    HydraulicLossWeights,
)


def _prediction_and_target():
    branches = {
        branch: {
            "node_depth": torch.zeros(1, 1, 1),
            "node_flooding_rate": torch.zeros(1, 1, 1),
        }
        for branch in HydraulicTrajectoryLoss.BRANCHES
    }
    prediction = {
        "branches": branches,
        "pfv_delta": torch.zeros(1),
        "tfv_delta": torch.zeros(1),
        "peak_delta": torch.zeros(1),
    }
    target = {
        f"trajectory_depth_{branch}": torch.zeros(1, 1, 1)
        for branch in HydraulicTrajectoryLoss.BRANCHES
    }
    target.update(
        {
            f"trajectory_flood_{branch}": torch.zeros(1, 1, 1)
            for branch in HydraulicTrajectoryLoss.BRANCHES
        }
    )
    target.update(
        {
            "pfv_delta": torch.full((1,), 100.0),
            "tfv_delta": torch.full((1,), 100.0),
            "peak_delta": torch.full((1,), 100.0),
        }
    )
    return prediction, target


def test_kpi_loss_scales_keep_pfv_tfv_peak_effects_on_comparable_scale():
    prediction, target = _prediction_and_target()
    weights = HydraulicLossWeights(
        depth=0.0,
        node_flooding=0.0,
        storage=0.0,
        facility_flow=0.0,
        outfall_flow=0.0,
        kpi_consistency=1.0,
    )
    raw = HydraulicTrajectoryLoss(weights, require_storage_targets=False, require_facility_flow_targets=False, require_outfall_flow_targets=False)
    scaled = HydraulicTrajectoryLoss(
        weights,
        require_storage_targets=False,
        require_facility_flow_targets=False,
        require_outfall_flow_targets=False,
        kpi_scales={"pfv_delta": 100.0, "tfv_delta": 100.0, "peak_delta": 100.0},
    )
    raw_loss = raw.total(raw(prediction, target))
    scaled_loss = scaled.total(scaled(prediction, target))
    assert float(scaled_loss) < float(raw_loss)
    assert torch.isfinite(scaled_loss)


def test_action_effect_loss_supervises_candidate_relative_to_no_control():
    prediction, target = _prediction_and_target()
    target["trajectory_flood_candidate"] = torch.full((1, 1, 1), 2.0)
    loss_fn = HydraulicTrajectoryLoss(
        HydraulicLossWeights(action_effect=1.0),
        require_storage_targets=False,
        require_facility_flow_targets=False,
        require_outfall_flow_targets=False,
    )
    losses = loss_fn(prediction, target)
    assert float(losses["action_effect"]) > 0.0
    assert torch.isfinite(loss_fn.total(losses))


def test_pfv_action_effect_loss_can_focus_on_priority_nodes():
    prediction, target = _prediction_and_target()
    target["trajectory_flood_candidate"] = torch.tensor([[[2.0]]])
    target["trajectory_flood_no_control"] = torch.tensor([[[0.0]]])
    loss_fn = HydraulicTrajectoryLoss(
        HydraulicLossWeights(pfv_action_effect=1.0),
        require_storage_targets=False,
        require_facility_flow_targets=False,
        require_outfall_flow_targets=False,
        action_effect_indices=torch.tensor([0]),
    )
    losses = loss_fn(prediction, target)
    assert float(losses["pfv_action_effect"]) > 0.0


def test_pairwise_ranking_loss_uses_within_state_order():
    prediction = torch.tensor([1.0, -1.0], requires_grad=True)
    target = torch.tensor([-1.0, 1.0])
    group_id = torch.tensor([3, 3])
    loss = HydraulicTrajectoryLoss._pairwise_rank_loss(
        prediction, target, group_id
    )
    assert float(loss) > 0.0
    loss.backward()
    assert prediction.grad is not None
