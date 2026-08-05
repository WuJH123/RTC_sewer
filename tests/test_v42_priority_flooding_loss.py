from __future__ import annotations

import torch

from sewerrtc.v4.models_v42.hydraulic_trajectory_losses import (
    HydraulicLossWeights,
    HydraulicTrajectoryLoss,
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
