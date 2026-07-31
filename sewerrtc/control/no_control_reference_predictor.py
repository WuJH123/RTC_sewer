from __future__ import annotations

import numpy as np
import torch


def constant_default_action_sequence(default_action: np.ndarray, horizon_steps: int) -> np.ndarray:
    """Build the causal No-control continuation from known passive settings."""
    action = np.asarray(default_action, dtype=np.float32)
    if action.ndim != 1:
        raise ValueError("default No-control action must be a one-dimensional canonical vector")
    if not np.isfinite(action).all() or np.any(action < 0.0) or np.any(action > 1.0):
        raise ValueError("default No-control action contains invalid settings")
    return np.repeat(action[None, :], max(1, int(horizon_steps)), axis=0)


class OnlineNoControlReferencePredictor:
    """Predict a No-control horizon from current state and forecast only.

    This wrapper deliberately accepts no true future SWMM detail or KPI arrays.
    Offline true No-control trajectories remain evaluation labels outside this
    interface.
    """

    def __init__(self, surrogate: torch.nn.Module) -> None:
        self.surrogate = surrogate

    def predict(
        self,
        *,
        state: torch.Tensor,
        reference_action_seq: torch.Tensor,
        rain_seq: torch.Tensor,
        actuator_mask: torch.Tensor,
        actuator_features: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        action_node_map: torch.Tensor,
        priority_indices: torch.Tensor,
        storage_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = self.surrogate(
            state=state,
            candidate_action_seq=reference_action_seq,
            reference_action_seq=reference_action_seq,
            rain_seq=rain_seq,
            actuator_mask=actuator_mask,
            actuator_features=actuator_features,
            node_static=node_static,
            edge_index=edge_index,
            action_node_map=action_node_map,
            priority_indices=priority_indices,
            storage_indices=storage_indices,
        )
        return {
            "reference_risk_rate_seq": output["reference_risk_rate_seq"],
            "reference_PFV_H": output["reference_risk_rate_seq"][:, :, 0].sum(dim=1) * 300.0,
            "reference_TFV_H": output["reference_risk_rate_seq"][:, :, 1].sum(dim=1) * 300.0,
            "reference_peak_TFV_rate": output["reference_risk_rate_seq"][:, :, 1].max(dim=1).values,
            "reference_source": "online_predicted_no_control_reference",
        }
