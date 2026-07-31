"""Final V4.2 sparse-state adapter for the paper workflow.

This module is intentionally *not* a controller.  It turns sparse telemetry and
an already-trained/frozen GAT reconstructor into the causal current-state
package consumed by the hydraulic surrogate and MPC.

Scientific rules
----------------
* 60 min history = 13 frames at 5 min spacing, chronological t-60,...,t.
* Only current/past observations are accepted.  Future hydraulic truth is not
  an input.
* The frozen GAT reconstructs node depth.  Head/filling/headroom are derived
  only from physical node metadata, never from normalised graph features.
* GAT uncertainty is estimated from stochastic forward passes when requested.
* Storage volume and facility flow are never fabricated from depth.  If they
  are not observed/reconstructed by an audited source they remain unavailable.
* Historical actions are carried forward as causal state context; this adapter
  never chooses or modifies an action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .state_contract import TEMPORAL_FRAME_OFFSETS_MIN


HISTORY_FRAMES = len(TEMPORAL_FRAME_OFFSETS_MIN)


@dataclass(frozen=True)
class SparseStateEstimate:
    node_depth: torch.Tensor
    node_head: torch.Tensor
    node_filling_degree: torch.Tensor
    node_headroom: torch.Tensor
    node_uncertainty: torch.Tensor
    sensor_mask: torch.Tensor
    historical_actions: torch.Tensor
    facility_current_setting: torch.Tensor
    facility_previous_setting: torch.Tensor
    facility_setting_rate: torch.Tensor
    storage_depth: torch.Tensor
    storage_headroom: torch.Tensor
    storage_volume: torch.Tensor | None
    storage_volume_available: torch.Tensor
    facility_flow: torch.Tensor | None
    facility_flow_available: torch.Tensor
    ood_score: torch.Tensor | None
    ood_available: bool
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_depth": self.node_depth,
            "node_head": self.node_head,
            "node_filling_degree": self.node_filling_degree,
            "node_headroom": self.node_headroom,
            "node_uncertainty": self.node_uncertainty,
            "sensor_mask": self.sensor_mask,
            "historical_actions": self.historical_actions,
            "facility_current_setting": self.facility_current_setting,
            "facility_previous_setting": self.facility_previous_setting,
            "facility_setting_rate": self.facility_setting_rate,
            "storage_depth": self.storage_depth,
            "storage_headroom": self.storage_headroom,
            "storage_volume": self.storage_volume,
            "storage_volume_available": self.storage_volume_available,
            "facility_flow": self.facility_flow,
            "facility_flow_available": self.facility_flow_available,
            "ood_score": self.ood_score,
            "ood_available": self.ood_available,
            "metadata": self.metadata,
        }


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN/Inf")


def _as_batch_mask(mask: torch.Tensor, batch: int, nodes: int) -> torch.Tensor:
    if mask.ndim == 1:
        if mask.shape[0] != nodes:
            raise ValueError(f"sensor_mask expected {nodes} nodes, got {mask.shape[0]}")
        return mask[None, :].expand(batch, -1)
    if mask.ndim == 2 and mask.shape == (batch, nodes):
        return mask
    if mask.ndim == 3 and mask.shape[0] == batch and mask.shape[2] == nodes:
        # Caller supplied a mask history.  The GAT reconstructs the decision
        # state, so use only the causal t frame.
        return mask[:, -1, :]
    raise ValueError("sensor_mask must be [N], [B,N], or [B,T,N]")


def _history_shape_check(
    sparse_depth_history: torch.Tensor,
    rainfall_history: torch.Tensor,
    historical_actions: torch.Tensor,
) -> tuple[int, int, int]:
    if sparse_depth_history.ndim != 3:
        raise ValueError("sparse_depth_history must be [B,13,N]")
    B, T, N = sparse_depth_history.shape
    if T != HISTORY_FRAMES:
        raise ValueError(
            f"V4.2 requires {HISTORY_FRAMES} five-minute history frames, got {T}"
        )
    if rainfall_history.ndim == 3 and rainfall_history.shape[-1] == 1:
        rainfall_history = rainfall_history[..., 0]
    if rainfall_history.ndim != 2 or rainfall_history.shape != (B, T):
        raise ValueError("rainfall_history must be [B,13]")
    if historical_actions.ndim != 3 or historical_actions.shape[:2] != (B, T):
        raise ValueError("historical_actions must be [B,13,A]")
    return B, T, N


def _physical_node_arrays(
    node_invert_m: torch.Tensor,
    node_max_depth_m: torch.Tensor,
    *,
    batch: int,
    nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    invert = torch.as_tensor(node_invert_m, device=device, dtype=dtype).reshape(-1)
    max_depth = torch.as_tensor(node_max_depth_m, device=device, dtype=dtype).reshape(-1)
    if invert.numel() != nodes or max_depth.numel() != nodes:
        raise ValueError("physical invert/max-depth arrays must have one value per graph node")
    _require_finite("node_invert_m", invert)
    _require_finite("node_max_depth_m", max_depth)
    if (max_depth < 0).any():
        raise ValueError("node_max_depth_m cannot be negative")
    return invert[None, :].expand(batch, -1), max_depth[None, :].expand(batch, -1)


def _mc_gat_depth(
    gat: nn.Module,
    *,
    sparse_depth: torch.Tensor,
    sensor_mask: torch.Tensor,
    rain: torch.Tensor,
    node_static: torch.Tensor,
    edge_index: torch.Tensor,
    mc_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mc_samples < 1:
        raise ValueError("mc_samples must be >= 1")
    was_training = gat.training
    predictions: list[torch.Tensor] = []
    try:
        # MC dropout needs training mode to activate dropout in GATConv while
        # gradients remain disabled.  With one sample use deterministic eval.
        gat.train(mc_samples > 1)
        with torch.no_grad():
            for _ in range(mc_samples):
                pred = gat(
                    sparse_depth=sparse_depth,
                    sensor_mask=sensor_mask,
                    rain=rain,
                    node_static=node_static,
                    edge_index=edge_index,
                )
                predictions.append(pred)
    finally:
        gat.train(was_training)
    stack = torch.stack(predictions, dim=0)
    mean = stack.mean(dim=0)
    if mc_samples == 1:
        std = torch.zeros_like(mean)
    else:
        std = stack.std(dim=0, unbiased=False)
    return mean, std


def build_sparse_state_estimate(
    gat: nn.Module,
    *,
    sparse_depth_history: torch.Tensor,
    sensor_mask: torch.Tensor,
    rainfall_history: torch.Tensor,
    historical_actions: torch.Tensor,
    node_static: torch.Tensor,
    edge_index: torch.Tensor,
    node_invert_m: torch.Tensor,
    node_max_depth_m: torch.Tensor,
    storage_node_mask: torch.Tensor | None = None,
    storage_volume_m3: torch.Tensor | None = None,
    storage_volume_available: torch.Tensor | None = None,
    facility_flow_m3s: torch.Tensor | None = None,
    facility_flow_available: torch.Tensor | None = None,
    ood_score: torch.Tensor | None = None,
    mc_samples: int = 16,
) -> SparseStateEstimate:
    """Build the causal Step-1 state package from a frozen GAT.

    Parameters use physical units unless explicitly described otherwise.
    ``node_static`` may be normalised for the GAT, but physical head/filling
    calculations use the separate ``node_invert_m`` and ``node_max_depth_m``
    arrays only.
    """
    B, T, N = _history_shape_check(
        sparse_depth_history, rainfall_history, historical_actions
    )
    if hasattr(gat, "n_nodes") and int(getattr(gat, "n_nodes")) != N:
        raise ValueError(f"GAT expects {getattr(gat, 'n_nodes')} nodes, got {N}")
    if node_static.ndim != 2 or node_static.shape[0] != N:
        raise ValueError("node_static must be [N,F]")
    _require_finite("sparse_depth_history", sparse_depth_history)
    _require_finite("rainfall_history", rainfall_history)
    _require_finite("historical_actions", historical_actions)

    current_sparse = sparse_depth_history[:, -1, :]
    current_mask = _as_batch_mask(sensor_mask, B, N).to(
        device=current_sparse.device, dtype=current_sparse.dtype
    )
    if ((current_mask < 0) | (current_mask > 1)).any():
        raise ValueError("sensor_mask must be in [0,1]")
    current_rain = rainfall_history[:, -1].to(
        device=current_sparse.device, dtype=current_sparse.dtype
    )

    depth, uncertainty = _mc_gat_depth(
        gat,
        sparse_depth=current_sparse,
        sensor_mask=current_mask,
        rain=current_rain,
        node_static=node_static,
        edge_index=edge_index,
        mc_samples=int(mc_samples),
    )
    _require_finite("GAT depth", depth)
    _require_finite("GAT uncertainty", uncertainty)

    invert, max_depth = _physical_node_arrays(
        node_invert_m,
        node_max_depth_m,
        batch=B,
        nodes=N,
        device=depth.device,
        dtype=depth.dtype,
    )
    head = invert + depth
    positive_capacity = max_depth > 1.0e-8
    filling = torch.full_like(depth, float("nan"))
    filling[positive_capacity] = depth[positive_capacity] / max_depth[positive_capacity]
    headroom = max_depth - depth

    A = historical_actions.shape[-1]
    current_setting = historical_actions[:, -1, :]
    previous_setting = historical_actions[:, -2, :]
    # Per-minute rate over the final 5-minute state interval.  The 10-minute
    # controller naturally yields repeated actions on alternate state frames.
    setting_rate = (current_setting - previous_setting) / 5.0

    if storage_node_mask is None:
        storage_mask = torch.zeros(N, dtype=torch.bool, device=depth.device)
    else:
        storage_mask = torch.as_tensor(storage_node_mask, device=depth.device).bool().reshape(-1)
        if storage_mask.numel() != N:
            raise ValueError("storage_node_mask must have one value per graph node")
    storage_depth = depth[:, storage_mask]
    storage_headroom = headroom[:, storage_mask]

    if storage_volume_m3 is not None:
        storage_volume = torch.as_tensor(
            storage_volume_m3, device=depth.device, dtype=depth.dtype
        )
        if storage_volume.ndim != 2 or storage_volume.shape != storage_depth.shape:
            raise ValueError("storage_volume_m3 must be [B,n_storage]")
        if storage_volume_available is None:
            volume_available = torch.isfinite(storage_volume)
        else:
            volume_available = torch.as_tensor(
                storage_volume_available, device=depth.device
            ).bool()
            if volume_available.shape != storage_volume.shape:
                raise ValueError("storage_volume_available shape mismatch")
        # Do not silently convert missing storage volume to zero.
        storage_volume = torch.where(
            volume_available, storage_volume, torch.full_like(storage_volume, float("nan"))
        )
    else:
        storage_volume = None
        volume_available = torch.zeros_like(storage_depth, dtype=torch.bool)

    if facility_flow_m3s is not None:
        facility_flow = torch.as_tensor(
            facility_flow_m3s, device=depth.device, dtype=depth.dtype
        )
        if facility_flow.ndim != 2 or facility_flow.shape != (B, A):
            raise ValueError("facility_flow_m3s must be [B,A]")
        if facility_flow_available is None:
            flow_available = torch.isfinite(facility_flow)
        else:
            flow_available = torch.as_tensor(
                facility_flow_available, device=depth.device
            ).bool()
            if flow_available.shape != facility_flow.shape:
                raise ValueError("facility_flow_available shape mismatch")
        facility_flow = torch.where(
            flow_available, facility_flow, torch.full_like(facility_flow, float("nan"))
        )
    else:
        facility_flow = None
        flow_available = torch.zeros((B, A), dtype=torch.bool, device=depth.device)

    if ood_score is not None:
        ood = torch.as_tensor(ood_score, device=depth.device, dtype=depth.dtype).reshape(-1)
        if ood.numel() != B or not torch.isfinite(ood).all():
            raise ValueError("ood_score must be finite [B]")
        ood_available = True
    else:
        ood = None
        ood_available = False

    return SparseStateEstimate(
        node_depth=depth,
        node_head=head,
        node_filling_degree=filling,
        node_headroom=headroom,
        node_uncertainty=uncertainty,
        sensor_mask=current_mask,
        historical_actions=historical_actions,
        facility_current_setting=current_setting,
        facility_previous_setting=previous_setting,
        facility_setting_rate=setting_rate,
        storage_depth=storage_depth,
        storage_headroom=storage_headroom,
        storage_volume=storage_volume,
        storage_volume_available=volume_available,
        facility_flow=facility_flow,
        facility_flow_available=flow_available,
        ood_score=ood,
        ood_available=ood_available,
        metadata={
            "role": "state_estimation_only_not_policy",
            "history_frame_offsets_min": list(TEMPORAL_FRAME_OFFSETS_MIN),
            "history_frame_count": HISTORY_FRAMES,
            "mc_samples": int(mc_samples),
            "uses_future_hydraulic_truth": False,
            "physical_head_source": "node_invert_m_plus_reconstructed_depth",
            "storage_volume_fabricated": False,
            "facility_flow_fabricated": False,
        },
    )
