"""Final V4.2 sparse-state adapter for the paper workflow.

The adapter is deliberately not a controller.  It turns sparse telemetry and a
trained/frozen state reconstructor into the causal current-state package used by
the hydraulic surrogate and MPC.

Two reconstructor contracts are recognised:

``formal temporal V4.2``
    Consumes 13x5-min sparse depth/mask/rain/action history, node/link static
    attributes and the action-node map, and returns depth mean + uncertainty.
``legacy Project4 single-snapshot``
    Consumes only current sparse depth/mask/rain and node static attributes.
    It remains usable for historical node-depth validation but is explicitly
    marked *not sufficient* for the final Step-1 paper contract.

No mode fabricates storage volume, facility flow, OOD, head or filling degree.
Head/filling/headroom are derived only from physical INP metadata.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

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


def _mask_history(
    sensor_mask: torch.Tensor,
    *,
    batch: int,
    frames: int,
    nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.as_tensor(sensor_mask, device=device, dtype=dtype)
    if mask.ndim == 1:
        if mask.shape[0] != nodes:
            raise ValueError(f"sensor_mask expected {nodes} nodes, got {mask.shape[0]}")
        mask = mask[None, None, :].expand(batch, frames, -1)
    elif mask.ndim == 2:
        if mask.shape != (batch, nodes):
            raise ValueError("2-D sensor_mask must be [B,N]")
        mask = mask[:, None, :].expand(-1, frames, -1)
    elif mask.ndim == 3:
        if mask.shape != (batch, frames, nodes):
            raise ValueError("3-D sensor_mask must be [B,13,N]")
    else:
        raise ValueError("sensor_mask must be [N], [B,N], or [B,13,N]")
    if ((mask < 0) | (mask > 1)).any():
        raise ValueError("sensor_mask must be in [0,1]")
    return mask


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


def _forward_parameters(gat: nn.Module) -> set[str]:
    try:
        return set(inspect.signature(gat.forward).parameters)
    except (TypeError, ValueError):
        return set()


def _formal_temporal_forward(
    gat: nn.Module,
    *,
    sparse_depth_history: torch.Tensor,
    mask_history: torch.Tensor,
    rainfall_history: torch.Tensor,
    historical_actions: torch.Tensor,
    node_static: torch.Tensor,
    link_static: torch.Tensor,
    edge_index: torch.Tensor,
    action_node_map: torch.Tensor,
    mc_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mc_samples < 1:
        raise ValueError("mc_samples must be >= 1")
    was_training = gat.training
    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    try:
        gat.train(mc_samples > 1)
        with torch.no_grad():
            for _ in range(mc_samples):
                out = gat(
                    sparse_depth_history=sparse_depth_history,
                    sensor_mask_history=mask_history,
                    rainfall_history=rainfall_history,
                    historical_actions=historical_actions,
                    node_static=node_static,
                    link_static=link_static,
                    edge_index=edge_index,
                    action_node_map=action_node_map,
                )
                if hasattr(out, "depth_mean") and hasattr(out, "depth_std"):
                    mean = out.depth_mean
                    std = out.depth_std
                elif isinstance(out, dict) and "depth_mean" in out and "depth_std" in out:
                    mean = out["depth_mean"]
                    std = out["depth_std"]
                else:
                    raise TypeError(
                        "formal temporal reconstructor must return depth_mean and depth_std"
                    )
                means.append(mean)
                variances.append(std.square())
    finally:
        gat.train(was_training)
    mean_stack = torch.stack(means, dim=0)
    mean = mean_stack.mean(dim=0)
    aleatoric = torch.stack(variances, dim=0).mean(dim=0)
    epistemic = mean_stack.var(dim=0, unbiased=False) if mc_samples > 1 else torch.zeros_like(mean)
    std = torch.sqrt(torch.clamp_min(aleatoric + epistemic, 0.0))
    return mean, std


def _legacy_snapshot_forward(
    gat: nn.Module,
    *,
    current_sparse: torch.Tensor,
    current_mask: torch.Tensor,
    current_rain: torch.Tensor,
    node_static: torch.Tensor,
    edge_index: torch.Tensor,
    mc_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mc_samples < 1:
        raise ValueError("mc_samples must be >= 1")
    was_training = gat.training
    predictions: list[torch.Tensor] = []
    try:
        gat.train(mc_samples > 1)
        with torch.no_grad():
            for _ in range(mc_samples):
                pred = gat(
                    sparse_depth=current_sparse,
                    sensor_mask=current_mask,
                    rain=current_rain,
                    node_static=node_static,
                    edge_index=edge_index,
                )
                predictions.append(pred)
    finally:
        gat.train(was_training)
    stack = torch.stack(predictions, dim=0)
    mean = stack.mean(dim=0)
    std = (
        stack.std(dim=0, unbiased=False)
        if mc_samples > 1
        else torch.zeros_like(mean)
    )
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
    link_static: torch.Tensor | None = None,
    action_node_map: torch.Tensor | None = None,
    storage_node_mask: torch.Tensor | None = None,
    storage_volume_m3: torch.Tensor | None = None,
    storage_volume_available: torch.Tensor | None = None,
    facility_flow_m3s: torch.Tensor | None = None,
    facility_flow_available: torch.Tensor | None = None,
    ood_score: torch.Tensor | None = None,
    mc_samples: int = 16,
) -> SparseStateEstimate:
    """Build the causal Step-1 state package from a frozen reconstructor.

    Formal V4.2 temporal models must consume the full 13-frame/action/link-static
    contract.  Historical single-snapshot GATs are supported only so previous
    validation assets remain reproducible; metadata prevents them from being
    mistaken for a formal GAT-integrated state source.
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

    mask_history = _mask_history(
        sensor_mask,
        batch=B,
        frames=T,
        nodes=N,
        device=sparse_depth_history.device,
        dtype=sparse_depth_history.dtype,
    )
    current_sparse = sparse_depth_history[:, -1, :]
    current_mask = mask_history[:, -1, :]
    current_rain = rainfall_history[:, -1].to(
        device=current_sparse.device, dtype=current_sparse.dtype
    )

    parameters = _forward_parameters(gat)
    formal_temporal = "sparse_depth_history" in parameters
    if formal_temporal:
        if link_static is None:
            raise ValueError("formal temporal GAT requires link_static")
        if action_node_map is None:
            raise ValueError("formal temporal GAT requires action_node_map")
        link_static_t = torch.as_tensor(
            link_static,
            device=sparse_depth_history.device,
            dtype=sparse_depth_history.dtype,
        )
        action_node_map_t = torch.as_tensor(
            action_node_map,
            device=sparse_depth_history.device,
            dtype=sparse_depth_history.dtype,
        )
        _require_finite("link_static", link_static_t)
        _require_finite("action_node_map", action_node_map_t)
        depth, uncertainty = _formal_temporal_forward(
            gat,
            sparse_depth_history=sparse_depth_history,
            mask_history=mask_history,
            rainfall_history=rainfall_history,
            historical_actions=historical_actions,
            node_static=node_static,
            link_static=link_static_t,
            edge_index=edge_index,
            action_node_map=action_node_map_t,
            mc_samples=int(mc_samples),
        )
        reconstructor_contract = "formal_temporal_v42"
    else:
        depth, uncertainty = _legacy_snapshot_forward(
            gat,
            current_sparse=current_sparse,
            current_mask=current_mask,
            current_rain=current_rain,
            node_static=node_static,
            edge_index=edge_index,
            mc_samples=int(mc_samples),
        )
        reconstructor_contract = "legacy_single_snapshot"

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
    filling[positive_capacity] = (
        depth[positive_capacity] / max_depth[positive_capacity]
    )
    headroom = max_depth - depth

    A = historical_actions.shape[-1]
    current_setting = historical_actions[:, -1, :]
    previous_setting = historical_actions[:, -2, :]
    setting_rate = (current_setting - previous_setting) / 5.0

    if storage_node_mask is None:
        storage_mask = torch.zeros(N, dtype=torch.bool, device=depth.device)
    else:
        storage_mask = torch.as_tensor(
            storage_node_mask, device=depth.device
        ).bool().reshape(-1)
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
        storage_volume = torch.where(
            volume_available,
            storage_volume,
            torch.full_like(storage_volume, float("nan")),
        )
    else:
        storage_volume = None
        volume_available = torch.zeros_like(
            storage_depth, dtype=torch.bool
        )

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
            flow_available,
            facility_flow,
            torch.full_like(facility_flow, float("nan")),
        )
    else:
        facility_flow = None
        flow_available = torch.zeros(
            (B, A), dtype=torch.bool, device=depth.device
        )

    if ood_score is not None:
        ood = torch.as_tensor(
            ood_score, device=depth.device, dtype=depth.dtype
        ).reshape(-1)
        if ood.numel() != B or not torch.isfinite(ood).all():
            raise ValueError("ood_score must be finite [B]")
        ood_available = True
    else:
        ood = None
        ood_available = False

    formal_input_contract_satisfied = bool(formal_temporal)
    formal_online_state_eligible = bool(
        formal_input_contract_satisfied and ood_available
    )
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
            "reconstructor_contract": reconstructor_contract,
            "formal_step1_input_contract_satisfied": formal_input_contract_satisfied,
            "formal_online_state_eligible": formal_online_state_eligible,
            "history_frame_offsets_min": list(TEMPORAL_FRAME_OFFSETS_MIN),
            "history_frame_count": HISTORY_FRAMES,
            "mc_samples": int(mc_samples),
            "uses_future_hydraulic_truth": False,
            "physical_head_source": "node_invert_m_plus_reconstructed_depth",
            "storage_volume_fabricated": False,
            "facility_flow_fabricated": False,
            "ood_available": ood_available,
        },
    )
