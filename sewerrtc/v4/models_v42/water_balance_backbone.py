"""Water balance backbone: physics skeleton for mass conservation structure.

Computes V(t+1) = V(t) + Qin*dt - Qout*dt from predicted flows.
Used as a structural prior / regularization in the loss.
"""
from __future__ import annotations

import torch
from torch import nn


class WaterBalanceBackbone(nn.Module):
    """Physics skeleton network enforcing mass conservation structure.

    Given predicted node depths and graph topology, computes approximate
    flow balance residuals. This is not a learned model but a differentiable
    physics constraint used for regularization.

    The backbone assumes:
      - Storage nodes accumulate volume: dV/dt = Qin - Qout
      - Non-storage nodes: depth ≈ max(inflow_depth, outflow_depth)
      - Flooding occurs when depth > capacity
    """

    def __init__(
        self,
        n_nodes: int,
        node_max_depth: torch.Tensor | None = None,
        node_ponded_area: torch.Tensor | None = None,
        dt_sec: float = 600.0,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.dt_sec = float(dt_sec)
        # Register physical parameters as buffers (not learned)
        if node_max_depth is not None:
            self.register_buffer("node_max_depth", node_max_depth)
        else:
            self.register_buffer("node_max_depth", torch.ones(n_nodes) * 5.0)
        if node_ponded_area is not None:
            self.register_buffer("node_ponded_area", node_ponded_area)
        else:
            self.register_buffer("node_ponded_area", torch.ones(n_nodes) * 100.0)

    def forward(
        self,
        pred_depth_seq: torch.Tensor,
        edge_index: torch.Tensor,
        node_static: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute water balance residuals from predicted depth sequence.

        pred_depth_seq : [B, H, N] predicted depth trajectory
        edge_index     : [2, E]
        node_static    : [N, F] optional (used for storage/outfall masks)

        Returns dict with balance residuals and derived quantities.
        """
        B, H, N = pred_depth_seq.shape
        if N != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} nodes, got {N}")

        # Volume approximation: V = depth * ponded_area
        volume_seq = pred_depth_seq * self.node_ponded_area[None, None, :]  # [B, H, N]

        # Volume change between consecutive steps
        dV = volume_seq[:, 1:, :] - volume_seq[:, :-1, :]  # [B, H-1, N]

        # Approximate flow from depth gradients along edges
        src, dst = edge_index[0], edge_index[1]
        # Flow ∝ depth_difference (simplified Manning-like)
        flow_residuals = []
        for k in range(H - 1):
            depth_k = pred_depth_seq[:, k, :]  # [B, N]
            depth_k1 = pred_depth_seq[:, k + 1, :]
            # Net inflow at each node from edges
            flow_in = torch.zeros(B, N, device=pred_depth_seq.device, dtype=pred_depth_seq.dtype)
            flow_out = torch.zeros(B, N, device=pred_depth_seq.device, dtype=pred_depth_seq.dtype)
            # Flow from src to dst proportional to depth difference
            q = torch.relu(depth_k[:, src] - depth_k[:, dst])  # [B, E]
            flow_in.scatter_add_(1, dst[None, :, None].expand(B, -1, 1), q[:, :, None])
            flow_out.scatter_add_(1, src[None, :, None].expand(B, -1, 1), q[:, :, None])
            # Mass balance residual: dV/dt - (Qin - Qout)*dt
            residual = dV[:, k, :] - (flow_in - flow_out) * self.dt_sec
            flow_residuals.append(residual)

        mass_balance_seq = torch.stack(flow_residuals, dim=1)  # [B, H-1, N]

        # Flooding indicator: depth > max_depth
        flood_indicator = torch.relu(pred_depth_seq - self.node_max_depth[None, None, :])  # [B, H, N]

        return {
            "volume_seq": volume_seq,                     # [B, H, N]
            "mass_balance_residual": mass_balance_seq,    # [B, H-1, N]
            "flood_volume": flood_indicator,              # [B, H, N]
        }
