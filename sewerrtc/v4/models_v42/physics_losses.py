"""Physics losses: 8 physics-guided loss terms, each recorded separately.

Losses:
  1. mass_balance       — |Σinflow - Σoutflow - Δstorage|
  2. storage_continuity — storage change consistency
  3. non_negative       — ReLU(-pred_depth) penalty
  4. capacity_bounds    — ReLU(pred_depth - max_depth) penalty
  5. flooding_consistency — flood occurs iff depth > capacity
  6. shared_init_state  — candidate & reference start from same state
  7. kpi_trajectory_consistency — predicted KPI matches trajectory-derived KPI
  8. peak_consistency   — direct peak ≈ max(sequence)
"""
from __future__ import annotations

import torch
from torch import nn


class PhysicsLosses(nn.Module):
    """Compute 8 physics-guided loss terms from model predictions.

    All losses are differentiable scalars. Each is returned in a dict
    so they can be logged individually and weighted in the total loss.
    """

    def __init__(
        self,
        n_nodes: int,
        node_max_depth: torch.Tensor | None = None,
        dt_sec: float = 600.0,
        ponded_area: float = 100.0,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.dt_sec = float(dt_sec)
        self.ponded_area = float(ponded_area)
        if node_max_depth is not None:
            self.register_buffer("node_max_depth", node_max_depth)
        else:
            self.register_buffer("node_max_depth", torch.ones(n_nodes) * 5.0)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor] | None = None,
        edge_index: torch.Tensor | None = None,
        node_static: torch.Tensor | None = None,
        action_node_map: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        pred dict must contain:
            y_candidate : [B, H, N]
            y_reference : [B, H, N]
            delta       : [B, H, N]
            pfv_delta   : [B]  (from LocalPriorityDecoder)
            tfv_delta   : [B]  (from GlobalSystemDecoder)
            peak_flood_rate : [B]
            tfv_rate_seq : [B, H]

        target dict (optional) may contain:
            pfv_gt : [B]
            tfv_gt : [B]
            peak_gt : [B]
        """
        losses: dict[str, torch.Tensor] = {}
        y_cand = pred["y_candidate"]
        y_ref = pred["y_reference"]
        B, H, N = y_cand.shape

        # 1. Mass balance: |Σinflow - Σoutflow - Δstorage|
        #    Approximate: volume change should be consistent with edge flows
        if edge_index is not None and edge_index.numel() > 0:
            vol = y_cand * self.ponded_area  # [B, H, N]
            dV = vol[:, 1:, :] - vol[:, :-1, :]  # [B, H-1, N]
            src, dst = edge_index[0], edge_index[1]
            # Net flow at each node
            q = torch.relu(y_cand[:, :-1, src] - y_cand[:, :-1, dst])  # [B, H-1, E]
            flow_in = torch.zeros(B, H - 1, N, device=y_cand.device, dtype=y_cand.dtype)
            flow_out = torch.zeros(B, H - 1, N, device=y_cand.device, dtype=y_cand.dtype)
            flow_in.scatter_add_(2, dst[None, None, :].expand(B, H - 1, -1), q)
            flow_out.scatter_add_(2, src[None, None, :].expand(B, H - 1, -1), q)
            net_flow = (flow_in - flow_out) * self.dt_sec
            mass_res = (dV - net_flow).abs().mean()
        else:
            mass_res = torch.zeros((), device=y_cand.device, dtype=y_cand.dtype)
        losses["mass_balance"] = mass_res

        # 2. Storage continuity: temporal smoothness of depth
        #    Penalize large second-order differences
        if H >= 3:
            d1 = y_cand[:, 1:, :] - y_cand[:, :-1, :]
            d2 = d1[:, 1:, :] - d1[:, :-1, :]
            storage_cont = (d2 ** 2).mean()
        else:
            storage_cont = torch.zeros((), device=y_cand.device, dtype=y_cand.dtype)
        losses["storage_continuity"] = storage_cont

        # 3. Non-negative depth: ReLU(-pred_depth) penalty
        neg_penalty = torch.relu(-y_cand).mean() + torch.relu(-y_ref).mean()
        losses["non_negative"] = neg_penalty

        # 4. Capacity bounds: ReLU(pred_depth - max_depth) penalty
        cap_viol_c = torch.relu(y_cand - self.node_max_depth[None, None, :]).mean()
        cap_viol_r = torch.relu(y_ref - self.node_max_depth[None, None, :]).mean()
        losses["capacity_bounds"] = cap_viol_c + cap_viol_r

        # 5. Flooding consistency: flood occurs iff depth > capacity
        #    Predicted flood volume from trajectory vs actual overflow
        flood_from_depth_c = torch.relu(y_cand - self.node_max_depth[None, None, :])
        flood_from_depth_r = torch.relu(y_ref - self.node_max_depth[None, None, :])
        # The delta trajectory should be consistent with flooding change
        delta_flood = flood_from_depth_c - flood_from_depth_r
        flood_consistency = (delta_flood - pred["delta"]).abs().mean()
        losses["flooding_consistency"] = flood_consistency

        # 6. Shared initial state: candidate & reference start from same state
        init_diff = (y_cand[:, 0, :] - y_ref[:, 0, :]).abs().mean()
        losses["shared_init_state"] = init_diff

        # 7. KPI-trajectory consistency: predicted KPI delta matches trajectory-derived
        #    KPI delta.  Both PFV and TFV are computed from the *overflow* depth
        #    (max(depth - max_depth, 0)) so they are in depth units (m) and
        #    commensurable with pred_pfv / pred_tfv.  Mean over horizon keeps the
        #    scale in meters (matching the predicted delta) instead of m·steps.
        priority_flood_c = torch.relu(
            y_cand - self.node_max_depth[None, None, :]
        )  # [B, H, N]
        priority_flood_r = torch.relu(
            y_ref - self.node_max_depth[None, None, :]
        )  # [B, H, N]
        traj_pfv_c = priority_flood_c.sum(dim=2).mean(dim=1)  # [B]  m
        traj_pfv_r = priority_flood_r.sum(dim=2).mean(dim=1)  # [B]  m
        pred_pfv = pred.get("pfv_delta", torch.zeros(B, device=y_cand.device))
        kpi_pfv_consistency = ((traj_pfv_c - traj_pfv_r) - pred_pfv).abs().mean()

        # TFV trajectory proxy: system-wide mean depth (sum/N) per step.
        # Use the delta (Candidate − Reference) so it is commensurable with tfv_delta.
        tfv_seq_c = y_cand.sum(dim=2) / max(self.n_nodes, 1)  # [B, H]
        tfv_seq_r = y_ref.sum(dim=2) / max(self.n_nodes, 1)  # [B, H]
        traj_tfv_c = tfv_seq_c.mean(dim=1)  # [B]  m
        traj_tfv_r = tfv_seq_r.mean(dim=1)  # [B]  m
        pred_tfv = pred.get("tfv_delta", torch.zeros(B, device=y_cand.device))
        kpi_tfv_consistency = ((traj_tfv_c - traj_tfv_r) - pred_tfv).abs().mean()
        losses["kpi_trajectory_consistency"] = kpi_pfv_consistency + kpi_tfv_consistency

        # 8. Peak consistency: direct peak ≈ max(sequence)
        peak_seq = pred.get("tfv_rate_seq", torch.zeros(B, H, device=y_cand.device)).max(dim=1).values
        peak_direct = pred.get("peak_flood_rate", torch.zeros(B, device=y_cand.device))
        losses["peak_consistency"] = (peak_seq - peak_direct).abs().mean()

        return losses
