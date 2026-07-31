"""Temporal sparse-sensor GAT reconstructor for the final V4.2 paper line.

This architecture matches the Step-1 method contract rather than the historical
single-snapshot Project4 GAT.  It consumes 60 minutes of causal sparse sensor
history, sensor masks, rainfall, historical actions, graph topology and both
node/link static attributes.  It estimates full-network depth and an aleatoric
uncertainty scale.  Physical head and filling degree remain deterministic
post-processing using INP metadata and are not learned targets here.

The module is provided as the *formal architecture*.  Existing frozen Project4
weights are not shape-compatible and must not be silently loaded into it; the
formal model requires a new training/compatibility audit before Policy Lock.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

try:
    from torch_geometric.nn import GATv2Conv
except Exception as exc:  # pragma: no cover
    GATv2Conv = None
    _PYG_IMPORT_ERROR = exc


HISTORY_FRAMES = 13


def _require_pyg() -> None:
    if GATv2Conv is None:
        raise RuntimeError("torch_geometric with GATv2Conv is required") from _PYG_IMPORT_ERROR


def _batch_graph(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    batch_size: int,
    n_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be [2,E]")
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError("link_static must be [E,F_edge]")
    E = edge_index.shape[1]
    offsets = (
        torch.arange(batch_size, device=edge_index.device)
        .repeat_interleave(E)
        * n_nodes
    )
    return (
        edge_index.repeat(1, batch_size) + offsets.unsqueeze(0),
        edge_attr.repeat(batch_size, 1),
    )


@dataclass(frozen=True)
class TemporalGATOutput:
    depth_mean: torch.Tensor
    depth_std: torch.Tensor
    latent_state: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "depth_mean": self.depth_mean,
            "depth_std": self.depth_std,
            "latent_state": self.latent_state,
        }


class TemporalSparseGATReconstructorV42(nn.Module):
    """Causal temporal GAT state estimator with edge attributes and uncertainty."""

    def __init__(
        self,
        *,
        n_nodes: int,
        n_facilities: int,
        node_static_dim: int,
        link_static_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        gat_layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        _require_pyg()
        self.n_nodes = int(n_nodes)
        self.n_facilities = int(n_facilities)
        self.node_static_dim = int(node_static_dim)
        self.link_static_dim = int(link_static_dim)
        self.hidden_dim = int(hidden_dim)

        # Per node and time: observed depth, mask, rainfall, incident-action
        # context.  Static node attributes are fused after temporal encoding.
        self.temporal = nn.GRU(
            input_size=4,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.node_static_proj = nn.Linear(node_static_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        per_head = max(8, hidden_dim // heads)
        out_dim = per_head * heads
        self.gats = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(gat_layers):
            in_dim = hidden_dim if i == 0 else out_dim
            self.gats.append(
                GATv2Conv(
                    in_dim,
                    per_head,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=True,
                    edge_dim=link_static_dim,
                )
            )
            self.norms.append(nn.LayerNorm(out_dim))
            self.skips.append(
                nn.Identity()
                if in_dim == out_dim
                else nn.Linear(in_dim, out_dim, bias=False)
            )
        self.output_dim = out_dim
        self.depth_mean_head = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.depth_scale_head = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        sparse_depth_history: torch.Tensor,
        sensor_mask_history: torch.Tensor,
        rainfall_history: torch.Tensor,
        historical_actions: torch.Tensor,
        node_static: torch.Tensor,
        link_static: torch.Tensor,
        edge_index: torch.Tensor,
        action_node_map: torch.Tensor,
    ) -> TemporalGATOutput:
        if sparse_depth_history.ndim != 3:
            raise ValueError("sparse_depth_history must be [B,13,N]")
        B, T, N = sparse_depth_history.shape
        if T != HISTORY_FRAMES or N != self.n_nodes:
            raise ValueError(
                f"expected sparse history [B,{HISTORY_FRAMES},{self.n_nodes}]"
            )
        if sensor_mask_history.shape != sparse_depth_history.shape:
            raise ValueError("sensor_mask_history must match sparse_depth_history")
        if rainfall_history.ndim == 3 and rainfall_history.shape[-1] == 1:
            rainfall_history = rainfall_history[..., 0]
        if rainfall_history.shape != (B, T):
            raise ValueError("rainfall_history must be [B,13]")
        if historical_actions.shape != (B, T, self.n_facilities):
            raise ValueError(
                f"historical_actions must be [B,13,{self.n_facilities}]"
            )
        if node_static.ndim != 2 or node_static.shape != (
            self.n_nodes,
            self.node_static_dim,
        ):
            raise ValueError("node_static shape mismatch")
        if link_static.ndim != 2 or link_static.shape[1] != self.link_static_dim:
            raise ValueError("link_static shape mismatch")
        if action_node_map.shape != (self.n_facilities, self.n_nodes):
            raise ValueError("action_node_map shape mismatch")
        tensors = (
            sparse_depth_history,
            sensor_mask_history,
            rainfall_history,
            historical_actions,
            node_static,
            link_static,
        )
        if not all(torch.isfinite(x).all() for x in tensors):
            raise ValueError("formal temporal GAT inputs must be finite")
        if ((sensor_mask_history < 0) | (sensor_mask_history > 1)).any():
            raise ValueError("sensor masks must lie in [0,1]")

        incidence = action_node_map.abs()
        node_degree = incidence.sum(dim=0).clamp_min(1.0)
        node_action = (
            torch.einsum("bta,an->btn", historical_actions, incidence)
            / node_degree[None, None, :]
        )
        rain_node = rainfall_history[:, :, None].expand(B, T, N)
        observed = sparse_depth_history * sensor_mask_history
        temporal_features = torch.stack(
            [observed, sensor_mask_history, rain_node, node_action], dim=-1
        )
        seq = temporal_features.permute(0, 2, 1, 3).reshape(B * N, T, 4)
        _, h = self.temporal(seq)
        temporal_state = h[-1].reshape(B, N, self.hidden_dim)
        static_state = self.node_static_proj(node_static)[None, :, :]
        x = self.input_norm(temporal_state + static_state).reshape(
            B * N, self.hidden_dim
        )

        be, ba = _batch_graph(
            edge_index,
            link_static,
            batch_size=B,
            n_nodes=N,
        )
        for gat, norm, skip in zip(self.gats, self.norms, self.skips):
            z = gat(x, be, ba)
            x = norm(z + skip(x)).relu()
        latent = x.reshape(B, N, self.output_dim)

        raw_mean = self.depth_mean_head(x).reshape(B, N)
        predicted = torch.nn.functional.softplus(raw_mean)
        # At observed nodes the decision-time sensor is authoritative; preserve
        # it exactly.  Unobserved nodes use the learned reconstruction.
        current_mask = sensor_mask_history[:, -1, :]
        current_obs = sparse_depth_history[:, -1, :]
        depth_mean = current_mask * current_obs + (1.0 - current_mask) * predicted
        depth_std = torch.nn.functional.softplus(
            self.depth_scale_head(x).reshape(B, N)
        ) + 1.0e-6
        # A directly observed sensor still has measurement uncertainty; do not
        # force the predicted scale to zero.
        return TemporalGATOutput(
            depth_mean=depth_mean,
            depth_std=depth_std,
            latent_state=latent,
        )
