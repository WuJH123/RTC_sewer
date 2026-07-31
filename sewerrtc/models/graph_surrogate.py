from __future__ import annotations

import torch
from torch import nn

from .gat_reconstructor import batch_edge_index, require_pyg

try:
    from torch_geometric.nn import GATConv
except Exception:  # pragma: no cover
    GATConv = None


class PhysicsGuidedTemporalGraphSurrogate(nn.Module):
    """Action-aware temporal graph surrogate for SWMM-like drainage dynamics.

    The model learns:
      State_t + Action_{t:t+H} + Rain_{t:t+H} + Graph -> Depth_{t+1:t+H}, ΔPFV, ΔTFV, Δpeak

    It uses GATConv for spatial hydraulic propagation and GRUCell for temporal memory.
    """

    def __init__(
        self,
        n_nodes: int,
        n_actions: int,
        static_dim: int,
        horizon_steps: int = 6,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        require_pyg()
        self.n_nodes = int(n_nodes)
        self.n_actions = int(n_actions)
        self.static_dim = int(static_dim)
        self.horizon_steps = int(horizon_steps)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        in_dim = 3 + static_dim  # depth, actuator node signal, rain, static
        per_head = max(8, hidden_dim // heads)
        self.input = nn.Linear(in_dim, hidden_dim)
        self.gat = GATConv(hidden_dim, per_head, heads=heads, dropout=dropout, add_self_loops=True)
        self.gat_norm = nn.LayerNorm(per_head * heads)
        self.gru = nn.GRUCell(per_head * heads, per_head * heads)
        self.depth_delta = nn.Sequential(nn.Linear(per_head * heads, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        pooled_dim = per_head * heads + horizon_steps * 2
        self.risk_head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        # logits: [PFV improves, TFV/peak safe, PFV is non-zero].
        # The third head is important because priority-zone flood deltas are
        # usually zero-inflated; without it the classifier learns "always zero".
        self.cls_head = nn.Sequential(nn.Linear(pooled_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 3))
        self._edge_cache = {}

    def _batched_edges(self, edge_index: torch.Tensor, batch_size: int, n_nodes: int) -> torch.Tensor:
        key = (int(batch_size), int(n_nodes), str(edge_index.device), int(edge_index.shape[1]))
        cached = self._edge_cache.get(key)
        if cached is not None and cached.device == edge_index.device:
            return cached
        cached = batch_edge_index(edge_index, batch_size, n_nodes)
        if len(self._edge_cache) > 4:
            self._edge_cache.clear()
        self._edge_cache[key] = cached
        return cached

    def forward(
        self,
        state: torch.Tensor,
        action_seq: torch.Tensor,
        rain_seq: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        action_node_map: torch.Tensor,
    ) -> dict:
        if state.ndim != 2:
            raise ValueError("state must be [B, N]")
        B, N = state.shape
        H = action_seq.shape[1]
        if N != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} nodes, got {N}")
        if rain_seq.ndim == 2:
            rain_seq = rain_seq[:, :, None]
        if action_node_map.shape != (self.n_actions, self.n_nodes):
            raise ValueError(f"action_node_map must be [{self.n_actions}, {self.n_nodes}]")
        hidden = None
        depth = state
        preds = []
        be = self._batched_edges(edge_index, B, N)
        static = node_static[None, :, :].expand(B, N, -1)
        for k in range(H):
            action_signal = torch.matmul(action_seq[:, k, :], action_node_map).clamp(0.0, 1.0)
            rain_node = rain_seq[:, k, :].reshape(B, 1, 1).expand(B, N, 1)
            x = torch.cat([depth[:, :, None], action_signal[:, :, None], rain_node, static], dim=-1)
            x = self.input(x.reshape(B * N, -1)).relu()
            msg = self.gat(x, be).relu()
            msg = self.gat_norm(msg)
            hidden = msg if hidden is None else self.gru(msg, hidden)
            delta = self.depth_delta(hidden).reshape(B, N)
            depth = torch.relu(depth + 0.25 * torch.tanh(delta))
            preds.append(depth)
        pred_seq = torch.stack(preds, dim=1)  # [B, H, N]
        pooled = hidden.reshape(B, N, -1).mean(dim=1)
        rain_summary = rain_seq[:, :H, 0]
        action_summary = action_seq[:, :H, :].mean(dim=-1)
        risk_input = torch.cat([pooled, rain_summary, action_summary], dim=-1)
        risk_delta = self.risk_head(risk_input)
        logits = self.cls_head(risk_input)
        return {
            "pred_seq": pred_seq,
            "next_state": pred_seq[:, 0, :],
            "risk_delta": risk_delta,
            "logits": logits,
        }


# Backwards-compatible import name used by the controller.
ActionAwareGraphSurrogate = PhysicsGuidedTemporalGraphSurrogate
