"""Counterfactual twin-graph dynamics: shared-parameter multi-branch model.

F(state_history, rainfall, action_schedule) → 12-step depth trajectory

All branches share ALL parameters:
    Y_candidate = F(X, R, U_candidate)
    Y_nc        = F(X, R, U_no_control)
    Y_di        = F(X, R, U_dynamic_internal)
    Y_hold      = F(X, R, U_hold_previous)

Derived deltas:
    delta_nc = Y_candidate - Y_nc   (for PFV)
    delta_di = Y_candidate - Y_di   (for TFV/Peak)
"""
from __future__ import annotations

import torch
from torch import nn

from .graph_state_encoder import GraphStateEncoder, batch_edge_index
from .rainfall_encoder import RainfallEncoder
from .actuator_action_encoder import ActuatorActionEncoder

try:
    from torch_geometric.nn import GATConv
except Exception:  # pragma: no cover
    GATConv = None


class TwinGraphDynamics(nn.Module):
    """Multi-branch graph dynamics model with shared parameters.

    All branch trajectories are produced by the same function F
    with different action inputs, enabling counterfactual reasoning about
    control effects across Candidate, NC, DI, and Hold-Previous branches.
    """

    def __init__(
        self,
        n_nodes: int,
        n_facilities: int,
        n_static_features: int = 7,
        hidden_dim: int = 32,
        gat_heads: int = 4,
        n_gat_layers: int = 2,
        horizon: int = 12,
        history_frames: int = 13,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.n_facilities = int(n_facilities)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.history_frames = int(history_frames)

        # --- Sub-encoders ---
        self.graph_encoder = GraphStateEncoder(
            n_nodes=n_nodes,
            n_static_features=n_static_features,
            hidden_dim=hidden_dim,
            gat_heads=gat_heads,
            n_gat_layers=n_gat_layers,
            dropout=dropout,
        )
        state_embed_dim = self.graph_encoder.output_dim  # per_head * heads

        self.rainfall_encoder = RainfallEncoder(
            horizon=horizon,
            hidden_dim=hidden_dim,
        )

        self.action_encoder = ActuatorActionEncoder(
            n_facilities=n_facilities,
            hidden_dim=hidden_dim,
            horizon=horizon,
        )

        # --- Dynamics GRU: per-node temporal rollout ---
        # Input: state_embed + rain_embed + action_embed (per node)
        gru_input_dim = state_embed_dim + hidden_dim + hidden_dim
        self.dynamics_gru = nn.GRUCell(gru_input_dim, hidden_dim)

        # --- Depth head: per-node depth delta prediction ---
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # --- Edge cache ---
        self._edge_cache = {}

    def _batched_edges(self, edge_index: torch.Tensor, batch_size: int) -> torch.Tensor:
        key = (int(batch_size), int(self.n_nodes), str(edge_index.device), int(edge_index.shape[1]))
        cached = self._edge_cache.get(key)
        if cached is not None and cached.device == edge_index.device:
            return cached
        cached = batch_edge_index(edge_index, batch_size, self.n_nodes)
        if len(self._edge_cache) > 4:
            self._edge_cache.clear()
        self._edge_cache[key] = cached
        return cached

    def forward(
        self,
        state_history: torch.Tensor,
        rainfall: torch.Tensor,
        action_candidate: torch.Tensor,
        action_reference: torch.Tensor,
        edge_index: torch.Tensor,
        node_static: torch.Tensor,
        action_node_map: torch.Tensor,
        action_dynamic_internal: torch.Tensor | None = None,
        action_hold_previous: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        state_history   : [B, T_hist, N]
        rainfall        : [B, horizon]
        action_candidate: [B, horizon, n_facilities]
        action_reference: [B, horizon, n_facilities]  (NC)
        edge_index      : [2, E]
        node_static     : [N, F_static]
        action_node_map : [n_facilities, N]
        action_dynamic_internal: [B, horizon, n_facilities]  (optional, DI)
        action_hold_previous: [B, horizon, n_facilities]  (optional, Hold)

        Returns dict with:
            y_candidate : [B, horizon, N]
            y_reference : [B, horizon, N]  (NC)
            delta       : [B, horizon, N]  (candidate - NC)
            y_dynamic_internal : [B, horizon, N]  (if DI actions provided)
            y_hold_previous      : [B, horizon, N]  (if Hold actions provided)
            delta_di    : [B, horizon, N]  (candidate - DI, if DI provided)
        """
        B = state_history.shape[0]
        N = self.n_nodes

        # Encode state (shared)
        h_state = self.graph_encoder(state_history, edge_index, node_static)
        # h_state: [B*N, state_embed_dim]

        # Encode rainfall (shared)
        h_rain = self.rainfall_encoder(rainfall)
        # h_rain: [B, H, hidden_dim]

        # Encode actions (shared parameters, different inputs)
        h_act_c = self.action_encoder(action_candidate, action_node_map)
        h_act_r = self.action_encoder(action_reference, action_node_map)
        # h_act_c, h_act_r: [B, H, N, hidden_dim]

        # Rollout candidate trajectory
        y_candidate = self._rollout(h_state, h_rain, h_act_c)

        # Rollout reference trajectory (NC, same F, different action)
        y_reference = self._rollout(h_state, h_rain, h_act_r)

        # Delta (NC)
        delta = y_candidate - y_reference

        result = {
            "y_candidate": y_candidate,   # [B, H, N]
            "y_reference": y_reference,   # [B, H, N]
            "delta": delta,               # [B, H, N]
        }

        # Optional DI branch
        if action_dynamic_internal is not None:
            h_act_di = self.action_encoder(action_dynamic_internal, action_node_map)
            y_di = self._rollout(h_state, h_rain, h_act_di)
            result["y_dynamic_internal"] = y_di
            result["delta_di"] = y_candidate - y_di

        # Optional Hold-Previous branch
        if action_hold_previous is not None:
            h_act_hold = self.action_encoder(action_hold_previous, action_node_map)
            y_hold = self._rollout(h_state, h_rain, h_act_hold)
            result["y_hold_previous"] = y_hold

        return result

    def _rollout(
        self,
        h_state: torch.Tensor,
        h_rain: torch.Tensor,
        h_action: torch.Tensor,
    ) -> torch.Tensor:
        """GRU-based temporal rollout.

        h_state : [B*N, state_embed_dim]
        h_rain  : [B, H, hidden_dim]
        h_action: [B, H, N, hidden_dim]

        Returns: depths [B, H, N]
        """
        B = h_rain.shape[0]
        N = self.n_nodes
        H = self.horizon

        # h_state: [B*N, D] → use as initial hidden
        h = h_state  # [B*N, state_embed_dim]

        depths = []
        prev_depth = None

        for k in range(H):
            # Per-node rain embedding: broadcast [B, hidden_dim] → [B*N, hidden_dim]
            rain_k = h_rain[:, k, :]  # [B, D]
            rain_k = rain_k[:, None, :].expand(B, N, -1).reshape(B * N, -1)

            # Per-node action embedding
            act_k = h_action[:, k, :, :]  # [B, N, D]
            act_k = act_k.reshape(B * N, -1)

            # Concatenate: [B*N, state_dim + rain_dim + act_dim]
            inp = torch.cat([h, rain_k, act_k], dim=-1)

            # GRU step
            h = self.dynamics_gru(inp, h)  # [B*N, hidden_dim]

            # Depth delta prediction
            depth_delta = self.depth_head(h)  # [B*N, 1]
            depth_delta = depth_delta.reshape(B, N)

            # Residual depth update with bounded change
            if prev_depth is None:
                d = depth_delta
            else:
                d = prev_depth + 0.25 * torch.tanh(depth_delta)

            d = torch.relu(d)  # non-negative depth
            depths.append(d)
            prev_depth = d

        return torch.stack(depths, dim=1)  # [B, H, N]
