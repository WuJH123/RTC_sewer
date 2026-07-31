"""Actuator action encoder: maps facility actions to node-level embeddings."""
from __future__ import annotations

import torch
from torch import nn


class ActuatorActionEncoder(nn.Module):
    """Encode action schedules into per-node, per-step action embeddings.

    Input
    -----
    action_schedule : [B, horizon, n_facilities]
    action_node_map : [n_facilities, N]  (binary/soft assignment)

    Output
    ------
    action_embed : [B, horizon, N, hidden_dim]
    """

    def __init__(
        self,
        n_facilities: int,
        hidden_dim: int = 32,
        horizon: int = 12,
        facility_type_dim: int = 1,
    ):
        super().__init__()
        self.n_facilities = int(n_facilities)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)

        # Action value + change features → per-facility embedding
        # Features: [action_value, delta_action, |delta_action|, is_binary]
        feat_dim = 3 + facility_type_dim
        self.facility_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Lightweight temporal 1D conv over horizon steps
        self.temporal_conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=1
        )

    def forward(
        self,
        action_schedule: torch.Tensor,
        action_node_map: torch.Tensor,
        facility_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        action_schedule : [B, H, A]
        action_node_map : [A, N]
        facility_types  : [A, facility_type_dim] optional (binary vs continuous flag)
        """
        B, H, A = action_schedule.shape
        N = action_node_map.shape[1]
        if A != self.n_facilities:
            raise ValueError(f"Expected {self.n_facilities} facilities, got {A}")

        # Compute action change features
        # Pad with zero at t=0 for causal delta
        action_shifted = torch.zeros_like(action_schedule)
        action_shifted[:, 1:, :] = action_schedule[:, :-1, :]
        delta = action_schedule - action_shifted  # [B, H, A]
        abs_delta = delta.abs()

        if facility_types is None:
            # Default: all continuous (0.0)
            facility_types = torch.zeros(A, 1, device=action_schedule.device, dtype=action_schedule.dtype)
        if facility_types.ndim == 1:
            facility_types = facility_types[:, None]
        # Expand to [B, H, A, type_dim]
        ft = facility_types[None, None, :, :].expand(B, H, -1, -1)

        # Concatenate features: [B, H, A, feat_dim]
        features = torch.cat([
            action_schedule[:, :, :, None],
            delta[:, :, :, None],
            abs_delta[:, :, :, None],
            ft,
        ], dim=-1)

        # Encode per-facility: [B*H*A, feat_dim] → [B*H*A, hidden_dim]
        fac_embed = self.facility_encoder(features.reshape(B * H * A, -1))
        fac_embed = fac_embed.reshape(B, H, A, self.hidden_dim)

        # Map to nodes: [B, H, N, hidden_dim]
        # action_node_map: [A, N] → for each node, sum facility embeddings of the
        # facilities that act on it.  einsum "bhad,an->bhdn" contracts over the
        # facility axis (a) and leaves one embedding per (batch, horizon, node).
        node_embed = torch.einsum("bhad,an->bhdn", fac_embed, action_node_map)
        # Transpose to [B, H, N, D]
        node_embed = node_embed.permute(0, 1, 3, 2)  # [B, H, N, D]

        # Temporal conv per node: treat (B*N) as batch, H as sequence length
        BN = B * N
        node_flat = node_embed.permute(0, 2, 1, 3).reshape(BN, H, self.hidden_dim)
        node_flat = node_flat.transpose(1, 2)  # [BN, D, H]
        node_flat = self.temporal_conv(node_flat)
        node_flat = node_flat.transpose(1, 2)  # [BN, H, D]
        node_embed = node_flat.reshape(B, N, H, self.hidden_dim).permute(0, 2, 1, 3)

        return node_embed  # [B, H, N, hidden_dim]
