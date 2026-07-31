"""Rainfall encoder: per-horizon-step MLP embedding."""
from __future__ import annotations

import torch
from torch import nn


class RainfallEncoder(nn.Module):
    """Encode rainfall forecast into per-step embeddings.

    Input
    -----
    rainfall : [B, horizon] or [B, horizon, N] (spatially varying)

    Output
    ------
    rain_embed : [B, horizon, hidden_dim]
    """

    def __init__(self, horizon: int = 12, hidden_dim: int = 32, spatial: bool = False, n_nodes: int = 932):
        super().__init__()
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.spatial = bool(spatial)
        # Input dim: 1 if uniform, n_nodes if spatially varying
        in_dim = n_nodes if spatial else 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, rainfall: torch.Tensor) -> torch.Tensor:
        B = rainfall.shape[0]
        H = self.horizon
        if rainfall.ndim == 1:
            # [B] → [B, 1] broadcast
            rainfall = rainfall[:, None]
        if rainfall.ndim == 2:
            if rainfall.shape[1] == H:
                # [B, H] → [B, H, 1]
                rainfall = rainfall[:, :, None]
            else:
                # [B, N] spatial → [B, 1, N]
                rainfall = rainfall[:, None, :]
        # rainfall: [B, H, 1] or [B, H, N]
        rain_embed = self.mlp(rainfall)  # [B, H, hidden_dim]
        return rain_embed
