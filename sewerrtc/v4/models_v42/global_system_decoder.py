"""Global system decoder (TFV head): system-wide flooding aggregation.

System-wide aggregation: storage + outfall + flooding sequences.
Water balance constraint: TFV ≈ total inflow - total outflow - Δstorage.
Integrate flooding rate over 12 steps → TFV.
"""
from __future__ import annotations

import torch
from torch import nn


class GlobalSystemDecoder(nn.Module):
    """TFV head: system-wide total flooding volume prediction.

    Input
    -----
    delta_trajectory : [B, H, N]
    state_context    : [B*N, D]
    node_static      : [N, F]  (for storage/outfall masks)

    Output
    ------
    tfv_delta        : [B]  predicted TFV difference (m³)
    tfv_rate_seq     : [B, H]  per-step system-wide flooding rate
    peak_flood_rate  : [B]  max flooding rate across horizon
    """

    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int = 32,
        horizon: int = 12,
        state_embed_dim: int = 32,
        dt_sec: float = 600.0,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.horizon = int(horizon)
        self.dt_sec = float(dt_sec)

        # System-wide flooding rate regressor
        # Input: spatially aggregated delta + global state context
        reg_input_dim = 1 + state_embed_dim
        self.system_regressor = nn.Sequential(
            nn.Linear(reg_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Peak head: from 12-step rate sequence, take max
        self.peak_mlp = nn.Sequential(
            nn.Linear(horizon, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        delta_trajectory: torch.Tensor,
        state_context: torch.Tensor,
        node_static: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        delta_trajectory : [B, H, N]
        state_context    : [B*N, D]
        node_static      : [N, F] optional
        """
        B, H, N = delta_trajectory.shape
        D = state_context.shape[-1]

        # Global state context: mean over all nodes
        global_state = state_context.reshape(B, N, D).mean(dim=1)  # [B, D]

        # Spatially aggregated delta: mean over all nodes per step
        global_delta = delta_trajectory.mean(dim=-1)  # [B, H]

        # Per-step system-wide flooding rate
        flood_rates = []
        for k in range(H):
            dk = global_delta[:, k:k+1]  # [B, 1]
            inp = torch.cat([dk, global_state], dim=-1)
            rate = torch.relu(self.system_regressor(inp).squeeze(-1))  # [B]
            flood_rates.append(rate)

        tfv_rate_seq = torch.stack(flood_rates, dim=1)  # [B, H]

        # Integrate → TFV
        tfv_delta = tfv_rate_seq.sum(dim=1) * self.dt_sec  # [B]

        # Peak: direct prediction from rate sequence
        peak_from_seq = tfv_rate_seq.max(dim=1).values  # [B]
        peak_direct = self.peak_mlp(tfv_rate_seq).squeeze(-1)  # [B]
        # Blend: consistency between max(sequence) and direct prediction
        peak_flood_rate = 0.5 * (peak_from_seq + peak_direct)

        return {
            "tfv_delta": tfv_delta,
            "tfv_rate_seq": tfv_rate_seq,
            "peak_flood_rate": peak_flood_rate,
            "peak_from_seq": peak_from_seq,
            "peak_direct": peak_direct,
        }
