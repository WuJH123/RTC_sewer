"""Candidate-relative differentiable control surrogate for the V4.2 repair path.

This model is deliberately KPI/action-effect first.  The input state is the
causal state signature exported by the experience bank; no future SWMM target
is used as an input.  Hydraulic residual summaries are an auxiliary decoder
and are masked when the corresponding target is unavailable.
"""
from __future__ import annotations

import torch
from torch import nn


class CandidateRelativeDifferentiableControlSurrogateV2(nn.Module):
    """Explicitly encode candidate/reference action differences.

    ``action_sequence`` is [B, 12, 36].  H3 is retained as a differentiable
    variable for gradient checks; H4-H12 are still part of the frozen action
    representation and are not silently discarded.
    """

    def __init__(
        self,
        *,
        state_dim: int = 25,
        n_facilities: int = 36,
        horizon: int = 12,
        hidden_dim: int = 96,
        raw_action_baseline: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_facilities = int(n_facilities)
        self.horizon = int(horizon)
        self.raw_action_baseline = bool(raw_action_baseline)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        action_features = self.n_facilities * (4 if not raw_action_baseline else 1)
        self.action_step_encoder = nn.Sequential(
            nn.Linear(action_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.action_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.action_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )

        # FiLM-style state-action interaction makes the same action difference
        # state dependent without introducing an architecture search.
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.pfv_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 2))
        self.tfv_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1))
        # [depth_mean, flood_total, storage_mean, facility_flow_mean] residual
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * 4),
        )

    def encode_action(
        self,
        action_sequence: torch.Tensor,
        current_action: torch.Tensor,
        no_control_action: torch.Tensor,
        internal_action: torch.Tensor,
    ) -> torch.Tensor:
        if action_sequence.ndim != 3 or action_sequence.shape[1:] != (self.horizon, self.n_facilities):
            raise ValueError("action_sequence must be [B,12,36]")
        if self.raw_action_baseline:
            features = action_sequence
        else:
            features = torch.cat(
                [
                    action_sequence,
                    action_sequence - current_action[:, None, :],
                    action_sequence - no_control_action[:, None, :],
                    action_sequence - internal_action[:, None, :],
                ],
                dim=-1,
            )
        encoded = self.action_step_encoder(features)
        sequence, _ = self.action_gru(encoded)
        pooled = torch.cat([sequence[:, -1, :], sequence.mean(dim=1)], dim=-1)
        return self.action_pool(pooled)

    def encode_fused(
        self,
        state_signature: torch.Tensor,
        action_sequence: torch.Tensor,
        current_action: torch.Tensor,
        no_control_action: torch.Tensor,
        internal_action: torch.Tensor,
    ) -> torch.Tensor:
        state = self.state_encoder(state_signature)
        action = self.encode_action(action_sequence, current_action, no_control_action, internal_action)
        gamma, beta = self.film(state).chunk(2, dim=-1)
        action_modulated = action * (1.0 + torch.tanh(gamma)) + beta
        return self.fusion(torch.cat([state, action_modulated, state * action_modulated], dim=-1))

    def forward(
        self,
        state_signature: torch.Tensor,
        action_sequence: torch.Tensor,
        current_action: torch.Tensor,
        no_control_action: torch.Tensor,
        internal_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused = self.encode_fused(
            state_signature,
            action_sequence,
            current_action,
            no_control_action,
            internal_action,
        )
        pfv = self.pfv_head(fused)
        return {
            "mean_g_pfv": pfv[:, 0],
            "log_scale_g_pfv": pfv[:, 1].clamp(-8.0, 8.0),
            "delta_tfv": self.tfv_head(fused).squeeze(-1),
            "trajectory_residual": self.trajectory_decoder(fused).reshape(-1, self.horizon, 4),
        }
