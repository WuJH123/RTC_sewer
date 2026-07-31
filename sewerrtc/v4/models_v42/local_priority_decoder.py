"""Local priority decoder (PFV head): hurdle model for priority flooding volume.

Focuses on priority/sentinel nodes (k-hop local subgraph).
Hurdle model: binary classifier (will flooding occur?) + regressor (how much?).
12-step priority flooding rate → integrate to get PFV.
"""
from __future__ import annotations

import torch
from torch import nn


class LocalPriorityDecoder(nn.Module):
    """PFV head: hurdle model on priority node subgraph.

    Input
    -----
    delta_trajectory : [B, H, N]  (candidate - reference depth)
    state_context    : [B*N, D]   (node embeddings from graph encoder)
    priority_mask    : [N]        binary mask of priority nodes

    Output
    ------
    pfv_delta        : [B]  predicted PFV difference (m³)
    pfv_rate_seq     : [B, H]  per-step priority flooding rate
    flood_prob       : [B, H]  per-step flood probability
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

        # Hurdle binary classifier: will any priority node flood?
        # Input: mean delta over priority nodes + state context
        cls_input_dim = 1 + state_embed_dim
        self.flood_classifier = nn.Sequential(
            nn.Linear(cls_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Regressor: how much flooding?
        reg_input_dim = 1 + state_embed_dim
        self.flood_regressor = nn.Sequential(
            nn.Linear(reg_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        delta_trajectory: torch.Tensor,
        state_context: torch.Tensor,
        priority_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        delta_trajectory : [B, H, N]
        state_context    : [B*N, D]
        priority_mask    : [N] binary
        """
        B, H, N = delta_trajectory.shape
        D = state_context.shape[-1]

        # Extract priority node features
        pri_idx = priority_mask.nonzero(as_tuple=True)[0]  # [N_pri]
        if pri_idx.numel() == 0:
            # No priority nodes: return zeros
            return {
                "pfv_delta": torch.zeros(B, device=delta_trajectory.device),
                "pfv_rate_seq": torch.zeros(B, H, device=delta_trajectory.device),
                "flood_prob": torch.zeros(B, H, device=delta_trajectory.device),
            }

        # Mean delta over priority nodes per step: [B, H]
        pri_delta = delta_trajectory[:, :, pri_idx].mean(dim=-1)  # [B, H]

        # Mean state context over priority nodes: [B, D]
        pri_state = state_context.reshape(B, N, D)[:, pri_idx, :].mean(dim=1)  # [B, D]

        # Per-step hurdle
        flood_probs = []
        flood_rates = []
        for k in range(H):
            dk = pri_delta[:, k:k+1]  # [B, 1]
            cls_in = torch.cat([dk, pri_state], dim=-1)
            logit = self.flood_classifier(cls_in)  # [B, 1]
            prob = torch.sigmoid(logit).squeeze(-1)  # [B]
            rate = torch.relu(self.flood_regressor(cls_in).squeeze(-1))  # [B]
            # Hurdle: rate is zeroed if classifier says no flood
            gated_rate = prob * rate
            flood_probs.append(prob)
            flood_rates.append(gated_rate)

        flood_prob_seq = torch.stack(flood_probs, dim=1)   # [B, H]
        flood_rate_seq = torch.stack(flood_rates, dim=1)   # [B, H]

        # Integrate flooding rate over time → PFV (m³)
        pfv_delta = flood_rate_seq.sum(dim=1) * self.dt_sec  # [B]

        return {
            "pfv_delta": pfv_delta,
            "pfv_rate_seq": flood_rate_seq,
            "flood_prob": flood_prob_seq,
        }
