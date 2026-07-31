from __future__ import annotations

import torch
from torch import nn

from .gat_reconstructor import batch_edge_index, require_pyg

try:
    from torch_geometric.nn import GATConv
except Exception:  # pragma: no cover
    GATConv = None


PHASE_TO_INDEX = {"unknown": 0, "rising": 1, "peak": 2, "recession": 3}


def encode_phase_indices(phases: list[str] | tuple[str, ...] | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(phases):
        return phases.to(dtype=torch.long)
    return torch.as_tensor(
        [PHASE_TO_INDEX.get(str(phase).strip().lower(), 0) for phase in phases],
        dtype=torch.long,
    )


def causal_action_scale(masked_delta: torch.Tensor) -> torch.Tensor:
    """Return a causal residual magnitude without averaging over actuators.

    A setting change can affect hydraulics after the command itself has ended.
    The cumulative maximum therefore keeps the effect branch active after the
    first change while retaining exact zero effect for a reference sequence.
    """
    step_scale = masked_delta.abs().amax(dim=2, keepdim=True)
    return torch.cummax(step_scale, dim=1).values


def causal_active_actuator_mask(masked_delta: torch.Tensor, *, threshold: float = 1.0e-8) -> torch.Tensor:
    """Mark actuators that have changed at or before each horizon step."""
    changed = masked_delta.abs() > float(threshold)
    return torch.cummax(changed.to(torch.int8), dim=1).values.bool()


class RawJointActionSurrogate(nn.Module):
    """Raw-action residual surrogate with an explicit No-control reference branch.

    Candidate and reference actions remain tensors of shape ``[B, H, A]`` until
    they enter the per-actuator temporal encoder. The effect branch consumes
    only their masked difference, which makes a zero action difference produce
    an exactly zero predicted effect by construction.
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        n_actions: int,
        node_static_dim: int,
        actuator_feature_dim: int,
        horizon_steps: int = 6,
        hidden_dim: int = 96,
        heads: int = 4,
        dropout: float = 0.05,
        architecture_version: str = "legacy_v1",
    ) -> None:
        super().__init__()
        require_pyg()
        self.n_nodes = int(n_nodes)
        self.n_actions = int(n_actions)
        self.horizon_steps = int(horizon_steps)
        self.hidden_dim = int(hidden_dim)
        self.architecture_version = str(architecture_version)
        self.priority_aware = self.architecture_version in {
            "priority_aware_v2",
            "priority_aware_safety_v3",
            "priority_aware_safety_v4",
            "causal_phase_safety_v5",
            "causal_phase_direction_v6",
        }
        self.has_safety_classifiers = self.architecture_version in {
            "priority_aware_safety_v3",
            "priority_aware_safety_v4",
            "causal_phase_safety_v5",
            "causal_phase_direction_v6",
        }
        self.has_action_residual_heads = self.architecture_version in {
            "priority_aware_safety_v4",
            "causal_phase_safety_v5",
            "causal_phase_direction_v6",
        }
        self.has_phase_conditioning = self.architecture_version in {
            "causal_phase_safety_v5",
            "causal_phase_direction_v6",
        }
        self.has_direction_classifiers = self.architecture_version == "causal_phase_direction_v6"
        per_head = max(8, hidden_dim // max(1, heads))
        self.actuator_input = nn.Linear(1 + actuator_feature_dim, hidden_dim)
        self.actuator_identity = nn.Embedding(n_actions, hidden_dim) if self.priority_aware else None
        self.actuator_temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.node_query = nn.Linear(1 + node_static_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, max(1, heads), dropout=dropout, batch_first=True)
        self.node_input = nn.Linear(hidden_dim * 2 + 3 + node_static_dim, hidden_dim)
        self.gat = GATConv(hidden_dim, per_head, heads=max(1, heads), dropout=dropout, add_self_loops=True)
        self.node_norm = nn.LayerNorm(per_head * max(1, heads))
        self.node_state_head = nn.Sequential(nn.Linear(per_head * max(1, heads), hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        risk_contexts = 3 if self.priority_aware else 1
        risk_input_dim = per_head * max(1, heads) * risk_contexts + 1
        self.reference_risk_head = nn.Sequential(nn.Linear(risk_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.effect_risk_head = nn.Sequential(nn.Linear(risk_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.effect_scale_head = nn.Sequential(nn.Linear(risk_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.safety_classification_head = (
            nn.Sequential(
                nn.Linear(risk_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),
            )
            if self.has_safety_classifiers
            else None
        )
        action_effect_dim = hidden_dim + 5
        self.effect_action_residual_head = (
            nn.Sequential(nn.Linear(action_effect_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
            if self.has_action_residual_heads
            else None
        )
        self.safety_action_residual_head = (
            nn.Sequential(nn.Linear(action_effect_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
            if self.has_action_residual_heads
            else None
        )
        self.phase_effect_heads = (
            nn.ModuleList([
                nn.Sequential(nn.Linear(action_effect_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
                for _ in range(4)
            ])
            if self.has_phase_conditioning
            else None
        )
        self.phase_safety_heads = (
            nn.ModuleList([
                nn.Sequential(nn.Linear(action_effect_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
                for _ in range(4)
            ])
            if self.has_phase_conditioning
            else None
        )
        self.horizon_action_temporal = (
            nn.GRU(action_effect_dim, hidden_dim, batch_first=True)
            if self.has_phase_conditioning
            else None
        )
        self.horizon_effect_head = (
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
            if self.has_phase_conditioning
            else None
        )
        self.horizon_scale_head = (
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
            if self.has_phase_conditioning
            else None
        )
        self.horizon_safety_head = (
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
            if self.has_phase_conditioning
            else None
        )
        self.horizon_direction_head = (
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
            if self.has_direction_classifiers
            else None
        )
        for head in (self.effect_action_residual_head, self.safety_action_residual_head):
            if head is not None:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        if self.has_phase_conditioning:
            for collection in (self.phase_effect_heads, self.phase_safety_heads):
                for head in collection:
                    nn.init.zeros_(head[-1].weight)
                    nn.init.zeros_(head[-1].bias)
            for head in (self.horizon_effect_head, self.horizon_safety_head, self.horizon_direction_head):
                if head is None:
                    continue
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _action_effect_features(
        delta_tokens: torch.Tensor,
        residual: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        active = active_mask.to(delta_tokens.dtype)
        denominator = torch.clamp(active.sum(dim=1, keepdim=True), min=1.0)
        pooled = (delta_tokens * active[:, :, None]).sum(dim=1) / denominator
        signed_mean = (residual * active).sum(dim=1, keepdim=True) / denominator
        absolute_mean = (residual.abs() * active).sum(dim=1, keepdim=True) / denominator
        positive_max = torch.where(active_mask, residual, torch.zeros_like(residual)).clamp(min=0.0).amax(dim=1, keepdim=True)
        negative_min = torch.where(active_mask, residual, torch.zeros_like(residual)).clamp(max=0.0).amin(dim=1, keepdim=True)
        active_fraction = active.mean(dim=1, keepdim=True)
        return torch.cat([pooled, signed_mean, absolute_mean, positive_max, negative_min, active_fraction], dim=1)

    def _encode_actions(
        self,
        sequence: torch.Tensor,
        actuator_mask: torch.Tensor,
        actuator_features: torch.Tensor,
    ) -> torch.Tensor:
        # [B,H,A] -> [B,A,H,D] -> [B,H,A,D]. Retaining every GRU token is
        # essential: an early release and a late release must not share one
        # horizon-level action embedding.
        B, H, A = sequence.shape
        features = actuator_features[None, :, None, :].expand(B, A, H, -1)
        action = sequence.transpose(1, 2).unsqueeze(-1)
        tokens = torch.cat([action, features], dim=-1)
        tokens = self.actuator_input(tokens).relu().reshape(B * A, H, self.hidden_dim)
        if self.actuator_identity is not None:
            identity = self.actuator_identity.weight[:, None, :].expand(A, H, -1)
            identity = identity[None].expand(B, -1, -1, -1).reshape(B * A, H, self.hidden_dim)
            tokens = tokens + identity
        encoded, _ = self.actuator_temporal(tokens)
        encoded = encoded.reshape(B, A, H, self.hidden_dim).transpose(1, 2)
        return encoded * actuator_mask[:, None, :, None]

    @staticmethod
    def _phase_expert(
        heads: nn.ModuleList,
        features: torch.Tensor,
        phase_index: torch.Tensor,
    ) -> torch.Tensor:
        predictions = torch.stack([head(features) for head in heads], dim=1)
        gather = phase_index[:, None, None].expand(-1, 1, predictions.shape[2])
        return predictions.gather(1, gather).squeeze(1)

    def _node_context(
        self,
        state: torch.Tensor,
        node_static: torch.Tensor,
        action_tokens: torch.Tensor,
        actuator_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N = state.shape
        queries = self.node_query(torch.cat([state[:, :, None], node_static[None].expand(B, -1, -1)], dim=-1))
        active = actuator_mask > 0.0
        empty = ~active.any(dim=1)
        safe_active = active.clone()
        if bool(empty.any()):
            safe_active[empty, 0] = True
        key_padding = ~safe_active
        attended, _ = self.cross_attention(queries, action_tokens, action_tokens, key_padding_mask=key_padding, need_weights=False)
        if bool(empty.any()):
            attended = attended.masked_fill(empty[:, None, None], 0.0)
        return attended

    def forward(
        self,
        *,
        state: torch.Tensor,
        candidate_action_seq: torch.Tensor,
        reference_action_seq: torch.Tensor,
        rain_seq: torch.Tensor,
        actuator_mask: torch.Tensor,
        actuator_features: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        action_node_map: torch.Tensor,
        priority_indices: torch.Tensor,
        storage_indices: torch.Tensor,
        phase_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if state.ndim != 2 or state.shape[1] != self.n_nodes:
            raise ValueError(f"state must be [B,{self.n_nodes}]")
        if candidate_action_seq.shape != reference_action_seq.shape:
            raise ValueError("candidate_action_seq and reference_action_seq must share shape")
        B, H, A = candidate_action_seq.shape
        if A != self.n_actions:
            raise ValueError(f"expected {self.n_actions} actions, got {A}")
        if rain_seq.ndim == 2:
            rain_seq = rain_seq[:, :, None]
        if actuator_mask.ndim == 1:
            actuator_mask = actuator_mask[None].expand(B, -1)
        if actuator_mask.shape != (B, A):
            raise ValueError("actuator_mask must be [B,A]")
        if action_node_map.shape != (A, self.n_nodes):
            raise ValueError("action_node_map must be [A,N]")
        if phase_index is None:
            phase_index = torch.zeros(B, dtype=torch.long, device=state.device)
        phase_index = phase_index.to(device=state.device, dtype=torch.long).reshape(-1)
        if phase_index.shape != (B,) or bool(((phase_index < 0) | (phase_index > 3)).any()):
            raise ValueError("phase_index must be [B] with values in {0,1,2,3}")

        masked_reference = reference_action_seq * actuator_mask[:, None, :]
        masked_delta = (candidate_action_seq - reference_action_seq) * actuator_mask[:, None, :]
        reference_tokens = self._encode_actions(masked_reference, actuator_mask, actuator_features)
        delta_tokens = self._encode_actions(masked_delta, actuator_mask, actuator_features)
        delta_scale = causal_action_scale(masked_delta)
        delta_active_mask = causal_active_actuator_mask(masked_delta) & (actuator_mask[:, None, :] > 0.0)

        depth = state
        reference_risks, delta_risks, delta_scales, node_states, priority_depths, storage_levels = [], [], [], [], [], []
        safety_logits = []
        horizon_action_features = []
        batched_edges = batch_edge_index(edge_index, B, self.n_nodes)
        static = node_static[None].expand(B, -1, -1)
        for step in range(H):
            rain = rain_seq[:, step, :1]
            reference_context = self._node_context(state, node_static, reference_tokens[:, step], actuator_mask)
            delta_context = self._node_context(
                state,
                node_static,
                delta_tokens[:, step],
                delta_active_mask[:, step].to(actuator_mask.dtype),
            )
            action_signal = torch.matmul(masked_reference[:, step, :], action_node_map).clamp(0.0, 1.0)
            delta_signal = torch.matmul(masked_delta[:, step, :], action_node_map)
            node_input = torch.cat(
                [depth[:, :, None], reference_context, delta_context, action_signal[:, :, None], delta_signal[:, :, None], static],
                dim=-1,
            )
            node_hidden = self.node_input(node_input.reshape(B * self.n_nodes, -1)).relu()
            node_hidden = self.node_norm(self.gat(node_hidden, batched_edges).relu()).reshape(B, self.n_nodes, -1)
            pooled = node_hidden.mean(dim=1)
            if self.priority_aware:
                priority_pooled = (
                    node_hidden.index_select(1, priority_indices).mean(dim=1)
                    if priority_indices.numel() else pooled
                )
                storage_pooled = (
                    node_hidden.index_select(1, storage_indices).mean(dim=1)
                    if storage_indices.numel() else pooled
                )
                risk_features = torch.cat([pooled, priority_pooled, storage_pooled, rain], dim=-1)
            else:
                risk_features = torch.cat([pooled, rain], dim=-1)
            reference_risks.append(torch.nn.functional.softplus(self.reference_risk_head(risk_features)))
            raw_effect = self.effect_risk_head(risk_features)
            if self.effect_action_residual_head is not None:
                action_effect_features = self._action_effect_features(
                    delta_tokens[:, step],
                    masked_delta[:, step, :],
                    delta_active_mask[:, step],
                )
                raw_effect = raw_effect + self.effect_action_residual_head(action_effect_features)
                horizon_action_features.append(action_effect_features)
                if self.phase_effect_heads is not None:
                    raw_effect = raw_effect + self._phase_expert(
                        self.phase_effect_heads, action_effect_features, phase_index
                    )
            delta_risks.append(raw_effect * delta_scale[:, step, :])
            delta_scales.append(torch.nn.functional.softplus(self.effect_scale_head(risk_features)) * delta_scale[:, step, :])
            if self.safety_classification_head is not None:
                logits = self.safety_classification_head(risk_features)
                if self.safety_action_residual_head is not None:
                    logits = logits + self.safety_action_residual_head(action_effect_features)
                if self.phase_safety_heads is not None:
                    logits = logits + self._phase_expert(
                        self.phase_safety_heads, action_effect_features, phase_index
                    )
                safety_logits.append(logits)
            state_delta = self.node_state_head(node_hidden).squeeze(-1)
            depth = torch.relu(depth + 0.10 * torch.tanh(state_delta) + 0.02 * rain)
            node_states.append(depth)
            priority_depths.append(depth.index_select(1, priority_indices).mean(dim=1) if priority_indices.numel() else depth.mean(dim=1))
            storage_levels.append(depth.index_select(1, storage_indices).mean(dim=1) if storage_indices.numel() else depth.mean(dim=1))

        reference_rate = torch.stack(reference_risks, dim=1)
        delta_rate = torch.stack(delta_risks, dim=1)
        delta_sigma_rate = torch.stack(delta_scales, dim=1)
        candidate_rate = torch.relu(reference_rate + delta_rate)
        pfv_rate, tfv_rate, peak_rate = candidate_rate.unbind(dim=2)
        dt_sec = 300.0
        delta_pfv = delta_rate[:, :, 0].sum(dim=1) * dt_sec
        delta_tfv = delta_rate[:, :, 1].sum(dim=1) * dt_sec
        delta_peak = candidate_rate[:, :, 1].max(dim=1).values - reference_rate[:, :, 1].max(dim=1).values
        sigma_pfv = torch.sqrt(torch.square(delta_sigma_rate[:, :, 0]).sum(dim=1)) * dt_sec
        sigma_tfv = torch.sqrt(torch.square(delta_sigma_rate[:, :, 1]).sum(dim=1)) * dt_sec
        # Channel 2 represents the running peak effect; its final horizon step
        # aligns with max(TFV_candidate)-max(TFV_reference).
        sigma_peak = delta_sigma_rate[:, -1, 2]
        horizon_hidden = None
        if self.horizon_action_temporal is not None and horizon_action_features:
            horizon_sequence = torch.stack(horizon_action_features, dim=1)
            _, horizon_hidden_state = self.horizon_action_temporal(horizon_sequence)
            horizon_hidden = horizon_hidden_state[-1]
            effect_gate = masked_delta.abs().amax(dim=(1, 2), keepdim=False)[:, None]
            aggregate_correction = self.horizon_effect_head(horizon_hidden) * effect_gate
            delta_pfv = delta_pfv + aggregate_correction[:, 0]
            delta_tfv = delta_tfv + aggregate_correction[:, 1]
            delta_peak = delta_peak + aggregate_correction[:, 2]
            horizon_sigma = torch.nn.functional.softplus(self.horizon_scale_head(horizon_hidden)) * effect_gate
            sigma_pfv = torch.sqrt(torch.square(sigma_pfv) + torch.square(horizon_sigma[:, 0]))
            sigma_tfv = torch.sqrt(torch.square(sigma_tfv) + torch.square(horizon_sigma[:, 1]))
            sigma_peak = torch.sqrt(torch.square(sigma_peak) + torch.square(horizon_sigma[:, 2]))

        def probability_nonpositive(mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            z = -mean / torch.clamp(sigma, min=1.0e-8)
            return 0.5 * (1.0 + torch.erf(z / (2.0 ** 0.5)))

        result = {
            "reference_risk_rate_seq": reference_rate,
            "delta_risk_rate_seq": delta_rate,
            "delta_risk_log_scale_seq": torch.log(torch.clamp(delta_sigma_rate, min=1.0e-8)),
            "delta_risk_sigma_seq": delta_sigma_rate,
            "candidate_risk_rate_seq": candidate_rate,
            "PFV_rate_seq": pfv_rate,
            "TFV_rate_seq": tfv_rate,
            "priority_depth_seq": torch.stack(priority_depths, dim=1),
            "storage_level_seq": torch.stack(storage_levels, dim=1),
            "node_state_seq": torch.stack(node_states, dim=1),
            "peak_TFV_rate": tfv_rate.max(dim=1).values,
            "PFV_H": pfv_rate.sum(dim=1) * dt_sec,
            "TFV_H": tfv_rate.sum(dim=1) * dt_sec,
            "delta_PFV_H": delta_pfv,
            "delta_TFV_H": delta_tfv,
            "delta_peak": delta_peak,
            "delta_PFV_sigma": sigma_pfv,
            "delta_TFV_sigma": sigma_tfv,
            "delta_peak_sigma": sigma_peak,
            "PFV_noninferiority_probability": probability_nonpositive(delta_pfv, sigma_pfv),
            "TFV_improvement_probability": probability_nonpositive(delta_tfv, sigma_tfv),
            "peak_safe_probability": probability_nonpositive(delta_peak, sigma_peak),
            "phase_index": phase_index,
        }
        if safety_logits:
            horizon_safety_logits = torch.stack(safety_logits, dim=1).mean(dim=1)
            if self.horizon_safety_head is not None and horizon_hidden is not None:
                horizon_safety_logits = horizon_safety_logits + self.horizon_safety_head(horizon_hidden)
            result.update({
                "safety_classification_logits": horizon_safety_logits,
                "PFV_noninferiority_classifier_probability": torch.sigmoid(horizon_safety_logits[:, 0]),
                "TFV_improvement_classifier_probability": torch.sigmoid(horizon_safety_logits[:, 1]),
                "peak_safe_classifier_probability": torch.sigmoid(horizon_safety_logits[:, 2]),
            })
        if self.horizon_direction_head is not None and horizon_hidden is not None:
            effect_gate = masked_delta.abs().amax(dim=(1, 2), keepdim=False)[:, None]
            direction_logits = self.horizon_direction_head(horizon_hidden) * effect_gate
            result.update({
                "direction_classification_logits": direction_logits,
                "PFV_direction_improve_probability": torch.sigmoid(direction_logits[:, 0]),
                "TFV_direction_improve_probability": torch.sigmoid(direction_logits[:, 1]),
                "peak_direction_improve_probability": torch.sigmoid(direction_logits[:, 2]),
            })
        return result
