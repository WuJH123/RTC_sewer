"""Trajectory-first multi-reference hydraulic surrogate for the V4.2 paper line.

The formal model is a shared dynamics function

    Y_b = F_theta(X[t-60:t], R[t:t+120], U_b, G)

executed for Candidate, No-control, Dynamic Internal and Hold Previous.  Unlike
legacy V4.2 KPI heads, this module predicts hydraulic trajectories first and
*derives* PFV/TFV/Peak from predicted node flooding-rate trajectories.

Predicted branch trajectories
-----------------------------
* node depth [m]
* node flooding rate [m3/s]
* storage volume [m3] at declared storage nodes
* managed-facility flow [m3/s]
* outfall flow [m3/s] at declared outfall nodes
* system flooding-rate sequence [m3/s]

The storage/facility/outfall heads are learnable hydraulic outputs and therefore
must be supervised by real SWMM/telemetry targets before formal admission.  The
module does not fabricate missing targets and it is not a Policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .actuator_action_encoder import ActuatorActionEncoder
from .rainfall_encoder import RainfallEncoder

try:
    from torch_geometric.nn import GATConv
except Exception as exc:  # pragma: no cover
    GATConv = None
    _PYG_IMPORT_ERROR = exc


@dataclass(frozen=True)
class HydraulicKPIBundle:
    pfv_m3: torch.Tensor
    tfv_m3: torch.Tensor
    peak_m3s: torch.Tensor


def _require_pyg() -> None:
    if GATConv is None:
        raise RuntimeError(
            "torch_geometric is required for V4.2 hydraulic surrogate"
        ) from _PYG_IMPORT_ERROR


def _batch_edge_index(
    edge_index: torch.Tensor, batch_size: int, n_nodes: int
) -> torch.Tensor:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    offsets = (
        torch.arange(batch_size, device=edge_index.device)
        .repeat_interleave(edge_index.shape[1])
        * n_nodes
    )
    return edge_index.repeat(1, batch_size) + offsets.unsqueeze(0)


def _as_index_tensor(
    indices: Iterable[int], *, device: torch.device
) -> torch.Tensor:
    values = [int(i) for i in indices]
    if not values:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.as_tensor(values, dtype=torch.long, device=device)


class HydraulicHistoryEncoder(nn.Module):
    """Encode 13-frame hydraulic state + historical actions before GAT."""

    def __init__(
        self,
        *,
        n_nodes: int,
        n_facilities: int,
        state_feature_dim: int,
        static_feature_dim: int,
        hidden_dim: int,
        gat_heads: int,
        gat_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _require_pyg()
        self.n_nodes = int(n_nodes)
        self.n_facilities = int(n_facilities)
        self.state_feature_dim = int(state_feature_dim)
        self.hidden_dim = int(hidden_dim)

        # Each node receives its own hydraulic feature vector plus one causal
        # action-context scalar formed only from facilities touching that node.
        self.temporal_gru = nn.GRU(
            input_size=state_feature_dim + 1,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.input_proj = nn.Linear(
            hidden_dim + static_feature_dim, hidden_dim
        )

        per_head = max(4, hidden_dim // gat_heads)
        gat_out = per_head * gat_heads
        self.gats = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()
        for layer in range(gat_layers):
            layer_in = hidden_dim if layer == 0 else gat_out
            self.gats.append(
                GATConv(
                    layer_in,
                    per_head,
                    heads=gat_heads,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            self.norms.append(nn.LayerNorm(gat_out))
            self.skips.append(
                nn.Identity()
                if layer_in == gat_out
                else nn.Linear(layer_in, gat_out, bias=False)
            )
        self.output_dim = gat_out

    def forward(
        self,
        state_history: torch.Tensor,
        historical_actions: torch.Tensor,
        edge_index: torch.Tensor,
        node_static: torch.Tensor,
        action_node_map: torch.Tensor,
    ) -> torch.Tensor:
        if state_history.ndim == 3:
            state_history = state_history.unsqueeze(-1)
        if state_history.ndim != 4:
            raise ValueError(
                "state_history must be [B,T,N] or [B,T,N,F]"
            )
        B, T, N, F = state_history.shape
        if N != self.n_nodes or F != self.state_feature_dim:
            raise ValueError(
                f"state_history expected [B,T,{self.n_nodes},{self.state_feature_dim}], "
                f"got {tuple(state_history.shape)}"
            )
        if T != 13:
            raise ValueError(
                f"formal V4.2 history requires 13 frames, got {T}"
            )
        if historical_actions.ndim != 3 or historical_actions.shape != (
            B,
            T,
            self.n_facilities,
        ):
            raise ValueError(
                f"historical_actions must be [B,13,{self.n_facilities}]"
            )
        if action_node_map.shape != (self.n_facilities, self.n_nodes):
            raise ValueError("action_node_map shape mismatch")
        if node_static.ndim != 2 or node_static.shape[0] != N:
            raise ValueError("node_static must be [N,F_static]")

        # Normalise only the incidence aggregation, not hydraulic values.
        incidence = action_node_map.abs()
        denom = incidence.sum(dim=0).clamp_min(1.0)
        node_action = (
            torch.einsum("bta,an->btn", historical_actions, incidence)
            / denom[None, None, :]
        )
        temporal_input = torch.cat(
            [state_history, node_action.unsqueeze(-1)], dim=-1
        )
        seq = temporal_input.permute(0, 2, 1, 3).reshape(
            B * N, T, F + 1
        )
        _, h_last = self.temporal_gru(seq)
        hist = h_last[-1].reshape(B, N, self.hidden_dim)

        static = node_static[None, :, :].expand(B, -1, -1)
        x = self.input_proj(
            torch.cat([hist, static], dim=-1).reshape(B * N, -1)
        ).relu()
        be = _batch_edge_index(edge_index, B, N)
        for gat, norm, skip in zip(self.gats, self.norms, self.skips):
            h = gat(x, be)
            x = norm(h + skip(x)).relu()
        return x.reshape(B, N, -1)


class MultiReferenceHydraulicSurrogate(nn.Module):
    """Shared four-branch hydraulic dynamics with trajectory-derived KPIs."""

    BRANCHES = (
        "candidate",
        "no_control",
        "dynamic_internal",
        "hold_previous",
    )

    def __init__(
        self,
        *,
        n_nodes: int,
        n_facilities: int,
        state_feature_dim: int = 1,
        static_feature_dim: int = 7,
        hidden_dim: int = 32,
        gat_heads: int = 4,
        gat_layers: int = 2,
        horizon: int = 12,
        dt_sec: float = 600.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.n_facilities = int(n_facilities)
        self.horizon = int(horizon)
        self.dt_sec = float(dt_sec)
        self.hidden_dim = int(hidden_dim)

        self.history_encoder = HydraulicHistoryEncoder(
            n_nodes=n_nodes,
            n_facilities=n_facilities,
            state_feature_dim=state_feature_dim,
            static_feature_dim=static_feature_dim,
            hidden_dim=hidden_dim,
            gat_heads=gat_heads,
            gat_layers=gat_layers,
            dropout=dropout,
        )
        state_dim = self.history_encoder.output_dim
        self.rain_encoder = RainfallEncoder(
            horizon=horizon, hidden_dim=hidden_dim
        )
        self.action_encoder = ActuatorActionEncoder(
            n_facilities=n_facilities,
            hidden_dim=hidden_dim,
            horizon=horizon,
        )
        # Action effects must propagate through the frozen hydraulic graph
        # during the forecast, not only at the two facility endpoint nodes.
        # Without this layer, PFV nodes outside an actuator's direct incidence
        # receive no candidate-action signal in the node-wise rollout.
        per_head = max(4, hidden_dim // gat_heads)
        rollout_dim = per_head * gat_heads
        self.rollout_gat = GATConv(
            hidden_dim,
            per_head,
            heads=gat_heads,
            dropout=dropout,
            add_self_loops=True,
        )
        self.rollout_norm = nn.LayerNorm(rollout_dim)
        self.rollout_skip = (
            nn.Identity()
            if rollout_dim == hidden_dim
            else nn.Linear(hidden_dim, rollout_dim, bias=False)
        )
        # Control influence can propagate upstream as well as downstream.  The
        # directed hydraulic edge list is retained for the GAT dynamics, but
        # this separate incidence prior must be bidirectional.  On the frozen
        # Wuhan graph, four directed hops reached only 1/8 PFV_CORE nodes;
        # twelve undirected hops cover the complete actuator-to-PFV envelope.
        self.action_diffusion_hops = 12
        self.action_diffusion_decay = 0.6
        self.action_diffusion_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Preserve actuator identity for remote PFV nodes.  The scalar graph
        # diffusion is a locality prior; it is not sufficient by itself after
        # long-path averaging across 36 facilities.
        self.global_action_proj = nn.Sequential(
            nn.Linear(n_facilities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_depth_residual = nn.Linear(hidden_dim, 1)
        self.action_flood_residual = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.action_depth_residual.weight)
        nn.init.zeros_(self.action_depth_residual.bias)
        nn.init.zeros_(self.action_flood_residual.weight)
        nn.init.zeros_(self.action_flood_residual.bias)
        self.state_to_hidden = nn.Linear(state_dim, hidden_dim)
        self.dynamics = nn.GRUCell(hidden_dim * 3, hidden_dim)

        self.depth_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.flood_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.storage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.outfall_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.facility_flow_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _diffuse_action_signal(
        self,
        action_schedule: torch.Tensor,
        action_node_map: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Diffuse facility settings over a frozen graph for action context."""
        B, H, _ = action_schedule.shape
        N = action_node_map.shape[1]
        direct = torch.einsum(
            "bha,an->bhn", action_schedule, action_node_map.abs()
        )
        nodes = torch.arange(N, device=edge_index.device)
        self_edges = torch.stack([nodes, nodes])
        undirected_edges = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edges = torch.cat([undirected_edges, self_edges], dim=1)
        degree = torch.bincount(edges[1], minlength=N).to(action_schedule.dtype)
        values = 1.0 / degree[edges[1]].clamp_min(1.0)
        adjacency = torch.sparse_coo_tensor(
            torch.stack([edges[1], edges[0]]),
            values,
            size=(N, N),
            device=action_schedule.device,
        ).coalesce()
        current = direct.reshape(B * H, N)
        total = current.clone()
        weight = 1.0
        for hop in range(1, self.action_diffusion_hops + 1):
            current = torch.sparse.mm(adjacency, current.T).T
            hop_weight = self.action_diffusion_decay**hop
            total = total + hop_weight * current
            weight += hop_weight
        return (total / weight).reshape(B, H, N)

    def _rollout_branch(
        self,
        *,
        initial_state: torch.Tensor,
        rain_embed: torch.Tensor,
        action_schedule: torch.Tensor,
        action_node_map: torch.Tensor,
        edge_index: torch.Tensor,
        storage_indices: torch.Tensor,
        outfall_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, N, _ = initial_state.shape
        if action_schedule.shape != (
            B,
            self.horizon,
            self.n_facilities,
        ):
            raise ValueError(
                f"action schedule must be [B,{self.horizon},{self.n_facilities}]"
            )
        action_embed = self.action_encoder(
            action_schedule, action_node_map
        )
        action_signal = self._diffuse_action_signal(
            action_schedule, action_node_map, edge_index
        )
        action_diffused_embed = self.action_diffusion_proj(
            action_signal.unsqueeze(-1)
        )
        batched_edge_index = _batch_edge_index(edge_index, B, N)
        h = self.state_to_hidden(initial_state).reshape(
            B * N, self.hidden_dim
        )

        depths: list[torch.Tensor] = []
        floods: list[torch.Tensor] = []
        hidden_steps: list[torch.Tensor] = []
        prev_depth: torch.Tensor | None = None
        for k in range(self.horizon):
            rain_k = (
                rain_embed[:, k, :][:, None, :]
                .expand(B, N, -1)
                .reshape(B * N, -1)
            )
            act_k = (
                action_embed[:, k, :, :] + action_diffused_embed[:, k, :, :]
                + self.global_action_proj(action_schedule[:, k, :])[:, None, :]
            ).reshape(B * N, -1)
            action_conditioned = h + act_k
            propagated = self.rollout_gat(action_conditioned, batched_edge_index)
            h_context = self.rollout_norm(
                propagated + self.rollout_skip(action_conditioned)
            ).relu()
            inp = torch.cat([h_context, rain_k, act_k], dim=-1)
            h = self.dynamics(inp, h)
            hidden_node = h.reshape(B, N, self.hidden_dim)
            hidden_steps.append(hidden_node)

            action_node = act_k.reshape(B, N, self.hidden_dim)
            raw_depth = (
                self.depth_head(h).reshape(B, N)
                + self.action_depth_residual(action_node).squeeze(-1)
            )
            if prev_depth is None:
                depth = torch.nn.functional.softplus(raw_depth)
            else:
                depth = torch.relu(
                    prev_depth + 0.25 * torch.tanh(raw_depth)
                )
            flood = torch.nn.functional.softplus(
                self.flood_head(h).reshape(B, N)
                + self.action_flood_residual(action_node).squeeze(-1)
            )
            depths.append(depth)
            floods.append(flood)
            prev_depth = depth

        hidden_seq = torch.stack(hidden_steps, dim=1)
        node_depth = torch.stack(depths, dim=1)
        node_flood = torch.stack(floods, dim=1)
        system_flood = node_flood.sum(dim=-1)

        if storage_indices.numel():
            storage_hidden = hidden_seq.index_select(2, storage_indices)
            storage_volume = torch.nn.functional.softplus(
                self.storage_head(storage_hidden).squeeze(-1)
            )
        else:
            storage_volume = hidden_seq.new_zeros(
                (B, self.horizon, 0)
            )

        if outfall_indices.numel():
            outfall_hidden = hidden_seq.index_select(2, outfall_indices)
            outfall_flow = torch.nn.functional.softplus(
                self.outfall_head(outfall_hidden).squeeze(-1)
            )
        else:
            outfall_flow = hidden_seq.new_zeros(
                (B, self.horizon, 0)
            )

        incidence = action_node_map.abs()
        denom = incidence.sum(dim=1).clamp_min(1.0)
        facility_hidden = torch.einsum(
            "an,bhnd->bhad", incidence, hidden_seq
        )
        facility_hidden = (
            facility_hidden / denom[None, None, :, None]
        )
        facility_input = torch.cat(
            [facility_hidden, action_schedule.unsqueeze(-1)], dim=-1
        )
        facility_flow = self.facility_flow_head(
            facility_input
        ).squeeze(-1)

        return {
            "node_depth": node_depth,
            "node_flooding_rate": node_flood,
            "storage_volume": storage_volume,
            "facility_flow": facility_flow,
            "outfall_flow": outfall_flow,
            "system_flooding_rate": system_flood,
            "hidden_sequence": hidden_seq,
        }

    def _derive_kpi(
        self,
        branch: dict[str, torch.Tensor],
        priority_indices: torch.Tensor,
    ) -> HydraulicKPIBundle:
        flood = branch["node_flooding_rate"]
        if priority_indices.numel() == 0:
            raise ValueError("priority_node_indices cannot be empty")
        priority_rate = flood.index_select(
            2, priority_indices
        ).sum(dim=-1)
        system_rate = branch["system_flooding_rate"]
        return HydraulicKPIBundle(
            pfv_m3=priority_rate.sum(dim=1) * self.dt_sec,
            tfv_m3=system_rate.sum(dim=1) * self.dt_sec,
            peak_m3s=system_rate.max(dim=1).values,
        )

    def forward(
        self,
        *,
        state_history: torch.Tensor,
        historical_actions: torch.Tensor,
        rainfall_forecast: torch.Tensor,
        action_candidate: torch.Tensor,
        action_no_control: torch.Tensor,
        action_dynamic_internal: torch.Tensor,
        action_hold_previous: torch.Tensor,
        edge_index: torch.Tensor,
        node_static: torch.Tensor,
        action_node_map: torch.Tensor,
        priority_node_indices: torch.Tensor,
        storage_node_indices: torch.Tensor | None = None,
        outfall_node_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Roll all four branches with one shared F_theta.

        No branch may be omitted or silently substituted. Returned scientific
        KPI deltas are deterministic functions of predicted flooding-rate
        trajectories; there are no free-standing KPI regression heads.
        """
        B = state_history.shape[0]
        if rainfall_forecast.shape != (B, self.horizon):
            raise ValueError(
                f"rainfall_forecast must be [B,{self.horizon}]"
            )
        if action_node_map.shape != (
            self.n_facilities,
            self.n_nodes,
        ):
            raise ValueError("action_node_map shape mismatch")
        priority_idx = torch.as_tensor(
            priority_node_indices,
            dtype=torch.long,
            device=state_history.device,
        ).reshape(-1)
        storage_idx = _as_index_tensor(
            [] if storage_node_indices is None else storage_node_indices,
            device=state_history.device,
        )
        outfall_idx = _as_index_tensor(
            [] if outfall_node_indices is None else outfall_node_indices,
            device=state_history.device,
        )
        for name, idx in (
            ("priority", priority_idx),
            ("storage", storage_idx),
            ("outfall", outfall_idx),
        ):
            if idx.numel() and (
                (idx < 0).any() or (idx >= self.n_nodes).any()
            ):
                raise ValueError(f"{name} node index outside graph")

        initial = self.history_encoder(
            state_history,
            historical_actions,
            edge_index,
            node_static,
            action_node_map,
        )
        rain_embed = self.rain_encoder(rainfall_forecast)
        schedules = {
            "candidate": action_candidate,
            "no_control": action_no_control,
            "dynamic_internal": action_dynamic_internal,
            "hold_previous": action_hold_previous,
        }
        branches: dict[str, dict[str, torch.Tensor]] = {}
        kpis: dict[str, HydraulicKPIBundle] = {}
        for name in self.BRANCHES:
            branches[name] = self._rollout_branch(
                initial_state=initial,
                rain_embed=rain_embed,
                action_schedule=schedules[name],
                action_node_map=action_node_map,
                edge_index=edge_index,
                storage_indices=storage_idx,
                outfall_indices=outfall_idx,
            )
            kpis[name] = self._derive_kpi(
                branches[name], priority_idx
            )

        candidate = kpis["candidate"]
        no_control = kpis["no_control"]
        dynamic_internal = kpis["dynamic_internal"]

        # Integrals are linear, so compute the counterfactual *rate difference*
        # before summing.  This avoids catastrophic cancellation when C and NC
        # (or C and DI) each have large absolute volumes but a small scientific
        # delta.  Peak is intentionally different: by definition it is
        # max(C)-max(DI), not max(C-DI).
        cand_flood = branches["candidate"]["node_flooding_rate"]
        nc_flood = branches["no_control"]["node_flooding_rate"]
        cand_priority = cand_flood.index_select(2, priority_idx)
        nc_priority = nc_flood.index_select(2, priority_idx)
        pfv_delta = (
            (cand_priority - nc_priority).sum(dim=(1, 2))
            * self.dt_sec
        )
        tfv_delta = (
            (
                branches["candidate"]["system_flooding_rate"]
                - branches["dynamic_internal"]["system_flooding_rate"]
            ).sum(dim=1)
            * self.dt_sec
        )
        peak_delta = candidate.peak_m3s - dynamic_internal.peak_m3s

        return {
            "branches": branches,
            "kpi_candidate": {
                "pfv_m3": candidate.pfv_m3,
                "tfv_m3": candidate.tfv_m3,
                "peak_m3s": candidate.peak_m3s,
            },
            "kpi_no_control": {
                "pfv_m3": no_control.pfv_m3,
                "tfv_m3": no_control.tfv_m3,
                "peak_m3s": no_control.peak_m3s,
            },
            "kpi_dynamic_internal": {
                "pfv_m3": dynamic_internal.pfv_m3,
                "tfv_m3": dynamic_internal.tfv_m3,
                "peak_m3s": dynamic_internal.peak_m3s,
            },
            "pfv_delta": pfv_delta,
            "tfv_delta": tfv_delta,
            "peak_delta": peak_delta,
            "metadata": {
                "role": "hydraulic_surrogate_not_policy",
                "shared_parameters_across_branches": True,
                "kpis_derived_from_flooding_rate_trajectory": True,
                "volume_delta_integrated_from_rate_difference": True,
                "peak_definition": "max_candidate_minus_max_dynamic_internal",
                "future_graph_message_passing": True,
                "action_effect_contract": "bidirectional_graph_propagated_action_effect_v3",
                "action_influence_hops": self.action_diffusion_hops,
                "action_influence_decay": self.action_diffusion_decay,
                "action_influence_bidirectional": True,
                "global_action_identity_context": True,
                "explicit_action_hydraulic_residuals": True,
                "dt_sec": self.dt_sec,
            },
        }
