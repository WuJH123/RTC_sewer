"""Physically defensible regularisation terms for V4.2.

Important
---------
Older V4.2 code estimated link flow from depth differences, assumed a uniform
100 m² ponded area, forced Candidate and Reference to be equal at t+10, and
compared depth proxies (metres) with KPI targets expressed in m³ / z-score
space.  Those terms were not SWMM mass balance constraints and could actively
suppress the control-effect signal.

This module therefore follows a fail-safe rule:

* keep only constraints that are valid for the variables the current model
  actually predicts (non-negative depth, configured depth capacity and a mild
  temporal smoothness regulariser);
* return graph-connected zeros for unsupported physics terms so existing
  logging schemas remain stable;
* only activate mass/KPI/peak consistency when the caller supplies explicit,
  unit-compatible residuals/quantities.

When the model is extended to predict link flow, storage volume, external
inflow, outfall flow and node flooding rate, true continuity constraints can be
added without changing these public loss keys.
"""
from __future__ import annotations

import torch
from torch import nn


class PhysicsLosses(nn.Module):
    """Compute physics/constraint losses without inventing hydraulic variables."""

    def __init__(
        self,
        n_nodes: int,
        node_max_depth: torch.Tensor | None = None,
        dt_sec: float = 600.0,
        ponded_area: float | None = None,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.dt_sec = float(dt_sec)
        # ``ponded_area`` is retained only for API compatibility.  It is not
        # used to fabricate storage volume from depth.
        self.ponded_area = None if ponded_area is None else float(ponded_area)
        if node_max_depth is not None:
            self.register_buffer("node_max_depth", node_max_depth.float())
        else:
            self.register_buffer("node_max_depth", torch.ones(n_nodes) * 5.0)

    @staticmethod
    def _zero_connected(x: torch.Tensor) -> torch.Tensor:
        """Return a differentiable zero connected to the prediction graph."""
        return x.sum() * 0.0

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor] | None = None,
        edge_index: torch.Tensor | None = None,
        node_static: torch.Tensor | None = None,
        action_node_map: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if "y_candidate" not in pred or "y_reference" not in pred:
            raise KeyError("PhysicsLosses requires y_candidate and y_reference")

        y_cand = pred["y_candidate"]
        y_ref = pred["y_reference"]
        if y_cand.shape != y_ref.shape:
            raise ValueError("Candidate and reference depth trajectories must align")
        _, H, N = y_cand.shape
        if N != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} nodes, got {N}")

        zero = self._zero_connected(y_cand)
        losses: dict[str, torch.Tensor] = {}

        # 1. True mass balance is only available if the model/caller supplies an
        # explicit residual in volume units.  Never infer flow from depth
        # differences.
        if "mass_balance_residual_m3" in pred:
            losses["mass_balance"] = pred["mass_balance_residual_m3"].abs().mean()
        else:
            losses["mass_balance"] = zero

        # 2. A mild temporal smoothness term.  This is a regulariser rather than
        # a claim of exact SWMM storage continuity, but is unit-consistent.
        if H >= 3:
            d1 = y_cand[:, 1:, :] - y_cand[:, :-1, :]
            d2 = d1[:, 1:, :] - d1[:, :-1, :]
            losses["storage_continuity"] = (d2 ** 2).mean()
        else:
            losses["storage_continuity"] = zero

        # 3. Non-negative depths for every predicted branch that exists.
        branch_depths = [y_cand, y_ref]
        for key in ("y_dynamic_internal", "y_hold_previous"):
            if key in pred:
                branch_depths.append(pred[key])
        losses["non_negative"] = sum(torch.relu(-x).mean() for x in branch_depths)

        # 4. Configured depth-capacity bound.  Outfalls/unknown zero depths are
        # sanitised by the data loader before reaching this module.
        max_depth = self.node_max_depth[None, None, :]
        losses["capacity_bounds"] = sum(
            torch.relu(x - max_depth).mean() for x in branch_depths
        )

        # 5. Flooding consistency cannot be inferred from depth exceedance alone
        # in SWMM.  It is activated only when an explicit residual is provided.
        if "flooding_consistency_residual_m3s" in pred:
            losses["flooding_consistency"] = pred[
                "flooding_consistency_residual_m3s"
            ].abs().mean()
        else:
            losses["flooding_consistency"] = zero

        # 6. Candidate and references share the *checkpoint* state by data
        # construction.  They are allowed to diverge immediately after actions
        # are applied, so no t+10 equality penalty is imposed here.
        losses["shared_init_state"] = zero

        # 7. KPI/trajectory consistency is valid only when both sides are given
        # in matching physical units.  The old depth-proxy-vs-m³ comparison is
        # deliberately disabled.
        kpi_terms = []
        if "pfv_from_trajectory_m3" in pred and "pfv_delta" in pred:
            kpi_terms.append(
                (pred["pfv_from_trajectory_m3"] - pred["pfv_delta"]).abs().mean()
            )
        if "tfv_from_trajectory_m3" in pred and "tfv_delta" in pred:
            kpi_terms.append(
                (pred["tfv_from_trajectory_m3"] - pred["tfv_delta"]).abs().mean()
            )
        losses["kpi_trajectory_consistency"] = sum(kpi_terms) if kpi_terms else zero

        # 8. Peak consistency requires a true total-flooding-rate sequence in
        # m³/s.  A sequence of mean node depths is not an acceptable proxy.
        if "peak_rate_sequence_m3s" in pred and "peak_delta" in pred:
            peak_from_seq = pred["peak_rate_sequence_m3s"].max(dim=1).values
            losses["peak_consistency"] = (
                peak_from_seq - pred["peak_delta"]
            ).abs().mean()
        else:
            losses["peak_consistency"] = zero

        return losses
