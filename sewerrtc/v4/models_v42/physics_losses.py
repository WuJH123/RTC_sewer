"""Physically defensible regularisation terms for V4.2.

Important
---------
Older V4.2 code estimated link flow from depth differences, assumed a uniform
100 m² ponded area, forced Candidate and Reference to be equal at t+10, and
compared depth proxies (metres) with KPI targets expressed in m³ / z-score
space.  Those terms were not SWMM mass-balance constraints and could actively
suppress the control-effect signal.

A second subtle issue is that ``node_feature_matrix`` standardises static node
features.  A z-scored ``max_depth`` must never be used as a physical depth
limit in metres.  Capacity loss is therefore fail-closed unless the caller
explicitly declares that ``node_max_depth`` is in physical metres.
"""
from __future__ import annotations

import torch
from torch import nn


class PhysicsLosses(nn.Module):
    """Compute only constraints supported by explicitly valid physical inputs."""

    def __init__(
        self,
        n_nodes: int,
        node_max_depth: torch.Tensor | None = None,
        dt_sec: float = 600.0,
        ponded_area: float | None = None,
        *,
        node_max_depth_is_physical_m: bool = False,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.dt_sec = float(dt_sec)
        self.ponded_area = None if ponded_area is None else float(ponded_area)
        self.capacity_active = bool(
            node_max_depth is not None and node_max_depth_is_physical_m
        )
        if self.capacity_active:
            depth = node_max_depth.detach().float().clone()
            if depth.ndim != 1 or depth.numel() != self.n_nodes:
                raise ValueError(
                    f"Physical node_max_depth must have shape [{self.n_nodes}], "
                    f"got {tuple(depth.shape)}"
                )
            if not torch.isfinite(depth).all():
                raise ValueError("Physical node_max_depth contains NaN/Inf")
            # Zero/negative limits commonly denote outfalls or unknown limits;
            # they are not valid capacity constraints.  Mark them unbounded.
            depth = torch.where(depth > 0.0, depth, torch.full_like(depth, float("inf")))
            self.register_buffer("node_max_depth", depth)
        else:
            self.register_buffer("node_max_depth", torch.full((self.n_nodes,), float("inf")))

    @staticmethod
    def _zero_connected(x: torch.Tensor) -> torch.Tensor:
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

        # True mass balance is only available if the caller supplies a residual
        # in physical volume units.  Never infer flow from node-depth differences.
        if "mass_balance_residual_m3" in pred:
            losses["mass_balance"] = pred["mass_balance_residual_m3"].abs().mean()
        else:
            losses["mass_balance"] = zero

        # Mild temporal curvature regularisation; this is not labelled as exact
        # SWMM storage continuity in scientific reporting.
        if H >= 3:
            d1 = y_cand[:, 1:, :] - y_cand[:, :-1, :]
            d2 = d1[:, 1:, :] - d1[:, :-1, :]
            losses["storage_continuity"] = (d2 ** 2).mean()
        else:
            losses["storage_continuity"] = zero

        branch_depths = [y_cand, y_ref]
        for key in ("y_dynamic_internal", "y_hold_previous"):
            if key in pred:
                branch_depths.append(pred[key])
        losses["non_negative"] = sum(torch.relu(-x).mean() for x in branch_depths)

        # Capacity bounds are active only when raw INP max-depth values in metres
        # have been explicitly supplied.  Standardised graph features are not
        # accepted as physical limits.
        if self.capacity_active:
            max_depth = self.node_max_depth[None, None, :]
            losses["capacity_bounds"] = sum(
                torch.relu(x - max_depth).mean() for x in branch_depths
            )
        else:
            losses["capacity_bounds"] = zero

        if "flooding_consistency_residual_m3s" in pred:
            losses["flooding_consistency"] = pred[
                "flooding_consistency_residual_m3s"
            ].abs().mean()
        else:
            losses["flooding_consistency"] = zero

        # Same-state equality belongs at the checkpoint in the data contract.
        # The first predicted t+10 state is allowed to diverge after actions.
        losses["shared_init_state"] = zero

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

        if "peak_rate_sequence_m3s" in pred and "peak_delta" in pred:
            peak_from_seq = pred["peak_rate_sequence_m3s"].max(dim=1).values
            losses["peak_consistency"] = (
                peak_from_seq - pred["peak_delta"]
            ).abs().mean()
        else:
            losses["peak_consistency"] = zero

        return losses
