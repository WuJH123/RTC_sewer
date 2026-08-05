"""Losses for the trajectory-first V4.2 hydraulic surrogate.

Formal training supervises hydraulics, not an independent KPI shortcut.  KPI
losses may be used only on the already-derived PFV/TFV/Peak outputs and cannot
replace node flooding-rate supervision.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HydraulicLossWeights:
    depth: float = 1.0
    node_flooding: float = 1.0
    storage: float = 0.5
    facility_flow: float = 0.5
    outfall_flow: float = 0.5
    kpi_consistency: float = 0.2
    priority_flooding: float = 0.0
    action_effect: float = 0.0
    tfv_direction: float = 0.0
    tfv_ranking: float = 0.0


class HydraulicTrajectoryLoss(nn.Module):
    """Four-branch hydraulic supervision with fail-closed target coverage."""

    BRANCHES = ("candidate", "no_control", "dynamic_internal", "hold_previous")
    TARGET_PREFIX = {
        "candidate": "candidate",
        "no_control": "no_control",
        "dynamic_internal": "dynamic_internal",
        "hold_previous": "hold_previous",
    }

    def __init__(
        self,
        weights: HydraulicLossWeights | None = None,
        *,
        require_storage_targets: bool = True,
        require_facility_flow_targets: bool = True,
        require_outfall_flow_targets: bool = True,
        priority_node_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or HydraulicLossWeights()
        self.require_storage_targets = bool(require_storage_targets)
        self.require_facility_flow_targets = bool(require_facility_flow_targets)
        self.require_outfall_flow_targets = bool(require_outfall_flow_targets)
        self.priority_node_indices = (
            None
            if priority_node_indices is None
            else priority_node_indices.detach().to(dtype=torch.long).reshape(-1)
        )

    @staticmethod
    def _masked_smooth_l1(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if mask is None:
            if not torch.isfinite(target).all():
                raise ValueError("target contains NaN/Inf without availability mask")
            return nn.functional.smooth_l1_loss(pred, target, reduction="mean")
        mask = mask.bool()
        if mask.shape != target.shape:
            try:
                mask = mask.expand_as(target)
            except RuntimeError as exc:
                raise ValueError("availability mask shape mismatch") from exc
        valid = mask & torch.isfinite(target)
        if not bool(valid.any()):
            # No fabricated zero target.  Caller/audit decides whether zero
            # coverage is admissible for a development-only dataset.
            return pred.sum() * 0.0
        return nn.functional.smooth_l1_loss(pred[valid], target[valid], reduction="mean")

    @staticmethod
    def _target_key(branch: str, quantity: str) -> str:
        return f"trajectory_{quantity}_{HydraulicTrajectoryLoss.TARGET_PREFIX[branch]}"

    def _require_target_group(
        self,
        target: dict[str, torch.Tensor],
        quantity: str,
        required: bool,
    ) -> None:
        missing = [
            self._target_key(branch, quantity)
            for branch in self.BRANCHES
            if self._target_key(branch, quantity) not in target
        ]
        if required and missing:
            raise KeyError(
                f"Formal hydraulic supervision missing {quantity} targets: {missing}"
            )

    def forward(
        self,
        pred: dict,
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if "branches" not in pred:
            raise KeyError("trajectory-first model output is missing branches")
        branches = pred["branches"]
        missing_branches = [b for b in self.BRANCHES if b not in branches]
        if missing_branches:
            raise KeyError(f"model output missing branches: {missing_branches}")

        self._require_target_group(target, "depth", True)
        self._require_target_group(target, "flood", True)
        self._require_target_group(target, "storage_volume", self.require_storage_targets)
        self._require_target_group(target, "facility_flow", self.require_facility_flow_targets)
        self._require_target_group(target, "outfall_flow", self.require_outfall_flow_targets)

        losses: dict[str, torch.Tensor] = {}
        depth_terms: list[torch.Tensor] = []
        flood_terms: list[torch.Tensor] = []
        storage_terms: list[torch.Tensor] = []
        facility_terms: list[torch.Tensor] = []
        outfall_terms: list[torch.Tensor] = []

        for branch in self.BRANCHES:
            bpred = branches[branch]
            depth_terms.append(
                self._masked_smooth_l1(
                    bpred["node_depth"], target[self._target_key(branch, "depth")]
                )
            )
            flood_terms.append(
                self._masked_smooth_l1(
                    bpred["node_flooding_rate"],
                    target[self._target_key(branch, "flood")],
                )
            )

            storage_key = self._target_key(branch, "storage_volume")
            if storage_key in target:
                storage_terms.append(
                    self._masked_smooth_l1(
                        bpred["storage_volume"],
                        target[storage_key],
                        target.get(storage_key + "_available"),
                    )
                )
            facility_key = self._target_key(branch, "facility_flow")
            if facility_key in target:
                facility_terms.append(
                    self._masked_smooth_l1(
                        bpred["facility_flow"],
                        target[facility_key],
                        target.get(facility_key + "_available"),
                    )
                )
            outfall_key = self._target_key(branch, "outfall_flow")
            if outfall_key in target:
                outfall_terms.append(
                    self._masked_smooth_l1(
                        bpred["outfall_flow"],
                        target[outfall_key],
                        target.get(outfall_key + "_available"),
                    )
                )

        zero = branches["candidate"]["node_depth"].sum() * 0.0
        losses["depth_trajectory"] = torch.stack(depth_terms).mean()
        losses["flooding_trajectory"] = torch.stack(flood_terms).mean()
        losses["storage_trajectory"] = (
            torch.stack(storage_terms).mean() if storage_terms else zero
        )
        losses["facility_flow_trajectory"] = (
            torch.stack(facility_terms).mean() if facility_terms else zero
        )
        losses["outfall_flow_trajectory"] = (
            torch.stack(outfall_terms).mean() if outfall_terms else zero
        )

        # These are consistency terms on trajectory-derived outputs only.  There
        # is no separate free-standing KPI head in the formal model.
        kpi_terms: list[torch.Tensor] = []
        for key in ("pfv_delta", "tfv_delta", "peak_delta"):
            if key in target:
                if key not in pred:
                    raise KeyError(f"derived KPI {key} missing from model output")
                if not torch.isfinite(target[key]).all():
                    raise ValueError(f"target {key} contains NaN/Inf")
                kpi_terms.append(
                    nn.functional.smooth_l1_loss(pred[key], target[key], reduction="mean")
                )
        losses["derived_kpi_consistency"] = (
            torch.stack(kpi_terms).mean() if kpi_terms else zero
        )
        if "tfv_delta" in target:
            tfv_target = target["tfv_delta"].reshape(-1)
            tfv_pred = pred["tfv_delta"].reshape(-1)
            # Balance the minority improving cases without changing the
            # regression target or the trajectory-derived KPI definition.
            scale = tfv_target.detach().abs().median().clamp_min(1.0)
            sign = torch.where(tfv_target < 0.0, -1.0, 1.0)
            direction = nn.functional.softplus(-(sign * tfv_pred / scale))
            negative = tfv_target < 0.0
            positive = ~negative
            n_negative = int(negative.sum().item())
            n_positive = int(positive.sum().item())
            if n_negative and n_positive:
                class_weight = torch.where(
                    negative,
                    torch.full_like(direction, 0.5 / n_negative),
                    torch.full_like(direction, 0.5 / n_positive),
                )
                direction = (direction * class_weight).sum()
            else:
                direction = direction.mean()
            losses["tfv_direction"] = direction
        else:
            losses["tfv_direction"] = zero
        group_index = target.get("_state_index")
        ranking_terms: list[torch.Tensor] = []
        if group_index is not None and "tfv_delta" in target:
            groups = group_index.reshape(-1)
            tfv_target = target["tfv_delta"].reshape(-1)
            tfv_pred = pred["tfv_delta"].reshape(-1)
            for group in torch.unique(groups):
                idx = torch.nonzero(groups == group, as_tuple=False).reshape(-1)
                if idx.numel() < 2:
                    continue
                target_group = tfv_target.index_select(0, idx)
                pred_group = tfv_pred.index_select(0, idx)
                target_diff = target_group[None, :] - target_group[:, None]
                pred_diff = pred_group[None, :] - pred_group[:, None]
                upper = torch.triu(
                    torch.ones_like(target_diff, dtype=torch.bool), diagonal=1
                )
                informative = upper & (target_diff.abs() > 1.0)
                if bool(informative.any()):
                    scale = target_group.detach().abs().median().clamp_min(1.0)
                    margin = target_diff[informative].sign() * pred_diff[informative] / scale
                    ranking_terms.append(nn.functional.softplus(-margin).mean())
        losses["tfv_ranking"] = (
            torch.stack(ranking_terms).mean() if ranking_terms else zero
        )
        priority_terms: list[torch.Tensor] = []
        if self.priority_node_indices is not None and self.priority_node_indices.numel():
            indices = self.priority_node_indices.to(
                device=branches["candidate"]["node_flooding_rate"].device
            )
            for branch in self.BRANCHES:
                priority_terms.append(
                    self._masked_smooth_l1(
                        branches[branch]["node_flooding_rate"].index_select(2, indices),
                        target[self._target_key(branch, "flood")].index_select(2, indices),
                    )
                )
        losses["priority_flooding_trajectory"] = (
            torch.stack(priority_terms).mean() if priority_terms else zero
        )
        action_effect_terms: list[torch.Tensor] = []
        candidate_flood = branches["candidate"]["node_flooding_rate"]
        candidate_target_flood = target[self._target_key("candidate", "flood")]
        for reference in ("no_control", "dynamic_internal", "hold_previous"):
            action_effect_terms.append(
                self._masked_smooth_l1(
                    candidate_flood - branches[reference]["node_flooding_rate"],
                    candidate_target_flood
                    - target[self._target_key(reference, "flood")],
                )
            )
        losses["action_effect_trajectory"] = torch.stack(action_effect_terms).mean()
        return losses

    def total(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        w = self.weights
        return (
            w.depth * losses["depth_trajectory"]
            + w.node_flooding * losses["flooding_trajectory"]
            + w.storage * losses["storage_trajectory"]
            + w.facility_flow * losses["facility_flow_trajectory"]
            + w.outfall_flow * losses["outfall_flow_trajectory"]
            + w.kpi_consistency * losses["derived_kpi_consistency"]
            + w.priority_flooding * losses["priority_flooding_trajectory"]
            + w.action_effect * losses["action_effect_trajectory"]
            + w.tfv_direction * losses["tfv_direction"]
            + w.tfv_ranking * losses["tfv_ranking"]
        )
