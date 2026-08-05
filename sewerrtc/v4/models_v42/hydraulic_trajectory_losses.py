"""Losses for the trajectory-first V4.2 hydraulic surrogate.

Formal training supervises hydraulics, not an independent KPI shortcut.  KPI
losses may be used only on the already-derived PFV/TFV/Peak outputs and cannot
replace node flooding-rate supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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
    action_effect: float = 0.0
    pfv_action_effect: float = 0.0
    pfv_ranking: float = 0.0
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
        kpi_scales: Mapping[str, float] | None = None,
        trajectory_scales: Mapping[str, float] | None = None,
        action_effect_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or HydraulicLossWeights()
        self.require_storage_targets = bool(require_storage_targets)
        self.require_facility_flow_targets = bool(require_facility_flow_targets)
        self.require_outfall_flow_targets = bool(require_outfall_flow_targets)
        self.kpi_scales = {
            str(key): max(float(value), 1.0e-6)
            for key, value in (kpi_scales or {}).items()
        }
        self.trajectory_scales = {
            str(key): max(float(value), 1.0e-6)
            for key, value in (trajectory_scales or {}).items()
        }
        self.action_effect_indices = action_effect_indices

    def _scaled(self, value: torch.Tensor, key: str) -> torch.Tensor:
        return value / self.trajectory_scales.get(key, 1.0)

    @staticmethod
    def _pairwise_rank_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        group_id: torch.Tensor | None,
    ) -> torch.Tensor:
        if group_id is None:
            return prediction.sum() * 0.0
        terms: list[torch.Tensor] = []
        for group in torch.unique(group_id):
            idx = torch.nonzero(group_id == group, as_tuple=False).flatten()
            for left in range(int(idx.numel())):
                for right in range(left + 1, int(idx.numel())):
                    i, j = idx[left], idx[right]
                    delta = target[i] - target[j]
                    if float(torch.abs(delta).detach()) <= 1.0e-6:
                        continue
                    direction = 1.0 if float(delta.detach()) < 0.0 else -1.0
                    terms.append(
                        torch.nn.functional.softplus(
                            direction * (prediction[i] - prediction[j])
                        )
                    )
        return torch.stack(terms).mean() if terms else prediction.sum() * 0.0

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
                    self._scaled(bpred["node_depth"], "depth"),
                    self._scaled(target[self._target_key(branch, "depth")], "depth"),
                )
            )
            flood_terms.append(
                self._masked_smooth_l1(
                    self._scaled(bpred["node_flooding_rate"], "flood"),
                    self._scaled(target[self._target_key(branch, "flood")], "flood"),
                )
            )

            storage_key = self._target_key(branch, "storage_volume")
            if storage_key in target:
                storage_terms.append(
                    self._masked_smooth_l1(
                        self._scaled(bpred["storage_volume"], "storage_volume"),
                        self._scaled(target[storage_key], "storage_volume"),
                        target.get(storage_key + "_available"),
                    )
                )
            facility_key = self._target_key(branch, "facility_flow")
            if facility_key in target:
                facility_terms.append(
                    self._masked_smooth_l1(
                        self._scaled(bpred["facility_flow"], "facility_flow"),
                        self._scaled(target[facility_key], "facility_flow"),
                        target.get(facility_key + "_available"),
                    )
                )
            outfall_key = self._target_key(branch, "outfall_flow")
            if outfall_key in target:
                outfall_terms.append(
                    self._masked_smooth_l1(
                        self._scaled(bpred["outfall_flow"], "outfall_flow"),
                        self._scaled(target[outfall_key], "outfall_flow"),
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

        # Directly supervise the counterfactual action effect.  The shared
        # branch losses can fit the common hydraulic baseline while leaving
        # candidate-vs-No-control differences nearly constant, even when the
        # derived PFV/TFV labels vary materially.
        losses["action_effect"] = self._masked_smooth_l1(
            self._scaled(
                branches["candidate"]["node_flooding_rate"]
                - branches["no_control"]["node_flooding_rate"],
                "action_effect",
            ),
            self._scaled(
                target[self._target_key("candidate", "flood")]
                - target[self._target_key("no_control", "flood")],
                "action_effect",
            ),
        )
        if self.action_effect_indices is not None:
            idx = self.action_effect_indices.to(
                branches["candidate"]["node_flooding_rate"].device
            )
            cand_effect = (
                branches["candidate"]["node_flooding_rate"]
                .index_select(2, idx)
                - branches["no_control"]["node_flooding_rate"].index_select(2, idx)
            )
            target_effect = (
                target[self._target_key("candidate", "flood")]
                .index_select(2, idx)
                - target[self._target_key("no_control", "flood")].index_select(2, idx)
            )
            losses["pfv_action_effect"] = self._masked_smooth_l1(
                self._scaled(cand_effect, "action_effect"),
                self._scaled(target_effect, "action_effect"),
            )
        else:
            losses["pfv_action_effect"] = zero

        # These are consistency terms on trajectory-derived outputs only.  There
        # is no separate free-standing KPI head in the formal model.
        kpi_terms: list[torch.Tensor] = []
        for key in ("pfv_delta", "tfv_delta", "peak_delta"):
            if key in target:
                if key not in pred:
                    raise KeyError(f"derived KPI {key} missing from model output")
                if not torch.isfinite(target[key]).all():
                    raise ValueError(f"target {key} contains NaN/Inf")
                scale = self.kpi_scales.get(key, 1.0)
                kpi_terms.append(
                    nn.functional.smooth_l1_loss(
                        pred[key] / scale,
                        target[key] / scale,
                        reduction="mean",
                    )
                )
        losses["derived_kpi_consistency"] = (
            torch.stack(kpi_terms).mean() if kpi_terms else zero
        )
        group_id = target.get("state_group_id")
        losses["pfv_ranking"] = self._pairwise_rank_loss(
            pred["pfv_delta"] / self.kpi_scales.get("pfv_delta", 1.0),
            target["pfv_delta"] / self.kpi_scales.get("pfv_delta", 1.0),
            group_id,
        )
        losses["tfv_ranking"] = self._pairwise_rank_loss(
            pred["tfv_delta"] / self.kpi_scales.get("tfv_delta", 1.0),
            target["tfv_delta"] / self.kpi_scales.get("tfv_delta", 1.0),
            group_id,
        )
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
            + w.action_effect * losses["action_effect"]
            + w.pfv_action_effect * losses["pfv_action_effect"]
            + w.pfv_ranking * losses["pfv_ranking"]
            + w.tfv_ranking * losses["tfv_ranking"]
        )
