"""Trajectory and KPI losses for the V4.2 multi-reference surrogate.

The shared dynamics model predicts Candidate, No-control, Dynamic Internal and
Hold-Previous branches.  PFV is supervised against No-control; TFV and Peak are
supervised against Dynamic Internal.  This module therefore keeps the branch
roles explicit instead of silently reusing one generic reference.
"""
from __future__ import annotations

import torch
from torch import nn


class TrajectoryLosses(nn.Module):
    """Compute branch trajectory and dead-zone-aware KPI losses."""

    def __init__(
        self,
        pfv_dead_zone: float = 1.0,
        tfv_dead_zone: float = 1.0,
        peak_dead_zone: float = 0.001,
        trajectory_weight: float = 1.0,
        delta_weight: float = 0.5,
        kpi_weight: float = 0.3,
        norm_std: dict[str, float] | None = None,
    ):
        super().__init__()
        if norm_std is not None:
            self.pfv_dead_zone = float(pfv_dead_zone) / max(
                norm_std.get("pfv_delta", 1.0), 1e-8
            )
            self.tfv_dead_zone = float(tfv_dead_zone) / max(
                norm_std.get("tfv_delta", 1.0), 1e-8
            )
            self.peak_dead_zone = float(peak_dead_zone) / max(
                norm_std.get("peak_delta", 1.0), 1e-8
            )
        else:
            self.pfv_dead_zone = float(pfv_dead_zone)
            self.tfv_dead_zone = float(tfv_dead_zone)
            self.peak_dead_zone = float(peak_dead_zone)
        self.trajectory_weight = float(trajectory_weight)
        self.delta_weight = float(delta_weight)
        self.kpi_weight = float(kpi_weight)

    @staticmethod
    def _zero(pred: dict[str, torch.Tensor]) -> torch.Tensor:
        return pred["y_candidate"].sum() * 0.0

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        zero = self._zero(pred)

        # Supervise every branch that is part of the formal four-reference
        # contract.  Missing DI/Hold predictions are a data/model contract error
        # rather than a reason to fall back to NC.
        branch_pairs = [
            ("y_candidate", "depth_candidate"),
            ("y_reference", "depth_reference"),
            ("y_dynamic_internal", "depth_dynamic_internal"),
            ("y_hold_previous", "depth_hold_previous"),
        ]
        branch_losses = []
        for pred_key, target_key in branch_pairs:
            if target_key in target:
                if pred_key not in pred:
                    raise KeyError(
                        f"Target {target_key} is present but model did not produce {pred_key}"
                    )
                branch_losses.append(
                    nn.functional.smooth_l1_loss(
                        pred[pred_key], target[target_key], reduction="mean"
                    )
                )
        losses["depth_trajectory"] = (
            torch.stack(branch_losses).mean() if branch_losses else zero
        )

        # Counterfactual trajectory deltas for the two scientific references.
        delta_losses = []
        if "depth_candidate" in target and "depth_reference" in target:
            target_delta_nc = target["depth_candidate"] - target["depth_reference"]
            delta_losses.append(
                nn.functional.smooth_l1_loss(
                    pred["delta"], target_delta_nc, reduction="mean"
                )
            )
        if "depth_dynamic_internal" in target:
            if "delta_di" not in pred:
                raise KeyError("DI target present but delta_di prediction is missing")
            target_delta_di = (
                target["depth_candidate"] - target["depth_dynamic_internal"]
            )
            delta_losses.append(
                nn.functional.smooth_l1_loss(
                    pred["delta_di"], target_delta_di, reduction="mean"
                )
            )
        losses["delta_trajectory"] = (
            torch.stack(delta_losses).mean() if delta_losses else zero
        )

        if "pfv_delta" in target and "pfv_delta" in pred:
            pfv_err = pred["pfv_delta"] - target["pfv_delta"]
            losses["pfv_kpi"] = torch.relu(
                pfv_err.abs() - self.pfv_dead_zone
            ).mean()
        else:
            losses["pfv_kpi"] = zero

        if "tfv_delta" in target and "tfv_delta" in pred:
            tfv_err = pred["tfv_delta"] - target["tfv_delta"]
            losses["tfv_kpi"] = torch.relu(
                tfv_err.abs() - self.tfv_dead_zone
            ).mean()
        else:
            losses["tfv_kpi"] = zero

        # Unified key: Peak is a *delta* relative to Dynamic Internal.
        if "peak_delta" in target and "peak_delta" in pred:
            peak_err = pred["peak_delta"] - target["peak_delta"]
            losses["peak_kpi"] = torch.relu(
                peak_err.abs() - self.peak_dead_zone
            ).mean()
        else:
            losses["peak_kpi"] = zero

        return losses

    def total(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.trajectory_weight * losses["depth_trajectory"]
            + self.delta_weight * losses["delta_trajectory"]
            + self.kpi_weight
            * (losses["pfv_kpi"] + losses["tfv_kpi"] + losses["peak_kpi"])
        )
