"""Trajectory losses: depth trajectory MSE/SmoothL1 for candidate and reference.

Includes:
  - Depth trajectory loss (candidate + reference)
  - Delta trajectory loss
  - Dead-zone-aware loss for PFV, TFV, Peak
"""
from __future__ import annotations

import torch
from torch import nn


class TrajectoryLosses(nn.Module):
    """Compute trajectory-level losses for depth predictions.

    Dead zones: below these thresholds, errors are not penalized.
      PFV dead_zone  = 1.0 m³
      TFV dead_zone  = 1.0 m³
      Peak dead_zone = 0.001 m³/s
    """

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
        # When targets are z-score normalized, scale dead zones by std
        # so the threshold remains meaningful in normalized space
        if norm_std is not None:
            self.pfv_dead_zone = float(pfv_dead_zone) / max(norm_std.get("pfv_delta", 1.0), 1e-8)
            self.tfv_dead_zone = float(tfv_dead_zone) / max(norm_std.get("tfv_delta", 1.0), 1e-8)
            self.peak_dead_zone = float(peak_dead_zone) / max(norm_std.get("peak_delta", 1.0), 1e-8)
        else:
            self.pfv_dead_zone = float(pfv_dead_zone)
            self.tfv_dead_zone = float(tfv_dead_zone)
            self.peak_dead_zone = float(peak_dead_zone)
        self.trajectory_weight = float(trajectory_weight)
        self.delta_weight = float(delta_weight)
        self.kpi_weight = float(kpi_weight)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        pred dict:
            y_candidate : [B, H, N]
            y_reference : [B, H, N]
            delta       : [B, H, N]
            pfv_delta   : [B]
            tfv_delta   : [B]
            peak_flood_rate : [B]

        target dict:
            depth_candidate : [B, H, N]
            depth_reference : [B, H, N]
            pfv_gt          : [B]
            tfv_gt          : [B]
            peak_gt         : [B]
        """
        losses: dict[str, torch.Tensor] = {}

        # Depth trajectory loss (candidate + reference)
        if "depth_candidate" in target and "depth_reference" in target:
            traj_c = nn.functional.smooth_l1_loss(
                pred["y_candidate"], target["depth_candidate"], reduction="mean"
            )
            traj_r = nn.functional.smooth_l1_loss(
                pred["y_reference"], target["depth_reference"], reduction="mean"
            )
            losses["depth_trajectory"] = (traj_c + traj_r) * 0.5
        else:
            losses["depth_trajectory"] = torch.zeros((), device=pred["y_candidate"].device)

        # Delta trajectory loss
        if "depth_candidate" in target and "depth_reference" in target:
            target_delta = target["depth_candidate"] - target["depth_reference"]
            losses["delta_trajectory"] = nn.functional.smooth_l1_loss(
                pred["delta"], target_delta, reduction="mean"
            )
        else:
            losses["delta_trajectory"] = torch.zeros((), device=pred["y_candidate"].device)

        # Dead-zone-aware KPI losses
        # PFV — data key is "pfv_delta" (Candidate − Reference convention)
        if "pfv_delta" in target and "pfv_delta" in pred:
            pfv_err = pred["pfv_delta"] - target["pfv_delta"]
            # Dead zone: no penalty if |error| < dead_zone
            pfv_dz = torch.relu(pfv_err.abs() - self.pfv_dead_zone)
            losses["pfv_kpi"] = pfv_dz.mean()
        else:
            losses["pfv_kpi"] = torch.zeros((), device=pred["y_candidate"].device)

        # TFV — data key is "tfv_delta"
        if "tfv_delta" in target and "tfv_delta" in pred:
            tfv_err = pred["tfv_delta"] - target["tfv_delta"]
            tfv_dz = torch.relu(tfv_err.abs() - self.tfv_dead_zone)
            losses["tfv_kpi"] = tfv_dz.mean()
        else:
            losses["tfv_kpi"] = torch.zeros((), device=pred["y_candidate"].device)

        # Peak — data key is "peak_delta"
        if "peak_delta" in target and "peak_flood_rate" in pred:
            peak_err = pred["peak_flood_rate"] - target["peak_delta"]
            peak_dz = torch.relu(peak_err.abs() - self.peak_dead_zone)
            losses["peak_kpi"] = peak_dz.mean()
        else:
            losses["peak_kpi"] = torch.zeros((), device=pred["y_candidate"].device)

        return losses

    def total(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Weighted sum of all trajectory losses."""
        return (
            self.trajectory_weight * losses["depth_trajectory"]
            + self.delta_weight * losses["delta_trajectory"]
            + self.kpi_weight * (losses["pfv_kpi"] + losses["tfv_kpi"] + losses["peak_kpi"])
        )
