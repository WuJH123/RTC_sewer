"""Uncertainty estimation: seed ensemble and MC dropout.

Includes:
  - Seed ensemble standard deviation (5 seeds)
  - Optional: MC dropout for single-model uncertainty
  - Uncertainty-error correlation metric
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class EnsembleUncertainty(nn.Module):
    """Seed ensemble uncertainty estimation.

    Wraps multiple model instances (trained with different seeds) and
    computes prediction mean and standard deviation.
    """

    def __init__(self, models: Sequence[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Run all models and compute ensemble statistics.

        Returns dict with:
            mean   : ensemble mean of predictions
            std    : ensemble standard deviation
            preds  : list of individual model outputs
        """
        all_preds = []
        for model in self.models:
            pred = model(*args, **kwargs)
            all_preds.append(pred)

        # Aggregate key outputs
        if "y_candidate" in all_preds[0]:
            y_stack = torch.stack([p["y_candidate"] for p in all_preds], dim=0)
            y_mean = y_stack.mean(dim=0)
            y_std = y_stack.std(dim=0)
        else:
            y_mean = y_std = None

        if "delta" in all_preds[0]:
            d_stack = torch.stack([p["delta"] for p in all_preds], dim=0)
            d_mean = d_stack.mean(dim=0)
            d_std = d_stack.std(dim=0)
        else:
            d_mean = d_std = None

        return {
            "y_candidate_mean": y_mean,
            "y_candidate_std": y_std,
            "delta_mean": d_mean,
            "delta_std": d_std,
            "preds": all_preds,
        }


class MCDropoutUncertainty(nn.Module):
    """MC dropout uncertainty for a single model.

    Enables dropout at inference time and runs multiple forward passes
    to estimate predictive uncertainty.
    """

    def __init__(self, model: nn.Module, n_forward: int = 10, dropout_rate: float = 0.1):
        super().__init__()
        self.model = model
        self.n_forward = int(n_forward)
        self.dropout_rate = float(dropout_rate)

    def _enable_dropout(self) -> None:
        """Enable dropout layers while keeping other modules in eval mode."""
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                module.p = self.dropout_rate

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Run MC dropout forward passes.

        Returns dict with:
            mean : MC mean
            std  : MC standard deviation
        """
        self._enable_dropout()
        all_preds = []
        with torch.no_grad():
            for _ in range(self.n_forward):
                pred = self.model(*args, **kwargs)
                all_preds.append(pred)

        # Stack key outputs
        if "y_candidate" in all_preds[0]:
            y_stack = torch.stack([p["y_candidate"] for p in all_preds], dim=0)
            y_mean = y_stack.mean(dim=0)
            y_std = y_stack.std(dim=0)
        else:
            y_mean = y_std = None

        if "delta" in all_preds[0]:
            d_stack = torch.stack([p["delta"] for p in all_preds], dim=0)
            d_mean = d_stack.mean(dim=0)
            d_std = d_stack.std(dim=0)
        else:
            d_mean = d_std = None

        return {
            "y_candidate_mean": y_mean,
            "y_candidate_std": y_std,
            "delta_mean": d_mean,
            "delta_std": d_std,
        }


def uncertainty_error_correlation(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainty: torch.Tensor,
) -> torch.Tensor:
    """Compute correlation between uncertainty and prediction error.

    A well-calibrated model should have higher uncertainty where errors
    are larger. Returns Pearson correlation coefficient.

    predictions : [B, ...]
    targets     : [B, ...]
    uncertainty : [B, ...]  (std or variance)
    """
    errors = (predictions - targets).abs().reshape(-1)
    unc = uncertainty.reshape(-1)
    if errors.numel() < 2:
        return torch.zeros((), device=predictions.device)
    # Pearson correlation
    errors_centered = errors - errors.mean()
    unc_centered = unc - unc.mean()
    cov = (errors_centered * unc_centered).mean()
    std_e = errors_centered.pow(2).mean().sqrt().clamp(min=1e-8)
    std_u = unc_centered.pow(2).mean().sqrt().clamp(min=1e-8)
    return cov / (std_e * std_u)
