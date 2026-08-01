"""Formal training utilities for V4.2 Step-1 sparse-state reconstruction.

This module deliberately separates *representation pretraining* from *formal
Wuhan target-domain evaluation*.  Source/unknown-domain hydraulic windows may
help learn topology/dynamics, but they cannot authorize the paper claim.

The loss also trains the reconstructor's ``depth_std`` head.  The earlier smoke
trainer optimized only depth mean, leaving the aleatoric scale effectively
untrained; such a scale cannot support uncertainty-aware MPC evidence.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from sewerrtc.models.temporal_sparse_gat_v42 import TemporalGATOutput
from sewerrtc.v4.v42_step1_dataset import Step1Sample


@dataclass(frozen=True)
class Step1LossWeights:
    global_depth: float = 1.0
    priority_depth: float = 3.0
    wet_priority_depth: float = 1.0
    heteroscedastic_nll: float = 0.25


@dataclass(frozen=True)
class Step1Split:
    auxiliary_pretrain_indices: tuple[int, ...]
    target_train_indices: tuple[int, ...]
    target_validation_indices: tuple[int, ...]
    target_calibration_indices: tuple[int, ...]
    target_train_groups: tuple[str, ...]
    target_validation_groups: tuple[str, ...]
    target_calibration_groups: tuple[str, ...]

    @property
    def target_group_count(self) -> int:
        return len(
            set(self.target_train_groups)
            | set(self.target_validation_groups)
            | set(self.target_calibration_groups)
        )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if not bool(mask.any()):
        return values.sum() * 0.0
    return values[mask].mean()


def step1_reconstruction_loss(
    output: TemporalGATOutput,
    target_depth: torch.Tensor,
    sensor_mask_current: torch.Tensor,
    priority_node_mask: torch.Tensor,
    *,
    weights: Step1LossWeights | None = None,
    wet_threshold_m: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Train mean and uncertainty on unobserved nodes.

    ``priority_node_mask`` is a 1-D boolean mask in canonical graph-node order.
    Priority weighting applies only to nodes that are unobserved in the current
    sensor layout, preventing exact sensor passthrough from masquerading as
    reconstruction skill.
    """
    w = weights or Step1LossWeights()
    mean = output.depth_mean
    std = output.depth_std.clamp_min(1.0e-4)
    if mean.shape != target_depth.shape or std.shape != target_depth.shape:
        raise ValueError("Step1 prediction/target shape mismatch")
    if sensor_mask_current.shape != target_depth.shape:
        raise ValueError("sensor_mask_current must match target_depth")
    if priority_node_mask.ndim != 1 or priority_node_mask.numel() != target_depth.shape[-1]:
        raise ValueError("priority_node_mask must contain one flag per graph node")
    if not torch.isfinite(target_depth).all():
        raise ValueError("target_depth contains NaN/Inf")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Step1 model output contains NaN/Inf")

    unobserved = sensor_mask_current < 0.5
    priority = priority_node_mask.to(target_depth.device).bool()[None, :].expand_as(unobserved)
    priority_unobserved = unobserved & priority
    wet_priority = priority_unobserved & (target_depth >= float(wet_threshold_m))

    sq = (mean - target_depth).square()
    global_depth = _masked_mean(sq, unobserved)
    priority_depth = _masked_mean(sq, priority_unobserved)
    wet_priority_depth = _masked_mean(sq, wet_priority)

    # Gaussian NLL up to an additive constant.  This is what actually trains the
    # aleatoric scale head; post-hoc calibration may rescale it but must not be
    # asked to rescue an untrained random uncertainty head.
    z2 = sq / std.square()
    nll = 0.5 * (z2 + 2.0 * torch.log(std))
    heteroscedastic_nll = _masked_mean(nll, unobserved)

    total = (
        w.global_depth * global_depth
        + w.priority_depth * priority_depth
        + w.wet_priority_depth * wet_priority_depth
        + w.heteroscedastic_nll * heteroscedastic_nll
    )
    return {
        "total": total,
        "global_depth": global_depth,
        "priority_depth": priority_depth,
        "wet_priority_depth": wet_priority_depth,
        "heteroscedastic_nll": heteroscedastic_nll,
    }


def _group_rank(group: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()


def build_formal_step1_split(
    samples: Sequence[Step1Sample],
    *,
    seed: int = 42,
) -> Step1Split:
    """Create a target-domain train/validation/calibration split.

    Auxiliary domains are returned only in ``auxiliary_pretrain_indices`` and
    never enter formal validation/calibration.  At least three independent
    target rainfall groups are required so the three roles cannot overlap.
    """
    target_groups = sorted(
        {
            str(sample.split_group)
            for sample in samples
            if sample.step1_domain_role == "target_formal"
        },
        key=lambda g: _group_rank(g, seed),
    )
    if len(target_groups) < 3:
        raise ValueError(
            "formal Step1 requires >=3 independent target-domain rainfall groups "
            "for train/validation/calibration isolation"
        )

    # Reserve at least one group each for validation and calibration.  For
    # larger populations use roughly 15%/15%; the remainder is target training.
    n = len(target_groups)
    n_val = max(1, int(round(n * 0.15)))
    n_cal = max(1, int(round(n * 0.15)))
    if n_val + n_cal >= n:
        n_val = 1
        n_cal = 1
    val_groups = set(target_groups[:n_val])
    cal_groups = set(target_groups[n_val : n_val + n_cal])
    train_groups = set(target_groups[n_val + n_cal :])
    if not train_groups:
        raise ValueError("formal Step1 split has no target training rainfall group")

    aux_idx: list[int] = []
    train_idx: list[int] = []
    val_idx: list[int] = []
    cal_idx: list[int] = []
    for i, sample in enumerate(samples):
        group = str(sample.split_group)
        if sample.step1_domain_role != "target_formal":
            aux_idx.append(i)
        elif group in val_groups:
            val_idx.append(i)
        elif group in cal_groups:
            cal_idx.append(i)
        elif group in train_groups:
            train_idx.append(i)
        else:
            raise RuntimeError(f"target group {group!r} was not assigned")

    if not train_idx or not val_idx or not cal_idx:
        raise ValueError("formal Step1 split produced an empty target partition")
    if set(train_groups) & set(val_groups) or set(train_groups) & set(cal_groups) or set(val_groups) & set(cal_groups):
        raise RuntimeError("target rainfall groups leaked across Step1 partitions")

    return Step1Split(
        auxiliary_pretrain_indices=tuple(aux_idx),
        target_train_indices=tuple(train_idx),
        target_validation_indices=tuple(val_idx),
        target_calibration_indices=tuple(cal_idx),
        target_train_groups=tuple(sorted(train_groups)),
        target_validation_groups=tuple(sorted(val_groups)),
        target_calibration_groups=tuple(sorted(cal_groups)),
    )


def split_summary(split: Step1Split) -> dict[str, object]:
    return {
        "auxiliary_pretrain_samples": len(split.auxiliary_pretrain_indices),
        "target_train_samples": len(split.target_train_indices),
        "target_validation_samples": len(split.target_validation_indices),
        "target_calibration_samples": len(split.target_calibration_indices),
        "target_train_groups": list(split.target_train_groups),
        "target_validation_groups": list(split.target_validation_groups),
        "target_calibration_groups": list(split.target_calibration_groups),
        "target_group_count": split.target_group_count,
        "formal_validation_uses_auxiliary_domain": False,
        "formal_calibration_uses_auxiliary_domain": False,
    }
