from __future__ import annotations

import torch
from torch import nn


def temporal_surrogate_loss(
    pred: dict,
    target_seq: torch.Tensor,
    risk_delta: torch.Tensor,
    edge_index: torch.Tensor | None = None,
    priority_weight: torch.Tensor | None = None,
    direction_pos_weight: torch.Tensor | None = None,
    lambda_volume: float = 0.10,
    lambda_flood: float = 0.20,
    lambda_direction: float = 0.30,
    lambda_smooth: float = 0.005,
) -> tuple[torch.Tensor, dict]:
    pred_seq = pred["pred_seq"]
    if priority_weight is None:
        priority_weight = torch.ones(target_seq.shape[-1], dtype=target_seq.dtype, device=target_seq.device)
    w = priority_weight[None, None, :]
    depth_loss = torch.mean(nn.functional.smooth_l1_loss(pred_seq, target_seq, reduction="none") * w)
    delta_loss = nn.functional.smooth_l1_loss(pred["risk_delta"], risk_delta)

    pred_volume = pred_seq.sum(dim=-1)
    true_volume = target_seq.sum(dim=-1)
    volume_loss = nn.functional.smooth_l1_loss(pred_volume, true_volume)

    threshold = torch.quantile(target_seq.detach(), 0.90)
    flood_y = (target_seq >= threshold).float()
    flood_logits = (pred_seq - threshold) / (0.10 + threshold.abs() * 0.05)
    flood_loss = nn.functional.binary_cross_entropy_with_logits(flood_logits, flood_y)

    y_pfv_improve = (risk_delta[:, 0] < 0).float()
    y_safe = ((risk_delta[:, 1] <= 0) & (risk_delta[:, 2] <= 0)).float()
    y_pfv_nonzero = (torch.abs(risk_delta[:, 0]) > 1e-6).float()
    if direction_pos_weight is not None:
        pfv_pos_weight = direction_pos_weight[0].to(dtype=pred["logits"].dtype, device=pred["logits"].device)
        safe_pos_weight = direction_pos_weight[1].to(dtype=pred["logits"].dtype, device=pred["logits"].device)
        nonzero_pos_weight = direction_pos_weight[2].to(dtype=pred["logits"].dtype, device=pred["logits"].device) if len(direction_pos_weight) > 2 else None
    else:
        pfv_pos_weight = None
        safe_pos_weight = None
        nonzero_pos_weight = None
    cls_loss = nn.functional.binary_cross_entropy_with_logits(
        pred["logits"][:, 0],
        y_pfv_improve,
        pos_weight=pfv_pos_weight,
    )
    cls_loss = cls_loss + nn.functional.binary_cross_entropy_with_logits(
        pred["logits"][:, 1],
        y_safe,
        pos_weight=safe_pos_weight,
    )
    if pred["logits"].shape[1] >= 3:
        cls_loss = cls_loss + nn.functional.binary_cross_entropy_with_logits(
            pred["logits"][:, 2],
            y_pfv_nonzero,
            pos_weight=nonzero_pos_weight,
        )

    smooth_loss = torch.zeros((), dtype=target_seq.dtype, device=target_seq.device)
    if edge_index is not None and edge_index.numel() > 0:
        src, dst = edge_index
        smooth_loss = torch.mean(torch.abs(pred_seq[:, :, src] - pred_seq[:, :, dst]))

    total = depth_loss + 0.5 * delta_loss + lambda_volume * volume_loss + lambda_flood * flood_loss + lambda_direction * cls_loss + lambda_smooth * smooth_loss
    parts = {
        "depth_loss": float(depth_loss.detach().cpu()),
        "delta_loss": float(delta_loss.detach().cpu()),
        "volume_loss": float(volume_loss.detach().cpu()),
        "flood_loss": float(flood_loss.detach().cpu()),
        "direction_loss": float(cls_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
    }
    return total, parts


def surrogate_loss(pred: dict, next_state: torch.Tensor, risk_delta: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Compatibility wrapper for legacy callers."""
    target_seq = next_state[:, None, :]
    if "pred_seq" not in pred:
        pred = {**pred, "pred_seq": pred["next_state"][:, None, :]}
    return temporal_surrogate_loss(pred, target_seq, risk_delta)
