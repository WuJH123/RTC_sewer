from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def action_excitation(action_sequence: torch.Tensor) -> torch.Tensor:
    """Return per-sample, per-actuator temporal excitation for ``[B,H,A]`` actions."""
    if action_sequence.ndim != 3:
        raise ValueError("action_sequence must be [B,H,A]")
    action_range = action_sequence.amax(dim=1) - action_sequence.amin(dim=1)
    if action_sequence.shape[1] > 1:
        step_change = torch.abs(action_sequence[:, 1:] - action_sequence[:, :-1]).amax(dim=1)
        action_range = torch.maximum(action_range, step_change)
    return action_range


def action_rich_sample_weights(
    excitation: torch.Tensor,
    *,
    gain: float,
    minimum_excitation: float,
) -> torch.Tensor:
    if excitation.ndim != 2:
        raise ValueError("excitation must be [B,A]")
    active = excitation.amax(dim=1) >= float(minimum_excitation)
    return 1.0 + max(0.0, float(gain)) * active.to(excitation.dtype)


def actuator_neighbour_state_loss(
    predicted_local: torch.Tensor,
    target_local: torch.Tensor,
    *,
    excitation: torch.Tensor,
    action_local_map: torch.Tensor,
    scale: float | torch.Tensor,
) -> torch.Tensor:
    """Supervise changed facilities at their hydraulic neighbourhoods.

    Observational trajectories cannot provide causal effects, but they can
    require the shared encoder to retain which actuator changed and where its
    adjacent hydraulic response occurred.
    """
    if predicted_local.shape != target_local.shape or predicted_local.ndim != 3:
        raise ValueError("predicted_local and target_local must share [B,H,L]")
    if excitation.ndim != 2 or action_local_map.ndim != 2:
        raise ValueError("excitation and action_local_map must be [B,A] and [A,L]")
    if excitation.shape[1] != action_local_map.shape[0]:
        raise ValueError("actuator dimension mismatch")
    node_weight = torch.matmul(excitation, action_local_map).clamp(min=0.0)
    active = node_weight.sum(dim=1, keepdim=True) > 0.0
    normalized = node_weight / torch.clamp(node_weight.sum(dim=1, keepdim=True), min=1.0e-8)
    normalized = normalized * active.to(normalized.dtype)
    elementwise = F.smooth_l1_loss(
        predicted_local / scale,
        target_local / scale,
        reduction="none",
    )
    per_sample = (elementwise * normalized[:, None, :]).sum(dim=2).mean(dim=1)
    active_rows = active[:, 0]
    if not bool(active_rows.any()):
        return predicted_local.sum() * 0.0
    return per_sample[active_rows].mean()


def horizon_peak_metrics(
    target_risk_rate_seq: np.ndarray,
    predicted_risk_rate_seq: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Evaluate horizon peak from the TFV-rate sequence, never from an alias channel."""
    target = np.asarray(target_risk_rate_seq)
    prediction = np.asarray(predicted_risk_rate_seq)
    if target.shape != prediction.shape or target.ndim != 3 or target.shape[2] < 2:
        raise ValueError("target and prediction must share [N,H,C] with a TFV-rate channel")
    target_peak = np.max(target[:, :, 1], axis=1)
    predicted_peak = np.max(prediction[:, :, 1], axis=1)
    denominator = float(np.square(target_peak - target_peak.mean()).sum())
    r2 = (
        float(1.0 - np.square(target_peak - predicted_peak).sum() / denominator)
        if denominator > 1.0e-12
        else float("nan")
    )
    return {
        "target": target_peak,
        "prediction": predicted_peak,
        "MAE": float(np.abs(predicted_peak - target_peak).mean()),
        "R2": r2,
    }
