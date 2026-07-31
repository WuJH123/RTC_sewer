from __future__ import annotations

import math

import numpy as np
import torch


DYNAMICS_WARM_START_PREFIXES = (
    "actuator_input.",
    "actuator_identity.",
    "actuator_temporal.",
    "node_query.",
    "cross_attention.",
    "node_input.",
    "gat.",
    "node_norm.",
    "node_state_head.",
    "reference_risk_head.",
)

EFFECT_HEAD_PREFIXES = (
    "effect_risk_head.",
    "effect_action_residual_head.",
    "effect_scale_head.",
    "safety_classification_head.",
    "safety_action_residual_head.",
    "phase_effect_heads.",
    "phase_safety_heads.",
    "horizon_action_temporal.",
    "horizon_effect_head.",
    "horizon_scale_head.",
    "horizon_safety_head.",
    "horizon_direction_head.",
)

ACTION_FINE_TUNE_PREFIXES = (
    "actuator_input.",
    "actuator_identity.",
    "actuator_temporal.",
    "cross_attention.",
)

STATE_INTERACTION_FINE_TUNE_PREFIXES = (
    "node_query.",
    "node_input.",
    "node_norm.",
)


def resolve_event_group_indices(
    event_ids: list[str] | np.ndarray,
    split: list[str] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, set[str]]:
    events = np.asarray(event_ids).astype(str)
    if split is not None:
        labels = np.asarray(split).astype(str)
        if len(labels) != len(events) or set(labels) != {"train", "validation"}:
            raise ValueError("embedded split must align with rows and contain train and validation")
        train_idx = np.flatnonzero(labels == "train")
        validation_idx = np.flatnonzero(labels == "validation")
        train_events = set(events[train_idx])
        validation_events = set(events[validation_idx])
        if train_events & validation_events:
            raise ValueError("embedded split leaks events between train and validation")
        return train_idx, validation_idx, validation_events
    validation_events = set(sorted(set(events))[::4])
    validation_idx = np.flatnonzero(np.isin(events, sorted(validation_events)))
    train_idx = np.flatnonzero(~np.isin(events, sorted(validation_events)))
    return train_idx, validation_idx, validation_events


def configure_effect_training_parameters(
    model: torch.nn.Module,
    *,
    fine_tune_action_encoder: bool,
    fine_tune_state_interaction: bool = False,
    head_learning_rate: float,
    action_learning_rate_scale: float,
    state_learning_rate_scale: float = 0.02,
) -> tuple[list[dict], list[str]]:
    """Select effect heads and low-rate action/state interaction adapters."""
    head_parameters, action_parameters, state_parameters, names = [], [], [], []
    for name, parameter in model.named_parameters():
        is_head = name.startswith(EFFECT_HEAD_PREFIXES)
        is_action = bool(fine_tune_action_encoder and name.startswith(ACTION_FINE_TUNE_PREFIXES))
        is_state = bool(
            fine_tune_state_interaction
            and name.startswith(STATE_INTERACTION_FINE_TUNE_PREFIXES)
        )
        parameter.requires_grad = is_head or is_action or is_state
        if is_head:
            head_parameters.append(parameter)
            names.append(name)
        elif is_action:
            action_parameters.append(parameter)
            names.append(name)
        elif is_state:
            state_parameters.append(parameter)
            names.append(name)
    groups: list[dict] = []
    if head_parameters:
        groups.append({"params": head_parameters, "lr": float(head_learning_rate)})
    if action_parameters:
        groups.append(
            {
                "params": action_parameters,
                "lr": float(head_learning_rate) * float(action_learning_rate_scale),
            }
        )
    if state_parameters:
        groups.append(
            {
                "params": state_parameters,
                "lr": float(head_learning_rate) * float(state_learning_rate_scale),
            }
        )
    if not groups:
        raise ValueError("effect training selected no trainable parameters")
    return groups, sorted(names)


def load_dynamics_warm_start(
    model: torch.nn.Module,
    checkpoint: dict,
    *,
    node_ids: list[str],
    action_ids: list[str],
) -> list[str]:
    if list(map(str, checkpoint.get("node_ids", []))) != list(map(str, node_ids)):
        raise ValueError("Dynamics warm-start node order differs from the effect dataset")
    if list(map(str, checkpoint.get("action_ids", []))) != list(map(str, action_ids)):
        raise ValueError("Dynamics warm-start action order differs from the effect dataset")
    source = checkpoint.get("model", {})
    target = model.state_dict()
    compatible = {
        name: value
        for name, value in source.items()
        if name.startswith(DYNAMICS_WARM_START_PREFIXES)
        and name in target
        and tuple(value.shape) == tuple(target[name].shape)
    }
    if not compatible:
        raise ValueError("Dynamics warm-start has no compatible shared encoder parameters")
    model.load_state_dict(compatible, strict=False)
    return sorted(compatible)


def aggregate_effect_targets(
    reference_risk_rate_seq: torch.Tensor,
    delta_risk_rate_seq: torch.Tensor,
    *,
    dt_sec: float = 300.0,
) -> torch.Tensor:
    """Build PFV, TFV and peak effects with physically consistent semantics."""
    candidate = reference_risk_rate_seq + delta_risk_rate_seq
    delta_pfv = delta_risk_rate_seq[:, :, 0].sum(dim=1) * float(dt_sec)
    delta_tfv = delta_risk_rate_seq[:, :, 1].sum(dim=1) * float(dt_sec)
    delta_peak = (
        candidate[:, :, 1].max(dim=1).values
        - reference_risk_rate_seq[:, :, 1].max(dim=1).values
    )
    return torch.stack([delta_pfv, delta_tfv, delta_peak], dim=1)


def direction_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    tolerance: float,
) -> dict[str, float | int | None]:
    valid = torch.isfinite(target) & torch.isfinite(prediction) & (target.abs() > float(tolerance))
    count = int(valid.sum().item())
    if count == 0:
        return {"accuracy": None, "count": 0, "tolerance": float(tolerance)}
    accuracy = float(((prediction[valid] < 0) == (target[valid] < 0)).float().mean().item())
    return {"accuracy": accuracy, "count": count, "tolerance": float(tolerance)}


def noninferiority_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reference_volume: torch.Tensor,
    *,
    absolute_margin: float,
    relative_margin: float,
) -> dict[str, float | int | None]:
    margin = torch.maximum(
        torch.full_like(reference_volume, float(absolute_margin)),
        torch.clamp(reference_volume, min=0.0) * float(relative_margin),
    )
    target_safe = target <= margin
    predicted_safe = prediction <= margin
    unsafe = ~target_safe
    unsafe_count = int(unsafe.sum().item())
    unsafe_recall = (
        float((~predicted_safe[unsafe]).float().mean().item()) if unsafe_count else None
    )
    return {
        "classification_accuracy": float((predicted_safe == target_safe).float().mean().item()),
        "target_noninferior_fraction": float(target_safe.float().mean().item()),
        "unsafe_count": unsafe_count,
        "unsafe_recall": unsafe_recall,
        "margin_min": float(margin.min().item()),
        "margin_max": float(margin.max().item()),
    }


def robust_scale(
    values: torch.Tensor,
    *,
    quantile: float = 0.90,
    minimum: float = 1.0e-3,
    dimensions: tuple[int, ...] | None = None,
) -> torch.Tensor:
    absolute = values.detach().abs()
    if dimensions is None:
        scale = torch.quantile(absolute.reshape(-1), float(quantile))
    else:
        remaining = [index for index in range(absolute.ndim) if index not in dimensions]
        permutation = list(dimensions) + remaining
        flattened = absolute.permute(permutation).reshape(-1, *[absolute.shape[index] for index in remaining])
        scale = torch.quantile(flattened, float(quantile), dim=0)
    return torch.clamp(scale, min=float(minimum))


def finite_metric_at_least(value: float | None, threshold: float) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) >= float(threshold)


def binary_classification_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    """Return class-balanced metrics without hiding unsupported classes."""
    valid = torch.isfinite(probability) & torch.isfinite(target.float())
    probability = probability[valid]
    target = target[valid].bool()
    prediction = probability >= float(threshold)
    positives = int(target.sum().item())
    negatives = int((~target).sum().item())
    true_positive_rate = (
        float(prediction[target].float().mean().item()) if positives else None
    )
    true_negative_rate = (
        float((~prediction[~target]).float().mean().item()) if negatives else None
    )
    balanced_accuracy = (
        0.5 * (true_positive_rate + true_negative_rate)
        if true_positive_rate is not None and true_negative_rate is not None
        else None
    )
    return {
        "accuracy": float((prediction == target).float().mean().item()) if len(target) else None,
        "balanced_accuracy": balanced_accuracy,
        "positive_recall": true_positive_rate,
        "negative_recall": true_negative_rate,
        "positive_count": positives,
        "negative_count": negatives,
        "count": int(len(target)),
        "threshold": float(threshold),
    }


def build_effect_sampling_weights(
    event_ids: np.ndarray | list[str],
    aggregate_targets: np.ndarray,
    class_targets: np.ndarray,
    *,
    tolerances: np.ndarray,
    maximum_weight_ratio: float = 20.0,
    phases: np.ndarray | list[str] | None = None,
    actuator_signatures: np.ndarray | list[str] | None = None,
    deployment_mask: np.ndarray | list[bool] | None = None,
    replay_weight: float = 1.0,
) -> np.ndarray:
    """Balance events, phases, actuator identity, effect signs, and safety labels."""
    events = np.asarray(event_ids).astype(str)
    aggregate = np.asarray(aggregate_targets, dtype=np.float64)
    classes = np.asarray(class_targets, dtype=np.int8)
    tolerance = np.asarray(tolerances, dtype=np.float64)
    if aggregate.shape != classes.shape or aggregate.ndim != 2:
        raise ValueError("aggregate_targets and class_targets must share [N,C]")
    if len(events) != len(aggregate) or tolerance.shape != (aggregate.shape[1],):
        raise ValueError("event or tolerance dimensions do not align")

    unique_events, event_counts = np.unique(events, return_counts=True)
    event_count = dict(zip(unique_events.tolist(), event_counts.astype(float).tolist()))
    weights = np.asarray([1.0 / event_count[event] for event in events], dtype=np.float64)
    for channel in range(aggregate.shape[1]):
        valid = np.abs(aggregate[:, channel]) > tolerance[channel]
        negative_effect = aggregate[:, channel] < 0.0
        for mask in (valid & negative_effect, valid & ~negative_effect):
            count = int(mask.sum())
            if count:
                weights[mask] += 1.0 / count
    for labels, name in ((phases, "phases"), (actuator_signatures, "actuator_signatures")):
        if labels is None:
            continue
        values = np.asarray(labels).astype(str)
        if len(values) != len(weights):
            raise ValueError(f"{name} dimensions do not align")
        unique, counts = np.unique(values, return_counts=True)
        count_by_value = dict(zip(unique.tolist(), counts.astype(float).tolist()))
        weights += np.asarray([1.0 / count_by_value[value] for value in values], dtype=np.float64)
    for channel in range(classes.shape[1]):
        for label in (0, 1):
            mask = classes[:, channel] == label
            count = int(mask.sum())
            if count:
                weights[mask] += 1.0 / count
    positive = weights[weights > 0.0]
    if not len(positive):
        raise ValueError("sampling weights contain no positive values")
    cap = float(positive.min()) * max(1.0, float(maximum_weight_ratio))
    weights = np.minimum(weights, cap)
    if deployment_mask is not None:
        deployment = np.asarray(deployment_mask, dtype=bool)
        if deployment.shape != weights.shape:
            raise ValueError("deployment_mask dimensions do not align")
        if not 0.0 < float(replay_weight) <= 1.0:
            raise ValueError("replay_weight must be in (0, 1]")
        weights[~deployment] *= float(replay_weight)
    return weights / weights.sum()


def same_checkpoint_pairwise_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    checkpoint_ids: np.ndarray | list[str],
    *,
    scale: torch.Tensor,
    tolerances: torch.Tensor,
) -> torch.Tensor:
    """Rank candidate effects within the same hydraulic checkpoint.

    This uses the strict same-state pairing directly: candidates from different
    states are never compared. Lower PFV, TFV, and peak effects rank better.
    """
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share [B,C]")
    groups = np.asarray(checkpoint_ids).astype(str)
    if len(groups) != len(prediction):
        raise ValueError("checkpoint_ids must align with the batch")
    losses: list[torch.Tensor] = []
    for checkpoint_id in np.unique(groups):
        positions = np.flatnonzero(groups == checkpoint_id)
        if len(positions) < 2:
            continue
        local = torch.as_tensor(positions, dtype=torch.long, device=prediction.device)
        local_prediction = prediction.index_select(0, local)
        local_target = target.index_select(0, local)
        row, col = torch.triu_indices(len(positions), len(positions), offset=1, device=prediction.device)
        target_difference = local_target[row] - local_target[col]
        prediction_difference = local_prediction[row] - local_prediction[col]
        valid = target_difference.abs() > tolerances.to(prediction.device)[None, :]
        if not bool(valid.any()):
            continue
        # A negative target difference means the first candidate is better.
        better = (target_difference < 0.0).to(prediction.dtype)
        logits = -prediction_difference / torch.clamp(scale.to(prediction.device)[None, :], min=1.0e-8)
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits[valid], better[valid]))
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def conformal_sigma_multipliers(
    prediction: np.ndarray,
    target: np.ndarray,
    sigma: np.ndarray,
    *,
    coverage: float = 0.90,
) -> np.ndarray:
    """Calibrate aggregate uncertainty on an event-disjoint calibration fold."""
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    scale = np.asarray(sigma, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.shape != scale.shape or predicted.ndim != 2:
        raise ValueError("prediction, target, and sigma must share [N,C]")
    ratio = np.abs(predicted - observed) / np.maximum(scale, 1.0e-8)
    quantile = min(1.0, max(0.5, float(coverage)))
    multiplier = np.quantile(ratio, quantile, axis=0, method="higher")
    return np.maximum(multiplier, 1.0).astype(np.float32)


def apply_uncertainty_multipliers(
    outputs: dict[str, np.ndarray],
    multipliers: np.ndarray | list[float] | tuple[float, float, float],
) -> dict[str, np.ndarray]:
    """Apply saved aggregate conformal scales without mutating model outputs."""
    scale = np.asarray(multipliers, dtype=np.float32)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale < 1.0):
        raise ValueError("uncertainty multipliers must be three finite values >= 1")
    calibrated = dict(outputs)
    for channel, key in enumerate(("delta_PFV_sigma", "delta_TFV_sigma", "delta_peak_sigma")):
        if key not in calibrated:
            raise KeyError(f"missing aggregate uncertainty output: {key}")
        calibrated[key] = np.asarray(calibrated[key], dtype=np.float32) * scale[channel]
    return calibrated


def select_binary_threshold(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    minimum_negative_recall: float = 0.80,
) -> dict[str, float | None]:
    """Select a safety threshold on calibration data, prioritizing unsafe recall."""
    values = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(target).astype(bool)
    if values.shape != labels.shape or values.ndim != 1:
        raise ValueError("probability and target must share [N]")
    if not labels.any() or labels.all():
        return {"threshold": 0.5, "balanced_accuracy": None, "negative_recall": None}
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], values)))
    eligible: list[tuple[float, float, float]] = []
    fallback: list[tuple[float, float, float]] = []
    for threshold in candidates:
        prediction = values >= threshold
        positive_recall = float(prediction[labels].mean())
        negative_recall = float((~prediction[~labels]).mean())
        balanced = 0.5 * (positive_recall + negative_recall)
        row = (balanced, negative_recall, float(threshold))
        fallback.append(row)
        if negative_recall >= float(minimum_negative_recall):
            eligible.append(row)
    balanced, negative_recall, threshold = max(eligible or fallback)
    return {
        "threshold": threshold,
        "balanced_accuracy": balanced,
        "negative_recall": negative_recall,
    }
def gate_aligned_selection_score(
    metrics: dict[str, float | None],
    thresholds: dict[str, float],
) -> float:
    """Rank checkpoints by their weakest deployment-critical validation ratio."""
    names = (
        "tfv_direction",
        "peak_direction",
        "tfv_balanced_accuracy",
        "peak_balanced_accuracy",
        "peak_unsafe_recall",
    )
    ratios: list[float] = []
    for name in names:
        value = metrics.get(name)
        threshold = float(thresholds[name])
        if threshold <= 0.0:
            continue
        if value is None or not np.isfinite(float(value)):
            return float("-inf")
        ratios.append(float(value) / threshold)
    if not ratios:
        return float("-inf")
    return float(min(ratios) + 0.01 * np.mean(ratios))
