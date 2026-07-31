from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from sewerrtc.data.peak_label_semantics import (
    peak_label_semantics_valid,
    repair_paired_risk_rate_sequences,
)

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.models.raw_joint_action_surrogate import RawJointActionSurrogate, encode_phase_indices
from sewerrtc.models.raw_joint_training import (
    aggregate_effect_targets,
    binary_classification_metrics,
    build_effect_sampling_weights,
    configure_effect_training_parameters,
    conformal_sigma_multipliers,
    direction_accuracy,
    finite_metric_at_least,
    gate_aligned_selection_score,
    robust_scale,
    same_checkpoint_pairwise_ranking_loss,
    select_binary_threshold,
)


TENSOR_KEYS = (
    "state",
    "candidate_action_seq",
    "reference_action_seq",
    "rain_seq",
    "reference_risk_rate_seq",
    "delta_risk_rate_seq",
    "priority_depth_seq",
    "storage_level_seq",
    "target_state_seq",
)

SUPPORTED_LABEL_SEMANTICS = {
    "same_state_candidate_minus_no_control",
    "mixed_reference_residual10_core_conditioned",
}


def label_semantics_supported(value: object) -> bool:
    return str(value) in SUPPORTED_LABEL_SEMANTICS


def resolve_architecture_version(requested: object, warm_checkpoint: dict) -> str:
    requested_text = str(requested or "auto")
    if requested_text.lower() != "auto":
        return requested_text
    checkpoint_version = str(warm_checkpoint.get("architecture_version", "") or "").strip()
    return checkpoint_version or "priority_aware_safety_v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    return selected.mean() if selected.numel() else values.sum() * 0.0


def _channel_weights(*, peak_multiplier: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([1.0, 1.0, max(1.0e-6, float(peak_multiplier))], dtype=torch.float32, device=device)


def _apply_peak_direction_sampling_weight(
    weights: np.ndarray,
    aggregate_targets: np.ndarray,
    *,
    peak_tolerance: float,
    peak_multiplier: float,
) -> np.ndarray:
    out = np.asarray(weights, dtype=np.float64).copy()
    if float(peak_multiplier) <= 1.0:
        return out / out.sum()
    peak_active = np.abs(np.asarray(aggregate_targets, dtype=np.float64)[:, 2]) > float(peak_tolerance)
    out[peak_active] *= float(peak_multiplier)
    return out / out.sum()


def _classification_targets(
    reference_rate: torch.Tensor,
    aggregate_delta: torch.Tensor,
    *,
    pfv_abs_margin: float,
    pfv_rel_margin: float,
    tfv_improvement_deadband: float,
    peak_margin: float,
) -> torch.Tensor:
    reference_pfv = reference_rate[:, :, 0].sum(dim=1) * 300.0
    pfv_margin = torch.maximum(
        torch.full_like(reference_pfv, float(pfv_abs_margin)),
        torch.clamp(reference_pfv, min=0.0) * float(pfv_rel_margin),
    )
    return torch.stack(
        [
            aggregate_delta[:, 0] <= pfv_margin,
            aggregate_delta[:, 1] < -float(tfv_improvement_deadband),
            aggregate_delta[:, 2] <= float(peak_margin),
        ],
        dim=1,
    ).float()


def _class_support(target: np.ndarray, event_ids: np.ndarray) -> dict[str, int]:
    positive = target.astype(bool)
    return {
        "positive_rows": int(positive.sum()),
        "negative_rows": int((~positive).sum()),
        "positive_events": int(len(set(event_ids[positive]))),
        "negative_events": int(len(set(event_ids[~positive]))),
    }


def _bootstrap_ci(values: np.ndarray, event_ids: np.ndarray, statistic, *, seed: int = 20260713, draws: int = 1000) -> list[float] | None:
    events = np.asarray(sorted(set(event_ids.astype(str))))
    if not len(events):
        return None
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(draws)):
        chosen = rng.choice(events, size=len(events), replace=True)
        indices = np.concatenate([np.flatnonzero(event_ids == event) for event in chosen])
        result = statistic(values[indices])
        if result is not None and np.isfinite(result):
            samples.append(float(result))
    if not samples:
        return None
    return np.quantile(np.asarray(samples), [0.025, 0.975]).astype(float).tolist()


def _direction_stat(values: np.ndarray, tolerance: float) -> float | None:
    prediction, target = values[:, 0], values[:, 1]
    valid = np.abs(target) > float(tolerance)
    if not valid.any():
        return None
    return float(np.mean((prediction[valid] < 0) == (target[valid] < 0)))


def _direction_score_from_probability(probability: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert P(effect improves) into a signed score compatible with _direction_stat.

    The existing gate treats negative scores as improvement and positive scores
    as worsening. A probability above 0.5 should therefore map to a negative
    signed score.
    """
    return 0.5 - probability


def _split_fit_calibration_events(
    train_idx: np.ndarray,
    event_ids: np.ndarray,
    *,
    fraction: float,
    seed: int,
    requested_events: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, set[str]]:
    events = np.asarray(sorted(set(event_ids[train_idx].astype(str))))
    if requested_events is not None:
        calibration_events = {str(value) for value in requested_events}
        missing = calibration_events - set(events)
        if missing:
            raise ValueError(f"requested calibration events are not training events: {sorted(missing)}")
        if not calibration_events or calibration_events == set(events):
            raise ValueError("requested calibration events must be a non-empty proper subset of training events")
        calibration_mask = np.asarray([event in calibration_events for event in event_ids], dtype=bool)
        calibration_idx = train_idx[calibration_mask[train_idx]]
        fit_idx = train_idx[~calibration_mask[train_idx]]
        return fit_idx, calibration_idx, calibration_events
    if float(fraction) <= 0.0 or len(events) < 3:
        return train_idx.copy(), np.asarray([], dtype=np.int64), set()
    rng = np.random.default_rng(int(seed))
    shuffled = events.copy()
    rng.shuffle(shuffled)
    count = max(2, int(round(len(events) * min(float(fraction), 0.4))))
    count = min(count, len(events) - 1)
    calibration_events = set(shuffled[:count].tolist())
    calibration_mask = np.asarray([event in calibration_events for event in event_ids], dtype=bool)
    calibration_idx = train_idx[calibration_mask[train_idx]]
    fit_idx = train_idx[~calibration_mask[train_idx]]
    if not len(fit_idx) or not len(calibration_idx):
        raise ValueError("event-disjoint fit/calibration split is empty")
    return fit_idx, calibration_idx, calibration_events


def _selection_score(
    predicted: np.ndarray,
    target: np.ndarray,
    probabilities: np.ndarray,
    class_targets: np.ndarray,
    tolerances: np.ndarray,
) -> float:
    scores: list[float] = []
    for channel in range(3):
        # PFV is a non-inferiority task. Its raw improve/worsen sign remains a
        # diagnostic, while TFV and peak retain directional deployment value.
        if channel != 0:
            value = _direction_stat(
                np.column_stack([predicted[:, channel], target[:, channel]]),
                float(tolerances[channel]),
            )
            if value is not None:
                scores.append(float(value))
        labels = class_targets[:, channel].astype(bool)
        if labels.any() and (~labels).any():
            prediction = probabilities[:, channel] >= 0.5
            positive_recall = float(prediction[labels].mean())
            negative_recall = float((~prediction[~labels]).mean())
            scores.append(0.5 * (positive_recall + negative_recall))
    return float(np.mean(scores)) if scores else float("-inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--dataset", default="outputs/project6_36_temporal_joint_v2/effect_dataset/same_state_raw_joint_36_v3.npz")
    parser.add_argument("--warm-start", default="outputs/models_temporal_joint_36/raw_joint_36_same_state_v2.pt")
    parser.add_argument("--v2-report", default="outputs/models_temporal_joint_36/raw_joint_36_same_state_v2_train_report.json")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--fine-tune-action-encoder", action="store_true")
    parser.add_argument("--fine-tune-state-interaction", action="store_true")
    parser.add_argument("--action-learning-rate-scale", type=float, default=0.10)
    parser.add_argument("--state-learning-rate-scale", type=float, default=0.02)
    parser.add_argument("--direction-loss-weight", type=float, default=0.50)
    parser.add_argument("--peak-direction-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--peak-aggregate-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--peak-sequence-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--peak-direction-sample-weight", type=float, default=1.0)
    parser.add_argument("--direction-classification-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--direction-eval-exclude-candidate-kinds",
        default="",
        help="Comma-separated candidate_kind values excluded from deployment direction metrics.",
    )
    parser.add_argument("--pairwise-ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--classification-loss-weight", type=float, default=0.75)
    parser.add_argument("--reference-loss-weight", type=float, default=0.0)
    parser.add_argument("--architecture-version", default="auto")
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--balanced-epoch-multiplier", type=float, default=1.0)
    parser.add_argument("--calibration-event-fraction", type=float, default=0.0)
    parser.add_argument("--calibration-events-file")
    parser.add_argument("--selection-objective", choices=("average", "gate_aligned"), default="average")
    parser.add_argument("--offline-safety-sample-weight", type=float, default=1.0)
    parser.add_argument("--deployment-source-token", default="")
    parser.add_argument("--legacy-replay-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-coverage", type=float, default=0.90)
    parser.add_argument("--selection-every", type=int, default=5)
    parser.add_argument("--lr-plateau-patience", type=int, default=3)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--out-dir", default="outputs/models_temporal_joint_36_v3")
    parser.add_argument("--model-name", default="raw_joint_36_same_state_v3.pt")
    parser.add_argument("--report-name", default="raw_joint_36_same_state_v3_train_report.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    dataset_path = root / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    warm_path = root / args.warm_start if not Path(args.warm_start).is_absolute() else Path(args.warm_start)
    v2_report_path = root / args.v2_report if not Path(args.v2_report).is_absolute() else Path(args.v2_report)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    data = np.load(dataset_path, allow_pickle=True)
    label_semantics = str(data["label_semantics"].item())
    if not label_semantics_supported(label_semantics):
        raise ValueError(
            "v3 training requires supported same-state effect labels; "
            f"got label_semantics={label_semantics!r}"
        )
    event_ids = data["event_ids"].astype(str)
    split = data["split"].astype(str)
    phases = data["phase"].astype(str) if "phase" in data.files else np.asarray(["unknown"] * len(event_ids))
    phase_indices = encode_phase_indices(phases.tolist())
    if set(split) != {"train", "validation"}:
        raise ValueError(f"dataset split must contain train and validation, got {set(split)}")
    train_events = set(event_ids[split == "train"])
    validation_events = set(event_ids[split == "validation"])
    if train_events & validation_events:
        raise ValueError("event-group leakage detected between train and validation")
    train_idx = np.flatnonzero(split == "train")
    validation_idx = np.flatnonzero(split == "validation")
    requested_calibration_events = None
    if args.calibration_events_file:
        calibration_path = root / args.calibration_events_file if not Path(args.calibration_events_file).is_absolute() else Path(args.calibration_events_file)
        requested_calibration_events = {
            line.strip() for line in calibration_path.read_text(encoding="utf-8").splitlines() if line.strip()
        }
    fit_idx, calibration_idx, calibration_events = _split_fit_calibration_events(
        train_idx,
        event_ids,
        fraction=float(args.calibration_event_fraction),
        seed=int(args.seed),
        requested_events=requested_calibration_events,
    )
    fit_events = set(event_ids[fit_idx])
    if fit_events & calibration_events or calibration_events & validation_events:
        raise ValueError("event leakage detected across fit/calibration/validation")
    reference_risk_np, delta_risk_np = repair_paired_risk_rate_sequences(
        data["reference_risk_rate_seq"],
        data["delta_risk_rate_seq"],
    )
    tensor_arrays = {key: data[key] for key in TENSOR_KEYS}
    tensor_arrays["reference_risk_rate_seq"] = reference_risk_np
    tensor_arrays["delta_risk_rate_seq"] = delta_risk_np
    tensors = {key: torch.as_tensor(tensor_arrays[key], dtype=torch.float32) for key in TENSOR_KEYS}

    warm = torch.load(warm_path, map_location="cpu", weights_only=False)
    action_ids = data["action_ids"].astype(str).tolist()
    node_ids = data["node_ids"].astype(str).tolist()
    if action_ids != [str(item) for item in warm["action_ids"]]:
        raise ValueError("v3 action order differs from v2 warm-start checkpoint")
    if node_ids != [str(item) for item in warm["node_ids"]]:
        raise ValueError("v3 node order differs from v2 warm-start checkpoint")

    node_static = np.asarray(warm["node_static"], dtype=np.float32)
    edge_index = np.asarray(warm["edge_index"], dtype=np.int64)
    action_node_map = np.asarray(warm["action_node_map"], dtype=np.float32)
    actuator_features = np.asarray(warm["actuator_features"], dtype=np.float32)
    priority_indices = np.asarray(warm["priority_indices"], dtype=np.int64)
    storage_indices = np.asarray(warm["storage_indices"], dtype=np.int64)
    architecture_version = resolve_architecture_version(args.architecture_version, warm)
    model = RawJointActionSurrogate(
        n_nodes=len(node_ids),
        n_actions=len(action_ids),
        node_static_dim=node_static.shape[1],
        actuator_feature_dim=actuator_features.shape[1],
        horizon_steps=int(warm["horizon_steps"]),
        hidden_dim=int(warm["hidden_dim"]),
        heads=int(warm.get("heads", 4)),
        architecture_version=architecture_version,
    )
    incompatible = model.load_state_dict(warm["model"], strict=False)
    expected_missing_prefixes = ["safety_classification_head."]
    if architecture_version in {"priority_aware_safety_v4", "causal_phase_safety_v5", "causal_phase_direction_v6"}:
        expected_missing_prefixes.extend(("effect_action_residual_head.", "safety_action_residual_head."))
    if architecture_version in {"causal_phase_safety_v5", "causal_phase_direction_v6"}:
        expected_missing_prefixes.extend((
            "phase_effect_heads.", "phase_safety_heads.", "horizon_action_temporal.",
            "horizon_effect_head.", "horizon_scale_head.", "horizon_safety_head.",
        ))
    if architecture_version == "causal_phase_direction_v6":
        expected_missing_prefixes.extend((
            "phase_effect_heads.", "phase_safety_heads.", "horizon_action_temporal.",
            "horizon_effect_head.", "horizon_scale_head.", "horizon_safety_head.",
            "horizon_direction_head.",
        ))
    unexpected_missing = [
        key for key in incompatible.missing_keys if not any(key.startswith(prefix) for prefix in expected_missing_prefixes)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise ValueError(
            f"v2 warm-start incompatibility: missing={unexpected_missing}, unexpected={incompatible.unexpected_keys}"
        )
    model = model.to(device)
    parameter_groups, trainable_names = configure_effect_training_parameters(
        model,
        fine_tune_action_encoder=bool(args.fine_tune_action_encoder),
        fine_tune_state_interaction=bool(args.fine_tune_state_interaction),
        head_learning_rate=float(args.learning_rate),
        action_learning_rate_scale=float(args.action_learning_rate_scale),
        state_learning_rate_scale=float(args.state_learning_rate_scale),
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=1.0e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(args.lr_plateau_factor),
        patience=max(0, int(args.lr_plateau_patience)),
        min_lr=1.0e-7,
    )
    fixed = {
        "node_static": torch.as_tensor(node_static, device=device),
        "edge_index": torch.as_tensor(edge_index, dtype=torch.long, device=device),
        "action_node_map": torch.as_tensor(action_node_map, device=device),
        "actuator_features": torch.as_tensor(actuator_features, device=device),
        "priority_indices": torch.as_tensor(priority_indices, dtype=torch.long, device=device),
        "storage_indices": torch.as_tensor(storage_indices, dtype=torch.long, device=device),
    }
    mask = torch.ones((1, len(action_ids)), dtype=torch.float32, device=device)
    temporal = (((cfg.get("controller", {}) or {}).get("temporal_joint", {}) or {}))
    validation_cfg = temporal.get("training_validation", {}) or {}
    safety_cfg = temporal.get("safety", {}) or {}
    pfv_abs = float(safety_cfg.get("pfv_abs_margin_m3", 100.0))
    pfv_rel = float(safety_cfg.get("pfv_rel_margin", 0.005))
    peak_margin = float(safety_cfg.get("peak_margin", 0.0))
    tolerances = torch.tensor(
        [
            float(validation_cfg.get("pfv_direction_tolerance_m3", 1.0)),
            float(validation_cfg.get("tfv_direction_tolerance_m3", 100.0)),
            float(validation_cfg.get("peak_direction_tolerance", 0.1)),
        ],
        dtype=torch.float32,
    )

    train_rows = torch.as_tensor(fit_idx, dtype=torch.long)
    aggregate_target_train = aggregate_effect_targets(
        tensors["reference_risk_rate_seq"].index_select(0, train_rows),
        tensors["delta_risk_rate_seq"].index_select(0, train_rows),
    )
    aggregate_scale = robust_scale(aggregate_target_train, dimensions=(0,), minimum=0.1)
    delta_rate_scale = robust_scale(
        tensors["delta_risk_rate_seq"].index_select(0, train_rows),
        dimensions=(0, 1),
        minimum=0.01,
    )
    reference_rate_scale = robust_scale(
        tensors["reference_risk_rate_seq"].index_select(0, train_rows),
        dimensions=(0, 1),
        minimum=1.0,
    )
    train_class_targets = _classification_targets(
        tensors["reference_risk_rate_seq"].index_select(0, train_rows),
        aggregate_target_train,
        pfv_abs_margin=pfv_abs,
        pfv_rel_margin=pfv_rel,
        tfv_improvement_deadband=float(tolerances[1]),
        peak_margin=peak_margin,
    )
    positive = train_class_targets.sum(dim=0)
    negative = len(fit_idx) - positive
    positive_weight = torch.where(
        (positive > 0) & (negative > 0),
        negative / torch.clamp(positive, min=1.0),
        torch.ones_like(positive),
    ).to(device)

    direction_positive_weight = []
    for channel in range(3):
        channel_target = aggregate_target_train[:, channel]
        valid = channel_target.abs() > tolerances[channel]
        negative_effect = int((valid & (channel_target < 0)).sum().item())
        nonnegative_effect = int((valid & (channel_target >= 0)).sum().item())
        direction_positive_weight.append(
            float(nonnegative_effect / max(negative_effect, 1))
            if negative_effect and nonnegative_effect
            else 1.0
        )
    direction_positive_weight_tensor = torch.as_tensor(
        direction_positive_weight, dtype=torch.float32, device=device
    )

    if bool(args.balanced_sampling):
        fit_residual = tensor_arrays["candidate_action_seq"][fit_idx] - tensor_arrays["reference_action_seq"][fit_idx]
        fit_signatures = []
        for residual in fit_residual:
            changed = np.flatnonzero(np.any(np.abs(residual) > 1.0e-7, axis=0))
            fit_signatures.append("|".join(action_ids[index] for index in changed) or "noop")
        deployment_mask = None
        if str(args.deployment_source_token).strip():
            if "source_dataset" not in data.files:
                raise ValueError("deployment-source-token requires source_dataset metadata")
            fit_sources = data["source_dataset"].astype(str)[fit_idx]
            deployment_mask = np.asarray(
                [str(args.deployment_source_token) in source for source in fit_sources],
                dtype=bool,
            )
            if not deployment_mask.any():
                raise ValueError(
                    f"deployment-source-token matched no fit rows: {args.deployment_source_token}"
                )
        sampling_weights = build_effect_sampling_weights(
            event_ids[fit_idx],
            aggregate_target_train.numpy(),
            train_class_targets.numpy(),
            tolerances=tolerances.numpy(),
            phases=phases[fit_idx],
            actuator_signatures=np.asarray(fit_signatures),
            deployment_mask=deployment_mask,
            replay_weight=float(args.legacy_replay_weight),
        )
        if not 0.0 < float(args.offline_safety_sample_weight) <= 1.0:
            raise ValueError("offline-safety-sample-weight must be in (0, 1]")
        if float(args.offline_safety_sample_weight) < 1.0:
            fit_kinds = data["candidate_kind"].astype(str)[fit_idx]
            offline = np.isin(fit_kinds, ["strong_counterfactual", "strong_single_or_pair"])
            sampling_weights[offline] *= float(args.offline_safety_sample_weight)
            sampling_weights /= sampling_weights.sum()
        sampling_weights = _apply_peak_direction_sampling_weight(
            sampling_weights,
            aggregate_target_train.numpy(),
            peak_tolerance=float(tolerances[2]),
            peak_multiplier=float(args.peak_direction_sample_weight),
        )
    else:
        sampling_weights = None
        deployment_mask = None

    sampling_summary = {
        "deployment_source_token": str(args.deployment_source_token),
        "legacy_replay_weight": float(args.legacy_replay_weight),
        "fit_deployment_rows": int(deployment_mask.sum()) if deployment_mask is not None else None,
        "fit_replay_rows": int((~deployment_mask).sum()) if deployment_mask is not None else None,
        "deployment_sampling_probability": (
            float(sampling_weights[deployment_mask].sum())
            if sampling_weights is not None and deployment_mask is not None
            else None
        ),
    }

    rng = np.random.default_rng(int(args.seed))
    history: list[dict[str, float | int]] = []
    best_selection_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    selection_rounds_without_improvement = 0
    stopped_early = False

    def predict_rows(indices: np.ndarray) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        rows = {key: value[indices].to(device) for key, value in tensors.items()}
        with torch.no_grad():
            predicted = model(
                state=rows["state"],
                candidate_action_seq=rows["candidate_action_seq"],
                reference_action_seq=rows["reference_action_seq"],
                rain_seq=rows["rain_seq"],
                actuator_mask=mask.expand(len(indices), -1),
                phase_index=phase_indices[indices].to(device),
                **fixed,
            )
        return predicted, rows

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        if sampling_weights is None:
            epoch_indices = rng.permutation(fit_idx)
        else:
            epoch_size = max(
                len(fit_idx),
                int(np.ceil(len(fit_idx) * max(float(args.balanced_epoch_multiplier), 1.0))),
            )
            epoch_indices = rng.choice(
                fit_idx,
                size=epoch_size,
                replace=True,
                p=sampling_weights,
            )
        epoch_losses: dict[str, list[float]] = {}
        for start in range(0, len(epoch_indices), int(args.batch_size)):
            indices = epoch_indices[start : start + int(args.batch_size)]
            batch = {key: value[indices].to(device) for key, value in tensors.items()}
            output = model(
                state=batch["state"],
                candidate_action_seq=batch["candidate_action_seq"],
                reference_action_seq=batch["reference_action_seq"],
                rain_seq=batch["rain_seq"],
                actuator_mask=mask.expand(len(indices), -1),
                phase_index=phase_indices[indices].to(device),
                **fixed,
            )
            target_aggregate = aggregate_effect_targets(batch["reference_risk_rate_seq"], batch["delta_risk_rate_seq"])
            target_classes = _classification_targets(
                batch["reference_risk_rate_seq"],
                target_aggregate,
                pfv_abs_margin=pfv_abs,
                pfv_rel_margin=pfv_rel,
                tfv_improvement_deadband=float(tolerances[1]),
                peak_margin=peak_margin,
            )
            predicted_aggregate = torch.stack(
                [output["delta_PFV_H"], output["delta_TFV_H"], output["delta_peak"]], dim=1
            )
            active = (batch["candidate_action_seq"] - batch["reference_action_seq"]).abs().amax(dim=(1, 2)) > 1.0e-8
            sequence_channel_weight = _channel_weights(
                peak_multiplier=float(args.peak_sequence_loss_multiplier),
                device=device,
            )
            sequence_error = torch.nn.functional.smooth_l1_loss(
                output["delta_risk_rate_seq"] / delta_rate_scale.to(device)[None, None, :],
                batch["delta_risk_rate_seq"] / delta_rate_scale.to(device)[None, None, :],
                reduction="none",
            )
            sequence_error = (sequence_error * sequence_channel_weight[None, None, :]).mean(dim=(1, 2))
            l_sequence = _masked_mean(sequence_error, active)
            aggregate_channel_weight = _channel_weights(
                peak_multiplier=float(args.peak_aggregate_loss_multiplier),
                device=device,
            )
            aggregate_error = torch.nn.functional.smooth_l1_loss(
                predicted_aggregate / aggregate_scale.to(device)[None, :],
                target_aggregate / aggregate_scale.to(device)[None, :],
                reduction="none",
            )
            aggregate_error = (aggregate_error * aggregate_channel_weight[None, :]).mean(dim=1)
            l_aggregate = _masked_mean(aggregate_error, active)
            l_reference = torch.nn.functional.smooth_l1_loss(
                output["reference_risk_rate_seq"] / reference_rate_scale.to(device)[None, None, :],
                batch["reference_risk_rate_seq"] / reference_rate_scale.to(device)[None, None, :],
            )
            direction_terms = []
            direction_logits = output.get("direction_classification_logits")
            for channel in range(3):
                valid = active & (target_aggregate[:, channel].abs() > tolerances[channel].to(device))
                if bool(valid.any()):
                    labels = (target_aggregate[valid, channel] < 0).float()
                    if direction_logits is not None:
                        channel_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            direction_logits[valid, channel],
                            labels,
                            pos_weight=direction_positive_weight_tensor[channel],
                        )
                    else:
                        channel_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            -predicted_aggregate[valid, channel] / aggregate_scale[channel].to(device),
                            labels,
                            pos_weight=direction_positive_weight_tensor[channel],
                        )
                    if channel == 2:
                        channel_loss = channel_loss * float(args.peak_direction_loss_multiplier)
                    direction_terms.append(channel_loss)
            l_direction = torch.stack(direction_terms).mean() if direction_terms else predicted_aggregate.sum() * 0.0
            if direction_logits is not None:
                l_direction = l_direction * float(args.direction_classification_loss_weight)
            if float(args.pairwise_ranking_loss_weight) > 0.0:
                if "checkpoint_id" not in data.files:
                    raise ValueError("pairwise ranking requires checkpoint_id metadata")
                l_pairwise = same_checkpoint_pairwise_ranking_loss(
                    predicted_aggregate,
                    target_aggregate,
                    data["checkpoint_id"].astype(str)[indices],
                    scale=aggregate_scale.to(device),
                    tolerances=tolerances.to(device),
                )
            else:
                l_pairwise = predicted_aggregate.sum() * 0.0
            l_classification = torch.nn.functional.binary_cross_entropy_with_logits(
                output["safety_classification_logits"],
                target_classes,
                pos_weight=positive_weight,
            )
            normalized_sigma = torch.clamp(
                output["delta_risk_sigma_seq"] / delta_rate_scale.to(device)[None, None, :],
                min=0.05,
                max=20.0,
            )
            normalized_residual = (
                output["delta_risk_rate_seq"] - batch["delta_risk_rate_seq"]
            ) / delta_rate_scale.to(device)[None, None, :]
            uncertainty_error = (
                0.5 * (normalized_residual.detach() / normalized_sigma).square()
                + torch.log(normalized_sigma)
            ).mean(dim=(1, 2))
            l_uncertainty = _masked_mean(uncertainty_error, active)
            l_zero = _masked_mean(
                output["delta_risk_rate_seq"].abs().mean(dim=(1, 2)),
                ~active,
            )
            loss = (
                l_aggregate
                + 0.25 * l_sequence
                + float(args.direction_loss_weight) * l_direction
                + float(args.pairwise_ranking_loss_weight) * l_pairwise
                + float(args.classification_loss_weight) * l_classification
                + float(args.reference_loss_weight) * l_reference
                + 0.02 * l_uncertainty
                + l_zero
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            for key, value in {
                "total": loss,
                "aggregate": l_aggregate,
                "sequence": l_sequence,
                "direction": l_direction,
                "pairwise_ranking": l_pairwise,
                "classification": l_classification,
                "reference": l_reference,
                "uncertainty": l_uncertainty,
                "zero": l_zero,
            }.items():
                epoch_losses.setdefault(key, []).append(float(value.detach().cpu()))
        record = {"epoch": epoch, **{f"loss_{key}": float(np.mean(values)) for key, values in epoch_losses.items()}}
        if len(calibration_idx) and (
            epoch == 1
            or epoch % max(1, int(args.selection_every)) == 0
            or epoch == int(args.epochs)
        ):
            model.eval()
            calibration_output, calibration_batch = predict_rows(calibration_idx)
            calibration_predicted = torch.stack(
                [
                    calibration_output["delta_PFV_H"],
                    calibration_output["delta_TFV_H"],
                    calibration_output["delta_peak"],
                ],
                dim=1,
            )
            calibration_target = aggregate_effect_targets(
                calibration_batch["reference_risk_rate_seq"],
                calibration_batch["delta_risk_rate_seq"],
            )
            calibration_classes = _classification_targets(
                calibration_batch["reference_risk_rate_seq"],
                calibration_target,
                pfv_abs_margin=pfv_abs,
                pfv_rel_margin=pfv_rel,
                tfv_improvement_deadband=float(tolerances[1]),
                peak_margin=peak_margin,
            )
            calibration_probabilities = torch.stack(
                [
                    calibration_output["PFV_noninferiority_classifier_probability"],
                    calibration_output["TFV_improvement_classifier_probability"],
                    calibration_output["peak_safe_classifier_probability"],
                ],
                dim=1,
            )
            calibration_predicted_np = calibration_predicted.cpu().numpy()
            calibration_target_np = calibration_target.cpu().numpy()
            calibration_probability_np = calibration_probabilities.cpu().numpy()
            calibration_class_np = calibration_classes.cpu().numpy()
            calibration_direction_np = calibration_predicted_np
            if "direction_classification_logits" in calibration_output:
                calibration_direction_probability = torch.sigmoid(
                    calibration_output["direction_classification_logits"]
                ).cpu().numpy()
                calibration_direction_np = _direction_score_from_probability(calibration_direction_probability)
            if args.selection_objective == "gate_aligned":
                tfv_class = binary_classification_metrics(
                    calibration_probabilities[:, 1].cpu(), calibration_classes[:, 1].cpu()
                )
                peak_class = binary_classification_metrics(
                    calibration_probabilities[:, 2].cpu(), calibration_classes[:, 2].cpu()
                )
                selection_metrics = {
                    "tfv_direction": _direction_stat(
                        np.column_stack([calibration_direction_np[:, 1], calibration_target_np[:, 1]]),
                        float(tolerances[1]),
                    ),
                    "peak_direction": _direction_stat(
                        np.column_stack([calibration_direction_np[:, 2], calibration_target_np[:, 2]]),
                        float(tolerances[2]),
                    ),
                    "tfv_balanced_accuracy": tfv_class["balanced_accuracy"],
                    "peak_balanced_accuracy": peak_class["balanced_accuracy"],
                    "peak_unsafe_recall": peak_class["negative_recall"],
                }
                selection_thresholds = {
                    "tfv_direction": float(validation_cfg.get("min_tfv_direction_accuracy", 0.60)),
                    "peak_direction": float(validation_cfg.get("min_peak_direction_accuracy", 0.70)),
                    "tfv_balanced_accuracy": float(validation_cfg.get("min_tfv_improvement_balanced_accuracy", 0.0)),
                    "peak_balanced_accuracy": float(validation_cfg.get("min_peak_safe_balanced_accuracy", 0.0)),
                    "peak_unsafe_recall": float(validation_cfg.get("min_peak_unsafe_recall", 0.80)),
                }
                selection_score = gate_aligned_selection_score(selection_metrics, selection_thresholds)
                record["calibration_gate_aligned_metrics"] = selection_metrics
            else:
                selection_score = _selection_score(
                    calibration_predicted_np,
                    calibration_target_np,
                    calibration_probability_np,
                    calibration_class_np,
                    tolerances.numpy(),
                )
            record["calibration_selection_score"] = float(selection_score)
            if selection_score > best_selection_score:
                best_selection_score = float(selection_score)
                best_epoch = int(epoch)
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                selection_rounds_without_improvement = 0
            else:
                selection_rounds_without_improvement += 1
            scheduler.step(float(selection_score) if np.isfinite(float(selection_score)) else -1.0e12)
            record["learning_rates"] = [float(group["lr"]) for group in optimizer.param_groups]
            record["selection_rounds_without_improvement"] = int(selection_rounds_without_improvement)
        history.append(record)
        if epoch == 1 or epoch % 5 == 0 or epoch == int(args.epochs):
            print(json.dumps(_json_safe(record)))
        if (
            int(args.early_stopping_patience) > 0
            and selection_rounds_without_improvement >= int(args.early_stopping_patience)
        ):
            stopped_early = True
            print(json.dumps({
                "early_stopping": True,
                "epoch": int(epoch),
                "best_epoch": int(best_epoch),
                "selection_rounds_without_improvement": int(selection_rounds_without_improvement),
            }))
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    model.eval()
    uncertainty_multipliers = np.ones(3, dtype=np.float32)
    classification_thresholds = np.full(3, 0.5, dtype=np.float32)
    calibration_summary: dict[str, object] = {
        "enabled": bool(len(calibration_idx)),
        "events": sorted(calibration_events),
        "rows": int(len(calibration_idx)),
    }
    if len(calibration_idx):
        calibration_output, calibration_batch = predict_rows(calibration_idx)
        calibration_predicted = torch.stack(
            [
                calibration_output["delta_PFV_H"],
                calibration_output["delta_TFV_H"],
                calibration_output["delta_peak"],
            ],
            dim=1,
        )
        calibration_target = aggregate_effect_targets(
            calibration_batch["reference_risk_rate_seq"],
            calibration_batch["delta_risk_rate_seq"],
        )
        calibration_sigma = torch.stack(
            [
                calibration_output["delta_PFV_sigma"],
                calibration_output["delta_TFV_sigma"],
                calibration_output["delta_peak_sigma"],
            ],
            dim=1,
        )
        uncertainty_multipliers = conformal_sigma_multipliers(
            calibration_predicted.cpu().numpy(),
            calibration_target.cpu().numpy(),
            calibration_sigma.cpu().numpy(),
            coverage=float(args.uncertainty_coverage),
        )
        calibration_classes = _classification_targets(
            calibration_batch["reference_risk_rate_seq"],
            calibration_target,
            pfv_abs_margin=pfv_abs,
            pfv_rel_margin=pfv_rel,
            tfv_improvement_deadband=float(tolerances[1]),
            peak_margin=peak_margin,
        )
        calibration_probabilities = torch.stack(
            [
                calibration_output["PFV_noninferiority_classifier_probability"],
                calibration_output["TFV_improvement_classifier_probability"],
                calibration_output["peak_safe_classifier_probability"],
            ],
            dim=1,
        ).cpu().numpy()
        calibration_class_np = calibration_classes.cpu().numpy().astype(bool)
        threshold_details = []
        for channel, minimum_negative_recall in enumerate((0.90, 0.80, 0.90)):
            detail = select_binary_threshold(
                calibration_probabilities[:, channel],
                calibration_class_np[:, channel],
                minimum_negative_recall=minimum_negative_recall,
            )
            classification_thresholds[channel] = float(detail["threshold"])
            threshold_details.append(detail)
        calibration_summary.update(
            {
                "uncertainty_target_coverage": float(args.uncertainty_coverage),
                "uncertainty_multipliers": uncertainty_multipliers.astype(float).tolist(),
                "classification_thresholds": classification_thresholds.astype(float).tolist(),
                "classification_threshold_details": threshold_details,
                "selected_epoch": int(best_epoch),
                "selected_score": None if not np.isfinite(best_selection_score) else float(best_selection_score),
            }
        )

    with torch.no_grad():
        batch = {key: value[validation_idx].to(device) for key, value in tensors.items()}
        output = model(
            state=batch["state"],
            candidate_action_seq=batch["candidate_action_seq"],
            reference_action_seq=batch["reference_action_seq"],
            rain_seq=batch["rain_seq"],
            actuator_mask=mask.expand(len(validation_idx), -1),
            phase_index=phase_indices[validation_idx].to(device),
            **fixed,
        )
        zero = model(
            state=batch["state"],
            candidate_action_seq=batch["reference_action_seq"],
            reference_action_seq=batch["reference_action_seq"],
            rain_seq=batch["rain_seq"],
            actuator_mask=mask.expand(len(validation_idx), -1),
            phase_index=phase_indices[validation_idx].to(device),
            **fixed,
        )
        predicted_delta = torch.stack(
            [output["delta_PFV_H"], output["delta_TFV_H"], output["delta_peak"]], dim=1
        )
        target_delta = aggregate_effect_targets(batch["reference_risk_rate_seq"], batch["delta_risk_rate_seq"])
        target_classes = _classification_targets(
            batch["reference_risk_rate_seq"],
            target_delta,
            pfv_abs_margin=pfv_abs,
            pfv_rel_margin=pfv_rel,
            tfv_improvement_deadband=float(tolerances[1]),
            peak_margin=peak_margin,
        )
        probabilities = torch.stack(
            [
                output["PFV_noninferiority_classifier_probability"],
                output["TFV_improvement_classifier_probability"],
                output["peak_safe_classifier_probability"],
            ],
            dim=1,
        )
        direction_probabilities = (
            torch.sigmoid(output["direction_classification_logits"])
            if "direction_classification_logits" in output
            else None
        )

    validation_event_ids = event_ids[validation_idx]
    validation_candidate_kinds = data["candidate_kind"].astype(str)[validation_idx] if "candidate_kind" in data.files else np.asarray([""] * len(validation_idx))
    direction_excluded_kinds = {
        item.strip()
        for item in str(args.direction_eval_exclude_candidate_kinds).split(",")
        if item.strip()
    }
    direction_eval_mask = (
        ~np.isin(validation_candidate_kinds, sorted(direction_excluded_kinds))
        if direction_excluded_kinds
        else np.ones(len(validation_idx), dtype=bool)
    )
    direction_scores = (
        _direction_score_from_probability(direction_probabilities).cpu()
        if direction_probabilities is not None
        else predicted_delta.cpu()
    )
    direction = {
        name: direction_accuracy(
            direction_scores[direction_eval_mask, channel],
            target_delta[direction_eval_mask, channel].cpu(),
            tolerance=float(tolerances[channel]),
        )
        for channel, name in enumerate(("PFV", "TFV", "peak"))
    }
    class_names = ("PFV_noninferiority", "TFV_improvement", "peak_safe")
    classification = {
        name: binary_classification_metrics(
            probabilities[:, channel].cpu(),
            target_classes[:, channel].cpu(),
            threshold=float(classification_thresholds[channel]),
        )
        for channel, name in enumerate(class_names)
    }
    support = {
        name: _class_support(target_classes[:, channel].cpu().numpy(), validation_event_ids)
        for channel, name in enumerate(class_names)
    }
    uncertainty = torch.stack(
        [output["delta_PFV_sigma"], output["delta_TFV_sigma"], output["delta_peak_sigma"]], dim=1
    )
    uncertainty = uncertainty * torch.as_tensor(
        uncertainty_multipliers, dtype=uncertainty.dtype, device=uncertainty.device
    )[None, :]
    coverage90 = (torch.abs(predicted_delta - target_delta) <= 1.645 * torch.clamp(uncertainty, min=1.0e-8)).float().mean(dim=0)
    zero_relative_error = float(
        zero["delta_risk_rate_seq"].abs().sum().cpu()
        / torch.clamp(batch["reference_risk_rate_seq"].abs().sum(), min=1.0e-6).cpu()
    )
    predicted_np = predicted_delta.cpu().numpy()
    direction_np = direction_scores.numpy()
    target_np = target_delta.cpu().numpy()
    probability_np = probabilities.cpu().numpy()
    target_class_np = target_classes.cpu().numpy().astype(bool)
    per_event_rows = []
    for event_id in sorted(set(validation_event_ids)):
        selected = np.flatnonzero(validation_event_ids == event_id)
        row: dict[str, object] = {"event_id": event_id, "rows": len(selected)}
        for channel, name in enumerate(("PFV", "TFV", "peak")):
            result = _direction_stat(np.column_stack([direction_np[selected, channel], target_np[selected, channel]]), float(tolerances[channel]))
            row[f"{name}_direction_accuracy"] = result
            row[f"{name}_MAE"] = float(np.mean(np.abs(predicted_np[selected, channel] - target_np[selected, channel])))
        for channel, name in enumerate(class_names):
            row[f"{name}_positive"] = int(target_class_np[selected, channel].sum())
            row[f"{name}_negative"] = int((~target_class_np[selected, channel]).sum())
            row[f"{name}_accuracy"] = float(
                np.mean(
                    (probability_np[selected, channel] >= classification_thresholds[channel])
                    == target_class_np[selected, channel]
                )
            )
        per_event_rows.append(row)

    def subgroup_metrics(label: str, selected: np.ndarray) -> dict[str, object]:
        row: dict[str, object] = {"group": label, "rows": int(len(selected)), "events": int(len(set(validation_event_ids[selected])))}
        for channel, name in enumerate(("PFV", "TFV", "peak")):
            result = _direction_stat(
                np.column_stack([direction_np[selected, channel], target_np[selected, channel]]),
                float(tolerances[channel]),
            ) if len(selected) else None
            row[f"{name}_direction_accuracy"] = result
            row[f"{name}_direction_support"] = int(np.sum(np.abs(target_np[selected, channel]) > float(tolerances[channel])))
        return row

    validation_phases = phases[validation_idx]
    per_phase_rows = [
        subgroup_metrics(str(phase), np.flatnonzero(validation_phases == phase))
        for phase in sorted(set(validation_phases))
    ]
    validation_residual = (
        tensor_arrays["candidate_action_seq"][validation_idx]
        - tensor_arrays["reference_action_seq"][validation_idx]
    )
    per_actuator_rows = []
    for action_index, actuator_id in enumerate(action_ids):
        selected = np.flatnonzero(np.any(np.abs(validation_residual[:, :, action_index]) > 1.0e-7, axis=1))
        if len(selected):
            per_actuator_rows.append({"actuator_id": actuator_id, **subgroup_metrics(actuator_id, selected)})

    metrics = {
        "zero_action_relative_error": zero_relative_error,
        "PFV_direction_accuracy": direction["PFV"]["accuracy"],
        "PFV_direction_samples": direction["PFV"]["count"],
        "TFV_direction_accuracy": direction["TFV"]["accuracy"],
        "TFV_direction_samples": direction["TFV"]["count"],
        "peak_direction_accuracy": direction["peak"]["accuracy"],
        "peak_direction_samples": direction["peak"]["count"],
        "PFV_effect_MAE_m3": float(np.mean(np.abs(predicted_np[:, 0] - target_np[:, 0]))),
        "TFV_effect_MAE_m3": float(np.mean(np.abs(predicted_np[:, 1] - target_np[:, 1]))),
        "peak_effect_MAE": float(np.mean(np.abs(predicted_np[:, 2] - target_np[:, 2]))),
        "direction_source": "direction_classification_head" if direction_probabilities is not None else "aggregate_regression_sign",
        "classification": classification,
        "class_support": support,
        "PFV_false_safe_rate": (
            None
            if classification["PFV_noninferiority"]["negative_recall"] is None
            else 1.0 - float(classification["PFV_noninferiority"]["negative_recall"])
        ),
        "uncertainty_90pct_coverage": {
            name: float(coverage90[channel].cpu()) for channel, name in enumerate(("PFV", "TFV", "peak"))
        },
        "bootstrap_95pct_CI": {
            f"{name}_direction_accuracy": _bootstrap_ci(
                np.column_stack([direction_np[direction_eval_mask, channel], target_np[direction_eval_mask, channel]]),
                validation_event_ids[direction_eval_mask],
                lambda values, tolerance=float(tolerances[channel]): _direction_stat(values, tolerance),
            )
            for channel, name in enumerate(("PFV", "TFV", "peak"))
        },
        "direction_eval_scope": {
            "rows": int(direction_eval_mask.sum()),
            "events": int(len(set(validation_event_ids[direction_eval_mask]))),
            "excluded_candidate_kinds": sorted(direction_excluded_kinds),
        },
    }

    minimum_rows_per_class = int(validation_cfg.get("min_rows_per_class", 8))
    minimum_events_per_class = int(validation_cfg.get("min_events_per_class", 3))
    support_checks = {
        f"{name}_class_support": (
            item["positive_rows"] >= minimum_rows_per_class
            and item["negative_rows"] >= minimum_rows_per_class
            and item["positive_events"] >= minimum_events_per_class
            and item["negative_events"] >= minimum_events_per_class
        )
        for name, item in support.items()
    }
    minimum_direction_samples = int(validation_cfg.get("min_direction_samples", 12))
    gate_checks = {
        "zero_action": zero_relative_error < float(validation_cfg.get("max_zero_action_relative_error", 0.005)),
        "TFV_direction_sample_count": direction["TFV"]["count"] >= minimum_direction_samples,
        "peak_direction_sample_count": direction["peak"]["count"] >= minimum_direction_samples,
        "TFV_direction": finite_metric_at_least(direction["TFV"]["accuracy"], float(validation_cfg.get("min_tfv_direction_accuracy", 0.60))),
        "peak_direction": finite_metric_at_least(direction["peak"]["accuracy"], float(validation_cfg.get("min_peak_direction_accuracy", 0.70))),
        **support_checks,
        "PFV_unsafe_recall": finite_metric_at_least(classification["PFV_noninferiority"]["negative_recall"], float(validation_cfg.get("min_pfv_unsafe_recall", 0.80))),
        "PFV_false_safe_rate": metrics["PFV_false_safe_rate"] is not None and float(metrics["PFV_false_safe_rate"]) <= float(validation_cfg.get("max_pfv_false_safe_rate", 0.10)),
        "peak_unsafe_recall": finite_metric_at_least(classification["peak_safe"]["negative_recall"], float(validation_cfg.get("min_peak_unsafe_recall", 0.80))),
    }
    balanced_accuracy_checks = {
        "PFV_noninferiority_balanced_accuracy": finite_metric_at_least(classification["PFV_noninferiority"]["balanced_accuracy"], float(validation_cfg.get("min_pfv_noninferiority_balanced_accuracy", 0.0))),
        "TFV_improvement_balanced_accuracy": finite_metric_at_least(classification["TFV_improvement"]["balanced_accuracy"], float(validation_cfg.get("min_tfv_improvement_balanced_accuracy", 0.0))),
        "peak_safe_balanced_accuracy": finite_metric_at_least(classification["peak_safe"]["balanced_accuracy"], float(validation_cfg.get("min_peak_safe_balanced_accuracy", 0.0))),
    }
    if bool(validation_cfg.get("require_balanced_accuracy_checks", False)):
        gate_checks.update(balanced_accuracy_checks)
    gate_passed = bool(all(gate_checks.values()))
    smoke_required_checks = {
        "PFV_noninferiority": bool(
            gate_checks["PFV_noninferiority_class_support"]
            and gate_checks["PFV_unsafe_recall"]
            and gate_checks["PFV_false_safe_rate"]
        ),
        "TFV_improvement_direction": bool(
            gate_checks["TFV_direction_sample_count"]
            and gate_checks["TFV_direction"]
            and gate_checks["TFV_improvement_class_support"]
        ),
        "peak_safety_direction": bool(
            gate_checks["peak_direction_sample_count"]
            and gate_checks["peak_direction"]
            and gate_checks["peak_safe_class_support"]
            and gate_checks["peak_unsafe_recall"]
        ),
    }
    smoke_eligibility = {
        "passed": bool(gate_passed and all(smoke_required_checks.values())),
        "required_checks": smoke_required_checks,
        "diagnostic_balanced_accuracy_checks": balanced_accuracy_checks,
        "policy": "Smoke is blocked unless PFV noninferiority, TFV improvement direction, and peak safety direction all validate on event-group holdout data.",
    }
    diagnostic_checks = {
        "PFV_direction_sample_count": direction["PFV"]["count"] >= minimum_direction_samples,
        "PFV_direction_accuracy": finite_metric_at_least(
            direction["PFV"]["accuracy"], float(validation_cfg.get("target_pfv_direction_accuracy", 0.75))
        ),
    }
    failures = []
    for name, passed in gate_checks.items():
        if not passed:
            reason = "insufficient_support" if name.endswith("_class_support") or name.endswith("_sample_count") else "threshold_not_met"
            failures.append({"check": name, "reason": reason})

    out_dir = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    model_path = out_dir / args.model_name
    checkpoint = {
        "model": model.state_dict(),
        "node_ids": node_ids,
        "action_ids": action_ids,
        "node_static": node_static,
        "edge_index": edge_index,
        "action_node_map": action_node_map,
        "actuator_features": actuator_features,
        "priority_indices": priority_indices,
        "storage_indices": storage_indices,
        "hidden_dim": int(warm["hidden_dim"]),
        "heads": int(warm.get("heads", 4)),
        "horizon_steps": int(warm["horizon_steps"]),
        "architecture_version": architecture_version,
        "training_mask_scope": "canonical_36",
        "label_semantics": "same_state_candidate_minus_no_control",
        "peak_label_semantics_valid": peak_label_semantics_valid(reference_risk_np),
        "peak_label_definition": "delta peak is max(candidate TFV rate)-max(reference TFV rate)",
        "training_scales": {
            "delta_rate": delta_rate_scale.numpy(),
            "aggregate": aggregate_scale.numpy(),
        },
        "uncertainty_calibration": {
            "method": "event_disjoint_split_conformal_absolute_residual_over_sigma",
            "target_coverage": float(args.uncertainty_coverage),
            "aggregate_sigma_multipliers": uncertainty_multipliers,
            "calibration_events": sorted(calibration_events),
        },
        "classification_thresholds": {
            name: float(classification_thresholds[channel])
            for channel, name in enumerate(("PFV_noninferiority", "TFV_improvement", "peak_safe"))
        },
        "provenance": {
            "dataset": str(dataset_path.resolve()),
            "dataset_sha256": _sha256(dataset_path),
            "warm_start": str(warm_path.resolve()),
            "warm_start_sha256": _sha256(warm_path),
            "config": str((root / args.config if not Path(args.config).is_absolute() else Path(args.config)).resolve()),
            "config_sha256": _sha256(root / args.config if not Path(args.config).is_absolute() else Path(args.config)),
            "train_events": sorted(train_events),
            "fit_events": sorted(fit_events),
            "calibration_events": sorted(calibration_events),
            "validation_events": sorted(validation_events),
            "event_group_split": True,
            "fit_calibration_validation_event_disjoint": True,
            "validation_gate_passed": gate_passed,
            "deployment_status": "blocked_pending_human_review" if gate_passed else "blocked_validation_failed",
        },
    }
    torch.save(checkpoint, model_path)
    pd.DataFrame(per_event_rows).to_csv(out_dir / "raw_joint_36_same_state_v3_per_event_metrics.csv", index=False)
    pd.DataFrame(per_phase_rows).to_csv(out_dir / "raw_joint_36_same_state_v3_per_phase_metrics.csv", index=False)
    pd.DataFrame(per_actuator_rows).to_csv(out_dir / "raw_joint_36_same_state_v3_per_actuator_metrics.csv", index=False)
    v2_metrics = None
    if v2_report_path.exists():
        v2_metrics = json.loads(v2_report_path.read_text(encoding="utf-8")).get("metrics")
    report = {
        "model": str(model_path),
        "dataset": str(dataset_path),
        "warm_start": str(warm_path),
        "dataset_sha256": checkpoint["provenance"]["dataset_sha256"],
        "warm_start_sha256": checkpoint["provenance"]["warm_start_sha256"],
        "config_sha256": checkpoint["provenance"]["config_sha256"],
        "architecture_version": architecture_version,
        "requested_architecture_version": str(args.architecture_version),
        "device": str(device),
        "trainable_parameters": trainable_names,
        "state_and_GAT_encoder_frozen": not bool(args.fine_tune_state_interaction),
        "graph_convolution_frozen": True,
        "state_interaction_fine_tuned": bool(args.fine_tune_state_interaction),
        "action_encoder_fine_tuned": bool(args.fine_tune_action_encoder),
        "action_learning_rate_scale": float(args.action_learning_rate_scale),
        "state_learning_rate_scale": float(args.state_learning_rate_scale),
        "train_events": sorted(train_events),
        "fit_events": sorted(fit_events),
        "calibration_events": sorted(calibration_events),
        "validation_events": sorted(validation_events),
        "train_rows": len(train_idx),
        "fit_rows": len(fit_idx),
        "calibration_rows": len(calibration_idx),
        "validation_rows": len(validation_idx),
        "event_group_split": True,
        "fit_calibration_validation_event_disjoint": True,
        "balanced_sampling": bool(args.balanced_sampling),
        "balanced_epoch_multiplier": float(args.balanced_epoch_multiplier),
        "offline_safety_sample_weight": float(args.offline_safety_sample_weight),
        "sampling_summary": sampling_summary,
        "pairwise_ranking_loss_weight": float(args.pairwise_ranking_loss_weight),
        "peak_direction_loss_multiplier": float(args.peak_direction_loss_multiplier),
        "peak_aggregate_loss_multiplier": float(args.peak_aggregate_loss_multiplier),
        "peak_sequence_loss_multiplier": float(args.peak_sequence_loss_multiplier),
        "peak_direction_sample_weight": float(args.peak_direction_sample_weight),
        "direction_classification_loss_weight": float(args.direction_classification_loss_weight),
        "early_stopping": {
            "enabled": int(args.early_stopping_patience) > 0,
            "patience_selection_rounds": int(args.early_stopping_patience),
            "stopped_early": bool(stopped_early),
        },
        "checkpoint_selection_objective": str(args.selection_objective),
        "direction_positive_weights": direction_positive_weight,
        "best_calibration_epoch": int(best_epoch),
        "calibration": calibration_summary,
        "metrics": metrics,
        "per_phase_metrics": per_phase_rows,
        "per_actuator_metric_rows": len(per_actuator_rows),
        "v2_metrics": v2_metrics,
        "validation_gate_checks": gate_checks,
        "validation_diagnostic_checks": diagnostic_checks,
        "validation_gate_failures": failures,
        "validation_gate_passed": gate_passed,
        "rolling_horizon_smoke_eligibility": smoke_eligibility,
        "acceptance": "awaiting_human_review_before_smoke" if smoke_eligibility["passed"] else "not_eligible_for_closed_loop",
        "losses": history,
    }
    report_path = out_dir / args.report_name
    report_path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(_json_safe({
        "model": str(model_path),
        "report": str(report_path),
        "validation_gate_passed": gate_passed,
        "validation_gate_failures": failures,
        "metrics": metrics,
    }), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
