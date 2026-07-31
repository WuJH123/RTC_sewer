from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from sewerrtc.data.sharded_temporal_action_dataset import (
    event_group_split,
    iter_sharded_batches,
    load_sharded_index,
    prefetch_batches,
)
from sewerrtc.data.peak_label_semantics import (
    RISK_LABEL_CHANNELS,
    peak_label_semantics_valid,
    repair_observational_risk_rate_seq,
)
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.models.raw_joint_action_surrogate import RawJointActionSurrogate
from sewerrtc.models.raw_joint_assets import build_raw_joint_assets
from sewerrtc.models.observational_action_pretraining import (
    action_excitation,
    action_rich_sample_weights,
    actuator_neighbour_state_loss,
    horizon_peak_metrics,
)


def _as_tensors(
    batch: dict[str, np.ndarray],
    device: torch.device,
    *,
    pin_memory: bool,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key == "event_ids":
            continue
        if key == "risk_rate_seq":
            value = repair_observational_risk_rate_seq(value)
        tensor = torch.from_numpy(np.ascontiguousarray(value)).to(dtype=torch.float32)
        if pin_memory and device.type == "cuda":
            tensor = tensor.pin_memory()
        tensors[key] = tensor.to(device=device, non_blocking=bool(pin_memory and device.type == "cuda"))
    return tensors


def _collect_scale_sample(index, train_events: set[str], max_samples: int, seed: int) -> tuple[np.ndarray, float]:
    risk_rows, local_delta_rows = [], []
    local_lookup = [index.node_cols.index(name) for name in index.local_node_cols]
    for batch in iter_sharded_batches(
        index.shard_files,
        allowed_events=train_events,
        batch_size=512,
        max_samples=max_samples,
        seed=seed,
    ):
        risk_rows.append(repair_observational_risk_rate_seq(batch["risk_rate_seq"]))
        current_local = batch["state"][:, local_lookup]
        local_delta_rows.append(batch["local_state_seq"] - current_local[:, None, :])
    risk = np.concatenate(risk_rows, axis=0)
    local_delta = np.concatenate(local_delta_rows, axis=0)
    risk_scale = np.maximum(np.quantile(np.abs(risk), 0.75, axis=(0, 1)), 1.0).astype(np.float32)
    local_scale = float(max(np.quantile(np.abs(local_delta), 0.75), 0.05))
    return risk_scale, local_scale


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.square(target - target.mean()).sum())
    if denominator <= 1.0e-12:
        return float("nan")
    return float(1.0 - np.square(target - prediction).sum() / denominator)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain shared raw-joint dynamics encoders from observational [H,36] trajectories.")
    parser.add_argument("--config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--dataset-index", default="outputs/cache_temporal_action_pretrain_36/temporal_action_pretrain_36.npz")
    parser.add_argument("--out-dir", default="outputs/models_temporal_action_pretrain_36")
    parser.add_argument("--model-name", default="raw_joint_36_observational_dynamics.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--max-train-samples-per-epoch", type=int, default=5000)
    parser.add_argument("--max-validation-samples", type=int, default=4096)
    parser.add_argument("--scale-samples", type=int, default=5000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--warm-start",
        default="",
        help="Optional observational checkpoint used to initialize a new output directory.",
    )
    parser.add_argument("--action-rich-weight", type=float, default=2.0)
    parser.add_argument("--minimum-action-excitation", type=float, default=0.02)
    parser.add_argument("--actuator-neighbour-loss-weight", type=float, default=1.0)
    parser.add_argument("--risk-change-loss-weight", type=float, default=0.5)
    parser.add_argument("--min-relative-identity-sensitivity", type=float, default=1.0e-4)
    parser.add_argument("--min-relative-temporal-sensitivity", type=float, default=1.0e-3)
    parser.add_argument("--prefetch-depth", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    try:
        torch.set_num_interop_threads(max(1, min(4, int(args.cpu_threads))))
    except RuntimeError:
        pass
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    index_path = root / args.dataset_index if not Path(args.dataset_index).is_absolute() else Path(args.dataset_index)
    output_dir = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    pin_memory = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.benchmark = True
    index = load_sharded_index(index_path)
    node_ids = [name.split(":", 1)[1] if name.startswith("h:") else name for name in index.node_cols]
    action_ids = list(index.action_ids)
    local_indices = torch.as_tensor(
        [index.node_cols.index(name) for name in index.local_node_cols],
        dtype=torch.long,
        device=device,
    )
    train_events, validation_events = event_group_split(
        index.shard_files,
        validation_fraction=float(args.validation_fraction),
        seed=int(args.seed),
    )
    risk_scale_np, local_scale = _collect_scale_sample(index, train_events, int(args.scale_samples), int(args.seed))
    node_static, edge_index, action_node_map, actuator_features, priority_indices, storage_indices = build_raw_joint_assets(
        cfg, node_ids, action_ids
    )
    model = RawJointActionSurrogate(
        n_nodes=len(node_ids),
        n_actions=len(action_ids),
        node_static_dim=node_static.shape[1],
        actuator_feature_dim=actuator_features.shape[1],
        horizon_steps=6,
        hidden_dim=int(args.hidden_dim),
        heads=4,
        architecture_version="priority_aware_v2",
    ).to(device)
    warm_start_path = None
    if args.resume and args.warm_start:
        raise ValueError("--resume and --warm-start are mutually exclusive")
    if args.warm_start:
        warm_start_path = root / args.warm_start if not Path(args.warm_start).is_absolute() else Path(args.warm_start)
        warm = torch.load(warm_start_path, map_location="cpu", weights_only=False)
        if list(map(str, warm.get("node_ids", []))) != node_ids:
            raise ValueError("Warm-start checkpoint node order differs from the current dataset")
        if list(map(str, warm.get("action_ids", []))) != action_ids:
            raise ValueError("Warm-start checkpoint action order differs from the current dataset")
        incompatible = model.load_state_dict(warm["model"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                f"Warm-start checkpoint is incompatible: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    trainable_prefixes = (
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
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(trainable_prefixes)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=1.0e-5,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    fixed = {
        "node_static": torch.as_tensor(node_static, device=device),
        "edge_index": torch.as_tensor(edge_index, dtype=torch.long, device=device),
        "action_node_map": torch.as_tensor(action_node_map, device=device),
        "actuator_features": torch.as_tensor(actuator_features, device=device),
        "priority_indices": torch.as_tensor(priority_indices, dtype=torch.long, device=device),
        "storage_indices": torch.as_tensor(storage_indices, dtype=torch.long, device=device),
    }
    action_local_map = torch.as_tensor(
        action_node_map[:, local_indices.detach().cpu().numpy()],
        dtype=torch.float32,
        device=device,
    )
    mask = torch.ones((1, len(action_ids)), dtype=torch.float32, device=device)
    risk_scale = torch.as_tensor(risk_scale_np, device=device)[None, None, :]
    model_path = output_dir / args.model_name
    last_checkpoint_path = output_dir / f"{model_path.stem}.last.pt"
    progress_path = output_dir / "temporal_action_dynamics_pretrain_progress.json"
    history = []
    start_epoch = 1
    if args.resume and last_checkpoint_path.exists():
        resume = torch.load(last_checkpoint_path, map_location="cpu", weights_only=False)
        if list(map(str, resume["node_ids"])) != node_ids or list(map(str, resume["action_ids"])) != action_ids:
            raise ValueError("Resume checkpoint node/action order differs from the current dataset")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if amp_enabled and resume.get("scaler"):
            scaler.load_state_dict(resume["scaler"])
        history = list(resume.get("history", []))
        start_epoch = int(resume["epoch"]) + 1
    for epoch in range(start_epoch, int(args.epochs) + 1):
        model.train()
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        weighted_loss = 0.0
        seen = 0
        train_source = iter_sharded_batches(
            index.shard_files,
            allowed_events=train_events,
            batch_size=int(args.batch_size),
            max_samples=int(args.max_train_samples_per_epoch),
            seed=int(args.seed) + epoch,
        )
        for raw_batch in prefetch_batches(train_source, prefetch_depth=int(args.prefetch_depth)):
            batch = _as_tensors(raw_batch, device, pin_memory=pin_memory)
            action = batch["candidate_action_seq"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(
                    state=batch["state"],
                    candidate_action_seq=action,
                    reference_action_seq=action,
                    rain_seq=batch["rain_seq"],
                    actuator_mask=mask.expand(len(action), -1),
                    **fixed,
                )
                predicted_local = output["node_state_seq"].index_select(2, local_indices)
                excitation = action_excitation(action)
                sample_weights = action_rich_sample_weights(
                    excitation,
                    gain=float(args.action_rich_weight),
                    minimum_excitation=float(args.minimum_action_excitation),
                )
                risk_error = F.smooth_l1_loss(
                    output["reference_risk_rate_seq"] / risk_scale,
                    batch["risk_rate_seq"] / risk_scale,
                    reduction="none",
                ).mean(dim=(1, 2))
                local_error = F.smooth_l1_loss(
                    predicted_local / local_scale,
                    batch["local_state_seq"] / local_scale,
                    reduction="none",
                ).mean(dim=(1, 2))
                risk_loss = (risk_error * sample_weights).sum() / sample_weights.sum()
                local_loss = (local_error * sample_weights).sum() / sample_weights.sum()
                neighbour_loss = actuator_neighbour_state_loss(
                    predicted_local,
                    batch["local_state_seq"],
                    excitation=excitation,
                    action_local_map=action_local_map,
                    scale=local_scale,
                )
                if action.shape[1] > 1:
                    predicted_change = output["reference_risk_rate_seq"][:, 1:] - output["reference_risk_rate_seq"][:, :-1]
                    target_change = batch["risk_rate_seq"][:, 1:] - batch["risk_rate_seq"][:, :-1]
                    risk_change_loss = F.smooth_l1_loss(
                        predicted_change / risk_scale,
                        target_change / risk_scale,
                    )
                else:
                    risk_change_loss = risk_loss * 0.0
                loss = (
                    risk_loss
                    + 0.5 * local_loss
                    + float(args.actuator_neighbour_loss_weight) * neighbour_loss
                    + float(args.risk_change_loss_weight) * risk_change_loss
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            weighted_loss += float(loss.detach()) * len(action)
            seen += len(action)
        epoch_loss = weighted_loss / max(1, seen)
        elapsed_sec = time.perf_counter() - epoch_started
        throughput = float(seen / max(elapsed_sec, 1.0e-9))
        peak_gpu_memory_gb = (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
            if device.type == "cuda"
            else 0.0
        )
        history.append({
            "epoch": epoch,
            "loss": epoch_loss,
            "samples": seen,
            "elapsed_sec": elapsed_sec,
            "samples_per_sec": throughput,
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
        })
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if amp_enabled else None,
                "epoch": epoch,
                "history": history,
                "node_ids": node_ids,
                "action_ids": action_ids,
            },
            last_checkpoint_path,
        )
        progress_path.write_text(
            json.dumps(
                {
                    "status": "training",
                    "completed_epoch": epoch,
                    "target_epochs": int(args.epochs),
                    "last_loss": epoch_loss,
                    "samples_this_epoch": seen,
                    "checkpoint": str(last_checkpoint_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"epoch={epoch:03d} loss={epoch_loss:.6f} samples={seen} "
            f"samples_per_sec={throughput:.1f} peak_gpu_gb={peak_gpu_memory_gb:.2f}",
            flush=True,
        )

    model.eval()
    target_risk, predicted_risk, target_local, predicted_local, validation_event_rows = [], [], [], [], []
    validation_action_rich = []
    sensitivity_batch = None
    with torch.no_grad():
        validation_source = iter_sharded_batches(
            index.shard_files,
            allowed_events=validation_events,
            batch_size=int(args.batch_size),
            max_samples=int(args.max_validation_samples),
            seed=int(args.seed) + 10000,
        )
        for raw_batch in prefetch_batches(validation_source, prefetch_depth=int(args.prefetch_depth)):
            batch = _as_tensors(raw_batch, device, pin_memory=pin_memory)
            action = batch["candidate_action_seq"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(
                    state=batch["state"],
                    candidate_action_seq=action,
                    reference_action_seq=action,
                    rain_seq=batch["rain_seq"],
                    actuator_mask=mask.expand(len(action), -1),
                    **fixed,
                )
            target_risk.append(repair_observational_risk_rate_seq(raw_batch["risk_rate_seq"]))
            predicted_risk.append(output["reference_risk_rate_seq"].cpu().numpy())
            target_local.append(raw_batch["local_state_seq"])
            predicted_local.append(output["node_state_seq"].index_select(2, local_indices).cpu().numpy())
            validation_action_rich.append(
                (
                    action_excitation(action).amax(dim=1)
                    >= float(args.minimum_action_excitation)
                ).cpu().numpy()
            )
            validation_event_rows.extend(map(str, raw_batch["event_ids"]))
            if sensitivity_batch is None:
                sensitivity_batch = raw_batch
    target_risk_np = np.concatenate(target_risk)
    predicted_risk_np = np.concatenate(predicted_risk)
    target_local_np = np.concatenate(target_local)
    predicted_local_np = np.concatenate(predicted_local)
    validation_action_rich_np = np.concatenate(validation_action_rich).astype(bool)
    channel_names = list(RISK_LABEL_CHANNELS)
    metrics = {
        f"{name}_MAE": float(np.abs(predicted_risk_np[:, :, i] - target_risk_np[:, :, i]).mean())
        for i, name in enumerate(channel_names)
    }
    metrics.update({
        f"{name}_R2": _r2(target_risk_np[:, :, i], predicted_risk_np[:, :, i])
        for i, name in enumerate(channel_names)
    })
    horizon_peak = horizon_peak_metrics(target_risk_np, predicted_risk_np)
    metrics["horizon_peak_TFV_rate_MAE"] = float(horizon_peak["MAE"])
    metrics["horizon_peak_TFV_rate_R2"] = float(horizon_peak["R2"])
    metrics["peak_label_semantics_valid"] = peak_label_semantics_valid(target_risk_np)
    metrics["local_state_RMSE"] = float(np.sqrt(np.square(predicted_local_np - target_local_np).mean()))
    metrics["validation_samples"] = int(len(target_risk_np))
    metrics["action_rich_validation_samples"] = int(validation_action_rich_np.sum())
    if bool(validation_action_rich_np.any()):
        for i, name in enumerate(channel_names):
            metrics[f"action_rich_{name}_R2"] = _r2(
                target_risk_np[validation_action_rich_np, :, i],
                predicted_risk_np[validation_action_rich_np, :, i],
            )
        action_rich_peak = horizon_peak_metrics(
            target_risk_np[validation_action_rich_np],
            predicted_risk_np[validation_action_rich_np],
        )
        metrics["action_rich_horizon_peak_TFV_rate_MAE"] = float(action_rich_peak["MAE"])
        metrics["action_rich_horizon_peak_TFV_rate_R2"] = float(action_rich_peak["R2"])
        metrics["action_rich_local_state_RMSE"] = float(
            np.sqrt(
                np.square(
                    predicted_local_np[validation_action_rich_np]
                    - target_local_np[validation_action_rich_np]
                ).mean()
            )
        )

    identity_sensitivity = identity_local_sensitivity = temporal_sensitivity = 0.0
    identity_relative_sensitivity = temporal_relative_sensitivity = 0.0
    if sensitivity_batch is not None:
        batch = _as_tensors(sensitivity_batch, device, pin_memory=pin_memory)
        action = batch["candidate_action_seq"]
        with torch.no_grad():
            baseline_output = model(
                state=batch["state"], candidate_action_seq=action, reference_action_seq=action,
                rain_seq=batch["rain_seq"], actuator_mask=mask.expand(len(action), -1), **fixed,
            )
            baseline = baseline_output["reference_risk_rate_seq"]
            variation = action.std(dim=(0, 1))
            pair = torch.topk(variation, k=min(2, len(action_ids))).indices
            swapped = action.clone()
            if len(pair) == 2:
                swapped[:, :, pair[0]], swapped[:, :, pair[1]] = action[:, :, pair[1]], action[:, :, pair[0]]
            swapped_result = model(
                state=batch["state"], candidate_action_seq=swapped, reference_action_seq=swapped,
                rain_seq=batch["rain_seq"], actuator_mask=mask.expand(len(action), -1), **fixed,
            )
            swapped_output = swapped_result["reference_risk_rate_seq"]
            reversed_action = torch.flip(action, dims=[1])
            reversed_output = model(
                state=batch["state"], candidate_action_seq=reversed_action, reference_action_seq=reversed_action,
                rain_seq=batch["rain_seq"], actuator_mask=mask.expand(len(action), -1), **fixed,
            )["reference_risk_rate_seq"]
            identity_sensitivity = float(torch.abs(swapped_output - baseline).mean().cpu())
            identity_local_sensitivity = float(
                torch.abs(
                    swapped_result["node_state_seq"].index_select(2, local_indices)
                    - baseline_output["node_state_seq"].index_select(2, local_indices)
                ).mean().cpu()
            )
            temporal_sensitivity = float(torch.abs(reversed_output - baseline).mean().cpu())
            risk_denominator = float(torch.clamp(torch.abs(baseline).mean(), min=1.0e-6).cpu())
            identity_relative_sensitivity = identity_sensitivity / risk_denominator
            temporal_relative_sensitivity = temporal_sensitivity / risk_denominator
    metrics["actuator_identity_sensitivity"] = identity_sensitivity
    metrics["actuator_identity_local_state_sensitivity"] = identity_local_sensitivity
    metrics["temporal_order_sensitivity"] = temporal_sensitivity
    metrics["actuator_identity_relative_sensitivity"] = identity_relative_sensitivity
    metrics["temporal_order_relative_sensitivity"] = temporal_relative_sensitivity

    representation_checks = {
        "action_rich_validation_support": int(metrics["action_rich_validation_samples"]) >= 100,
        "identity_sensitivity": identity_relative_sensitivity
        >= float(args.min_relative_identity_sensitivity),
        "temporal_order_sensitivity": temporal_relative_sensitivity
        >= float(args.min_relative_temporal_sensitivity),
        "peak_label_semantics": bool(metrics["peak_label_semantics_valid"]),
        "finite_horizon_peak_metric": bool(np.isfinite(metrics["horizon_peak_TFV_rate_R2"])),
        "finite_action_rich_metrics": all(
            np.isfinite(metrics.get(f"action_rich_{name}_R2", np.nan))
            for name in channel_names[:2]
        ) and np.isfinite(metrics.get("action_rich_horizon_peak_TFV_rate_R2", np.nan)),
    }

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
        "hidden_dim": int(args.hidden_dim),
        "heads": 4,
        "horizon_steps": 6,
        "architecture_version": "priority_aware_v2",
        "label_semantics": "observational_temporal_dynamics_pretraining",
        "risk_label_channels": list(RISK_LABEL_CHANNELS),
        "peak_label_definition": "horizon peak is max(TFV_rate_seq); channel 2 is the running TFV-rate peak",
        "warm_start": str(warm_start_path) if warm_start_path else None,
        "training_scales": {"reference_rate": risk_scale_np, "local_state_delta": local_scale},
        "trainable_prefixes": trainable_prefixes,
        "observational_action_supervision": {
            "action_rich_weight": float(args.action_rich_weight),
            "minimum_action_excitation": float(args.minimum_action_excitation),
            "actuator_neighbour_loss_weight": float(args.actuator_neighbour_loss_weight),
            "risk_change_loss_weight": float(args.risk_change_loss_weight),
        },
    }
    torch.save(checkpoint, model_path)
    report = {
        "model": str(model_path),
        "dataset_index": str(index.path),
        "samples_available": int(index.sample_count),
        "train_events": sorted(train_events),
        "validation_events": sorted(validation_events),
        "event_group_split": True,
        "epochs": int(args.epochs),
        "max_train_samples_per_epoch": int(args.max_train_samples_per_epoch),
        "metrics": metrics,
        "history": history,
        "risk_scale": risk_scale_np.astype(float).tolist(),
        "local_scale": local_scale,
        "device": str(device),
        "performance": {
            "amp": amp_enabled,
            "tf32": bool(args.tf32 and device.type == "cuda"),
            "batch_size": int(args.batch_size),
            "prefetch_depth": int(args.prefetch_depth),
            "cpu_threads": int(args.cpu_threads),
            "last_epoch_samples_per_sec": float(history[-1]["samples_per_sec"]) if history else 0.0,
            "peak_gpu_memory_gb": float(history[-1]["peak_gpu_memory_gb"]) if history else 0.0,
        },
        "label_semantics": "observational dynamics only; not causal action-effect evidence",
        "risk_label_channels": list(RISK_LABEL_CHANNELS),
        "peak_label_definition": "horizon peak is max(TFV_rate_seq); channel 2 is the running TFV-rate peak",
        "warm_start": str(warm_start_path) if warm_start_path else None,
        "observational_action_supervision": checkpoint["observational_action_supervision"],
        "actuator_feature_dim": int(actuator_features.shape[1]),
        "action_representation_gate": {
            "checks": representation_checks,
            "passed": bool(all(representation_checks.values())),
            "scope": "observational action representation only; not a causal MPC safety gate",
            "min_relative_identity_sensitivity": float(args.min_relative_identity_sensitivity),
            "min_relative_temporal_sensitivity": float(args.min_relative_temporal_sensitivity),
        },
    }
    report_path = output_dir / "temporal_action_dynamics_pretrain_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    progress_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_epoch": int(args.epochs),
                "target_epochs": int(args.epochs),
                "model": str(model_path),
                "report": str(report_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
