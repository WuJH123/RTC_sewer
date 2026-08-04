"""Bounded-memory development/formal trainer for V4.2 Step-1 Temporal GAT.

This runner supersedes the eager formal trainer for large manifests.  It keeps
sensor layout and rainfall splits independent from the model random seed, uses
streaming metrics (no epoch-wide prediction arrays), selects checkpoints by
validation RMSE while the NLL weight changes, and leaves auxiliary pretraining
off unless an explicit compatibility allow-list is supplied.

The script deliberately does NOT write a passing paper ``evidence.json``.
Formal evidence remains gated by target-event diversity, multi-model-seed
robustness, uncertainty calibration and OOD calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.models.temporal_sparse_gat_v42 import TemporalSparseGATReconstructorV42
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_step1_streaming import (
    Step1StreamingDataset,
    split_target_groups,
    target_rainfall_groups,
)
from sewerrtc.v4.v42_step1_training import Step1LossWeights, step1_reconstruction_loss

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class _ScalarStats:
    count: int = 0
    sum_y: float = 0.0
    sum_y2: float = 0.0
    sse: float = 0.0
    sae: float = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        mask = mask.bool()
        n = int(mask.sum().item())
        if n == 0:
            return
        p = pred[mask]
        y = target[mask]
        err = p - y
        self.count += n
        self.sum_y += float(y.sum().item())
        self.sum_y2 += float((y * y).sum().item())
        self.sse += float((err * err).sum().item())
        self.sae += float(err.abs().sum().item())

    def metrics(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "nse": None, "rmse": None, "mae": None}
        denom = self.sum_y2 - (self.sum_y * self.sum_y) / float(self.count)
        nse = None if denom <= 1.0e-12 else 1.0 - self.sse / denom
        return {
            "count": int(self.count),
            "nse": None if nse is None else float(nse),
            "rmse": float(math.sqrt(self.sse / float(self.count))),
            "mae": float(self.sae / float(self.count)),
        }


class StreamingMetricAccumulator:
    """Exact streaming aggregate metrics without retaining prediction arrays."""

    def __init__(self) -> None:
        self.overall = _ScalarStats()
        self.priority = _ScalarStats()
        self.priority_wet = _ScalarStats()
        self.std_sum = 0.0
        self.std_count = 0

    def update(
        self,
        *,
        pred: torch.Tensor,
        std: torch.Tensor,
        target: torch.Tensor,
        observed: torch.Tensor,
        priority_mask: torch.Tensor,
        wet_threshold_m: float,
    ) -> None:
        with torch.no_grad():
            unobs = ~observed.bool()
            pri = priority_mask.bool()[None, :].expand_as(unobs)
            pri_unobs = unobs & pri
            wet = pri_unobs & (target >= float(wet_threshold_m))
            self.overall.update(pred, target, unobs)
            self.priority.update(pred, target, pri_unobs)
            self.priority_wet.update(pred, target, wet)
            n_std = int(unobs.sum().item())
            if n_std:
                self.std_sum += float(std[unobs].sum().item())
                self.std_count += n_std

    def result(self) -> dict[str, object]:
        return {
            "overall_unobserved": self.overall.metrics(),
            "priority_unobserved": self.priority.metrics(),
            "priority_wet_unobserved": self.priority_wet.metrics(),
            "mean_predicted_std_unobserved": (
                None if self.std_count == 0 else float(self.std_sum / self.std_count)
            ),
        }


def _rss_mb() -> float | None:
    if psutil is None:
        return None
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def _hash_strings(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def _state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))
    return h.hexdigest()


def _read_group_allowlist(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return tuple()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return tuple()
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        values = payload.get("groups", payload) if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            raise ValueError("aux allow-list JSON must be a list or {'groups': [...]} object")
        return tuple(sorted({str(x) for x in values}))
    return tuple(sorted({line.strip() for line in text.splitlines() if line.strip()}))


def _graph_tensors(graph, device: torch.device):
    return (
        torch.from_numpy(graph.node_static).to(device),
        torch.from_numpy(graph.link_static).to(device),
        torch.from_numpy(graph.edge_index).to(device),
        torch.from_numpy(graph.action_node_map).to(device),
    )


def _priority_mask(graph, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(graph.n_nodes, dtype=torch.bool, device=device)
    mask[get_pfv_core_node_indices(graph.node_ids)] = True
    return mask


def _nll_weight(epoch: int, target: float, warmup: int, ramp: int) -> float:
    if target <= 0.0:
        return 0.0
    if epoch <= warmup:
        return 0.0
    if ramp <= 0:
        return float(target)
    step = min(ramp, epoch - warmup)
    return float(target) * float(step) / float(ramp)


def _make_loader(
    dataset: Step1StreamingDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    # shuffle must remain False for IterableDataset; file/window order is
    # deterministically shuffled inside Step1StreamingDataset when requested.
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": False,
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(**kwargs)


def _write_runtime_status(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _run_epoch(
    *,
    model: torch.nn.Module,
    dataset: Step1StreamingDataset,
    graph,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    optimizer: torch.optim.Optimizer | None,
    weights: Step1LossWeights,
    wet_threshold_m: float,
    heartbeat_batches: int,
    epoch: int,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
    runtime_status_file: Path | None = None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    dataset.set_epoch(epoch)
    loader = _make_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda" if pin_memory is None else bool(pin_memory),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    node_static, link_static, edge_index, action_map = _graph_tensors(graph, device)
    pri_mask = _priority_mask(graph, device)
    metrics = StreamingMetricAccumulator()
    loss_sums = {
        "total": 0.0,
        "global_depth": 0.0,
        "priority_depth": 0.0,
        "wet_priority_depth": 0.0,
        "heteroscedastic_nll": 0.0,
    }
    windows_seen = 0
    batches = 0
    detail_files: set[str] = set()
    rainfall_groups: set[str] = set()
    physical_runs: set[str] = set()
    peak_rss = _rss_mb()
    started = time.time()
    last_status = started

    amp_enabled = os.environ.get("RTC_V42_STEP1_AMP") == "1" and device.type == "cuda"
    scaler = None
    if amp_enabled and training:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            sdh = batch["sparse_depth_history"].to(device, non_blocking=True)
            smh = batch["sensor_mask_history"].to(device, non_blocking=True)
            rain = batch["rainfall_history"].to(device, non_blocking=True)
            actions = batch["historical_actions"].to(device, non_blocking=True)
            target = batch["target_depth"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                out = model(
                    sparse_depth_history=sdh,
                    sensor_mask_history=smh,
                    rainfall_history=rain,
                    historical_actions=actions,
                    node_static=node_static,
                    link_static=link_static,
                    edge_index=edge_index,
                    action_node_map=action_map,
                )
                losses = step1_reconstruction_loss(
                    out,
                    target,
                    smh[:, -1, :],
                    pri_mask,
                    weights=weights,
                    wet_threshold_m=wet_threshold_m,
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                else:
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()

            actual_batch = int(target.shape[0])
            windows_seen += actual_batch
            batches += 1
            for key in loss_sums:
                loss_sums[key] += float(losses[key].detach().item()) * actual_batch
            metrics.update(
                pred=out.depth_mean.detach(),
                std=out.depth_std.detach(),
                target=target.detach(),
                observed=smh[:, -1, :].detach() >= 0.5,
                priority_mask=pri_mask,
                wet_threshold_m=wet_threshold_m,
            )
            detail_files.update(map(str, batch["detail_path"]))
            rainfall_groups.update(map(str, batch["split_group_key"]))
            physical_runs.update(map(str, batch["physical_identity_sha256"]))
            rss = _rss_mb()
            if rss is not None:
                peak_rss = rss if peak_rss is None else max(peak_rss, rss)
            if heartbeat_batches > 0 and batches % heartbeat_batches == 0:
                live = metrics.result()["overall_unobserved"]
                logger.info(
                    "epoch %d batch %d | windows %d/%d | NSE %s | RMSE %s | RSS %sMB | files %d",
                    epoch,
                    batches,
                    windows_seen,
                    len(dataset),
                    "NA" if live["nse"] is None else f"{live['nse']:.4f}",
                    "NA" if live["rmse"] is None else f"{live['rmse']:.4f}",
                    "NA" if rss is None else f"{rss:.0f}",
                    len(detail_files),
                )
            now = time.time()
            if runtime_status_file is not None and (
                now - last_status >= 5.0 or batches == 1
            ):
                _write_runtime_status(
                    runtime_status_file,
                    {
                        "stage": "step1",
                        "epoch": int(epoch),
                        "batch": int(batches),
                        "windows_seen": int(windows_seen),
                        "expected_windows": int(len(dataset)),
                        "windows_per_sec": float(windows_seen / max(now - started, 1e-9)),
                        "unique_detail_files": int(len(detail_files)),
                        "unique_rainfall_groups": int(len(rainfall_groups)),
                        "timestamp": time.time(),
                    },
                )
                last_status = now

    expected = len(dataset)
    # With num_workers=0 or correctly partitioned workers, every selected window
    # must be yielded exactly once.  Any mismatch is an evidence-integrity error.
    if windows_seen != expected:
        raise RuntimeError(
            f"Step1 streaming window count mismatch: expected={expected} actual={windows_seen}"
        )
    result = metrics.result()
    result.update(
        {
            "expected_windows": int(expected),
            "actual_windows_seen": int(windows_seen),
            "batches": int(batches),
            "unique_detail_files": int(len(detail_files)),
            "unique_physical_runs": int(len(physical_runs)),
            "unique_rainfall_groups": int(len(rainfall_groups)),
            "peak_rss_mb": peak_rss,
            "elapsed_s": float(time.time() - started),
            "loss_components": {
                key: float(value / max(1, windows_seen)) for key, value in loss_sums.items()
            },
        }
    )
    if runtime_status_file is not None:
        _write_runtime_status(
            runtime_status_file,
            {
                "stage": "step1",
                "epoch": int(epoch),
                "batch": int(batches),
                "windows_seen": int(windows_seen),
                "expected_windows": int(expected),
                "windows_per_sec": float(windows_seen / max(result["elapsed_s"], 1e-9)),
                "unique_detail_files": int(len(detail_files)),
                "unique_rainfall_groups": int(len(rainfall_groups)),
                "timestamp": time.time(),
                "completed": True,
            },
        )
    return result


def _save_checkpoint(path: Path, *, model, optimizer, epoch: int, meta: dict[str, object]) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "meta": meta,
        },
        path,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--model-seed", type=int, default=42)
    ap.add_argument("--sensor-layout-seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--aux-sampling-seed", type=int, default=42)
    ap.add_argument("--sensor-ratio", type=float, default=0.10)
    ap.add_argument("--validation-group", type=str, default=None)
    ap.add_argument("--calibration-group", type=str, default=None)
    ap.add_argument("--reserve-calibration", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--gat-layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--priority-weight", type=float, default=0.0)
    ap.add_argument("--wet-priority-weight", type=float, default=0.0)
    ap.add_argument("--nll-weight", type=float, default=0.0)
    ap.add_argument("--nll-warmup-epochs", type=int, default=5)
    ap.add_argument("--nll-ramp-epochs", type=int, default=5)
    ap.add_argument("--wet-threshold-m", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--heartbeat-batches", type=int, default=10)
    ap.add_argument("--aux-pretrain", action="store_true")
    ap.add_argument("--aux-allowlist", type=Path, default=None)
    ap.add_argument("--aux-epochs", type=int, default=3)
    ap.add_argument("--aux-max-windows-per-group", type=int, default=16)
    ap.add_argument("--aux-max-windows-per-run", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = args.manifest or (
        args.project_root
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat/dataset/step1_window_manifest.parquet"
    )
    outdir = args.output_dir or (
        args.project_root
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat"
        / f"streaming_model_seed_{args.model_seed}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    target_groups = target_rainfall_groups(manifest)
    split = split_target_groups(
        target_groups,
        split_seed=args.split_seed,
        validation_group=args.validation_group,
        calibration_group=args.calibration_group,
        reserve_calibration=args.reserve_calibration,
    )
    logger.info("target groups=%d train=%s val=%s cal=%s", len(target_groups), split["train"], split["validation"], split["calibration"])

    train_ds = Step1StreamingDataset(
        project_root=args.project_root,
        manifest_path=manifest,
        sensor_ratio=args.sensor_ratio,
        sensor_layout_seed=args.sensor_layout_seed,
        domain_roles=("target_formal",),
        allowed_groups=split["train"],
        shuffle_files=True,
        iteration_seed=args.model_seed,
    )
    val_ds = Step1StreamingDataset(
        project_root=args.project_root,
        manifest_path=manifest,
        sensor_ratio=args.sensor_ratio,
        sensor_layout_seed=args.sensor_layout_seed,
        domain_roles=("target_formal",),
        allowed_groups=split["validation"],
        shuffle_files=False,
        iteration_seed=args.model_seed,
    )
    if train_ds.sensor_layout_sha256 != val_ds.sensor_layout_sha256:
        raise RuntimeError("sensor layout changed between Step1 train and validation")
    graph = train_ds.graph

    model = TemporalSparseGATReconstructorV42(
        n_nodes=graph.n_nodes,
        n_facilities=graph.n_facilities,
        node_static_dim=graph.node_static.shape[1],
        link_static_dim=graph.link_static.shape[1],
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        gat_layers=args.gat_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)

    aux_report = None
    if args.aux_pretrain:
        allow = _read_group_allowlist(args.aux_allowlist)
        if not allow:
            raise ValueError(
                "--aux-pretrain requires --aux-allowlist produced by an explicit compatibility audit"
            )
        aux_ds = Step1StreamingDataset(
            project_root=args.project_root,
            manifest_path=manifest,
            sensor_ratio=args.sensor_ratio,
            sensor_layout_seed=args.sensor_layout_seed,
            domain_roles=("auxiliary_pretrain",),
            allowed_groups=allow,
            max_windows_per_group=args.aux_max_windows_per_group,
            max_windows_per_physical_run=args.aux_max_windows_per_run,
            sampling_seed=args.aux_sampling_seed,
            shuffle_files=True,
            iteration_seed=args.model_seed,
        )
        aux_history = []
        for epoch in range(1, args.aux_epochs + 1):
            weights = Step1LossWeights(
                priority_depth=0.0,
                wet_priority_depth=0.0,
                heteroscedastic_nll=0.0,
            )
            metrics = _run_epoch(
                model=model,
                dataset=aux_ds,
                graph=graph,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                optimizer=optimizer,
                weights=weights,
                wet_threshold_m=args.wet_threshold_m,
                heartbeat_batches=args.heartbeat_batches,
                epoch=epoch,
            )
            aux_history.append(metrics)
            logger.info("aux epoch %d NSE=%s", epoch, metrics["overall_unobserved"]["nse"])
        aux_report = {
            "allowlist_groups": list(allow),
            "selection": asdict(aux_ds.summary),
            "history": aux_history,
        }

    history: list[dict[str, object]] = []
    best_rmse = float("inf")
    best_epoch = -1
    stale = 0
    best_path = outdir / "best_model.pt"
    last_path = outdir / "last_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        nll_w = _nll_weight(epoch, args.nll_weight, args.nll_warmup_epochs, args.nll_ramp_epochs)
        weights = Step1LossWeights(
            global_depth=1.0,
            priority_depth=float(args.priority_weight),
            wet_priority_depth=float(args.wet_priority_weight),
            heteroscedastic_nll=float(nll_w),
        )
        train_metrics = _run_epoch(
            model=model,
            dataset=train_ds,
            graph=graph,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            optimizer=optimizer,
            weights=weights,
            wet_threshold_m=args.wet_threshold_m,
            heartbeat_batches=args.heartbeat_batches,
            epoch=epoch,
        )
        val_metrics = _run_epoch(
            model=model,
            dataset=val_ds,
            graph=graph,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            optimizer=None,
            weights=weights,
            wet_threshold_m=args.wet_threshold_m,
            heartbeat_batches=args.heartbeat_batches,
            epoch=epoch,
        )
        history.append(
            {
                "epoch": epoch,
                "nll_weight": float(nll_w),
                "train": train_metrics,
                "validation": val_metrics,
            }
        )
        val_rmse = val_metrics["overall_unobserved"]["rmse"]
        val_nse = val_metrics["overall_unobserved"]["nse"]
        pri_nse = val_metrics["priority_unobserved"]["nse"]
        logger.info(
            "epoch %d DONE | train_NSE=%s | val_NSE=%s | val_RMSE=%s | pri_NSE=%s | windows=%d/%d",
            epoch,
            "NA" if train_metrics["overall_unobserved"]["nse"] is None else f"{train_metrics['overall_unobserved']['nse']:.4f}",
            "NA" if val_nse is None else f"{val_nse:.4f}",
            "NA" if val_rmse is None else f"{val_rmse:.4f}",
            "NA" if pri_nse is None else f"{pri_nse:.4f}",
            train_metrics["actual_windows_seen"],
            train_metrics["expected_windows"],
        )
        meta = {
            "model_seed": args.model_seed,
            "sensor_layout_seed": args.sensor_layout_seed,
            "split_seed": args.split_seed,
            "aux_sampling_seed": args.aux_sampling_seed,
            "sensor_layout_sha256": train_ds.sensor_layout_sha256,
            "split": split,
            "epoch": epoch,
        }
        _save_checkpoint(last_path, model=model, optimizer=optimizer, epoch=epoch, meta=meta)
        if val_rmse is not None and float(val_rmse) < best_rmse:
            best_rmse = float(val_rmse)
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience:
                logger.info("early stop at epoch %d; best epoch=%d", epoch, best_epoch)
                break

    if best_epoch < 0:
        raise RuntimeError("Step1 streaming training produced no valid validation RMSE")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    final_weights = Step1LossWeights(
        global_depth=1.0,
        priority_depth=float(args.priority_weight),
        wet_priority_depth=float(args.wet_priority_weight),
        heteroscedastic_nll=float(args.nll_weight),
    )
    final_train = _run_epoch(
        model=model,
        dataset=train_ds,
        graph=graph,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        optimizer=None,
        weights=final_weights,
        wet_threshold_m=args.wet_threshold_m,
        heartbeat_batches=args.heartbeat_batches,
        epoch=best_epoch,
    )
    final_val = _run_epoch(
        model=model,
        dataset=val_ds,
        graph=graph,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        optimizer=None,
        weights=final_weights,
        wet_threshold_m=args.wet_threshold_m,
        heartbeat_batches=args.heartbeat_batches,
        epoch=best_epoch,
    )

    calibration_summary = None
    if split["calibration"]:
        cal_ds = Step1StreamingDataset(
            project_root=args.project_root,
            manifest_path=manifest,
            sensor_ratio=args.sensor_ratio,
            sensor_layout_seed=args.sensor_layout_seed,
            domain_roles=("target_formal",),
            allowed_groups=split["calibration"],
            shuffle_files=False,
        )
        calibration_summary = {
            "selection": asdict(cal_ds.summary),
            "note": "reserved only; no post-hoc uncertainty calibration performed by this trainer",
        }

    report = {
        "contract_id": "PROJECT6_V42_PAPER_WORKFLOW_V1",
        "stage": "step1_sparse_state_streaming_training",
        "status": "development_or_training_only_not_formal_evidence",
        "formal_reconstructor": "TemporalSparseGATReconstructorV42",
        "bounded_memory_streaming": True,
        "csv_projection_authority": "column_name_then_explicit_canonical_reorder",
        "fixed_sensor_layout": True,
        "sensor_ratio": float(args.sensor_ratio),
        "sensor_count": int(len(train_ds.sensor_indices)),
        "sensor_layout_sha256": train_ds.sensor_layout_sha256,
        "seeds": {
            "model_seed": int(args.model_seed),
            "sensor_layout_seed": int(args.sensor_layout_seed),
            "split_seed": int(args.split_seed),
            "aux_sampling_seed": int(args.aux_sampling_seed),
        },
        "split": split,
        "split_manifest_sha256": _hash_strings(
            ["train=" + ",".join(split["train"]), "validation=" + ",".join(split["validation"]), "calibration=" + ",".join(split["calibration"])]
        ),
        "train_selection": asdict(train_ds.summary),
        "validation_selection": asdict(val_ds.summary),
        "calibration_selection": calibration_summary,
        "auxiliary_pretraining": aux_report,
        "best_epoch": int(best_epoch),
        "checkpoint_selection": "minimum_validation_overall_unobserved_RMSE",
        "model_sha256": _state_hash(model),
        "final_train": final_train,
        "final_validation": final_val,
        "nll_schedule": {
            "target_weight": float(args.nll_weight),
            "warmup_epochs": int(args.nll_warmup_epochs),
            "ramp_epochs": int(args.nll_ramp_epochs),
        },
        "uncertainty_calibrated": False,
        "ood_calibrated": False,
        "formal_evidence_ready": False,
    }
    (outdir / "training_history.json").write_text(
        json.dumps(_json_safe(history), indent=2, allow_nan=False), encoding="utf-8"
    )
    (outdir / "formal_training_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    logger.info("wrote %s", outdir / "formal_training_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
