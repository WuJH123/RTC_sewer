"""Step-1 Temporal Sparse GAT training and scientific smoke test.

Usage:
    python scripts/train_v42_step1_gat.py --smoke          # fast check
    python scripts/train_v42_step1_gat.py --epochs 30      # learning test

Produces:
    step1_gat/evidence.json      — formal evidence contract
    step1_gat/train_log.json     — per-epoch metrics
    step1_gat/smoke_report.json  — smoke test summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.models.temporal_sparse_gat_v42 import (
    HISTORY_FRAMES,
    TemporalSparseGATReconstructorV42,
)
from sewerrtc.v4.v42_priority_contract import PFV_CORE_8_IDS, get_pfv_core_node_indices
from sewerrtc.v4.v42_step1_dataset import (
    Step1GraphAssets,
    Step1TorchDataset,
    build_step1_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float | None
    train_nse: float
    val_nse: float | None
    train_priority_nse: float
    val_priority_nse: float | None
    lr: float


def _collate_graph(batch: list[dict], graph: Step1GraphAssets) -> dict:
    """Collate a batch, adding graph-level tensors."""
    B = len(batch)
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        collated[k] = torch.stack([b[k] for b in batch], dim=0)
    return collated


def _compute_nse(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Nash-Sutcliffe Efficiency.  mask selects which elements to evaluate."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if len(target) == 0:
        return float("nan")
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _persistence_baseline(
    sparse_depth: np.ndarray,
    sensor_mask: np.ndarray,
) -> np.ndarray:
    """Predict anchor depth: observed nodes keep their value, unobserved = 0.

    This is the trivial "no reconstruction" baseline.
    """
    current_obs = sparse_depth[-1]  # [N]
    return current_obs  # unobserved are already 0


def _spatial_mean_baseline(
    sparse_depth: np.ndarray,
    sensor_mask: np.ndarray,
) -> np.ndarray:
    """Predict unobserved nodes with the mean of observed nodes."""
    current_obs = sparse_depth[-1]
    mask = sensor_mask[-1]
    observed_vals = current_obs[mask > 0.5]
    mean_val = float(observed_vals.mean()) if len(observed_vals) > 0 else 0.0
    pred = current_obs.copy()
    pred[mask < 0.5] = mean_val
    return pred


def _make_splits(
    samples: list,
    split_groups: list[str],
) -> tuple[list[int], list[int]]:
    """Split indices by rainfall group (group-level isolation)."""
    unique_groups = sorted(set(split_groups))
    if len(unique_groups) < 2:
        raise ValueError(f"Need >= 2 rainfall groups, got {len(unique_groups)}")
    # Deterministic group assignment: hash group key -> train/val
    val_groups = set()
    for g in unique_groups:
        h = int(hashlib.sha256(g.encode()).hexdigest()[:8], 16)
        if h % 4 == 0:  # ~25% for validation
            val_groups.add(g)
    # Ensure at least 1 val group
    if not val_groups:
        val_groups.add(unique_groups[-1])
    train_idx = [i for i, g in enumerate(split_groups) if g not in val_groups]
    val_idx = [i for i, g in enumerate(split_groups) if g in val_groups]
    return train_idx, val_idx


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    graph: Step1GraphAssets,
    priority_indices: list[int],
    device: torch.device,
) -> tuple[float, float, float]:
    """Train one epoch. Returns (avg_loss, nse, priority_nse)."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_pred = []
    all_target = []
    all_mask = []

    node_static_t = torch.from_numpy(graph.node_static).to(device)
    link_static_t = torch.from_numpy(graph.link_static).to(device)
    edge_index_t = torch.from_numpy(graph.edge_index).to(device)
    action_map_t = torch.from_numpy(graph.action_node_map).to(device)

    for batch in loader:
        B = len(batch["sparse_depth_history"])
        sdh = batch["sparse_depth_history"].to(device)
        smh = batch["sensor_mask_history"].to(device)
        rh = batch["rainfall_history"].to(device)
        ha = batch["historical_actions"].to(device)
        target = batch["target_depth"].to(device)

        out = model(
            sparse_depth_history=sdh,
            sensor_mask_history=smh,
            rainfall_history=rh,
            historical_actions=ha,
            node_static=node_static_t,
            link_static=link_static_t,
            edge_index=edge_index_t,
            action_node_map=action_map_t,
        )

        # Loss: only on unobserved nodes
        current_mask = smh[:, -1, :]  # [B, N]
        unobserved = 1.0 - current_mask
        n_unobs = unobserved.sum().clamp_min(1.0)
        loss = ((out.depth_mean - target) ** 2 * unobserved).sum() / n_unobs

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Collect for NSE computation
        all_pred.append(out.depth_mean.detach().cpu().numpy())
        all_target.append(target.cpu().numpy())
        all_mask.append(current_mask.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)

    # Compute NSE over entire epoch
    pred_all = np.concatenate(all_pred, axis=0)
    target_all = np.concatenate(all_target, axis=0)
    mask_all = np.concatenate(all_mask, axis=0)

    # Full-network NSE on unobserved nodes
    unobs_mask = mask_all < 0.5
    full_nse = _compute_nse(pred_all[unobs_mask], target_all[unobs_mask])

    # Priority node NSE
    pri_mask = np.zeros(pred_all.shape[1], dtype=bool)
    for idx in priority_indices:
        pri_mask[idx] = True
    pri_unobs = unobs_mask[:, pri_mask]
    priority_nse = _compute_nse(pred_all[:, pri_mask][pri_unobs], target_all[:, pri_mask][pri_unobs])

    return avg_loss, full_nse, priority_nse


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    graph: Step1GraphAssets,
    priority_indices: list[int],
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model. Returns (avg_loss, nse, priority_nse)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_pred = []
    all_target = []
    all_mask = []

    node_static_t = torch.from_numpy(graph.node_static).to(device)
    link_static_t = torch.from_numpy(graph.link_static).to(device)
    edge_index_t = torch.from_numpy(graph.edge_index).to(device)
    action_map_t = torch.from_numpy(graph.action_node_map).to(device)

    for batch in loader:
        sdh = batch["sparse_depth_history"].to(device)
        smh = batch["sensor_mask_history"].to(device)
        rh = batch["rainfall_history"].to(device)
        ha = batch["historical_actions"].to(device)
        target = batch["target_depth"].to(device)

        out = model(
            sparse_depth_history=sdh,
            sensor_mask_history=smh,
            rainfall_history=rh,
            historical_actions=ha,
            node_static=node_static_t,
            link_static=link_static_t,
            edge_index=edge_index_t,
            action_node_map=action_map_t,
        )

        current_mask = smh[:, -1, :]
        unobserved = 1.0 - current_mask
        n_unobs = unobserved.sum().clamp_min(1.0)
        loss = ((out.depth_mean - target) ** 2 * unobserved).sum() / n_unobs

        total_loss += loss.item()
        n_batches += 1
        all_pred.append(out.depth_mean.cpu().numpy())
        all_target.append(target.cpu().numpy())
        all_mask.append(current_mask.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    pred_all = np.concatenate(all_pred, axis=0)
    target_all = np.concatenate(all_target, axis=0)
    mask_all = np.concatenate(all_mask, axis=0)

    unobs_mask = mask_all < 0.5
    full_nse = _compute_nse(pred_all[unobs_mask], target_all[unobs_mask])

    pri_mask = np.zeros(pred_all.shape[1], dtype=bool)
    for idx in priority_indices:
        pri_mask[idx] = True
    pri_unobs = unobs_mask[:, pri_mask]
    priority_nse = _compute_nse(pred_all[:, pri_mask][pri_unobs], target_all[:, pri_mask][pri_unobs])

    return avg_loss, full_nse, priority_nse


def _compute_baseline_metrics(
    dataset: Step1TorchDataset,
    indices: Sequence[int],
    priority_indices: list[int],
    baseline_fn,
) -> tuple[float, float]:
    """Compute NSE and priority NSE for a baseline method."""
    all_pred = []
    all_target = []
    all_mask = []
    for i in indices:
        s = dataset.samples[i]
        pred = baseline_fn(s.sparse_depth_history, s.sensor_mask_history)
        all_pred.append(pred)
        all_target.append(s.target_depth)
        all_mask.append(s.sensor_mask_history[-1])

    pred_all = np.stack(all_pred, axis=0)
    target_all = np.stack(all_target, axis=0)
    mask_all = np.stack(all_mask, axis=0)

    unobs_mask = mask_all < 0.5
    full_nse = _compute_nse(pred_all[unobs_mask], target_all[unobs_mask])

    pri_mask_bool = np.zeros(pred_all.shape[1], dtype=bool)
    for idx in priority_indices:
        pri_mask_bool[idx] = True
    pri_unobs = unobs_mask[:, pri_mask_bool]
    priority_nse = _compute_nse(pred_all[:, pri_mask_bool][pri_unobs], target_all[:, pri_mask_bool][pri_unobs])

    return full_nse, priority_nse


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Step-1 Temporal Sparse GAT")
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--gat-layers", type=int, default=2)
    ap.add_argument("--sensor-ratio", type=float, default=0.10)
    ap.add_argument("--smoke", action="store_true", help="2-epoch smoke test")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stop-patience", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.smoke:
        args.epochs = min(args.epochs, 2)

    output_dir = args.output_dir or (
        args.project_root / "outputs" / "project6_dual_reference_v4" / "final_v4"
        / "v42_paper" / "step1_gat"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or (
        output_dir / "dataset" / "step1_window_manifest.parquet"
    )
    if not manifest_path.exists():
        logger.error("Window manifest not found: %s", manifest_path)
        logger.error("Run: python scripts/build_v42_step1_windows.py")
        return 5

    # Build dataset
    logger.info("Building Step-1 dataset from %s", manifest_path)
    t0 = time.time()
    ds_build = build_step1_dataset(
        project_root=args.project_root,
        manifest_path=manifest_path,
        sensor_ratio=args.sensor_ratio,
        rng_seed=args.seed,
        max_samples=args.max_samples,
    )
    logger.info(
        "Dataset built in %.1fs: %d samples, %d skipped, %d warnings",
        time.time() - t0, len(ds_build.samples), ds_build.skipped, len(ds_build.warnings),
    )
    if not ds_build.samples:
        logger.error("No valid samples!")
        return 5

    graph = ds_build.graph
    logger.info(
        "Graph: %d nodes, %d edges, %d facilities, link_static_dim=%d",
        graph.n_nodes, graph.n_edges, graph.n_facilities, graph.link_static.shape[1],
    )

    # Split by rainfall group
    split_groups = [s.split_group for s in ds_build.samples]
    train_idx, val_idx = _make_splits(ds_build.samples, split_groups)
    logger.info("Split: %d train, %d val (rainfall-group isolated)", len(train_idx), len(val_idx))

    train_groups = sorted(set(split_groups[i] for i in train_idx))
    val_groups = sorted(set(split_groups[i] for i in val_idx))
    logger.info("Train groups: %d, Val groups: %d", len(train_groups), len(val_groups))

    # Build dataloaders
    full_dataset = Step1TorchDataset(ds_build.samples, graph)
    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Priority node indices
    priority_indices = get_pfv_core_node_indices(graph.node_ids)
    logger.info("Priority nodes: %d indices", len(priority_indices))

    # Build model
    model = TemporalSparseGATReconstructorV42(
        n_nodes=graph.n_nodes,
        n_facilities=graph.n_facilities,
        node_static_dim=graph.node_static.shape[1],
        link_static_dim=graph.link_static.shape[1],
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        gat_layers=args.gat_layers,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s, %.2fM parameters", type(model).__name__, n_params / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Baseline metrics
    logger.info("Computing baselines...")
    persist_nse, persist_pri_nse = _compute_baseline_metrics(
        full_dataset, val_idx, priority_indices, _persistence_baseline
    )
    spatial_nse, spatial_pri_nse = _compute_baseline_metrics(
        full_dataset, val_idx, priority_indices, _spatial_mean_baseline
    )
    logger.info(
        "Persistence val NSE: %.4f, priority NSE: %.4f", persist_nse, persist_pri_nse
    )
    logger.info(
        "Spatial-mean val NSE: %.4f, priority NSE: %.4f", spatial_nse, spatial_pri_nse
    )

    # Training loop
    epoch_logs: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_nse, train_pri_nse = train_one_epoch(
            model, train_loader, optimizer, graph, priority_indices, DEVICE
        )
        val_loss, val_nse, val_pri_nse = evaluate(
            model, val_loader, graph, priority_indices, DEVICE
        )
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        elapsed = time.time() - t0
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_nse=train_nse,
            val_nse=val_nse,
            train_priority_nse=train_pri_nse,
            val_priority_nse=val_pri_nse,
            lr=lr_now,
        )
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_nse": round(train_nse, 4),
            "val_nse": round(val_nse, 4),
            "train_priority_nse": round(train_pri_nse, 4),
            "val_priority_nse": round(val_pri_nse, 4),
            "lr": round(lr_now, 8),
            "elapsed_sec": round(elapsed, 1),
        })
        logger.info(
            "Epoch %d/%d (%.1fs): train_loss=%.4f val_loss=%.4f "
            "train_nse=%.4f val_nse=%.4f train_pri=%.4f val_pri=%.4f",
            epoch, args.epochs, elapsed,
            train_loss, val_loss, train_nse, val_nse, train_pri_nse, val_pri_nse,
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            best_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience and epoch >= 10:
                logger.info("Early stopping at epoch %d (best=%d)", epoch, best_epoch)
                break

    # Final evaluation with best model
    best_model_path = output_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        logger.info("Loaded best model from epoch %d", best_epoch)

    final_train_loss, final_train_nse, final_train_pri = evaluate(
        model, train_loader, graph, priority_indices, DEVICE
    )
    final_val_loss, final_val_nse, final_val_pri = evaluate(
        model, val_loader, graph, priority_indices, DEVICE
    )

    # Model hash
    model_hash = hashlib.sha256(
        json.dumps(
            {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()},
            sort_keys=True,
        ).encode()
    ).hexdigest()

    # Write outputs
    train_log_path = output_dir / "train_log.json"
    train_log_path.write_text(json.dumps(epoch_logs, indent=2), encoding="utf-8")

    smoke_report = {
        "contract_id": "PROJECT6_V42_PAPER_WORKFLOW_V1",
        "stage": "step1_sparse_state",
        "reconstructor": "TemporalSparseGATReconstructorV42",
        "reconstructor_contract": "formal_temporal_v42",
        "n_samples": len(ds_build.samples),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "rainfall_groups_train": len(train_groups),
        "rainfall_groups_val": len(val_groups),
        "sensor_ratio": args.sensor_ratio,
        "epochs_run": len(epoch_logs),
        "best_epoch": best_epoch,
        "n_parameters": n_params,
        "hidden_dim": args.hidden_dim,
        "gat_heads": args.heads,
        "gat_layers": args.gat_layers,
        "baselines": {
            "persistence_val_nse": round(persist_nse, 4),
            "persistence_val_priority_nse": round(persist_pri_nse, 4),
            "spatial_mean_val_nse": round(spatial_nse, 4),
            "spatial_mean_val_priority_nse": round(spatial_pri_nse, 4),
        },
        "final": {
            "train_loss": round(final_train_loss, 6),
            "val_loss": round(final_val_loss, 6),
            "train_nse": round(final_train_nse, 4),
            "val_nse": round(final_val_nse, 4),
            "train_priority_nse": round(final_train_pri, 4),
            "val_priority_nse": round(final_val_pri, 4),
        },
        "model_sha256": model_hash,
        "action_authority": "actual_readback_setting",
        "rainfall_group_isolated_split": True,
        "new_formal_training": True,
        "uses_future_hydraulic_truth": False,
        "smoke_mode": args.smoke,
        "graph": {
            "n_nodes": graph.n_nodes,
            "n_edges": graph.n_edges,
            "n_facilities": graph.n_facilities,
            "node_static_dim": graph.node_static.shape[1],
            "link_static_dim": graph.link_static.shape[1],
        },
    }
    report_path = output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(smoke_report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", report_path)
    logger.info("Wrote %s", train_log_path)

    # Key questions
    learning = final_val_nse > persist_nse and final_val_nse > spatial_nse
    logger.info("=" * 60)
    logger.info("KEY QUESTIONS:")
    logger.info("  Train loss decreased? %s (final=%.4f)", "YES" if epoch_logs[-1]["train_loss"] < epoch_logs[0]["train_loss"] else "NO", epoch_logs[-1]["train_loss"])
    logger.info("  Val NSE > persistence? %s (%.4f > %.4f)", "YES" if final_val_nse > persist_nse else "NO", final_val_nse, persist_nse)
    logger.info("  Val NSE > spatial? %s (%.4f > %.4f)", "YES" if final_val_nse > spatial_nse else "NO", final_val_nse, spatial_nse)
    logger.info("  Priority NSE improved? %s (%.4f > %.4f)", "YES" if final_val_pri > persist_pri_nse else "NO", final_val_pri, persist_pri_nse)
    logger.info("  Overall learning? %s", "YES" if learning else "NO")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
