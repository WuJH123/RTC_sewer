"""4-fold Leave-One-Event-Out (LOEO) A0 baseline for Step-1 development closure.

Section 9 of the formal Step1 plan:
- 4 target rainfall groups
- 4-fold Leave-One-Rainfall-Group-Out
- Each fold: 3 groups train, 1 group validation
- A0 config: aux-pretrain OFF, priority_weight=0, wet_priority_weight=0, NLL_weight=0
- model_seed=42, sensor_layout_seed=42
- Max 20 epochs, patience=6, checkpoint by validation overall-unobserved RMSE
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(r"E:\RTC_sewer\Project6")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.models.temporal_sparse_gat_v42 import TemporalSparseGATReconstructorV42
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_step1_streaming import (
    Step1StreamingDataset,
    target_rainfall_groups,
)
from sewerrtc.v4.v42_step1_training import Step1LossWeights, step1_reconstruction_loss

# Reuse helpers from the streaming trainer
from scripts.train_v42_step1_streaming import (
    StreamingMetricAccumulator,
    _graph_tensors,
    _hash_strings,
    _make_loader,
    _priority_mask,
    _rss_mb,
    _run_epoch,
    _state_hash,
)

try:
    import psutil
except Exception:
    psutil = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MANIFEST = (
    PROJECT_ROOT
    / "outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat/dataset/step1_window_manifest.parquet"
)
OUTDIR = (
    PROJECT_ROOT
    / "outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat/development"
)

# A0 hyperparameters (Section 9)
MODEL_SEED = 42
SENSOR_LAYOUT_SEED = 42
SENSOR_RATIO = 0.10
EPOCHS = 20
PATIENCE = 3
BATCH_SIZE = 64
LR = 3.0e-4
HIDDEN_DIM = 128
HEADS = 4
GAT_LAYERS = 3
HEARTBEAT_BATCHES = 10
NUM_WORKERS = 0
WET_THRESHOLD_M = 0.05


def _run_loeo_fold(
    *,
    fold_id: int,
    train_groups: tuple[str, ...],
    val_group: str,
    device: torch.device,
) -> dict:
    """Run a single LOEO fold and return the result dict."""
    logger.info("=" * 60)
    logger.info("FOLD %d: train=%s val=%s", fold_id, train_groups, val_group)
    logger.info("=" * 60)

    # Set seeds
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)

    # Create datasets
    train_ds = Step1StreamingDataset(
        project_root=PROJECT_ROOT,
        manifest_path=MANIFEST,
        sensor_ratio=SENSOR_RATIO,
        sensor_layout_seed=SENSOR_LAYOUT_SEED,
        domain_roles=("target_formal",),
        allowed_groups=train_groups,
        shuffle_files=True,
        iteration_seed=MODEL_SEED,
    )
    val_ds = Step1StreamingDataset(
        project_root=PROJECT_ROOT,
        manifest_path=MANIFEST,
        sensor_ratio=SENSOR_RATIO,
        sensor_layout_seed=SENSOR_LAYOUT_SEED,
        domain_roles=("target_formal",),
        allowed_groups=(val_group,),
        shuffle_files=False,
        iteration_seed=MODEL_SEED,
    )
    if train_ds.sensor_layout_sha256 != val_ds.sensor_layout_sha256:
        raise RuntimeError("sensor layout mismatch between train/val")

    graph = train_ds.graph
    logger.info(
        "train windows=%d val windows=%d sensor_sha=%s",
        len(train_ds), len(val_ds), train_ds.sensor_layout_sha256[:16],
    )

    # Build model
    model = TemporalSparseGATReconstructorV42(
        n_nodes=graph.n_nodes,
        n_facilities=graph.n_facilities,
        node_static_dim=graph.node_static.shape[1],
        link_static_dim=graph.link_static.shape[1],
        hidden_dim=HIDDEN_DIM,
        heads=HEADS,
        gat_layers=GAT_LAYERS,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1.0e-4)

    # A0 weights: all zero for priority and NLL
    weights = Step1LossWeights(
        global_depth=1.0,
        priority_depth=0.0,
        wet_priority_depth=0.0,
        heteroscedastic_nll=0.0,
    )

    # Training loop
    best_rmse = float("inf")
    best_epoch = -1
    stale = 0
    fold_dir = OUTDIR / f"loeo_fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best_model.pt"
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_metrics = _run_epoch(
            model=model, dataset=train_ds, graph=graph, device=device,
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, optimizer=optimizer,
            weights=weights, wet_threshold_m=WET_THRESHOLD_M,
            heartbeat_batches=HEARTBEAT_BATCHES, epoch=epoch,
        )
        val_metrics = _run_epoch(
            model=model, dataset=val_ds, graph=graph, device=device,
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, optimizer=None,
            weights=weights, wet_threshold_m=WET_THRESHOLD_M,
            heartbeat_batches=HEARTBEAT_BATCHES, epoch=epoch,
        )
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(epoch_record)

        val_rmse = val_metrics["overall_unobserved"]["rmse"]
        val_nse = val_metrics["overall_unobserved"]["nse"]
        train_nse = train_metrics["overall_unobserved"]["nse"]
        pri_nse = val_metrics["priority_unobserved"]["nse"]
        pri_wet = val_metrics["priority_wet_unobserved"]

        logger.info(
            "fold %d epoch %d | train_NSE=%s val_NSE=%s val_RMSE=%s pri_NSE=%s | windows=%d",
            fold_id, epoch,
            "NA" if train_nse is None else f"{train_nse:.4f}",
            "NA" if val_nse is None else f"{val_nse:.4f}",
            "NA" if val_rmse is None else f"{val_rmse:.4f}",
            "NA" if pri_nse is None else f"{pri_nse:.4f}",
            train_metrics["actual_windows_seen"],
        )

        if val_rmse is not None and float(val_rmse) < best_rmse:
            best_rmse = float(val_rmse)
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                logger.info("fold %d early stop at epoch %d; best=%d", fold_id, epoch, best_epoch)
                break

    if best_epoch < 0:
        raise RuntimeError(f"fold {fold_id} produced no valid validation RMSE")

    # Reload best model for final evaluation
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    final_train = _run_epoch(
        model=model, dataset=train_ds, graph=graph, device=device,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, optimizer=None,
        weights=weights, wet_threshold_m=WET_THRESHOLD_M,
        heartbeat_batches=HEARTBEAT_BATCHES, epoch=best_epoch,
    )
    final_val = _run_epoch(
        model=model, dataset=val_ds, graph=graph, device=device,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, optimizer=None,
        weights=weights, wet_threshold_m=WET_THRESHOLD_M,
        heartbeat_batches=HEARTBEAT_BATCHES, epoch=best_epoch,
    )

    # Save fold history
    fold_history_path = fold_dir / "training_history.json"
    # Use a simplified version for JSON serialization
    def _simplify(obj):
        if isinstance(obj, dict):
            return {k: _simplify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_simplify(v) for v in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        if isinstance(obj, np.generic):
            return _simplify(obj.item())
        return obj

    fold_history_path.write_text(
        json.dumps(_simplify(history), indent=2, allow_nan=False), encoding="utf-8"
    )

    val_overall = final_val["overall_unobserved"]
    val_priority = final_val["priority_unobserved"]
    val_wet = final_val["priority_wet_unobserved"]
    train_overall = final_train["overall_unobserved"]

    return {
        "fold_id": fold_id,
        "train_groups": list(train_groups),
        "validation_group": val_group,
        "train_windows": len(train_ds),
        "validation_windows": len(val_ds),
        "best_epoch": best_epoch,
        "best_val_rmse": best_rmse,
        "sensor_layout_sha256": train_ds.sensor_layout_sha256,
        "model_sha256": _state_hash(model),
        "train_overall_nse": train_overall["nse"],
        "train_overall_rmse": train_overall["rmse"],
        "train_overall_mae": train_overall["mae"],
        "val_overall_nse": val_overall["nse"],
        "val_overall_rmse": val_overall["rmse"],
        "val_overall_mae": val_overall["mae"],
        "val_priority_nse": val_priority["nse"],
        "val_priority_rmse": val_priority["rmse"],
        "val_priority_mae": val_priority["mae"],
        "val_wet_rmse": val_wet["rmse"],
        "val_wet_mae": val_wet["mae"],
        "val_wet_count": val_wet["count"],
        "peak_rss_mb": final_val.get("peak_rss_mb"),
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    groups = target_rainfall_groups(MANIFEST)
    logger.info("target rainfall groups: %d", len(groups))
    for i, g in enumerate(groups):
        logger.info("  group[%d]: %s", i, g[:32] + "...")

    if len(groups) != 4:
        raise RuntimeError(f"LOEO A0 expects exactly 4 target groups, got {len(groups)}")

    # 4-fold LOEO: each group is validation once
    fold_results = []
    for fold_id, val_group in enumerate(groups):
        train_groups = tuple(g for g in groups if g != val_group)
        result = _run_loeo_fold(
            fold_id=fold_id,
            train_groups=train_groups,
            val_group=val_group,
            device=device,
        )
        fold_results.append(result)
        logger.info(
            "FOLD %d COMPLETE: val_NSE=%s val_RMSE=%s best_epoch=%d",
            fold_id,
            "NA" if result["val_overall_nse"] is None else f"{result['val_overall_nse']:.4f}",
            f"{result['val_overall_rmse']:.4f}",
            result["best_epoch"],
        )

    # Write CSV summary
    csv_path = OUTDIR / "loeo_A0_summary.csv"
    fieldnames = [
        "fold_id", "validation_group", "train_groups",
        "train_windows", "validation_windows", "best_epoch", "best_val_rmse",
        "train_overall_nse", "train_overall_rmse", "train_overall_mae",
        "val_overall_nse", "val_overall_rmse", "val_overall_mae",
        "val_priority_nse", "val_priority_rmse", "val_priority_mae",
        "val_wet_rmse", "val_wet_mae", "val_wet_count",
        "peak_rss_mb",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in fold_results:
            row = dict(r)
            row["train_groups"] = ";".join(g[:16] + "..." for g in r["train_groups"])
            row["validation_group"] = r["validation_group"][:32] + "..."
            writer.writerow(row)

    # Write JSON summary
    json_path = OUTDIR / "loeo_A0_summary.json"

    def _safe(obj):
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(v) for v in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        if isinstance(obj, np.generic):
            return _safe(obj.item())
        return obj

    # Compute aggregate stats
    val_nses = [r["val_overall_nse"] for r in fold_results if r["val_overall_nse"] is not None]
    val_rmses = [r["val_overall_rmse"] for r in fold_results]
    val_maes = [r["val_overall_mae"] for r in fold_results]

    summary = {
        "contract": "PROJECT6_V42_STEP1_LOEO_A0",
        "config": {
            "model_seed": MODEL_SEED,
            "sensor_layout_seed": SENSOR_LAYOUT_SEED,
            "sensor_ratio": SENSOR_RATIO,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "priority_weight": 0.0,
            "wet_priority_weight": 0.0,
            "nll_weight": 0.0,
            "aux_pretrain": False,
        },
        "n_target_groups": len(groups),
        "folds": fold_results,
        "aggregate": {
            "val_nse_mean": float(np.mean(val_nses)) if val_nses else None,
            "val_nse_std": float(np.std(val_nses)) if len(val_nses) > 1 else None,
            "val_nse_min": float(np.min(val_nses)) if val_nses else None,
            "val_nse_max": float(np.max(val_nses)) if val_nses else None,
            "val_rmse_mean": float(np.mean(val_rmses)),
            "val_rmse_std": float(np.std(val_rmses)),
            "val_mae_mean": float(np.mean(val_maes)),
            "val_mae_std": float(np.std(val_maes)),
        },
    }
    json_path.write_text(
        json.dumps(_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
    )

    # Write markdown report
    md_path = OUTDIR / "loeo_A0_report.md"
    lines = [
        "# Step1 LOEO A0 Development Report",
        "",
        "## Configuration",
        "",
        f"- Model seed: {MODEL_SEED}",
        f"- Sensor layout seed: {SENSOR_LAYOUT_SEED}",
        f"- Sensor ratio: {SENSOR_RATIO}",
        f"- Epochs: {EPOCHS}, Patience: {PATIENCE}",
        f"- Priority weight: 0, Wet priority weight: 0, NLL weight: 0",
        f"- Aux pretrain: OFF",
        f"- Target rainfall groups: {len(groups)}",
        "",
        "## Fold Results",
        "",
        "| Fold | Val Group (short) | Train Win | Val Win | Best Epoch | Train NSE | Val NSE | Val RMSE | Pri NSE | Wet RMSE |",
        "|------|-------------------|-----------|---------|------------|-----------|---------|----------|---------|----------|",
    ]
    for r in fold_results:
        vg = r["validation_group"][:12] + "..."
        tnse = f"{r['train_overall_nse']:.4f}" if r["train_overall_nse"] is not None else "NA"
        vnse = f"{r['val_overall_nse']:.4f}" if r["val_overall_nse"] is not None else "NA"
        vrmse = f"{r['val_overall_rmse']:.4f}"
        pnse = f"{r['val_priority_nse']:.4f}" if r["val_priority_nse"] is not None else "NA"
        wrmse = f"{r['val_wet_rmse']:.4f}" if r["val_wet_rmse"] is not None else "NA"
        lines.append(
            f"| {r['fold_id']} | {vg} | {r['train_windows']} | {r['validation_windows']} "
            f"| {r['best_epoch']} | {tnse} | {vnse} | {vrmse} | {pnse} | {wrmse} |"
        )

    lines.extend([
        "",
        "## Aggregate",
        "",
    ])
    agg = summary["aggregate"]
    if agg["val_nse_mean"] is not None:
        lines.append(f"- Val NSE: {agg['val_nse_mean']:.4f} +/- {agg['val_nse_std']:.4f}" if agg["val_nse_std"] is not None else f"- Val NSE: {agg['val_nse_mean']:.4f}")
        lines.append(f"  - Range: [{agg['val_nse_min']:.4f}, {agg['val_nse_max']:.4f}]")
    lines.append(f"- Val RMSE: {agg['val_rmse_mean']:.4f} +/- {agg['val_rmse_std']:.4f}")
    lines.append(f"- Val MAE: {agg['val_mae_mean']:.4f} +/- {agg['val_mae_std']:.4f}")

    lines.extend([
        "",
        "## Architecture Verdict",
        "",
    ])
    # Section 10 verdict logic
    nse_above_08 = sum(1 for n in val_nses if n is not None and n >= 0.8)
    nse_above_05 = sum(1 for n in val_nses if n is not None and n >= 0.5)
    n_folds = len(val_nses)

    if nse_above_08 >= 3:
        verdict = "ARCHITECTURE_GO"
        lines.append(f"**{verdict}**: {nse_above_08}/{n_folds} folds have Val NSE >= 0.8")
    elif nse_above_05 >= 2:
        verdict = "EVENT_OOD_LIMITED"
        lines.append(f"**{verdict}**: Some folds perform well but others struggle, suggesting event-specific OOD challenges")
    else:
        verdict = "TARGET_DATA_LIMITED"
        lines.append(f"**{verdict}**: Most folds show Val NSE < 0.5, suggesting the 4-group target data is insufficient for robust cross-event generalization")

    lines.extend([
        "",
        f"- Folds with NSE >= 0.8: {nse_above_08}/{n_folds}",
        f"- Folds with NSE >= 0.5: {nse_above_05}/{n_folds}",
        "",
        "## Files",
        "",
        f"- CSV: `{csv_path.relative_to(PROJECT_ROOT)}`",
        f"- JSON: `{json_path.relative_to(PROJECT_ROOT)}`",
        f"- Fold checkpoints: `{(OUTDIR / 'loeo_fold_*').relative_to(PROJECT_ROOT)}`",
    ])

    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("LOEO A0 complete. Verdict: %s", verdict)
    logger.info("Outputs: %s, %s, %s", csv_path, json_path, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
