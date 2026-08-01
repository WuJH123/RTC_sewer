"""Formal single-seed trainer for V4.2 Step-1 Temporal Sparse GAT.

This is intentionally separate from ``train_v42_step1_gat.py`` (the original
smoke/learning script).  Formal training obeys three additional contracts:

1. one frozen sensor deployment per experiment;
2. source/unknown-domain windows may be auxiliary pretraining only, while
   validation and uncertainty calibration populations are target-domain only;
3. the model's aleatoric ``depth_std`` head is trained by heteroscedastic NLL.

The script writes a training report and checkpoint.  It does **not** write a
passing ``evidence.json`` because uncertainty calibration, OOD calibration and
multi-seed robustness are later formal gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.models.temporal_sparse_gat_v42 import TemporalSparseGATReconstructorV42
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_step1_dataset import Step1TorchDataset, build_step1_dataset
from sewerrtc.v4.v42_step1_training import (
    Step1LossWeights,
    build_formal_step1_split,
    split_summary,
    step1_reconstruction_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _nse(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return float("nan")
    den = float(np.sum((target - target.mean()) ** 2))
    num = float(np.sum((target - pred) ** 2))
    if den <= 1.0e-12:
        return 1.0 if num <= 1.0e-12 else 0.0
    return 1.0 - num / den


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if pred.size == 0:
        return {"nse": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    err = pred - target
    return {
        "nse": float(_nse(pred, target)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
    }


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


def _run_epoch(
    model,
    loader,
    graph,
    device,
    *,
    optimizer=None,
    weights: Step1LossWeights,
    wet_threshold_m: float,
):
    training = optimizer is not None
    model.train(training)
    node_static, link_static, edge_index, action_map = _graph_tensors(graph, device)
    pri_mask = _priority_mask(graph, device)
    loss_sum = 0.0
    batches = 0
    pred_all: list[np.ndarray] = []
    target_all: list[np.ndarray] = []
    mask_all: list[np.ndarray] = []
    std_all: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
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
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            loss_sum += float(losses["total"].detach().cpu())
            batches += 1
            pred_all.append(out.depth_mean.detach().cpu().numpy())
            std_all.append(out.depth_std.detach().cpu().numpy())
            target_all.append(target.detach().cpu().numpy())
            mask_all.append(smh[:, -1, :].detach().cpu().numpy())

    pred = np.concatenate(pred_all, axis=0)
    std = np.concatenate(std_all, axis=0)
    target = np.concatenate(target_all, axis=0)
    observed = np.concatenate(mask_all, axis=0) >= 0.5
    unobserved = ~observed
    pri = np.zeros(pred.shape[1], dtype=bool)
    pri[get_pfv_core_node_indices(graph.node_ids)] = True
    pri_unobs = unobserved[:, pri]

    result = {
        "loss": loss_sum / max(1, batches),
        "overall_unobserved": _metrics(pred[unobserved], target[unobserved]),
        "priority_unobserved": _metrics(pred[:, pri][pri_unobs], target[:, pri][pri_unobs]),
        "mean_predicted_std_unobserved": float(std[unobserved].mean()),
    }
    wet = pri_unobs & (target[:, pri] >= float(wet_threshold_m))
    result["priority_wet_unobserved"] = _metrics(
        pred[:, pri][wet], target[:, pri][wet]
    )
    result["priority_wet_count"] = int(wet.sum())
    return result


def _state_hash(model) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sensor-ratio", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--gat-layers", type=int, default=3)
    ap.add_argument("--pretrain-epochs", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--priority-weight", type=float, default=3.0)
    ap.add_argument("--wet-priority-weight", type=float, default=1.0)
    ap.add_argument("--nll-weight", type=float, default=0.25)
    ap.add_argument("--wet-threshold-m", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = args.output_dir or (
        args.project_root
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat"
        / f"formal_seed_{args.seed}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or (
        args.project_root
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat/dataset/step1_window_manifest.parquet"
    )

    built = build_step1_dataset(
        project_root=args.project_root,
        manifest_path=manifest,
        sensor_ratio=args.sensor_ratio,
        rng_seed=args.seed,
    )
    split = build_formal_step1_split(built.samples, seed=args.seed)
    graph = built.graph
    dataset = Step1TorchDataset(built.samples, graph)
    weights = Step1LossWeights(
        priority_depth=args.priority_weight,
        wet_priority_depth=args.wet_priority_weight,
        heteroscedastic_nll=args.nll_weight,
    )

    def loader(indices, shuffle: bool):
        return DataLoader(
            Subset(dataset, list(indices)),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

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

    history: list[dict] = []
    if split.auxiliary_pretrain_indices and args.pretrain_epochs > 0:
        aux_loader = loader(split.auxiliary_pretrain_indices, True)
        for epoch in range(1, args.pretrain_epochs + 1):
            train_metrics = _run_epoch(
                model,
                aux_loader,
                graph,
                device,
                optimizer=optimizer,
                weights=weights,
                wet_threshold_m=args.wet_threshold_m,
            )
            history.append({"phase": "auxiliary_pretrain", "epoch": epoch, **train_metrics})
            logger.info("aux epoch %d loss %.5f", epoch, train_metrics["loss"])

    train_loader = loader(split.target_train_indices, True)
    val_loader = loader(split.target_validation_indices, False)
    best_loss = float("inf")
    best_epoch = -1
    patience = 0
    best_path = outdir / "best_model.pt"
    for epoch in range(1, args.finetune_epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            graph,
            device,
            optimizer=optimizer,
            weights=weights,
            wet_threshold_m=args.wet_threshold_m,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            graph,
            device,
            optimizer=None,
            weights=weights,
            wet_threshold_m=args.wet_threshold_m,
        )
        history.append(
            {
                "phase": "target_finetune",
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
            }
        )
        logger.info(
            "target epoch %d loss %.5f val %.5f val_NSE %.4f pri_NSE %.4f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["overall_unobserved"]["nse"],
            val_metrics["priority_unobserved"]["nse"],
        )
        if val_metrics["loss"] < best_loss:
            best_loss = float(val_metrics["loss"])
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_epoch < 0:
        raise RuntimeError("formal Step1 training never produced a checkpoint")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    final_train = _run_epoch(
        model,
        train_loader,
        graph,
        device,
        optimizer=None,
        weights=weights,
        wet_threshold_m=args.wet_threshold_m,
    )
    final_val = _run_epoch(
        model,
        val_loader,
        graph,
        device,
        optimizer=None,
        weights=weights,
        wet_threshold_m=args.wet_threshold_m,
    )
    calibration_loader = loader(split.target_calibration_indices, False)
    calibration_uncalibrated = _run_epoch(
        model,
        calibration_loader,
        graph,
        device,
        optimizer=None,
        weights=weights,
        wet_threshold_m=args.wet_threshold_m,
    )

    report = {
        "contract_id": "PROJECT6_V42_PAPER_WORKFLOW_V1",
        "stage": "step1_sparse_state_training",
        "status": "trained_requires_uncertainty_and_ood_calibration",
        "formal_reconstructor": "TemporalSparseGATReconstructorV42",
        "fixed_sensor_layout": True,
        "sensor_ratio": args.sensor_ratio,
        "sensor_layout_sha256": built.sensor_layout_sha256,
        "sensor_count": int(len(built.sensor_indices)),
        "target_domain_validation_only": True,
        "auxiliary_domain_used_for_formal_validation": False,
        "aleatoric_scale_trained": True,
        "bidirectional_graph": True,
        "physical_edge_attributes": True,
        "split": split_summary(split),
        "best_epoch": best_epoch,
        "model_sha256": _state_hash(model),
        "final_target_train": final_train,
        "final_target_validation": final_val,
        "target_calibration_before_posthoc_calibration": calibration_uncalibrated,
        "uncertainty_calibrated": False,
        "ood_calibrated": False,
        "formal_evidence_ready": False,
        "warnings": built.warnings[:100],
        "skipped_samples": built.skipped,
    }
    (outdir / "training_history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False), encoding="utf-8"
    )
    (outdir / "formal_training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    logger.info("wrote %s", outdir / "formal_training_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
