"""Train a small development-only V4.2 Step-2 control-core surrogate.

This runner is a fast scientific screen, not formal paper evidence.  It trains
``MultiReferenceHydraulicSurrogate`` on the development-only four-reference
source-domain core dataset produced by :mod:`v42_fast_feasibility`.

Only depth, node flooding-rate and trajectory-derived KPI deltas are supervised.
Outfall/storage/facility-flow claims are deliberately disabled in this pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.models_v42.hydraulic_multi_reference import MultiReferenceHydraulicSurrogate
from sewerrtc.v4.models_v42.hydraulic_trajectory_losses import HydraulicLossWeights, HydraulicTrajectoryLoss
from sewerrtc.v4.v42_fast_feasibility import FAST_CONTRACT_ID
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology

BRANCHES = ("candidate", "no_control", "dynamic_internal", "hold_previous")


def _arr(value: str) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=np.float32)


def _stack(frame: pd.DataFrame, column: str) -> torch.Tensor:
    return torch.from_numpy(np.stack([_arr(v) for v in frame[column]], axis=0))


def _hash_model(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _split_groups(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    groups = sorted(frame["split_group_key"].astype(str).unique())
    if len(groups) < 2:
        raise ValueError("fast Step2 requires at least two rainfall groups")
    ranked = sorted(
        groups,
        key=lambda g: hashlib.sha256(f"{seed}:{g}".encode("utf-8")).hexdigest(),
    )
    n_val = max(1, int(round(0.2 * len(ranked))))
    n_val = min(n_val, len(ranked) - 1)
    val_groups = ranked[:n_val]
    train_groups = ranked[n_val:]
    train = frame[frame["split_group_key"].astype(str).isin(train_groups)].copy()
    val = frame[frame["split_group_key"].astype(str).isin(val_groups)].copy()
    if train.empty or val.empty:
        raise ValueError("fast Step2 group split produced empty train/validation")
    return train, val, train_groups, val_groups


def _tensorise(frame: pd.DataFrame) -> dict[str, torch.Tensor]:
    data: dict[str, torch.Tensor] = {
        "history_depth": _stack(frame, "history_depth"),
        "history_actions": _stack(frame, "history_actions_readback"),
        "rainfall": _stack(frame, "rainfall_forecast"),
        "pfv_delta": torch.as_tensor(frame["pfv_delta"].to_numpy(np.float32)),
        "tfv_delta": torch.as_tensor(frame["tfv_delta"].to_numpy(np.float32)),
        "peak_delta": torch.as_tensor(frame["peak_delta"].to_numpy(np.float32)),
    }
    for branch in BRANCHES:
        data[f"action_{branch}"] = _stack(frame, f"action_{branch}_readback")
        data[f"depth_{branch}"] = _stack(frame, f"trajectory_depth_{branch}")
        data[f"flood_{branch}"] = _stack(frame, f"trajectory_flood_{branch}")
    return data


def _nse(pred: np.ndarray, target: np.ndarray) -> float | None:
    y = target.astype(np.float64).reshape(-1)
    p = pred.astype(np.float64).reshape(-1)
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-12:
        return None
    return float(1.0 - np.sum((p - y) ** 2) / denom)


def _sign_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred <= 0.0) == (target <= 0.0)))


def _batch_indices(n: int, batch: int, *, shuffle: bool, seed: int) -> list[np.ndarray]:
    idx = np.arange(n)
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)
    return [idx[i : i + batch] for i in range(0, n, batch)]


def _forward(model, batch, graph_tensors, priority_idx, device):
    edge_index, node_static, action_map = graph_tensors
    return model(
        state_history=batch["history_depth"].to(device),
        historical_actions=batch["history_actions"].to(device),
        rainfall_forecast=batch["rainfall"].to(device),
        action_candidate=batch["action_candidate"].to(device),
        action_no_control=batch["action_no_control"].to(device),
        action_dynamic_internal=batch["action_dynamic_internal"].to(device),
        action_hold_previous=batch["action_hold_previous"].to(device),
        edge_index=edge_index,
        node_static=node_static,
        action_node_map=action_map,
        priority_node_indices=priority_idx,
        storage_node_indices=None,
        outfall_node_indices=None,
    )


def _targets(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    target: dict[str, torch.Tensor] = {}
    for branch in BRANCHES:
        target[f"trajectory_depth_{branch}"] = batch[f"depth_{branch}"].to(device)
        target[f"trajectory_flood_{branch}"] = batch[f"flood_{branch}"].to(device)
    for key in ("pfv_delta", "tfv_delta", "peak_delta"):
        target[key] = batch[key].to(device)
    return target


def _slice(data: dict[str, torch.Tensor], idx: np.ndarray) -> dict[str, torch.Tensor]:
    ti = torch.as_tensor(idx, dtype=torch.long)
    return {k: v.index_select(0, ti) for k, v in data.items()}


def _evaluate(model, data, graph_tensors, priority_idx, device, batch_size, loss_fn):
    model.eval()
    total_loss = 0.0
    total_n = 0
    depth_pred: list[np.ndarray] = []
    depth_true: list[np.ndarray] = []
    flood_pred: list[np.ndarray] = []
    flood_true: list[np.ndarray] = []
    kp_pred = {k: [] for k in ("pfv_delta", "tfv_delta", "peak_delta")}
    kp_true = {k: [] for k in kp_pred}
    with torch.no_grad():
        for idx in _batch_indices(len(next(iter(data.values()))), batch_size, shuffle=False, seed=0):
            batch = _slice(data, idx)
            pred = _forward(model, batch, graph_tensors, priority_idx, device)
            target = _targets(batch, device)
            losses = loss_fn(pred, target)
            loss = loss_fn.total(losses)
            total_loss += float(loss.item()) * len(idx)
            total_n += len(idx)
            for branch in BRANCHES:
                depth_pred.append(pred["branches"][branch]["node_depth"].cpu().numpy())
                depth_true.append(batch[f"depth_{branch}"].numpy())
                flood_pred.append(pred["branches"][branch]["node_flooding_rate"].cpu().numpy())
                flood_true.append(batch[f"flood_{branch}"].numpy())
            for key in kp_pred:
                kp_pred[key].append(pred[key].detach().cpu().numpy())
                kp_true[key].append(batch[key].numpy())
    dp = np.concatenate(depth_pred, axis=0)
    dt = np.concatenate(depth_true, axis=0)
    fp = np.concatenate(flood_pred, axis=0)
    ft = np.concatenate(flood_true, axis=0)
    report = {
        "loss": float(total_loss / max(1, total_n)),
        "depth_nse": _nse(dp, dt),
        "depth_rmse_m": float(np.sqrt(np.mean((dp - dt) ** 2))),
        "flood_mae_m3s": float(np.mean(np.abs(fp - ft))),
    }
    for key in kp_pred:
        p = np.concatenate(kp_pred[key])
        y = np.concatenate(kp_true[key])
        report[f"{key}_mae"] = float(np.mean(np.abs(p - y)))
        report[f"{key}_sign_accuracy"] = _sign_accuracy(p, y)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--gat-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frame = pd.read_parquet(args.manifest) if args.manifest.suffix.lower() == ".parquet" else pd.read_csv(args.manifest)
    if frame.empty:
        raise ValueError("fast Step2 manifest is empty")
    if not bool(frame["development_only"].astype(bool).all()):
        raise RuntimeError("fast trainer accepts development-only pilot data only")
    train_f, val_f, train_groups, val_groups = _split_groups(frame, args.seed)
    train = _tensorise(train_f)
    val = _tensorise(val_f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = _load_graph_topology(args.project_root)
    edge_index = torch.from_numpy(graph["edge_index"].astype(np.int64)).to(device)
    node_static = torch.from_numpy(graph["node_static"].astype(np.float32)).to(device)
    action_map = torch.from_numpy(graph["action_node_map"].astype(np.float32)).to(device)
    priority_idx = torch.as_tensor(get_pfv_core_node_indices(list(graph["node_ids"])), dtype=torch.long, device=device)

    model = MultiReferenceHydraulicSurrogate(
        n_nodes=int(graph["n_nodes"]),
        n_facilities=int(graph["n_facilities"]),
        state_feature_dim=1,
        static_feature_dim=int(graph["node_static"].shape[1]),
        hidden_dim=args.hidden_dim,
        gat_heads=4,
        gat_layers=args.gat_layers,
        horizon=12,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = HydraulicTrajectoryLoss(
        HydraulicLossWeights(
            depth=0.5,
            node_flooding=2.0,
            storage=0.0,
            facility_flow=0.0,
            outfall_flow=0.0,
            kpi_consistency=0.5,
        ),
        require_storage_targets=False,
        require_facility_flow_targets=False,
        require_outfall_flow_targets=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_model.pt"
    history = []
    best = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for idx in _batch_indices(len(train_f), args.batch_size, shuffle=True, seed=args.seed + epoch):
            batch = _slice(train, idx)
            optimizer.zero_grad(set_to_none=True)
            pred = _forward(model, batch, (edge_index, node_static, action_map), priority_idx, device)
            target = _targets(batch, device)
            losses = loss_fn(pred, target)
            loss = loss_fn.total(losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += float(loss.detach().item()) * len(idx)
            seen += len(idx)
        val_report = _evaluate(model, val, (edge_index, node_static, action_map), priority_idx, device, args.batch_size, loss_fn)
        row = {"epoch": epoch, "train_loss": running / max(1, seen), "validation": val_report}
        history.append(row)
        print(json.dumps(row, allow_nan=False))
        if val_report["loss"] < best:
            best = float(val_report["loss"])
            stale = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    train_report = _evaluate(model, train, (edge_index, node_static, action_map), priority_idx, device, args.batch_size, loss_fn)
    val_report = _evaluate(model, val, (edge_index, node_static, action_map), priority_idx, device, args.batch_size, loss_fn)
    model_sha = _hash_model(model)
    report = {
        "contract_id": FAST_CONTRACT_ID,
        "stage": "step2_fast_control_core",
        "development_only": True,
        "formal_mainline_authorized": False,
        "formal_model": "MultiReferenceHydraulicSurrogate",
        "outfall_supervised": False,
        "storage_supervised": False,
        "facility_flow_supervised": False,
        "trajectory_first": True,
        "train_cases": int(len(train_f)),
        "validation_cases": int(len(val_f)),
        "train_rainfall_groups": train_groups,
        "validation_rainfall_groups": val_groups,
        "model_sha256": model_sha,
        "train": train_report,
        "validation": val_report,
        "history": history,
    }
    (args.output_dir / "fast_step2_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (args.output_dir / "validation_groups.json").write_text(json.dumps({"groups": val_groups}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
