from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sewerrtc.data.tensor_cache import load_npz
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.models.graph_surrogate import PhysicsGuidedTemporalGraphSurrogate
from sewerrtc.models.losses import temporal_surrogate_loss


def _fast_tensor_batches(
    tensors: list[torch.Tensor],
    batch_size: int,
    sample_weights: torch.Tensor | None,
    max_samples: int,
    seed: int,
):
    n = int(tensors[0].shape[0])
    if n <= 0:
        return
    total = n if int(max_samples) <= 0 else min(int(max_samples), n)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    if sample_weights is not None:
        idx_all = torch.multinomial(sample_weights.cpu().float(), total, replacement=True, generator=gen)
    else:
        idx_all = torch.randperm(n, generator=gen)[:total]
    for start in range(0, total, int(batch_size)):
        idx = idx_all[start : start + int(batch_size)]
        yield tuple(t.index_select(0, idx) for t in tensors)


def _fast_numpy_batches(
    arrays: list[np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    sample_weights: np.ndarray | None,
    max_samples: int,
    seed: int,
):
    indices = np.asarray(indices, dtype=np.int64)
    n = int(len(indices))
    if n <= 0:
        return
    total = n if int(max_samples) <= 0 else min(int(max_samples), n)
    rng = np.random.default_rng(int(seed))
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if len(weights) != n:
            raise ValueError("sample_weights must align with indices")
        if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0:
            weights = np.ones(n, dtype=np.float64)
        weights = weights / weights.sum()
        picked = rng.choice(indices, size=total, replace=True, p=weights)
    else:
        picked = rng.permutation(indices)[:total]
    for start in range(0, total, int(batch_size)):
        idx = picked[start : start + int(batch_size)]
        yield tuple(torch.as_tensor(a[idx]) for a in arrays)


def _clean_float32(arr: np.ndarray, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> np.ndarray:
    out = arr.astype(np.float32, copy=False)
    np.nan_to_num(out, copy=False, nan=nan, posinf=posinf, neginf=neginf)
    return out


def _train_val_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    val_n = max(1, int(int(n) * float(val_fraction)))
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(int(n))
    return indices[val_n:].astype(np.int64), indices[:val_n].astype(np.int64)


def _build_graph_tensors(cfg: dict, node_ids: list[str], action_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_table = pd.read_csv(cfg_path(cfg, "outputs.audit") / "node_table.csv")
    link_table = pd.read_csv(cfg_path(cfg, "outputs.audit") / "link_table.csv")
    static_cols = ["invert", "max_depth", "ponded_area", "degree_in", "degree_out", "is_storage", "is_outfall"]
    for c in static_cols:
        if c not in node_table:
            node_table[c] = 0.0
        node_table[c] = pd.to_numeric(node_table[c], errors="coerce").fillna(0.0)
    node_table = node_table.set_index("node_id")
    static = []
    for n in node_ids:
        if n in node_table.index:
            static.append(node_table.loc[n, static_cols].to_numpy(dtype=np.float32))
        else:
            static.append(np.zeros(len(static_cols), dtype=np.float32))
    static = np.asarray(static, dtype=np.float32)
    mean, std = static.mean(axis=0, keepdims=True), static.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    static = (static - mean) / std
    idx = {n: i for i, n in enumerate(node_ids)}
    edges = []
    for _, r in link_table.iterrows():
        a, b = str(r.get("from_node", "")), str(r.get("to_node", ""))
        if a in idx and b in idx:
            edges.append((idx[a], idx[b]))
            edges.append((idx[b], idx[a]))
    if not edges:
        edges = [(i, i) for i in range(len(node_ids))]
    edge_index = np.asarray(edges, dtype=np.int64).T

    link_table = link_table.set_index("link_id")
    amap = np.zeros((len(action_ids), len(node_ids)), dtype=np.float32)
    for j, aid in enumerate(action_ids):
        lid = aid.split(":", 1)[1] if aid.startswith("a:") else aid
        if lid in link_table.index:
            r = link_table.loc[lid]
            for nid, w in [(str(r.get("from_node", "")), 1.0), (str(r.get("to_node", "")), 0.6)]:
                if nid in idx:
                    amap[j, idx[nid]] = max(amap[j, idx[nid]], w)
        if amap[j].sum() <= 0:
            amap[j, :] = 1.0 / len(node_ids)
        else:
            amap[j] /= max(1.0, amap[j].sum())
    return static.astype(np.float32), edge_index, amap


def _evaluate_in_batches(
    model: PhysicsGuidedTemporalGraphSurrogate,
    arrays: list[np.ndarray],
    indices: np.ndarray,
    node_static: torch.Tensor,
    edge_index: torch.Tensor,
    action_node_map: torch.Tensor,
    device: torch.device,
    eval_batch_size: int,
    risk_delta_scale: torch.Tensor,
    max_samples: int = 0,
) -> dict:
    vh_np, va_np, vr_np, vy_np, vd_np = arrays
    eval_indices = np.asarray(indices, dtype=np.int64)
    if max_samples and 0 < int(max_samples) < len(eval_indices):
        take = np.linspace(0, len(eval_indices) - 1, int(max_samples), dtype=np.int64)
        eval_indices = eval_indices[take]
    eval_batch = max(1, int(eval_batch_size))
    total = 0
    depth_sse = 0.0
    depth_count = 0
    delta_abs = 0.0
    delta_count = 0
    pfv_dir_ok = 0
    tfv_dir_ok = 0
    peak_dir_ok = 0
    safe_ok = 0
    pfv_nonzero_total = 0
    pfv_nonzero_ok = 0
    pfv_improve_total = 0
    pfv_improve_hit = 0
    pfv_neutral_total = 0
    pfv_neutral_false_improve = 0
    pfv_nonzero_cls_ok = 0
    pfv_nonzero_cls_hit = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(eval_indices), eval_batch):
            batch_idx = eval_indices[start : start + eval_batch]
            end = start + len(batch_idx)
            vh = torch.as_tensor(vh_np[batch_idx], dtype=torch.float32, device=device)
            va = torch.as_tensor(va_np[batch_idx], dtype=torch.float32, device=device)
            vr = torch.as_tensor(vr_np[batch_idx], dtype=torch.float32, device=device)
            vy = torch.as_tensor(vy_np[batch_idx], dtype=torch.float32, device=device)
            vd = torch.as_tensor(vd_np[batch_idx], dtype=torch.float32, device=device)
            pred = model(vh, va, vr, node_static, edge_index, action_node_map)
            pred_delta = pred["risk_delta"].float() * risk_delta_scale[None, :]
            diff = pred["pred_seq"] - vy
            depth_sse += float(torch.sum(diff * diff).detach().cpu())
            depth_count += int(diff.numel())
            delta_abs += float(torch.sum(torch.abs(pred_delta - vd)).detach().cpu())
            delta_count += int(vd.numel())
            pfv_dir_ok += int(((pred_delta[:, 0] < 0) == (vd[:, 0] < 0)).sum().detach().cpu())
            tfv_dir_ok += int(((pred_delta[:, 1] < 0) == (vd[:, 1] < 0)).sum().detach().cpu())
            peak_dir_ok += int(((pred_delta[:, 2] < 0) == (vd[:, 2] < 0)).sum().detach().cpu())
            safe_y = ((vd[:, 1] <= 0) & (vd[:, 2] <= 0)).float()
            safe_pred = (torch.sigmoid(pred["logits"][:, 1]) >= 0.5).float()
            safe_ok += int((safe_pred == safe_y).sum().detach().cpu())
            pfv_nonzero = torch.abs(vd[:, 0]) > 1.0
            pfv_improve = vd[:, 0] < -1.0
            pfv_neutral = torch.abs(vd[:, 0]) <= 1.0
            pfv_pred_improve = pred_delta[:, 0] < -1.0
            if pred["logits"].shape[1] >= 3:
                pfv_pred_nonzero = torch.sigmoid(pred["logits"][:, 2]) >= 0.5
            else:
                pfv_pred_nonzero = torch.abs(pred_delta[:, 0]) > 1.0
            pfv_nonzero_total += int(pfv_nonzero.sum().detach().cpu())
            pfv_nonzero_ok += int(((pred_delta[:, 0] < 0) == (vd[:, 0] < 0))[pfv_nonzero].sum().detach().cpu())
            pfv_nonzero_cls_ok += int((pfv_pred_nonzero == pfv_nonzero).sum().detach().cpu())
            pfv_nonzero_cls_hit += int((pfv_pred_nonzero & pfv_nonzero).sum().detach().cpu())
            pfv_improve_total += int(pfv_improve.sum().detach().cpu())
            pfv_improve_hit += int((pfv_pred_improve & pfv_improve).sum().detach().cpu())
            pfv_neutral_total += int(pfv_neutral.sum().detach().cpu())
            pfv_neutral_false_improve += int((pfv_pred_improve & pfv_neutral).sum().detach().cpu())
            total += int(end - start)
            del vh, va, vr, vy, vd, pred, pred_delta, diff, safe_y, safe_pred
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "val_horizon_depth_RMSE": float(np.sqrt(depth_sse / max(1, depth_count))),
        "val_delta_MAE": float(delta_abs / max(1, delta_count)),
        "PFV_direction_accuracy": float(pfv_dir_ok / max(1, total)),
        "TFV_direction_accuracy": float(tfv_dir_ok / max(1, total)),
        "peak_direction_accuracy": float(peak_dir_ok / max(1, total)),
        "safe_classification_accuracy": float(safe_ok / max(1, total)),
        "PFV_nonzero_direction_accuracy": float(pfv_nonzero_ok / max(1, pfv_nonzero_total)),
        "PFV_nonzero_class_accuracy": float(pfv_nonzero_cls_ok / max(1, total)),
        "PFV_nonzero_recall": float(pfv_nonzero_cls_hit / max(1, pfv_nonzero_total)),
        "PFV_improve_recall": float(pfv_improve_hit / max(1, pfv_improve_total)),
        "PFV_neutral_false_improve_rate": float(pfv_neutral_false_improve / max(1, pfv_neutral_total)),
        "PFV_nonzero_samples": int(pfv_nonzero_total),
        "PFV_improve_samples": int(pfv_improve_total),
        "val_samples_used": int(total),
    }


def _risk_scale_from_train(risk_delta_train: np.ndarray) -> np.ndarray:
    scales = []
    for j in range(risk_delta_train.shape[1]):
        x = np.asarray(risk_delta_train[:, j], dtype=np.float64)
        nz = np.abs(x[np.abs(x) > 1.0])
        if len(nz) >= 32:
            s = float(np.nanpercentile(nz, 95))
        else:
            s = float(np.nanpercentile(np.abs(x), 99))
        if not np.isfinite(s) or s < 1.0:
            s = 1.0
        scales.append(s)
    return np.asarray(scales, dtype=np.float32)


def _direction_pos_weights(risk_delta_train_scaled: np.ndarray) -> np.ndarray:
    y_pfv = risk_delta_train_scaled[:, 0] < 0
    y_safe = (risk_delta_train_scaled[:, 1] <= 0) & (risk_delta_train_scaled[:, 2] <= 0)
    y_pfv_nonzero = np.abs(risk_delta_train_scaled[:, 0]) > 1e-6
    weights = []
    for y in (y_pfv, y_safe, y_pfv_nonzero):
        pos = float(np.sum(y))
        neg = float(len(y) - pos)
        if pos < 1:
            weights.append(1.0)
        else:
            weights.append(min(100.0, max(1.0, neg / pos)))
    return np.asarray(weights, dtype=np.float32)


def _delta_distribution(name: str, risk_delta_arr: np.ndarray) -> dict:
    pfv = np.asarray(risk_delta_arr[:, 0], dtype=np.float64)
    tfv = np.asarray(risk_delta_arr[:, 1], dtype=np.float64)
    peak = np.asarray(risk_delta_arr[:, 2], dtype=np.float64)
    return {
        "split": name,
        "samples": int(len(pfv)),
        "pfv_improve": int(np.sum(pfv < -1.0)),
        "pfv_worse": int(np.sum(pfv > 1.0)),
        "pfv_zero": int(np.sum(np.abs(pfv) <= 1.0)),
        "pfv_improve_frac": float(np.mean(pfv < -1.0)),
        "pfv_nonzero_frac": float(np.mean(np.abs(pfv) > 1.0)),
        "tfv_improve_frac": float(np.mean(tfv < -1.0)),
        "tfv_worse_frac": float(np.mean(tfv > 1.0)),
        "peak_improve_frac": float(np.mean(peak < -1.0)),
        "peak_worse_frac": float(np.mean(peak > 1.0)),
    }


def _sample_weights(
    risk_delta_arr: np.ndarray,
    pfv_improve_weight: float,
    pfv_worse_weight: float,
    pfv_nonzero_weight: float,
) -> np.ndarray:
    pfv = np.asarray(risk_delta_arr[:, 0], dtype=np.float64)
    w = np.ones(len(pfv), dtype=np.float64)
    w[np.abs(pfv) > 1.0] *= max(1.0, float(pfv_nonzero_weight))
    w[pfv < -1.0] *= max(1.0, float(pfv_improve_weight))
    w[pfv > 1.0] *= max(1.0, float(pfv_worse_weight))
    if not np.all(np.isfinite(w)) or float(np.sum(w)) <= 0:
        w = np.ones(len(pfv), dtype=np.float64)
    return w


def _checkpoint_payload(
    model: PhysicsGuidedTemporalGraphSurrogate,
    opt: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_score: float,
    metrics: dict,
    cfg: dict,
    state: np.ndarray,
    action_seq: np.ndarray,
    node_ids: list[str],
    action_cols: list[str],
    node_static_np: np.ndarray,
    edge_index_np: np.ndarray,
    action_node_map_np: np.ndarray,
    risk_delta_scale_np: np.ndarray,
) -> dict:
    raw_model = getattr(model, "_orig_mod", model)
    return {
        "model": raw_model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "best_score": float(best_score),
        "metrics": dict(metrics),
        "n_nodes": int(state.shape[1]),
        "n_actions": int(action_seq.shape[2]),
        "static_dim": int(node_static_np.shape[1]),
        "horizon_steps": int(action_seq.shape[1]),
        "hidden_dim": int(cfg["training"]["hidden_dim"]),
        "gat_heads": int(cfg["training"].get("gat_heads", 4)),
        "node_ids": node_ids,
        "action_ids": action_cols,
        "node_static": node_static_np,
        "edge_index": edge_index_np,
        "action_node_map": action_node_map_np,
        "risk_delta_scale": risk_delta_scale_np,
    }


def _save_checkpoint(
    path: Path,
    model: PhysicsGuidedTemporalGraphSurrogate,
    opt: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_score: float,
    metrics: dict,
    cfg: dict,
    state: np.ndarray,
    action_seq: np.ndarray,
    node_ids: list[str],
    action_cols: list[str],
    node_static_np: np.ndarray,
    edge_index_np: np.ndarray,
    action_node_map_np: np.ndarray,
    risk_delta_scale_np: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            model,
            opt,
            scaler,
            epoch,
            best_score,
            metrics,
            cfg,
            state,
            action_seq,
            node_ids,
            action_cols,
            node_static_np,
            edge_index_np,
            action_node_map_np,
            risk_delta_scale_np,
        ),
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--val-max-samples", type=int, default=4096)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--no-full-val-at-end", action="store_true")
    ap.add_argument("--resume-checkpoint", default="")
    ap.add_argument("--no-pfv-oversample", action="store_true")
    ap.add_argument("--pfv-improve-sample-weight", type=float, default=50.0)
    ap.add_argument("--pfv-worse-sample-weight", type=float, default=20.0)
    ap.add_argument("--pfv-nonzero-sample-weight", type=float, default=8.0)
    ap.add_argument("--max-train-samples-per-epoch", type=int, default=49152)
    ap.add_argument("--use-dataloader", action="store_true")
    ap.add_argument("--compile-model", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dev = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    cache = load_npz(
        cfg_path(cfg, "outputs.cache") / "transition_cache.npz",
        keys=["state", "action_seq", "rain_seq", "target_seq", "risk_delta", "node_cols", "action_cols"],
    )
    state = _clean_float32(cache["state"])
    action_seq = _clean_float32(cache["action_seq"], nan=0.0, posinf=1.0, neginf=0.0)
    rain_seq = _clean_float32(cache["rain_seq"])
    target_seq = _clean_float32(cache["target_seq"])
    risk_delta = _clean_float32(cache["risk_delta"])
    node_cols = [str(x) for x in cache["node_cols"]]
    action_cols = [str(x) for x in cache["action_cols"]]
    node_ids = [c.split(":", 1)[1] for c in node_cols]
    node_static_np, edge_index_np, action_node_map_np = _build_graph_tensors(cfg, node_ids, action_cols)
    node_static = torch.tensor(node_static_np, dtype=torch.float32, device=dev)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=dev)
    action_node_map = torch.tensor(action_node_map_np, dtype=torch.float32, device=dev)
    priority_ids = set((cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines())
    priority_weight = np.ones(len(node_ids), dtype=np.float32)
    for i, n in enumerate(node_ids):
        if n in priority_ids:
            priority_weight[i] = float(cfg["training"].get("lambda_priority", 2.0))
    priority_weight_t = torch.tensor(priority_weight, dtype=torch.float32, device=dev)
    train_idx, val_idx = _train_val_indices(
        len(state),
        float(cfg["training"]["val_fraction"]),
        int(cfg["experiment"].get("random_seed", 2026)),
    )
    arrays = [state, action_seq, rain_seq, target_seq, risk_delta]
    train_risk_delta = risk_delta[train_idx]
    val_risk_delta = risk_delta[val_idx]
    split_report = pd.DataFrame([
        _delta_distribution("train", train_risk_delta),
        _delta_distribution("val", val_risk_delta),
        _delta_distribution("all", risk_delta),
    ])
    risk_delta_scale_np = _risk_scale_from_train(train_risk_delta)
    risk_delta_scale = torch.tensor(risk_delta_scale_np, dtype=torch.float32, device=dev)
    train_delta_scaled = train_risk_delta / risk_delta_scale_np[None, :]
    direction_pos_weight_np = _direction_pos_weights(train_delta_scaled)
    direction_pos_weight = torch.tensor(direction_pos_weight_np, dtype=torch.float32, device=dev)
    batch = int(args.batch_size or cfg["training"]["batch_size"])
    if args.batch_size <= 0 and dev.type == "cuda":
        batch = max(batch, 128)
    eval_batch = int(args.eval_batch_size or (64 if dev.type == "cuda" else min(batch, 64)))
    sample_weights_np = None
    if not args.no_pfv_oversample:
        sample_weights_np = _sample_weights(
            train_risk_delta,
            pfv_improve_weight=float(args.pfv_improve_sample_weight),
            pfv_worse_weight=float(args.pfv_worse_sample_weight),
            pfv_nonzero_weight=float(args.pfv_nonzero_sample_weight),
        )
    if args.use_dataloader:
        print("[train_surrogate] --use-dataloader ignored; using memory-safe numpy index batches")
    model = PhysicsGuidedTemporalGraphSurrogate(
        state.shape[1],
        action_seq.shape[2],
        node_static_np.shape[1],
        action_seq.shape[1],
        int(cfg["training"]["hidden_dim"]),
        int(cfg["training"].get("gat_heads", 4)),
    ).to(dev)
    if args.compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[train_surrogate] torch.compile enabled")
        except Exception as exc:
            print(f"[train_surrogate] torch.compile unavailable; continuing uncompiled: {exc}")
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))
    use_amp = dev.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    epochs = args.epochs or int(cfg["training"]["debug_epochs"])
    out = ensure_dir(cfg_path(cfg, "outputs.models"))
    diag = ensure_dir(cfg_path(cfg, "outputs.diagnostics"))
    split_report.to_csv(diag / "surrogate_delta_split_distribution.csv", index=False)
    latest_path = out / "graph_surrogate_latest.pt"
    best_path = out / "graph_surrogate_best.pt"
    final_path = out / "graph_surrogate.pt"
    interrupted_path = out / "graph_surrogate_interrupted.pt"
    start_epoch = 1
    best_score = math.inf
    history_rows = []
    if args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt:
                try:
                    scaler.load_state_dict(ckpt["scaler"])
                except Exception:
                    pass
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_score = float(ckpt.get("best_score", math.inf))
            print(f"[train_surrogate] resumed {ckpt_path} from epoch={start_epoch - 1} best_score={best_score:.6f}")
    print(
        f"[train_surrogate] samples={len(state)} train={len(train_idx)} val={len(val_idx)} "
        f"batch_size={batch} eval_batch_size={eval_batch} amp={use_amp} "
        f"num_workers={int(args.num_workers)} val_every={int(args.val_every)} "
        f"val_max_samples={int(args.val_max_samples)} save_every={int(args.save_every)} "
        f"max_train_samples_per_epoch={int(args.max_train_samples_per_epoch)} "
        f"fast_batcher=True "
        f"risk_delta_scale={risk_delta_scale_np.tolist()} "
        f"direction_pos_weight={direction_pos_weight_np.tolist()} "
        f"pfv_oversample={not args.no_pfv_oversample} device={dev}"
    )
    print(split_report.to_string(index=False))
    last_metrics = {}
    try:
        for ep in range(start_epoch, epochs + 1):
            losses = []
            model.train()
            batch_iter = _fast_numpy_batches(
                arrays,
                train_idx,
                batch,
                sample_weights_np if not args.no_pfv_oversample else None,
                int(args.max_train_samples_per_epoch),
                int(cfg["experiment"].get("random_seed", 2026)) + ep,
            )
            for h, aseq, rseq, yseq, d in batch_iter:
                h = h.to(dev, non_blocking=True)
                aseq = aseq.to(dev, non_blocking=True)
                rseq = rseq.to(dev, non_blocking=True)
                yseq = yseq.to(dev, non_blocking=True)
                d = d.to(dev, non_blocking=True)
                d_scaled = d / risk_delta_scale[None, :]
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(h, aseq, rseq, node_static, edge_index, action_node_map)
                pred_for_loss = {
                    "pred_seq": pred["pred_seq"].float(),
                    "risk_delta": pred["risk_delta"].float(),
                    "logits": pred["logits"].float(),
                }
                with torch.cuda.amp.autocast(enabled=False):
                    loss, _parts = temporal_surrogate_loss(
                        pred_for_loss,
                        yseq,
                        d_scaled.float(),
                        edge_index=edge_index,
                        priority_weight=priority_weight_t,
                        direction_pos_weight=direction_pos_weight,
                        lambda_volume=float(cfg["training"].get("lambda_volume", 0.10)),
                        lambda_flood=float(cfg["training"].get("lambda_flood", 0.20)),
                        lambda_direction=float(cfg["training"].get("lambda_direction", 0.30)),
                    )
                if not torch.isfinite(loss):
                    print("[train_surrogate] non-finite loss skipped")
                    continue
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(losses)) if losses else float("nan")
            should_validate = (ep == epochs) or (int(args.val_every) > 0 and ep % int(args.val_every) == 0)
            metrics = {"epoch": ep, "loss": train_loss}
            if should_validate:
                eval_metrics = _evaluate_in_batches(
                    model,
                    arrays,
                    val_idx,
                    node_static,
                    edge_index,
                    action_node_map,
                    dev,
                    eval_batch,
                    risk_delta_scale,
                    max_samples=int(args.val_max_samples),
                )
                direction_penalty = (
                    2000.0 * (1.0 - eval_metrics.get("PFV_nonzero_direction_accuracy", 0.0))
                    + 1000.0 * (1.0 - eval_metrics.get("peak_direction_accuracy", 0.0))
                )
                score = float(
                    eval_metrics["val_delta_MAE"]
                    + 100.0 * eval_metrics["val_horizon_depth_RMSE"]
                    + direction_penalty
                )
                best_marker = ""
                if score < best_score:
                    best_score = score
                    best_marker = " best"
                    _save_checkpoint(
                        best_path,
                        model,
                        opt,
                        scaler,
                        ep,
                        best_score,
                        eval_metrics,
                        cfg,
                        state,
                        action_seq,
                        node_ids,
                        action_cols,
                        node_static_np,
                        edge_index_np,
                        action_node_map_np,
                        risk_delta_scale_np,
                    )
                    _save_checkpoint(
                        final_path,
                        model,
                        opt,
                        scaler,
                        ep,
                        best_score,
                        eval_metrics,
                        cfg,
                        state,
                        action_seq,
                        node_ids,
                        action_cols,
                        node_static_np,
                        edge_index_np,
                        action_node_map_np,
                        risk_delta_scale_np,
                    )
                metrics.update(eval_metrics)
                metrics["score"] = score
                metrics["best_score"] = best_score
                last_metrics = dict(eval_metrics)
                print(
                    f"epoch={ep:03d} loss={train_loss:.6f} "
                    f"val_RMSE={eval_metrics['val_horizon_depth_RMSE']:.5f} "
                    f"val_delta_MAE={eval_metrics['val_delta_MAE']:.5f} "
                    f"PFV_dir={eval_metrics['PFV_direction_accuracy']:.3f} "
                    f"PFV_nz_dir={eval_metrics['PFV_nonzero_direction_accuracy']:.3f} "
                    f"PFV_recall={eval_metrics['PFV_improve_recall']:.3f} "
                    f"PFV_nz_recall={eval_metrics['PFV_nonzero_recall']:.3f} "
                    f"peak_dir={eval_metrics['peak_direction_accuracy']:.3f}{best_marker}"
                )
            else:
                print(f"epoch={ep:03d} loss={train_loss:.6f}")
            history_rows.append(metrics)
            should_save_latest = (ep == epochs) or (int(args.save_every) > 0 and ep % int(args.save_every) == 0)
            if should_save_latest:
                _save_checkpoint(
                    latest_path,
                    model,
                    opt,
                    scaler,
                    ep,
                    best_score,
                    last_metrics,
                    cfg,
                    state,
                    action_seq,
                    node_ids,
                    action_cols,
                    node_static_np,
                    edge_index_np,
                    action_node_map_np,
                    risk_delta_scale_np,
                )
            pd.DataFrame(history_rows).to_csv(diag / "surrogate_training_history.csv", index=False)
    except KeyboardInterrupt:
        _save_checkpoint(
            interrupted_path,
            model,
            opt,
            scaler,
            max(start_epoch - 1, ep if "ep" in locals() else 0),
            best_score,
            last_metrics,
            cfg,
            state,
            action_seq,
            node_ids,
            action_cols,
            node_static_np,
            edge_index_np,
            action_node_map_np,
            risk_delta_scale_np,
        )
        print(f"[train_surrogate] interrupted; saved {interrupted_path}")
        raise
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=dev, weights_only=False)
        model.load_state_dict(best_ckpt["model"])
        print(f"[train_surrogate] loaded best checkpoint for final report: {best_path}")
    metrics = _evaluate_in_batches(
        model,
        arrays,
        val_idx,
        node_static,
        edge_index,
        action_node_map,
        dev,
        eval_batch,
        risk_delta_scale,
        max_samples=0 if not args.no_full_val_at_end else int(args.val_max_samples),
    )
    metrics["horizon_steps"] = int(action_seq.shape[1])
    metrics["batch_size"] = int(batch)
    metrics["eval_batch_size"] = int(eval_batch)
    if not final_path.exists():
        _save_checkpoint(
            final_path,
            model,
            opt,
            scaler,
            epochs,
            best_score,
            metrics,
            cfg,
            state,
            action_seq,
            node_ids,
            action_cols,
            node_static_np,
            edge_index_np,
            action_node_map_np,
            risk_delta_scale_np,
        )
    pd.DataFrame([metrics]).to_csv(diag / "surrogate_report.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
