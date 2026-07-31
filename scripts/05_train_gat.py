from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from sewerrtc.data.tensor_cache import load_npz
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.models.gat_reconstructor import SparseGATReconstructor


def _metrics(pred: np.ndarray, y: np.ndarray, prefix: str = "") -> dict:
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    denom = float(np.sum((y - y.mean()) ** 2))
    nse = float(1.0 - np.sum((pred - y) ** 2) / denom) if denom > 1e-9 else 0.0
    return {f"{prefix}RMSE": rmse, f"{prefix}MAE": mae, f"{prefix}NSE": nse}


def _build_static_and_edges(cfg: dict, node_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
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
    return static.astype(np.float32), edge_index


def _edge_smoothness(pred: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    return torch.mean(torch.abs(pred[:, src] - pred[:, dst]))


def _predict_in_batches(
    model: SparseGATReconstructor,
    val_x: np.ndarray,
    val_r: np.ndarray,
    mask_t: torch.Tensor,
    node_static: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    preds = []
    eval_batch = max(1, min(int(batch_size), 64))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(val_x), eval_batch):
            end = min(len(val_x), start + eval_batch)
            vx = torch.tensor(val_x[start:end], dtype=torch.float32, device=device)
            vr = torch.tensor(val_r[start:end], dtype=torch.float32, device=device)
            pred = model(vx, mask_t.expand(len(vx), -1), vr, node_static, edge_index)
            preds.append(pred.detach().cpu().numpy())
            del vx, vr, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return np.concatenate(preds, axis=0) if preds else np.zeros_like(val_x)


def _event_grouped_indices(event_ids: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    event_ids = np.asarray(event_ids).astype(str)
    events = np.asarray(sorted(set(event_ids.tolist())), dtype=object)
    rng = np.random.default_rng(int(seed))
    if len(events) < 2:
        idx = np.arange(len(event_ids))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * float(val_fraction))))
        return idx[n_val:], idx[:n_val], events.astype(str).tolist(), events.astype(str).tolist()
    rng.shuffle(events)
    n_val_events = min(len(events) - 1, max(1, int(round(len(events) * float(val_fraction)))))
    val_events = sorted(str(x) for x in events[:n_val_events])
    train_events = sorted(str(x) for x in events[n_val_events:])
    train_idx = np.flatnonzero(np.isin(event_ids, train_events))
    val_idx = np.flatnonzero(np.isin(event_ids, val_events))
    return train_idx, val_idx, train_events, val_events


def _add_unseen_metrics(metrics: dict, pred: np.ndarray, truth: np.ndarray, sensor_mask: np.ndarray, priority_indices: list[int]) -> None:
    unseen = np.flatnonzero(np.asarray(sensor_mask) < 0.5)
    if len(unseen):
        metrics.update(_metrics(pred[:, unseen], truth[:, unseen], "unsensed_"))
    priority_unseen = [i for i in priority_indices if i in set(unseen.tolist())]
    if priority_unseen:
        metrics.update(_metrics(pred[:, priority_unseen], truth[:, priority_unseen], "priority_unsensed_"))
    metrics["unsensed_node_count"] = int(len(unseen))
    metrics["priority_unsensed_node_count"] = int(len(priority_unseen))


def _load_verified_initial_checkpoint(
    path: str | Path,
    model: SparseGATReconstructor,
    *,
    node_ids: list[str],
    node_static: np.ndarray,
    edge_index: np.ndarray,
    sensor_count: int,
    hidden_dim: int,
    gat_heads: int,
) -> None:
    """Load a GAT only when its graph and sensor contract match this run."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"GAT initialization checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checks = {
        "node_ids": list(map(str, checkpoint.get("node_ids", []))) == list(map(str, node_ids)),
        "n_nodes": int(checkpoint.get("n_nodes", -1)) == len(node_ids),
        "static_dim": int(checkpoint.get("static_dim", -1)) == int(node_static.shape[1]),
        "hidden_dim": int(checkpoint.get("hidden_dim", -1)) == int(hidden_dim),
        "gat_heads": int(checkpoint.get("gat_heads", -1)) == int(gat_heads),
        "sensor_count": int(checkpoint.get("sensor_count", -1)) == int(sensor_count),
        "node_static": np.array_equal(np.asarray(checkpoint.get("node_static")), np.asarray(node_static)),
        "edge_index": np.array_equal(np.asarray(checkpoint.get("edge_index")), np.asarray(edge_index)),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"GAT initialization checkpoint is incompatible with this run: {failed}")
    model.load_state_dict(checkpoint["model"], strict=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-train-samples-per-epoch", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--score-full-weight", type=float, default=-1.0)
    ap.add_argument("--score-priority-weight", type=float, default=-1.0)
    ap.add_argument("--init-checkpoint", default="", help="Verified compatible GAT checkpoint used for warm start or evaluation.")
    ap.add_argument("--eval-only", action="store_true", help="Evaluate and re-save --init-checkpoint without further optimization.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dev = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    cache = load_npz(cfg_path(cfg, "outputs.cache") / "transition_cache.npz", keys=["state", "rain", "node_cols", "event_ids"])
    state = cache["state"].astype(np.float32)
    rain = cache["rain"].astype(np.float32)
    node_cols = [str(x) for x in cache["node_cols"]]
    node_ids = [c.split(":", 1)[1] for c in node_cols]
    node_static_np, edge_index_np = _build_static_and_edges(cfg, node_ids)
    node_static = torch.tensor(node_static_np, dtype=torch.float32, device=dev)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=dev)
    sensors = pd.read_csv(cfg_path(cfg, "outputs.design") / "sensor_nodes.csv")
    sensor_ids = set(sensors["node_id"].astype(str))
    mask = np.zeros(len(node_ids), dtype=np.float32)
    for i, n in enumerate(node_ids):
        if n in sensor_ids:
            mask[i] = 1.0
    priority_ids = set((cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines())
    domain_path = cfg_path(cfg, "outputs.design") / "priority_domain_nodes.txt"
    if domain_path.exists():
        priority_domain_ids = set(domain_path.read_text(encoding="utf-8").splitlines())
    else:
        priority_domain_ids = set(priority_ids)
    priority_weight = np.ones(len(node_ids), dtype=np.float32)
    priority_domain_mask = np.zeros(len(node_ids), dtype=np.float32)
    for i, n in enumerate(node_ids):
        if n in priority_domain_ids:
            priority_weight[i] = float(cfg["training"].get("lambda_priority", 2.0))
            priority_domain_mask[i] = 1.0
    split_seed = int(cfg["experiment"].get("random_seed", 2026))
    train_idx, val_idx, train_events, val_events = _event_grouped_indices(
        cache["event_ids"], float(cfg["training"]["val_fraction"]), split_seed
    )
    train_x, val_x = state[train_idx], state[val_idx]
    train_r, val_r = rain[train_idx], rain[val_idx]
    max_val_samples = int(args.max_val_samples or cfg["training"].get("gat_max_val_samples", 0) or 0)
    if max_val_samples and len(val_x) > max_val_samples:
        val_rng = np.random.default_rng(split_seed + 31)
        val_take = np.sort(val_rng.choice(len(val_x), size=max_val_samples, replace=False))
        val_x, val_r = val_x[val_take], val_r[val_take]
    batch = int(cfg["training"]["batch_size"])
    epochs = args.epochs or int(cfg["training"]["debug_epochs"])
    base_ds = TensorDataset(torch.tensor(train_x), torch.tensor(train_r))
    model = SparseGATReconstructor(
        len(node_ids),
        node_static_np.shape[1],
        int(cfg["training"]["hidden_dim"]),
        int(cfg["training"].get("gat_heads", 4)),
    ).to(dev)
    if args.init_checkpoint:
        _load_verified_initial_checkpoint(
            args.init_checkpoint,
            model,
            node_ids=node_ids,
            node_static=node_static_np,
            edge_index=edge_index_np,
            sensor_count=int(mask.sum()),
            hidden_dim=int(cfg["training"]["hidden_dim"]),
            gat_heads=int(cfg["training"].get("gat_heads", 4)),
        )
    if args.eval_only:
        if not args.init_checkpoint:
            raise ValueError("--eval-only requires --init-checkpoint")
        epochs = 0
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))
    mask_t = torch.tensor(mask[None, :], dtype=torch.float32, device=dev)
    weight_t = torch.tensor(priority_weight[None, :], dtype=torch.float32, device=dev)
    unsensed_loss_weight = float(cfg["training"].get("lambda_unsensed", 1.0))
    unsensed_weight_t = 1.0 + max(0.0, unsensed_loss_weight - 1.0) * (1.0 - mask_t)
    domain_mask_t = torch.tensor(priority_domain_mask[None, :], dtype=torch.float32, device=dev)
    high_depth_threshold = float((cfg.get("risk_stratification", {}) or {}).get("high_priority_depth_threshold_m", 0.20))
    high_depth_weight = float(cfg["training"].get("lambda_high_depth", 0.0))
    max_train_samples = int(args.max_train_samples_per_epoch or cfg["training"].get("gat_max_train_samples_per_epoch", 0) or 0)
    eval_every = int(args.eval_every or cfg["training"].get("gat_eval_every", 0) or 0)
    patience = int(args.patience or cfg["training"].get("gat_patience", 0) or 0)
    full_score_weight = float(args.score_full_weight)
    if full_score_weight < 0:
        full_score_weight = float(cfg["training"].get("gat_score_full_weight", 0.45))
    priority_score_weight = float(args.score_priority_weight)
    if priority_score_weight < 0:
        priority_score_weight = float(cfg["training"].get("gat_score_priority_weight", 0.55))
    score_norm = max(1e-6, full_score_weight + priority_score_weight)
    full_score_weight /= score_norm
    priority_score_weight /= score_norm
    train_rng = np.random.default_rng(int(cfg["experiment"].get("random_seed", 2026)) + 17)
    sensor_dropout = float(cfg["training"].get("sensor_dropout", 0.10))
    out = ensure_dir(cfg_path(cfg, "outputs.models"))
    diag = ensure_dir(cfg_path(cfg, "outputs.diagnostics"))
    pd.DataFrame({"event_id": train_events}).to_csv(diag / "gat_train_events.csv", index=False)
    pd.DataFrame({"event_id": val_events}).to_csv(diag / "gat_val_events.csv", index=False)

    def save_checkpoint(metrics: dict, epoch: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "n_nodes": len(node_ids),
                "static_dim": int(node_static_np.shape[1]),
                "hidden_dim": int(cfg["training"]["hidden_dim"]),
                "gat_heads": int(cfg["training"].get("gat_heads", 4)),
                "node_ids": node_ids,
                "node_static": node_static_np,
                "edge_index": edge_index_np,
                "best_epoch": int(epoch),
                "sensor_count": int(mask.sum()),
                "unsensed_loss_weight": float(unsensed_loss_weight),
                "priority_domain_sensor_count": int(sum(1 for i, n in enumerate(node_ids) if n in priority_domain_ids and mask[i] > 0)),
                "metrics": metrics,
                "split_strategy": "event_id_grouped",
                "train_events": train_events,
                "val_events": val_events,
            },
            out / "gat_sr0p10.pt",
        )

    best_score = -float("inf")
    best_metrics: dict | None = None
    best_epoch = 0
    stale_evals = 0
    for ep in range(1, epochs + 1):
        if max_train_samples and max_train_samples < len(base_ds):
            sub_idx = train_rng.choice(len(base_ds), size=max_train_samples, replace=False).tolist()
            ds = Subset(base_ds, sub_idx)
        else:
            ds = base_ds
        dl = DataLoader(ds, batch_size=batch, shuffle=True)
        model.train()
        losses = []
        for h, r in dl:
            h, r = h.to(dev), r.to(dev)
            batch_mask = mask_t.expand(len(h), -1)
            if sensor_dropout > 0.0:
                keep = (torch.rand_like(batch_mask) >= sensor_dropout).float()
                batch_mask = batch_mask * keep
            pred = model(h, batch_mask, r, node_static, edge_index)
            per_node_loss = torch.nn.functional.smooth_l1_loss(pred, h, reduction="none")
            combined_weight = weight_t * unsensed_weight_t
            recon_loss = torch.sum(per_node_loss * combined_weight) / torch.clamp(
                combined_weight.expand_as(per_node_loss).sum(), min=1.0
            )
            high_mask = ((h >= high_depth_threshold).float() * domain_mask_t).detach()
            high_den = torch.clamp(high_mask.sum(), min=1.0)
            high_loss = torch.sum(per_node_loss * high_mask) / high_den
            smooth = _edge_smoothness(pred, edge_index)
            loss = recon_loss + high_depth_weight * high_loss + 0.005 * smooth
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        msg = f"epoch={ep:03d} loss={np.mean(losses):.6f}"
        should_eval = (eval_every > 0 and ep % eval_every == 0) or ep == epochs
        if should_eval:
            pred = _predict_in_batches(model, val_x, val_r, mask_t, node_static, edge_index, dev, batch)
            metrics = _metrics(pred, val_x)
            pr_idx = [i for i, n in enumerate(node_ids) if n in priority_ids]
            _add_unseen_metrics(metrics, pred, val_x, mask, pr_idx)
            if pr_idx:
                metrics.update(_metrics(pred[:, pr_idx], val_x[:, pr_idx], "priority_"))
                threshold = np.quantile(val_x[:, pr_idx], 0.90)
                hit = ((pred[:, pr_idx] >= threshold) & (val_x[:, pr_idx] >= threshold)).sum()
                miss_den = max(1, (val_x[:, pr_idx] >= threshold).sum())
                false = ((pred[:, pr_idx] >= threshold) & (val_x[:, pr_idx] < threshold)).sum()
                false_den = max(1, (val_x[:, pr_idx] < threshold).sum())
                metrics["priority_high_depth_hit_rate"] = float(hit / miss_den)
                metrics["priority_high_depth_false_alarm_rate"] = float(false / false_den)
            metrics["epoch"] = int(ep)
            metrics["sensor_count"] = int(mask.sum())
            metrics["priority_domain_sensor_count"] = int(sum(1 for i, n in enumerate(node_ids) if n in priority_domain_ids and mask[i] > 0))
            full_nse = float(metrics.get("unsensed_NSE", metrics.get("NSE", 0.0)))
            priority_nse = float(metrics.get("priority_NSE", full_nse))
            score = full_score_weight * full_nse + priority_score_weight * priority_nse
            metrics["gat_selection_score"] = float(score)
            metrics["gat_score_full_weight"] = float(full_score_weight)
            metrics["gat_score_priority_weight"] = float(priority_score_weight)
            if score > best_score:
                best_score = score
                best_metrics = dict(metrics)
                best_epoch = ep
                stale_evals = 0
                save_checkpoint(metrics, ep)
                msg += (
                    f" NSE={metrics.get('NSE', float('nan')):.4f}"
                    f" priority_NSE={metrics.get('priority_NSE', float('nan')):.4f}"
                    f" score={score:.4f} best"
                )
            else:
                stale_evals += 1
                msg += (
                    f" NSE={metrics.get('NSE', float('nan')):.4f}"
                    f" priority_NSE={metrics.get('priority_NSE', float('nan')):.4f}"
                    f" score={score:.4f}"
                )
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        print(msg)
        if patience and stale_evals >= patience:
            print(f"[train_gat] early_stop epoch={ep} best_epoch={best_epoch} best_score={best_score:.6f}")
            break
    if best_metrics is None:
        pred = _predict_in_batches(model, val_x, val_r, mask_t, node_static, edge_index, dev, batch)
        best_metrics = _metrics(pred, val_x)
        pr_idx = [i for i, n in enumerate(node_ids) if n in priority_ids]
        _add_unseen_metrics(best_metrics, pred, val_x, mask, pr_idx)
        if pr_idx:
            best_metrics.update(_metrics(pred[:, pr_idx], val_x[:, pr_idx], "priority_"))
        best_metrics["epoch"] = int(epochs)
        best_metrics["sensor_count"] = int(mask.sum())
        best_metrics["priority_domain_sensor_count"] = int(sum(1 for i, n in enumerate(node_ids) if n in priority_domain_ids and mask[i] > 0))
        full_nse = float(best_metrics.get("unsensed_NSE", best_metrics.get("NSE", 0.0)))
        priority_nse = float(best_metrics.get("priority_NSE", full_nse))
        best_metrics["gat_selection_score"] = float(full_score_weight * full_nse + priority_score_weight * priority_nse)
        best_metrics["gat_score_full_weight"] = float(full_score_weight)
        best_metrics["gat_score_priority_weight"] = float(priority_score_weight)
        best_epoch = int(epochs)
        save_checkpoint(best_metrics, best_epoch)
    pd.DataFrame([best_metrics]).to_csv(diag / "gat_reconstruction_report.csv", index=False)
    print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
