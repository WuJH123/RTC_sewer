from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from sewerrtc.io.project_paths import cfg_path, load_config, resolve_gat_model_path
from sewerrtc.models.gat_reconstructor import SparseGATReconstructor


def metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if true.size == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "NSE": np.nan, "n_values": 0}
    error = pred - true
    denominator = float(np.sum((true - np.mean(true)) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "NSE": float(1.0 - np.sum(error ** 2) / denominator) if denominator > 0 else np.nan,
        "n_values": int(true.size),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan_project6.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-events", type=int, default=30)
    ap.add_argument("--out-dir", default="outputs/audits")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_gat_model_path(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    node_ids = [str(value) for value in checkpoint["node_ids"]]
    static = torch.tensor(checkpoint["node_static"], dtype=torch.float32, device=device)
    edge_index = torch.tensor(checkpoint["edge_index"], dtype=torch.long, device=device)
    model = SparseGATReconstructor(
        int(checkpoint.get("n_nodes", len(node_ids))), int(checkpoint.get("static_dim", static.shape[1])),
        int(checkpoint.get("hidden_dim", 256)), int(checkpoint.get("gat_heads", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    node_index = {node: idx for idx, node in enumerate(node_ids)}
    sensors = pd.read_csv(cfg_path(cfg, "outputs.design") / "sensor_nodes.csv")["node_id"].astype(str).tolist()
    sensor_idx = [node_index[node] for node in sensors if node in node_index]
    mask = torch.zeros(len(node_ids), dtype=torch.float32, device=device); mask[sensor_idx] = 1.0
    priority = [x.strip() for x in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text().splitlines() if x.strip()]
    priority_idx = [node_index[node] for node in priority if node in node_index]
    nodes = pd.read_csv(cfg_path(cfg, "outputs.audit") / "node_table.csv")
    storage_nodes = nodes.loc[nodes["node_type"].astype(str).eq("storage"), "node_id"].astype(str).tolist()
    storage_idx = [node_index[node] for node in storage_nodes if node in node_index]
    actuators = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    neighbour_nodes = sorted(set(actuators["from_node"].astype(str)) | set(actuators["to_node"].astype(str)))
    neighbour_idx = [node_index[node] for node in neighbour_nodes if node in node_index]
    val_events = pd.read_csv(root / "outputs" / "surrogate_all109" / "horizon_surrogate_val_events.csv")["event_id"].astype(str).tolist()[:args.max_events]
    detail_dir = cfg_path(cfg, "outputs.data_bank_train") / "trajectories"
    group_true = {key: [] for key in ("priority", "storage", "actuator_neighbour", "high_risk")}
    group_pred = {key: [] for key in group_true}
    used = []
    for event_id in val_events:
        path = detail_dir / f"{event_id}__no_control_detail.csv"
        if not path.exists():
            continue
        detail = pd.read_csv(path, low_memory=False)
        true = np.zeros((len(detail), len(node_ids)), dtype=np.float32)
        for idx, node in enumerate(node_ids):
            column = f"h:{node}"
            if column in detail:
                true[:, idx] = pd.to_numeric(detail[column], errors="coerce").fillna(0.0).to_numpy(np.float32)
        rain = pd.to_numeric(detail.get("rainfall_mm_h", pd.Series(0.0, index=detail.index)), errors="coerce").fillna(0.0).to_numpy(np.float32)
        predictions = []
        with torch.no_grad():
            for start in range(0, len(detail), 64):
                observed = torch.tensor(true[start:start+64], dtype=torch.float32, device=device)
                rain_batch = torch.tensor(rain[start:start+64, None], dtype=torch.float32, device=device)
                predictions.append(model(observed * mask[None, :], mask, rain_batch, static, edge_index).cpu().numpy())
        pred = np.concatenate(predictions)
        for name, indices in (("priority", priority_idx), ("storage", storage_idx), ("actuator_neighbour", neighbour_idx)):
            if indices:
                group_true[name].append(true[:, indices]); group_pred[name].append(pred[:, indices])
        if priority_idx:
            high = np.max(true[:, priority_idx], axis=1) > 0.20
            if high.any():
                group_true["high_risk"].append(true[high]); group_pred["high_risk"].append(pred[high])
        used.append(event_id)
    records = []
    for name in group_true:
        true = np.concatenate(group_true[name]) if group_true[name] else np.empty((0,), dtype=float)
        pred = np.concatenate(group_pred[name]) if group_pred[name] else np.empty((0,), dtype=float)
        records.append({"group": name, **metrics(true, pred)})
    out = root / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "gat_local_accuracy.csv"; pd.DataFrame(records).to_csv(csv_path, index=False)
    report = {
        "path": str(csv_path), "checkpoint": str(checkpoint_path), "device": str(device),
        "events": used, "priority_nodes": len(priority_idx), "storage_nodes": len(storage_idx),
        "actuator_neighbour_nodes": len(neighbour_idx), "priority_sensor_overlap": len(set(priority) & set(sensors)),
        "metrics": records,
    }
    (out / "gat_local_accuracy_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
