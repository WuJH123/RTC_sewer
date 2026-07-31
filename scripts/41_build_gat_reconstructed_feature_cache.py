from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from sewerrtc.data.dataset_fingerprint import source_file_fingerprint
from sewerrtc.data.gat_feature_cache import gat_feature_cache_path
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config, resolve_gat_model_path
from sewerrtc.models.gat_reconstructor import SparseGATReconstructor


def _load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    node_ids = [str(x) for x in ckpt["node_ids"]]
    static = torch.tensor(ckpt["node_static"], dtype=torch.float32, device=device)
    edge_index = torch.tensor(ckpt["edge_index"], dtype=torch.long, device=device)
    model = SparseGATReconstructor(
        int(ckpt.get("n_nodes", len(node_ids))),
        int(ckpt.get("static_dim", static.shape[1])),
        int(ckpt.get("hidden_dim", 256)),
        int(ckpt.get("gat_heads", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, node_ids, static, edge_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan_project6.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--out-dir", default="", help="Isolated feature-cache directory for this model/action schema.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    model_path = resolve_gat_model_path(cfg)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained GAT checkpoint: {model_path}")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested for GAT feature cache but unavailable")
    model, node_ids, static, edge_index = _load_model(model_path, device)
    node_index = {node: i for i, node in enumerate(node_ids)}
    sensors = pd.read_csv(cfg_path(cfg, "outputs.design") / "sensor_nodes.csv")["node_id"].astype(str).tolist()
    priority = (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines()
    sensor_idx = [node_index[x] for x in sensors if x in node_index]
    priority_idx = [node_index[x] for x in priority if x in node_index]
    if not sensor_idx or not priority_idx:
        raise ValueError("Sensor or priority nodes do not overlap the GAT node order")
    mask = torch.zeros(len(node_ids), dtype=torch.float32, device=device)
    mask[sensor_idx] = 1.0

    detail_dir = cfg_path(cfg, "outputs.data_bank_train") / "trajectories"
    files = sorted(detail_dir.glob("*_detail.csv"))
    rainfall_table = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv")
    allowed_events = set(rainfall_table["event_id"].astype(str))
    files = [p for p in files if p.name.rpartition("__")[0] in allowed_events]
    schedule_path = cfg_path(cfg, "outputs.data_bank_train") / "trajectory_schedule.csv"
    if schedule_path.exists():
        schedule = pd.read_csv(schedule_path)
        allowed_keys = set(zip(schedule["event_id"].astype(str), schedule["policy_id"].astype(str)))
        files = [
            p for p in files
            if (lambda parts: bool(parts[1]) and (parts[0], parts[2]) in allowed_keys)(
                p.stem.removesuffix("_detail").rpartition("__")
            )
        ]
    if args.max_files:
        files = files[: int(args.max_files)]
    configured_cache = (cfg.get("outputs", {}) or {}).get("gat_features", "outputs/gat_reconstructed_features")
    cache_dir = ensure_dir(Path(args.out_dir) if args.out_dir else root / configured_cache)
    gat_fingerprint = source_file_fingerprint([model_path])
    completed = skipped = 0
    for i, path in enumerate(files, start=1):
        out = gat_feature_cache_path(cache_dir, path)
        source_fingerprint = source_file_fingerprint([path])
        if args.resume and out.exists():
            try:
                old = np.load(out, allow_pickle=False)
                if str(old["source_fingerprint"].item()) == source_fingerprint and str(old["gat_fingerprint"].item()) == gat_fingerprint:
                    skipped += 1
                    continue
            except Exception:
                pass
        detail = pd.read_csv(path)
        n = len(detail)
        true = np.zeros((n, len(node_ids)), dtype=np.float32)
        for j, node in enumerate(node_ids):
            col = f"h:{node}"
            if col in detail:
                true[:, j] = pd.to_numeric(detail[col], errors="coerce").fillna(0.0).to_numpy(np.float32)
        rain = pd.to_numeric(detail.get("rainfall_mm_h", pd.Series(0.0, index=detail.index)), errors="coerce").fillna(0.0).to_numpy(np.float32)
        reconstructed = []
        with torch.no_grad():
            for start in range(0, n, max(1, int(args.batch_size))):
                observed = torch.tensor(true[start : start + args.batch_size], dtype=torch.float32, device=device)
                sparse = observed * mask[None, :]
                rain_batch = torch.tensor(rain[start : start + args.batch_size, None], dtype=torch.float32, device=device)
                pred = model(sparse, mask, rain_batch, static, edge_index)
                reconstructed.append(pred.detach().cpu().numpy())
        recon = np.concatenate(reconstructed, axis=0) if reconstructed else np.empty((0, len(node_ids)), dtype=np.float32)
        priority_recon = recon[:, priority_idx]
        np.savez_compressed(
            out,
            source_fingerprint=np.asarray(source_fingerprint),
            gat_fingerprint=np.asarray(gat_fingerprint),
            row_count=np.asarray(n, dtype=np.int64),
            current_depth_mean=np.mean(recon, axis=1).astype(np.float32),
            current_depth_p95=np.quantile(recon, 0.95, axis=1).astype(np.float32),
            current_depth_max=np.max(recon, axis=1).astype(np.float32),
            priority_depth_mean=np.mean(priority_recon, axis=1).astype(np.float32),
            priority_depth_max=np.max(priority_recon, axis=1).astype(np.float32),
        )
        completed += 1
        if i % 100 == 0 or i == len(files):
            print(f"[gat_feature_cache] {i}/{len(files)} completed={completed} skipped={skipped}")
    report = {
        "files": len(files),
        "completed": completed,
        "skipped": skipped,
        "device": str(device),
        "gat_model": str(model_path),
        "gat_fingerprint": gat_fingerprint,
        "cache_dir": str(cache_dir),
    }
    (cache_dir / "gat_feature_cache_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
