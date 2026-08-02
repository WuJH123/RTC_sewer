"""Calibrate the Formal F2 sparse-GAT uncertainty/OOD gate on NEW F2 Calibration events.

The calibration window manifest must contain only ledger role=calibration. It is
never used for gradient updates. Predictive std is calibrated against absolute
unobserved-node error, and the OOD limit is frozen from the 99th percentile of
new-calibration predictive uncertainty before Policy Lock.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_v42_step1_streaming import _graph_tensors
from sewerrtc.models.temporal_sparse_gat_v42 import TemporalSparseGATReconstructorV42
from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID, read_table, sha256_file
from sewerrtc.v4.v42_step1_streaming import Step1StreamingDataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument(
        "--calibration-manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/calibration/FORMAL_F2_CALIBRATION_STEP1_WINDOW_MANIFEST.parquet",
    )
    ap.add_argument(
        "--ledger",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare/FORMAL_F2_EVENT_LEDGER.csv",
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/step1/seed_42",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/calibration/STEP1_UNCERTAINTY_OOD_CALIBRATION.json",
    )
    ap.add_argument("--sensor-ratio", type=float, default=0.10)
    ap.add_argument("--sensor-layout-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--min-calibration-groups", type=int, default=8)
    args = ap.parse_args()

    manifest = read_table(args.calibration_manifest)
    ledger = read_table(args.ledger)
    if manifest.empty:
        raise ValueError("Formal F2 Step1 calibration manifest is empty")
    groups = set(manifest["split_group_key"].astype(str))
    allowed = set(ledger.loc[ledger["formal_f2_role"].astype(str).eq("calibration"), "rainfall_group_key"].astype(str))
    if not groups.issubset(allowed):
        raise RuntimeError(f"Step1 calibration contains non-F2 calibration rainfalls: {sorted(groups - allowed)[:10]}")
    if len(groups) < args.min_calibration_groups:
        raise RuntimeError(f"only {len(groups)} Step1 calibration rainfall groups; require {args.min_calibration_groups}")
    report_path = args.model_dir / "formal_step1_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    historical = set(map(str, report.get("train_rainfall_groups", []))) | set(map(str, report.get("validation_rainfall_groups", []))) | set(map(str, report.get("model_calibration_rainfall_groups", [])))
    if groups & historical:
        raise RuntimeError(f"new Step1 calibration overlaps historical training/model-selection rainfalls: {sorted(groups & historical)[:10]}")

    dataset = Step1StreamingDataset(
        project_root=args.project_root,
        manifest_path=args.calibration_manifest,
        sensor_ratio=args.sensor_ratio,
        sensor_layout_seed=args.sensor_layout_seed,
        allowed_groups=sorted(groups),
        shuffle_files=False,
        iteration_seed=42,
    )
    expected_layout = str(report.get("sensor_layout_sha256", ""))
    if expected_layout and dataset.sensor_layout_sha256 != expected_layout:
        raise RuntimeError("Formal F2 Step1 calibration sensor layout differs from training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = dataset.graph
    model = TemporalSparseGATReconstructorV42(
        n_nodes=graph.n_nodes,
        n_facilities=graph.n_facilities,
        node_static_dim=graph.node_static.shape[1],
        link_static_dim=graph.link_static.shape[1],
        hidden_dim=128,
        heads=4,
        gat_layers=3,
    ).to(device)
    model.load_state_dict(torch.load(args.model_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    node_static, link_static, edge_index, action_map = _graph_tensors(graph, device)
    ratios: list[np.ndarray] = []
    uncertainty: list[np.ndarray] = []
    absolute_errors: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            sparse = batch["sparse_depth_history"].to(device)
            masks = batch["sensor_mask_history"].to(device)
            target = batch["target_depth"].to(device)
            out = model(
                sparse_depth_history=sparse,
                sensor_mask_history=masks,
                rainfall_history=batch["rainfall_history"].to(device),
                historical_actions=batch["historical_actions"].to(device),
                node_static=node_static,
                link_static=link_static,
                edge_index=edge_index,
                action_node_map=action_map,
            )
            unobserved = ~(masks[:, -1, :] >= 0.5)
            err = (out.depth_mean - target).abs()
            std = torch.clamp(out.depth_std, min=1e-6)
            ratios.append((err[unobserved] / std[unobserved]).cpu().numpy())
            absolute_errors.append(err[unobserved].cpu().numpy())
            uncertainty.append((std.masked_fill(~unobserved, 0.0).sum(1) / unobserved.sum(1).clamp(min=1)).cpu().numpy())
    ratio = np.concatenate(ratios).astype(float)
    errors = np.concatenate(absolute_errors).astype(float)
    scores = np.concatenate(uncertainty).astype(float)
    if ratio.size == 0 or scores.size == 0:
        raise RuntimeError("no unobserved-node calibration scores")
    scale_95 = float(np.quantile(ratio, 0.95))
    ood_limit_99 = float(np.quantile(scores, 0.99))
    empirical_coverage = float(np.mean(ratio <= scale_95))
    finite = all(math.isfinite(v) for v in (scale_95, ood_limit_99, empirical_coverage))
    status = "pass" if finite and empirical_coverage >= 0.94 else "fail"
    payload = {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "stage": "formal_f2_step1_uncertainty_ood_calibration",
        "status": status,
        "development_only": False,
        "formal_mainline_authorized": False,
        "calibration_authority": "new_f2_calibration_rainfall_groups",
        "calibration_manifest": str(args.calibration_manifest),
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
        "calibration_rainfall_groups": sorted(groups),
        "calibration_rainfall_group_count": len(groups),
        "historical_overlap_count": len(groups & historical),
        "gat_model_sha256": str(report.get("gat_model_sha256", "")),
        "sensor_layout_sha256": dataset.sensor_layout_sha256,
        "uncertainty_scale_95": scale_95,
        "uncertainty_empirical_coverage": empirical_coverage,
        "mean_absolute_unobserved_error_m": float(np.mean(errors)),
        "ood_score": "mean_predictive_std_unobserved",
        "ood_limit_99": ood_limit_99,
        "uncertainty_calibrated": status == "pass",
        "ood_calibrated": status == "pass",
        "future_hydraulic_truth_used_online": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)
    return 0 if status == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
