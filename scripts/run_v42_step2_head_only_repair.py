"""Bounded seed-42 candidate-relative Step-2 head repair.

Uses existing authoritative PFV/TFV labels only.  The hydraulic surrogate is
frozen; this script trains small PFV/TFV action-effect heads and compares them
with the untouched seed-42 model.  It never starts SWMM and never consumes
Calibration/Challenge/Locked rainfall groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_v42_step2_fast import _forward, _graph_indices, _tensorise
from scripts.train_v42_step2_formal_f2 import _add_causal_dynamic_internal_input, _rank
from sewerrtc.control.authoritative_control_metrics_v42 import action_sha256
from sewerrtc.v4.models_v42.hydraulic_multi_reference import MultiReferenceHydraulicSurrogate
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology, build_surrogate_action_node_map


def _arr(value: object) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=np.float32)


def _pair_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    values = []
    for i, j in combinations(range(len(actual)), 2):
        da = float(actual[j] - actual[i])
        dp = float(predicted[j] - predicted[i])
        if abs(da) > 1.0e-8 and abs(dp) > 1.0e-8:
            values.append(np.sign(da) == np.sign(dp))
    return float(np.mean(values)) if values else None


def _safe_spearman(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(actual) < 3 or np.ptp(actual) == 0.0 or np.ptp(predicted) == 0.0:
        return None
    value = spearmanr(actual, predicted).statistic
    return float(value) if np.isfinite(value) else None


def _top5_recall(g: np.ndarray, tfv: np.ndarray, pg: np.ndarray, ptfv: np.ndarray) -> float | None:
    safe = np.flatnonzero(g <= 100.0)
    pred_safe = np.flatnonzero(pg <= 100.0)
    if not len(safe) or not len(pred_safe):
        return None
    truth_top = safe[np.argsort(tfv[safe])[:5]]
    pred_top = pred_safe[np.argsort(ptfv[pred_safe])[:5]]
    return float(len(set(truth_top.tolist()) & set(pred_top.tolist())) / min(5, len(truth_top)))


def _metrics(features: torch.Tensor, labels: pd.DataFrame, prediction: np.ndarray) -> dict[str, object]:
    g = labels["pfv_budget_metric_m3"].to_numpy(float)
    tfv = labels["tfv_delta_m3"].to_numpy(float)
    group_rows = []
    for state, positions in labels.groupby("state_key", sort=True).groups.items():
        idx = np.asarray(list(positions), dtype=int)
        if len(idx) < 3:
            continue
        group_rows.append({
            "state_key": str(state),
            "rows": int(len(idx)),
            "pfv_spearman": _safe_spearman(g[idx], prediction[idx, 0]),
            "tfv_spearman": _safe_spearman(tfv[idx], prediction[idx, 1]),
            "pfv_pairwise": _pair_accuracy(g[idx], prediction[idx, 0]),
            "tfv_pairwise": _pair_accuracy(tfv[idx], prediction[idx, 1]),
            "top5_recall": _top5_recall(g[idx], tfv[idx], prediction[idx, 0], prediction[idx, 1]),
        })
    def med(key: str) -> float | None:
        values = [float(row[key]) for row in group_rows if row.get(key) is not None and np.isfinite(row[key])]
        return float(np.median(values)) if values else None
    return {
        "rows": int(len(labels)),
        "states": int(labels.state_key.nunique()),
        "pfv_mae": float(np.mean(np.abs(prediction[:, 0] - g))),
        "tfv_mae": float(np.mean(np.abs(prediction[:, 1] - tfv))),
        "median_pfv_spearman": med("pfv_spearman"),
        "median_tfv_spearman": med("tfv_spearman"),
        "median_pfv_pairwise": med("pfv_pairwise"),
        "median_tfv_pairwise": med("tfv_pairwise"),
        "median_top5_recall": med("top5_recall"),
        "state_metrics": group_rows,
    }


class CandidateRelativeHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _pair_loss(pred: torch.Tensor, target: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rank_terms = []
    direction_terms = []
    for group in torch.unique(state):
        idx = torch.nonzero(state == group, as_tuple=False).reshape(-1)
        if idx.numel() < 2:
            continue
        for k in range(2):
            p = pred.index_select(0, idx)[:, k]
            y = target.index_select(0, idx)[:, k]
            scale = y.detach().abs().median().clamp_min(1.0)
            diff_y = y[None, :] - y[:, None]
            diff_p = p[None, :] - p[:, None]
            upper = torch.triu(torch.ones_like(diff_y, dtype=torch.bool), diagonal=1)
            informative = upper & (diff_y.abs() > 1.0e-6)
            if bool(informative.any()):
                signed = diff_y[informative].sign()
                rank_terms.append(nn.functional.softplus(-signed * diff_p[informative] / scale).mean())
                direction_terms.append(nn.functional.smooth_l1_loss(diff_p[informative] / scale, diff_y[informative] / scale))
    zero = pred.sum() * 0.0
    return (torch.stack(rank_terms).mean() if rank_terms else zero, torch.stack(direction_terms).mean() if direction_terms else zero)


def _loss_scale_audit(labels: pd.DataFrame) -> dict[str, float]:
    g = labels["pfv_budget_metric_m3"].to_numpy(float)
    t = labels["tfv_delta_m3"].to_numpy(float)
    return {
        "pfv_median_abs_m3": float(np.median(np.abs(g))),
        "tfv_median_abs_m3": float(np.median(np.abs(t))),
        "pfv_iqr_m3": float(np.subtract(*np.percentile(g, [75, 25]))),
        "tfv_iqr_m3": float(np.subtract(*np.percentile(t, [75, 25]))),
        "regression_normalization": "per-target median absolute magnitude, clamped at 1 m3",
        "ranking_weight": 0.5,
        "direction_weight": 0.5,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--experience-bank", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    required = [
        "state_key", "split_group_key", "history_depth", "history_actions_readback", "rainfall_forecast",
        "action_candidate_readback", "action_no_control_readback", "action_dynamic_internal_readback",
        "action_hold_previous_readback", "source_detail_path_dynamic_internal", "checkpoint_min",
        "candidate_action_sha256",
        "pfv_delta", "tfv_delta", "peak_delta",
    ]
    for branch in ("candidate", "no_control", "dynamic_internal", "hold_previous"):
        required.extend([f"trajectory_depth_{branch}", f"trajectory_flood_{branch}"])
    manifest = pd.read_parquet(args.manifest, columns=required)
    manifest = _add_causal_dynamic_internal_input(manifest, args.project_root)
    bank = pd.read_parquet(args.experience_bank, columns=[
        "state_key", "candidate_action_sha256", "canonical_candidate_action_sha256",
        "pfv_budget_metric_m3", "tfv_candidate_m3", "tfv_internal_m3",
    ])
    bank["action_key"] = bank["canonical_candidate_action_sha256"].fillna(bank["candidate_action_sha256"]).astype(str)
    lookup = bank.drop_duplicates(["state_key", "action_key"], keep="last").set_index(["state_key", "action_key"])
    manifest["action_key"] = manifest["action_candidate_readback"].map(lambda value: action_sha256(_arr(value)))
    keys = pd.MultiIndex.from_arrays([manifest.state_key.astype(str), manifest.action_key.astype(str)])
    joined = lookup.reindex(keys)
    if joined["pfv_budget_metric_m3"].isna().any():
        raise RuntimeError(f"experience-bank label alignment failed: {int(joined.pfv_budget_metric_m3.isna().sum())} rows")
    labels = pd.DataFrame({
        "state_key": manifest.state_key.astype(str).to_numpy(),
        "split_group_key": manifest.split_group_key.astype(str).to_numpy(),
        "pfv_budget_metric_m3": joined["pfv_budget_metric_m3"].to_numpy(float),
        "tfv_delta_m3": (joined["tfv_candidate_m3"] - joined["tfv_internal_m3"]).to_numpy(float),
    })
    if not np.isfinite(labels[["pfv_budget_metric_m3", "tfv_delta_m3"]].to_numpy(float)).all():
        raise RuntimeError("PFV/TFV labels contain NaN/Inf")

    groups = sorted(labels.split_group_key.unique())
    ranked = _rank(groups, 42)
    validation_groups = ranked[:8]
    holdout_groups = ranked[8:16]
    train_groups = ranked[16:]
    split = np.where(labels.split_group_key.isin(train_groups), "train", np.where(labels.split_group_key.isin(validation_groups), "validation", "holdout"))
    labels["split"] = split
    if tuple(map(len, (train_groups, validation_groups, holdout_groups))) != (65, 8, 8):
        raise RuntimeError("unexpected 65/8/8 rainfall-group split")

    graph = _load_graph_topology(args.project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_tensors = (
        torch.from_numpy(graph["edge_index"].astype(np.int64)).to(device),
        torch.from_numpy(graph["node_static"].astype(np.float32)).to(device),
        torch.from_numpy(build_surrogate_action_node_map(graph).astype(np.float32)).to(device),
        _graph_indices(graph, "is_storage", device), _graph_indices(graph, "is_outfall", device),
    )
    priority = torch.as_tensor(get_pfv_core_node_indices(list(graph["node_ids"])), dtype=torch.long, device=device)
    base = MultiReferenceHydraulicSurrogate(
        n_nodes=int(graph["n_nodes"]), n_facilities=int(graph["n_facilities"]), state_feature_dim=1,
        static_feature_dim=int(graph["node_static"].shape[1]), hidden_dim=64, gat_heads=4, gat_layers=3, horizon=12,
    ).to(device)
    base.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    base.eval()
    features = []
    baseline = []
    for start in range(0, len(manifest), int(args.batch_size)):
        frame = manifest.iloc[start : start + int(args.batch_size)].copy()
        data = _tensorise(frame)
        with torch.inference_mode():
            pred = _forward(base, data, graph_tensors, priority, device)
            hidden = torch.cat([
                pred["branches"][branch]["hidden_sequence"].mean(dim=(1, 2))
                for branch in ("candidate", "no_control", "dynamic_internal", "hold_previous")
            ], dim=1)
            candidate_h3 = data["action_candidate"][:, :3, :].reshape(len(frame), -1)
            delta_h3 = (data["action_candidate"][:, :3, :] - data["action_no_control"][:, :3, :]).reshape(len(frame), -1)
            feature = torch.cat([hidden.cpu(), candidate_h3, delta_h3], dim=1)
            base_g = pred["pfv_delta"].detach().cpu() - 0.05 * pred["kpi_no_control"]["pfv_m3"].detach().cpu()
            base_t = pred["tfv_delta"].detach().cpu()
            features.append(feature)
            baseline.append(torch.stack([base_g, base_t], dim=1))
    features = torch.cat(features).float()
    baseline_pred = torch.cat(baseline).numpy()
    scale = torch.as_tensor([
        max(float(np.median(np.abs(labels.pfv_budget_metric_m3))), 1.0),
        max(float(np.median(np.abs(labels.tfv_delta_m3))), 1.0),
    ], dtype=torch.float32)
    loss_audit = _loss_scale_audit(labels)
    (out / "LOSS_SCALE_AUDIT.json").write_text(json.dumps(loss_audit, indent=2), encoding="utf-8")

    reports: dict[str, object] = {
        "baseline": _metrics(features, labels, baseline_pred),
        "lineage": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "experience_bank": str(args.experience_bank.resolve()),
            "experience_bank_sha256": hashlib.sha256(args.experience_bank.read_bytes()).hexdigest(),
            "old_model": str(args.model.resolve()),
            "old_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "new_swmm_started": False,
            "calibration_groups_used_for_training": [],
            "train_groups": train_groups,
            "validation_groups": validation_groups,
            "holdout_groups": holdout_groups,
        },
        "variants": {},
        "development_gate": {"local_direction": 0.70, "pairwise": 0.75, "spearman": 0.60, "top5_recall": 0.70},
    }
    train_mask = labels.split.eq("train").to_numpy()
    validation_mask = labels.split.eq("validation").to_numpy()
    holdout_mask = labels.split.eq("holdout").to_numpy()
    reports["baseline"]["train"] = _metrics(features[train_mask], labels[train_mask].reset_index(drop=True), baseline_pred[train_mask])
    reports["baseline"]["validation"] = _metrics(features[validation_mask], labels[validation_mask].reset_index(drop=True), baseline_pred[validation_mask])
    reports["baseline"]["holdout"] = _metrics(features[holdout_mask], labels[holdout_mask].reset_index(drop=True), baseline_pred[holdout_mask])

    train_idx = np.flatnonzero(labels.split.eq("train").to_numpy())
    val_idx = np.flatnonzero(labels.split.eq("validation").to_numpy())
    hold_idx = np.flatnonzero(labels.split.eq("holdout").to_numpy())
    state_codes = {state: i for i, state in enumerate(sorted(labels.state_key.unique()))}
    state_tensor = torch.as_tensor([state_codes[x] for x in labels.state_key], dtype=torch.long)
    for variant, use_pairs in (("HEAD_ONLY_ACTION_REPAIR_V1", False), ("HEAD_ONLY_RANK_DIRECTION_V1", True)):
        torch.manual_seed(42)
        head = CandidateRelativeHead(features.shape[1]).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
        history = []
        best = None
        best_state = None
        stale = 0
        for epoch in range(1, int(args.epochs) + 1):
            head.train()
            order = train_idx.copy()
            np.random.RandomState(42 + epoch).shuffle(order)
            losses = []
            for start in range(0, len(order), 256):
                idx = torch.as_tensor(order[start : start + 256], dtype=torch.long)
                x = features.index_select(0, idx).to(device)
                y = torch.as_tensor(labels.iloc[order[start : start + 256]][["pfv_budget_metric_m3", "tfv_delta_m3"]].to_numpy(np.float32), device=device)
                pred = head(x)
                regression = nn.functional.smooth_l1_loss(pred / scale.to(device), y / scale.to(device))
                rank, direction = _pair_loss(pred, y, state_tensor.index_select(0, idx).to(device)) if use_pairs else (pred.sum() * 0.0, pred.sum() * 0.0)
                loss = regression + 0.5 * rank + 0.5 * direction
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            head.eval()
            with torch.inference_mode():
                val_pred = head(features.index_select(0, torch.as_tensor(val_idx)).to(device)).cpu().numpy()
            val_report = _metrics(features[val_idx], labels.iloc[val_idx].reset_index(drop=True), val_pred)
            key_values = [val_report.get("median_pfv_spearman"), val_report.get("median_tfv_spearman"), val_report.get("median_pfv_pairwise"), val_report.get("median_tfv_pairwise")]
            key = tuple(-float(value) if value is not None else 1.0 for value in key_values)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": val_report})
            if best is None or key < best:
                best, best_state, stale = key, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= int(args.patience):
                    break
        head.load_state_dict(best_state)
        with torch.inference_mode():
            all_pred = head(features.to(device)).cpu().numpy()
        model_path = out / f"{variant}_seed42.pt"
        torch.save(head.state_dict(), model_path)
        variant_report = {
            "status": "development_only",
            "variant": variant,
            "trained_parameters": sum(p.numel() for p in head.parameters()),
            "history": history,
            "train": _metrics(features[train_idx], labels.iloc[train_idx].reset_index(drop=True), all_pred[train_idx]),
            "validation": _metrics(features[val_idx], labels.iloc[val_idx].reset_index(drop=True), all_pred[val_idx]),
            "holdout": _metrics(features[hold_idx], labels.iloc[hold_idx].reset_index(drop=True), all_pred[hold_idx]),
            "head_path": str(model_path),
            "head_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        }
        reports["variants"][variant] = variant_report
    reports["selection"] = "head_only_variant_with_best untouched validation ranking; no calibration authority created"
    (out / "STEP2_HEAD_ONLY_REPAIR_SEED42_REPORT.json").write_text(json.dumps(reports, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(out / "STEP2_HEAD_ONLY_REPAIR_SEED42_REPORT.json"), "device": str(device), "rows": len(manifest), "variants": list(reports["variants"])}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
