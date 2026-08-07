"""Bounded same-state Step2 ranking/gradient audit against canonical truth."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_v42_step2_action_sensitivity import _iter_manifest_batches
from scripts.train_v42_step2_fast import _forward, _graph_indices, _tensorise
from sewerrtc.control.authoritative_control_metrics_v42 import action_sha256
from sewerrtc.v4.models_v42.hydraulic_multi_reference import MultiReferenceHydraulicSurrogate
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology, build_surrogate_action_node_map


def _sign(value: float, tolerance: float = 1.0e-8) -> int:
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def _pair_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    signs = []
    for i, j in combinations(range(len(actual)), 2):
        a = _sign(float(actual[j] - actual[i]))
        p = _sign(float(predicted[j] - predicted[i]))
        if a and p:
            signs.append(a == p)
    return float(np.mean(signs)) if signs else None


def _top5_recall(actual: np.ndarray, actual_safe: np.ndarray, predicted: np.ndarray, predicted_budget: np.ndarray) -> float | None:
    actual_ids = np.flatnonzero(actual_safe)
    predicted_ids = np.flatnonzero(predicted_budget <= 100.0)
    if not len(actual_ids) or not len(predicted_ids):
        return None
    actual_top = actual_ids[np.argsort(actual[actual_ids])[:5]]
    predicted_top = predicted_ids[np.argsort(predicted[predicted_ids])[:5]]
    return float(len(set(actual_top.tolist()) & set(predicted_top.tolist())) / min(5, len(actual_top)))


def _local_direction(actual: np.ndarray, gradients: np.ndarray, actions: np.ndarray, max_pairs: int = 200) -> float | None:
    pairs = []
    h3 = actions[:, :3].reshape(len(actions), -1)
    for i, j in combinations(range(len(actions)), 2):
        delta = h3[j] - h3[i]
        distance = float(np.mean(np.abs(delta)))
        if distance > 1.0e-7:
            pairs.append((distance, i, j, delta))
    pairs.sort(key=lambda item: item[0])
    signs = []
    for _, i, j, delta in pairs[:max_pairs]:
        truth = _sign(float(actual[j] - actual[i]))
        derivative = _sign(float(np.dot(gradients[i, :3].reshape(-1), delta)))
        if truth and derivative:
            signs.append(truth == derivative)
    return float(np.mean(signs)) if signs else None


def _load_truth(path: Path, state_keys: set[str]) -> pd.DataFrame:
    columns = [
        "state_key", "canonical_candidate_action_sha256", "candidate_action_sha256",
        "pfv_budget_metric_m3", "pfv_feasible", "tfv_candidate_m3", "tfv_internal_m3",
    ]
    frame = pd.read_parquet(path, columns=columns, filters=[("state_key", "in", sorted(state_keys))])
    frame["state_key"] = frame["state_key"].astype(str)
    frame["action_key"] = frame["canonical_candidate_action_sha256"].fillna(frame["candidate_action_sha256"]).astype(str)
    return frame.drop_duplicates(["state_key", "action_key"], keep="last").reset_index(drop=True)


def _safe_spearman(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(actual) < 3 or np.ptp(actual) == 0.0 or np.ptp(predicted) == 0.0:
        return None
    value = spearmanr(actual, predicted).statistic
    return float(value) if np.isfinite(value) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--experience-bank", type=Path, required=True)
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--state-keys", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    args = ap.parse_args()

    state_keys = {str(x) for x in args.state_keys}
    truth = _load_truth(args.experience_bank, state_keys)
    graph = _load_graph_topology(args.project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_map = build_surrogate_action_node_map(graph).astype(np.float32)
    graph_tensors = (
        torch.from_numpy(graph["edge_index"].astype(np.int64)).to(device),
        torch.from_numpy(graph["node_static"].astype(np.float32)).to(device),
        torch.from_numpy(action_map).to(device),
        _graph_indices(graph, "is_storage", device),
        _graph_indices(graph, "is_outfall", device),
    )
    priority = torch.as_tensor(get_pfv_core_node_indices(list(graph["node_ids"])), dtype=torch.long, device=device)
    model_rows: dict[int, dict[tuple[str, str], dict[str, object]]] = {}

    for seed in args.seeds:
        model = MultiReferenceHydraulicSurrogate(
            n_nodes=int(graph["n_nodes"]), n_facilities=int(graph["n_facilities"]),
            state_feature_dim=1, static_feature_dim=int(graph["node_static"].shape[1]),
            hidden_dim=64, gat_heads=4, gat_layers=3, horizon=12,
        ).to(device)
        model.load_state_dict(torch.load(args.model_root / f"seed_{seed}" / "best_model.pt", map_location=device, weights_only=True))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        rows: dict[tuple[str, str], dict[str, object]] = {}
        for frame in _iter_manifest_batches(args.manifest, state_keys, args.batch_size):
            data = _tensorise(frame)
            actions = data["action_candidate"].numpy()
            keys = [(str(state), action_sha256(action)) for state, action in zip(frame["state_key"], actions)]
            action = data["action_candidate"].to(device).detach().requires_grad_(True)
            local = dict(data)
            local["action_candidate"] = action
            prediction = _forward(model, local, graph_tensors, priority, device)
            tfv = prediction["tfv_delta"].reshape(-1)
            nc_pfv = prediction["kpi_no_control"]["pfv_m3"].reshape(-1)
            budget = prediction["pfv_delta"].reshape(-1) - 0.05 * torch.clamp(nc_pfv, min=0.0)
            grad_tfv = torch.autograd.grad(tfv.sum(), action, retain_graph=True)[0]
            grad_budget = torch.autograd.grad(budget.sum(), action)[0]
            for i, key in enumerate(keys):
                rows[key] = {
                    "pred_tfv": float(tfv[i].detach().cpu()),
                    "pred_budget": float(budget[i].detach().cpu()),
                    "grad_tfv": grad_tfv[i].detach().cpu().numpy(),
                    "grad_budget": grad_budget[i].detach().cpu().numpy(),
                    "action": actions[i],
                }
            del prediction, grad_tfv, grad_budget, action, local, data
            if device.type == "cuda":
                torch.cuda.empty_cache()
        model_rows[int(seed)] = rows
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_seed = []
    for seed in args.seeds:
        rows = model_rows[int(seed)]
        for state_key, group in truth.groupby("state_key", sort=True):
            items = [rows[(state_key, key)] for key in group["action_key"] if (state_key, key) in rows]
            if len(items) < 3:
                continue
            aligned = group[group["action_key"].map(lambda key: (state_key, key) in rows)].copy()
            predicted_tfv = np.asarray([item["pred_tfv"] for item in items], dtype=float)
            predicted_budget = np.asarray([item["pred_budget"] for item in items], dtype=float)
            actual_tfv = aligned["tfv_candidate_m3"].to_numpy(float)
            actual_budget = aligned["pfv_budget_metric_m3"].to_numpy(float)
            actual_safe = aligned["pfv_feasible"].to_numpy(bool)
            per_seed.append({
                "seed": int(seed), "state_key": state_key, "rows": int(len(items)),
                "tfv_pairwise_accuracy": _pair_accuracy(actual_tfv, predicted_tfv),
                "pfv_pairwise_accuracy": _pair_accuracy(actual_budget, predicted_budget),
                "tfv_spearman": _safe_spearman(actual_tfv, predicted_tfv),
                "pfv_spearman": _safe_spearman(actual_budget, predicted_budget),
                "top5_good_action_recall": _top5_recall(actual_tfv, actual_safe, predicted_tfv, predicted_budget),
                "tfv_local_direction_agreement": _local_direction(actual_tfv, np.asarray([item["grad_tfv"] for item in items]), np.asarray([item["action"] for item in items])),
                "pfv_local_direction_agreement": _local_direction(actual_budget, np.asarray([item["grad_budget"] for item in items]), np.asarray([item["action"] for item in items])),
                "predicted_safe_actual_unsafe_fraction": float(np.mean((predicted_budget <= 100.0) & ~actual_safe)),
                "actual_safe_fraction": float(np.mean(actual_safe)),
            })

    def median(key: str) -> float | None:
        values = [float(row[key]) for row in per_seed if row.get(key) is not None and np.isfinite(row[key])]
        return float(np.median(values)) if values else None

    summary = {
        "audit_id": "V42_STEP2_GRADIENT_TRUTH_AUDIT_V1", "read_only": True, "new_swmm_started": False,
        "manifest": str(args.manifest), "experience_bank": str(args.experience_bank),
        "model_root": str(args.model_root),
        "state_keys": sorted(state_keys), "states": len(state_keys), "truth_rows": int(len(truth)),
        "model_seeds": [int(x) for x in args.seeds], "action_map_nonzero": int(np.count_nonzero(action_map)),
        "per_seed_state_rows": per_seed,
        "median_tfv_local_direction_agreement": median("tfv_local_direction_agreement"),
        "median_pfv_local_direction_agreement": median("pfv_local_direction_agreement"),
        "median_tfv_pairwise_accuracy": median("tfv_pairwise_accuracy"),
        "median_pfv_pairwise_accuracy": median("pfv_pairwise_accuracy"),
        "median_tfv_spearman": median("tfv_spearman"), "median_pfv_spearman": median("pfv_spearman"),
        "mean_top5_good_action_recall": float(np.mean([row["top5_good_action_recall"] for row in per_seed if row.get("top5_good_action_recall") is not None])) if any(row.get("top5_good_action_recall") is not None for row in per_seed) else None,
        "mean_predicted_safe_actual_unsafe_fraction": float(np.mean([row["predicted_safe_actual_unsafe_fraction"] for row in per_seed])) if per_seed else None,
        "truth_rows_by_state": {
            str(state): int(len(group)) for state, group in truth.groupby("state_key", sort=True)
        },
        "development_gate_thresholds": {"local_direction": 0.70, "pairwise": 0.75, "spearman": 0.60, "top5_recall": 0.70},
    }
    summary["gradient_gate_pass"] = bool(
        summary["median_tfv_local_direction_agreement"] is not None
        and summary["median_pfv_local_direction_agreement"] is not None
        and summary["median_tfv_local_direction_agreement"] >= 0.70
        and summary["median_pfv_local_direction_agreement"] >= 0.70
        and summary["median_tfv_pairwise_accuracy"] is not None
        and summary["median_tfv_pairwise_accuracy"] >= 0.75
        and summary["median_pfv_pairwise_accuracy"] is not None
        and summary["median_pfv_pairwise_accuracy"] >= 0.75
        and summary["median_tfv_spearman"] is not None
        and summary["median_tfv_spearman"] >= 0.60
        and summary["mean_top5_good_action_recall"] is not None
        and summary["mean_top5_good_action_recall"] >= 0.70
    )
    summary["interpretation"] = "Mean-model ranking/direction audit only; no PFV UCB calibration authority is created by this report."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
