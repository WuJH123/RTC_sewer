"""V4.2 Single-Head Event-Level CV (§10).

Runs 5-fold GroupKFold (group=event_id) for each single-head task:
  10.1 PFV-only    — Ridge / HGB / Twin-head on pfv_delta
  10.2 TFV-only    — Ridge / HGB / Twin-head on tfv_delta
  10.3 Peak-only   — Ridge / HGB / Twin-head on peak_delta
  10.4 Ranking-only — pairwise ranking accuracy on pfv_delta direction

Each CV saves complete OOF predictions, per-fold metrics, and aggregate.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold

from sewerrtc.v4.v42_trainer import (
    load_v42_training_data,
    TwinWithKPIHeads,
    _build_model,
    _make_batch,
    HIDDEN_DIM,
)
from sewerrtc.v4.models_v42.ranking_losses import RankingLosses

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction (same as baseline_comparability)
# ---------------------------------------------------------------------------

def _extract_flat_features(data: dict) -> np.ndarray:
    """Extract flat feature vector per sample for Ridge/HGB."""
    state = data["state_history"].numpy()
    rain = data["rainfall"].numpy()
    act_c = data["action_candidate"].numpy()
    act_r = data["action_reference"].numpy()

    last_frame = state[:, -1, :]
    diff = act_c - act_r

    features = np.stack([
        last_frame.mean(axis=1), last_frame.std(axis=1), last_frame.max(axis=1),
        rain.sum(axis=1), rain.mean(axis=1), rain.max(axis=1),
        act_c.reshape(len(act_c), -1).sum(axis=1),
        act_c.reshape(len(act_c), -1).mean(axis=1),
        act_c.reshape(len(act_c), -1).max(axis=1),
        act_r.reshape(len(act_r), -1).sum(axis=1),
        act_r.reshape(len(act_r), -1).mean(axis=1),
        act_r.reshape(len(act_r), -1).max(axis=1),
        diff.reshape(len(diff), -1).sum(axis=1),
        diff.reshape(len(diff), -1).mean(axis=1),
        diff.reshape(len(diff), -1).max(axis=1),
    ], axis=1)
    return features.astype(np.float64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_full_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute R², MAE, sign accuracy, Spearman, false-safe, false-reject."""
    r2 = float(r2_score(target, pred))
    mae = float(mean_absolute_error(target, pred))

    mask = np.abs(target) > 1e-8
    if mask.sum() > 0:
        sign_acc = float(np.mean(np.sign(pred[mask]) == np.sign(target[mask])))
    else:
        sign_acc = 0.5

    # Spearman correlation
    if len(target) > 2:
        spearman_rho, spearman_p = stats.spearmanr(pred, target)
        spearman_rho = float(spearman_rho)
    else:
        spearman_rho = 0.0

    # False safe: model predicts improvement (pred<0) but target shows degradation (target>0)
    n_mask = mask.sum()
    if n_mask > 0:
        false_safe = float(np.mean(
            (pred[mask] < 0) & (target[mask] > 0)
        ))
        false_reject = float(np.mean(
            (pred[mask] > 0) & (target[mask] < 0)
        ))
    else:
        false_safe = 0.5
        false_reject = 0.5

    return {
        "r2": r2, "mae": mae, "sign_accuracy": sign_acc,
        "spearman_rho": spearman_rho,
        "false_safe_rate": false_safe,
        "false_reject_rate": false_reject,
        "n_samples": int(len(target)),
    }


# ---------------------------------------------------------------------------
# Twin-head feature extraction
# ---------------------------------------------------------------------------

def _extract_twin_features(
    data: dict, device: torch.device,
) -> np.ndarray:
    """Run Twin base model (random init) and extract KPI head input features."""
    n_nodes = data["n_nodes"]
    n_facilities = data["n_facilities"]
    model = _build_model("D", n_nodes, n_facilities, data["node_max_depth"]).to(device)
    model.eval()

    n = len(data["pfv_delta"])
    batch_size = 64
    all_feats = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = np.arange(start, end)
            batch = _make_batch(data, idx)
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            shared = {
                "edge_index": data["edge_index"].to(device),
                "node_static": data["node_static"].to(device),
                "action_node_map": data["action_node_map"].to(device),
            }
            for k, v in shared.items():
                batch_dev[k] = v

            pred = model(
                state_history=batch_dev["state_history"],
                rainfall=batch_dev["rainfall"],
                action_candidate=batch_dev["action_candidate"],
                action_reference=batch_dev["action_reference"],
                edge_index=batch_dev["edge_index"],
                node_static=batch_dev["node_static"],
                action_node_map=batch_dev["action_node_map"],
            )
            # Extract delta trajectory features
            delta = pred["delta"]  # [B, H, N]
            delta_mean = delta.mean(dim=(1, 2)).cpu().numpy()  # [B]
            delta_std = delta.reshape(delta.shape[0], -1).std(dim=1).cpu().numpy()
            delta_max = delta.reshape(delta.shape[0], -1).amax(dim=1).cpu().numpy()

            # Action diff features
            act_diff = (batch_dev["action_candidate"] - batch_dev["action_reference"])
            act_diff_sum = act_diff.reshape(act_diff.shape[0], -1).sum(dim=1).cpu().numpy()
            act_diff_mean = act_diff.reshape(act_diff.shape[0], -1).mean(dim=1).cpu().numpy()

            # y_candidate / y_reference stats
            y_c_mean = pred["y_candidate"].reshape(pred["y_candidate"].shape[0], -1).mean(dim=1).cpu().numpy()
            y_r_mean = pred["y_reference"].reshape(pred["y_reference"].shape[0], -1).mean(dim=1).cpu().numpy()

            feats = np.stack([
                delta_mean, delta_std, delta_max,
                act_diff_sum, act_diff_mean,
                y_c_mean, y_r_mean,
            ], axis=1)
            all_feats.append(feats)

    return np.concatenate(all_feats, axis=0)


# ---------------------------------------------------------------------------
# Single-head CV runner
# ---------------------------------------------------------------------------

def _run_single_head_cv(
    task_name: str,
    target_values: np.ndarray,
    features: np.ndarray,
    twin_features: np.ndarray,
    event_ids: np.ndarray,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Run 5-fold GroupKFold for a single head task."""
    gkf = GroupKFold(n_splits=n_folds)
    groups = event_ids

    oof_ridge = np.zeros(len(target_values))
    oof_hgb = np.zeros(len(target_values))
    oof_twin = np.zeros(len(target_values))
    fold_metrics: dict[str, list] = {"ridge": [], "hgb": [], "twin_head": []}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(np.arange(len(target_values)), target_values, groups)):
        y_train = target_values[train_idx]
        y_val = target_values[val_idx]

        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(features[train_idx], y_train)
        oof_ridge[val_idx] = ridge.predict(features[val_idx])

        # HGB
        hgb = HistGradientBoostingRegressor(
            max_iter=100, max_depth=5, learning_rate=0.1, random_state=42,
        )
        hgb.fit(features[train_idx], y_train)
        oof_hgb[val_idx] = hgb.predict(features[val_idx])

        # Twin-head (Ridge on Twin features)
        twin_ridge = Ridge(alpha=1.0)
        twin_ridge.fit(twin_features[train_idx], y_train)
        oof_twin[val_idx] = twin_ridge.predict(twin_features[val_idx])

        # Per-fold metrics
        for name, oof in [("ridge", oof_ridge), ("hgb", oof_hgb), ("twin_head", oof_twin)]:
            m = _compute_full_metrics(oof[val_idx], y_val)
            m["fold"] = fold_idx
            fold_metrics[name].append(m)

        logger.info("  Fold %d: ridge R²=%.4f  hgb R²=%.4f  twin R²=%.4f",
                     fold_idx,
                     fold_metrics["ridge"][-1]["r2"],
                     fold_metrics["hgb"][-1]["r2"],
                     fold_metrics["twin_head"][-1]["r2"])

    # Aggregate
    result: dict[str, Any] = {"task": task_name, "n_samples": len(target_values), "folds": {}}
    for name, oof in [("ridge", oof_ridge), ("hgb", oof_hgb), ("twin_head", oof_twin)]:
        overall = _compute_full_metrics(oof, target_values)
        fold_r2s = [m["r2"] for m in fold_metrics[name]]
        result["folds"][name] = {
            "oof_r2": overall["r2"],
            "oof_mae": overall["mae"],
            "oof_sign_accuracy": overall["sign_accuracy"],
            "oof_spearman_rho": overall["spearman_rho"],
            "oof_false_safe": overall["false_safe_rate"],
            "oof_false_reject": overall["false_reject_rate"],
            "fold_r2_mean": float(np.mean(fold_r2s)),
            "fold_r2_std": float(np.std(fold_r2s)),
            "per_fold": fold_metrics[name],
        }

    # Save OOF predictions
    result["oof_predictions"] = {
        "ridge": oof_ridge.tolist(),
        "hgb": oof_hgb.tolist(),
        "twin_head": oof_twin.tolist(),
        "target": target_values.tolist(),
    }

    return result


# ---------------------------------------------------------------------------
# Ranking-only CV
# ---------------------------------------------------------------------------

def _run_ranking_cv(
    data: dict, features: np.ndarray, twin_features: np.ndarray,
    event_ids: np.ndarray, device: torch.device, n_folds: int = 5,
) -> dict[str, Any]:
    """Ranking-only CV: predict PFV delta direction, measure ranking quality."""
    pfv = data["pfv_delta"].numpy()
    gkf = GroupKFold(n_splits=n_folds)

    oof_ridge = np.zeros(len(pfv))
    oof_hgb = np.zeros(len(pfv))
    oof_twin = np.zeros(len(pfv))
    fold_metrics: dict[str, list] = {"ridge": [], "hgb": [], "twin_head": []}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(np.arange(len(pfv)), pfv, event_ids)):
        y_train = pfv[train_idx]
        y_val = pfv[val_idx]

        ridge = Ridge(alpha=1.0)
        ridge.fit(features[train_idx], y_train)
        oof_ridge[val_idx] = ridge.predict(features[val_idx])

        hgb = HistGradientBoostingRegressor(
            max_iter=100, max_depth=5, learning_rate=0.1, random_state=42,
        )
        hgb.fit(features[train_idx], y_train)
        oof_hgb[val_idx] = hgb.predict(features[val_idx])

        twin_ridge = Ridge(alpha=1.0)
        twin_ridge.fit(twin_features[train_idx], y_train)
        oof_twin[val_idx] = twin_ridge.predict(twin_features[val_idx])

        # Ranking metrics: pairwise accuracy
        for name, oof in [("ridge", oof_ridge), ("hgb", oof_hgb), ("twin_head", oof_twin)]:
            pred_val = oof[val_idx]
            # Pairwise ranking accuracy
            n_pairs = 0
            n_correct = 0
            for i in range(len(val_idx)):
                for j in range(i + 1, min(i + 20, len(val_idx))):
                    if abs(y_val[i] - y_val[j]) > 1.0:  # dead zone
                        gt_better = y_val[i] < y_val[j]  # lower = better
                        pred_better = pred_val[i] < pred_val[j]
                        n_pairs += 1
                        if gt_better == pred_better:
                            n_correct += 1
            rank_acc = n_correct / max(n_pairs, 1)

            m = _compute_full_metrics(pred_val, y_val)
            m["ranking_accuracy"] = rank_acc
            m["n_pairs"] = n_pairs
            m["fold"] = fold_idx
            fold_metrics[name].append(m)

    result: dict[str, Any] = {"task": "ranking_only", "n_samples": len(pfv), "folds": {}}
    for name, oof in [("ridge", oof_ridge), ("hgb", oof_hgb), ("twin_head", oof_twin)]:
        overall = _compute_full_metrics(oof, pfv)
        fold_r2s = [m["r2"] for m in fold_metrics[name]]
        fold_rank_accs = [m["ranking_accuracy"] for m in fold_metrics[name]]
        result["folds"][name] = {
            "oof_r2": overall["r2"],
            "oof_mae": overall["mae"],
            "oof_sign_accuracy": overall["sign_accuracy"],
            "oof_spearman_rho": overall["spearman_rho"],
            "ranking_accuracy_mean": float(np.mean(fold_rank_accs)),
            "fold_r2_mean": float(np.mean(fold_r2s)),
            "fold_r2_std": float(np.std(fold_r2s)),
            "per_fold": fold_metrics[name],
        }

    result["oof_predictions"] = {
        "ridge": oof_ridge.tolist(),
        "hgb": oof_hgb.tolist(),
        "twin_head": oof_twin.tolist(),
        "target": pfv.tolist(),
    }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single_head_cv(
    project_root: str | Path,
    output_root: str | Path,
) -> dict:
    """Run all 4 single-head CVs."""
    project_root = Path(project_root)
    output_root = Path(output_root)
    audit_dir = output_root / "audits" / "v42_repair" / "single_head_cv"
    audit_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Single-head CV on device: %s", device)

    # Load data
    data = load_v42_training_data(project_root, output_root)
    event_ids = data["event_ids"]

    # Extract features
    logger.info("Extracting flat features...")
    features = _extract_flat_features(data)

    logger.info("Extracting Twin features...")
    twin_features = _extract_twin_features(data, device)

    results: dict[str, Any] = {}

    # 10.1 PFV-only
    logger.info("=== 10.1 PFV-only CV ===")
    results["pfv_only"] = _run_single_head_cv(
        "pfv_only", data["pfv_delta"].numpy(), features, twin_features, event_ids,
    )
    (audit_dir / "pfv_only_cv.json").write_text(
        json.dumps(results["pfv_only"], indent=2, default=str), encoding="utf-8",
    )

    # 10.2 TFV-only
    logger.info("=== 10.2 TFV-only CV ===")
    results["tfv_only"] = _run_single_head_cv(
        "tfv_only", data["tfv_delta"].numpy(), features, twin_features, event_ids,
    )
    (audit_dir / "tfv_only_cv.json").write_text(
        json.dumps(results["tfv_only"], indent=2, default=str), encoding="utf-8",
    )

    # 10.3 Peak-only
    logger.info("=== 10.3 Peak-only CV ===")
    results["peak_only"] = _run_single_head_cv(
        "peak_only", data["peak_delta"].numpy(), features, twin_features, event_ids,
    )
    (audit_dir / "peak_only_cv.json").write_text(
        json.dumps(results["peak_only"], indent=2, default=str), encoding="utf-8",
    )

    # 10.4 Ranking-only
    logger.info("=== 10.4 Ranking-only CV ===")
    results["ranking_only"] = _run_ranking_cv(
        data, features, twin_features, event_ids, device,
    )
    (audit_dir / "ranking_only_cv.json").write_text(
        json.dumps(results["ranking_only"], indent=2, default=str), encoding="utf-8",
    )

    # Summary
    summary = {}
    for task, res in results.items():
        summary[task] = {}
        for model_name, metrics in res["folds"].items():
            summary[task][model_name] = {
                "oof_r2": metrics["oof_r2"],
                "oof_sign_acc": metrics["oof_sign_accuracy"],
                "oof_spearman": metrics["oof_spearman_rho"],
            }

    (audit_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    # Gate check
    any_r2_positive = any(
        res["folds"]["twin_head"]["oof_r2"] > 0
        for res in results.values()
    )
    n_better_than_mean = sum(
        1 for task, res in results.items()
        if res["folds"]["twin_head"]["oof_r2"] > res["folds"]["ridge"]["oof_r2"]
        or res["folds"]["twin_head"]["oof_r2"] > res["folds"]["hgb"]["oof_r2"]
    )

    gate = {
        "any_oof_r2_positive": any_r2_positive,
        "n_twin_better_than_baseline": n_better_than_mean,
        "gate_pass": any_r2_positive,
    }
    (audit_dir / "cv_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8",
    )

    logger.info("CV gate: %s", "PASS" if gate["gate_pass"] else "FAIL")
    return {"results": summary, "gate": gate}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "final_v4"
    result = run_single_head_cv(PROJECT_ROOT, OUTPUT_ROOT)
    print(json.dumps(result, indent=2))
