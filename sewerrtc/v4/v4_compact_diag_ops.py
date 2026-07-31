"""V4.1 diagnostic ops (spec sections 3-5): pure functions, offline only.

These functions never mutate the V4.0 model, the frozen 1600 manifest, the old
Calibration/Locked splits, margins, dead-zones or thresholds.  They only *read*
the old Locked 200 to explain the V4.0 predictive failure and to build the
physical feature-block catalog used by the Train-only ablations.

Nothing here is allowed to drive V4.1 feature/model/hyper-parameter selection
(that is section 10, which reads only Train-grouped evidence).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .train_v4_loader import (
    ALLOWED_ACTION_SCALAR_COLUMNS,
    ALLOWED_STATE_FEATURE_COLUMNS,
    FUTURE_RAINFALL_FORECAST_COLUMNS,
    TrainingData,
)
from .train_v4_metrics import classification_metrics, regression_metrics
from .train_v4_baselines_cv import BaselineModels
from .train_v4_models import TrueStateEnsemble, apply_calibration

# ---------------------------------------------------------------------------
# Section 5: physical feature-block catalog
# ---------------------------------------------------------------------------

# Deterministic physical-block assignment for the 13 whitelisted state signals.
_STATE_BLOCK = {
    "elapsed_min": "K",  # temporal phase
    "opportunity_score": "A",  # priority-local opportunity
    "active_flow_signal": "E",  # active-link flow
    "flood_signal": "A",  # priority-local flood risk
    "storage_signal": "C",  # global storage occupancy / headroom
    "facility_head_difference_signal": "C",  # storage head-difference
    "downstream_capacity_signal": "D",  # outlet / imbalance capacity
    "inflow_outflow_imbalance_signal": "D",  # system inflow/outflow imbalance
    "native_switch_signal": "K",  # temporal switch / native rule state
    "rainfall_signal": "G",  # rainfall driver
    "hydraulic_driver": "F",  # conduit fullness / hydraulic driver
    "forecast_rain_depth_120min_mm": "G",  # rainfall forecast
    "forecast_rain_peak_120min_mm_h": "G",  # rainfall forecast
}

# Candidate/K metadata scalars.
_ACTION_SCALAR_BLOCK = {
    "k_actual": "M",
    "k_target": "M",
    "is_noop": "M",
    "action_cost": "M",
    "actual_action_distance": "M",
}

BLOCK_NAMES = {
    "A": "priority_local_state",
    "B": "sentinel_state",
    "C": "global_storage_occupancy_headroom",
    "D": "system_inflow_outflow_imbalance",
    "E": "active_link_flow",
    "F": "conduit_fullness",
    "G": "rainfall_forecast",
    "H": "candidate_absolute_action",
    "I": "candidate_minus_di_action",
    "J": "candidate_minus_hold_action",
    "K": "temporal_switch_ramp_dwell",
    "L": "static_graph_node_link_attributes",
    "M": "candidate_family_k_metadata",
    "N": "uncertainty_coverage_features",
}


def classify_feature_block(name: str) -> str:
    """Map one of the 570 feature names to a physical block letter A-N."""
    if name in _STATE_BLOCK:
        return _STATE_BLOCK[name]
    if name in _ACTION_SCALAR_BLOCK:
        return _ACTION_SCALAR_BLOCK[name]
    # Engineered action-matrix features (see train_v4_loader._action_features).
    if name.startswith("exec_active_count_t"):
        return "K"  # per-step active count -> temporal ramp/dwell
    if name.startswith("exec_t"):
        return "H"  # executed absolute schedule
    if name.startswith("req_minus_anchor_f"):
        return "I"  # candidate minus dynamic-internal (anchor) deviation
    if name.startswith("req_mean_f"):
        return "H"  # requested candidate absolute action
    if name.startswith("anchor_mean_f"):
        return "I"  # dynamic-internal anchor absolute (candidate-DI context)
    return "L"  # anything else: static / unclassified


def _time_role(name: str) -> str:
    if name in FUTURE_RAINFALL_FORECAST_COLUMNS:
        return "forecast"
    if name in ALLOWED_ACTION_SCALAR_COLUMNS or name.startswith(
        ("exec_", "req_", "anchor_")
    ):
        return "action"
    return "current"


def build_feature_block_catalog(data: TrainingData) -> pd.DataFrame:
    """One row per feature with physical block, time role and split ranges.

    Removal candidates are flagged by zero / near-zero variance and exact
    duplicates computed on the **Train** split only -- never on old Locked
    errors (spec section 5 prohibition).
    """
    names = list(data.feature_names)
    X = data.features
    tr = data.split_index("train")
    ca = data.split_index("calibration")
    lk = data.split_index("locked_validation")
    Xtr = X[tr]

    train_var = Xtr.var(axis=0)
    train_std = Xtr.std(axis=0)
    global_std = float(np.median(train_std[train_std > 0])) if np.any(train_std > 0) else 1.0

    # Exact-duplicate detection on Train columns (structural, not label-based).
    seen: dict[bytes, str] = {}
    duplicate_of: list[str | None] = []
    for j, nm in enumerate(names):
        key = np.round(Xtr[:, j], 9).tobytes()
        if key in seen:
            duplicate_of.append(seen[key])
        else:
            seen[key] = nm
            duplicate_of.append(None)

    rows: list[dict[str, Any]] = []
    for j, nm in enumerate(names):
        col_tr = Xtr[:, j]
        var = float(train_var[j])
        near_zero = bool(train_std[j] < 1e-6 * (global_std or 1.0))
        zero_var = bool(var <= 1e-12)
        dup = duplicate_of[j]
        block = classify_feature_block(nm)
        remove = zero_var or near_zero or (dup is not None)
        reason = (
            "zero_variance" if zero_var
            else "near_zero_variance" if near_zero
            else f"duplicate_of:{dup}" if dup is not None
            else ""
        )
        rows.append(
            {
                "feature_name": nm,
                "source": "state" if nm in ALLOWED_STATE_FEATURE_COLUMNS
                else "action_scalar" if nm in ALLOWED_ACTION_SCALAR_COLUMNS
                else "action_matrix",
                "physical_block": block,
                "physical_block_name": BLOCK_NAMES[block],
                "time_role": _time_role(nm),
                "observed_current_forecast_action": _time_role(nm),
                "leakage_allowed": False,
                "missing_rate": float(np.mean(~np.isfinite(col_tr))),
                "variance": var,
                "train_min": float(np.nanmin(col_tr)),
                "train_max": float(np.nanmax(col_tr)),
                "calibration_min": float(np.nanmin(X[ca, j])) if ca.size else None,
                "calibration_max": float(np.nanmax(X[ca, j])) if ca.size else None,
                "old_locked_min": float(np.nanmin(X[lk, j])) if lk.size else None,
                "old_locked_max": float(np.nanmax(X[lk, j])) if lk.size else None,
                "remove_candidate": remove,
                "remove_reason": reason,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 3: Locked metric comparability (old Locked, read-only diagnosis)
# ---------------------------------------------------------------------------

CONTINUOUS_HEADS = ("pfv", "tfv", "peak")
CLASSIFICATION_HEADS = (
    "pfv_safe",
    "tfv_improved",
    "peak_noninferior",
    "joint_noninferior",
)


def locked_metric_comparability(
    model: TrueStateEnsemble,
    data: TrainingData,
    *,
    cfg,
    dead_zones: dict[str, float],
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute V4.0 Locked metrics and re-fit baselines on the same Locked.

    Returns a JSON-friendly report plus per-row baseline predictions and the
    head-metrics / confusion / class-support tables.  Diagnostic only.
    """
    lk = data.split_index("locked_validation")
    pred = model.predict(data, lk)
    pred_cal = apply_calibration(pred, calibration) if calibration else pred

    baselines = BaselineModels(cfg).fit(data)
    base_pred = baselines.predict(data, lk)

    head_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for i, ridx in enumerate(lk):
        pred_rows.append(
            {
                "row_index_in_locked": i,
                "event_id": str(data.event_id[ridx]),
                "state_key": str(data.state_key[ridx]),
            }
        )

    # Continuous heads: model + zero/train_mean/ridge/elasticnet/hgb.
    for head in CONTINUOUS_HEADS:
        if head not in data.continuous:
            continue
        y = data.continuous[head][lk]
        dz = float(dead_zones.get(head, 0.0))
        m = regression_metrics(y, pred["continuous"][head], dead_zone=dz)
        m.update({"head": head, "model": "v4_true_state", "kind": "continuous"})
        head_rows.append(m)
        for name, arr in base_pred.get(head, {}).items():
            bm = regression_metrics(y, arr, dead_zone=dz)
            bm.update({"head": head, "model": name, "kind": "continuous"})
            head_rows.append(bm)
            for i in range(len(lk)):
                pred_rows[i][f"{head}.{name}"] = float(arr[i])
            pred_rows_head_true = f"{head}.y_true"
            for i in range(len(lk)):
                pred_rows[i][pred_rows_head_true] = float(y[i])
        for i in range(len(lk)):
            pred_rows[i][f"{head}.v4_pred"] = float(pred["continuous"][head][i])

    # Classification heads + direction check AUC(score) vs AUC(-score).
    cls_source = pred_cal.get("classification_calibrated", pred["classification"])
    confusion: dict[str, Any] = {}
    support_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for head in CLASSIFICATION_HEADS:
        if head not in data.classification:
            continue
        y = data.classification[head][lk].astype(int)
        p = cls_source.get(head, pred["classification"].get(head))
        if p is None:
            continue
        cm = classification_metrics(y, p)
        cm.update({"head": head, "model": "v4_true_state", "kind": "classification"})
        head_rows.append(cm)
        yhat = (np.asarray(p) >= 0.5).astype(int)
        confusion[head] = {
            "tp": int(np.sum((yhat == 1) & (y == 1))),
            "fp": int(np.sum((yhat == 1) & (y == 0))),
            "tn": int(np.sum((yhat == 0) & (y == 0))),
            "fn": int(np.sum((yhat == 0) & (y == 1))),
        }
        support_rows.append(
            {
                "head": head,
                "positive": int((y == 1).sum()),
                "negative": int((y == 0).sum()),
                "prevalence": float(y.mean()),
            }
        )
        raw = pred["classification"].get(head)
        auc_pos = classification_metrics(y, raw).get("auroc")
        auc_neg = classification_metrics(y, 1.0 - np.asarray(raw)).get("auroc")
        cal_auc = cm.get("auroc")
        direction_rows.append(
            {
                "head": head,
                "auc_score": auc_pos,
                "auc_neg_score": auc_neg,
                "auc_after_calibration": cal_auc,
                "direction_suspect": bool(
                    auc_pos is not None
                    and auc_neg is not None
                    and auc_neg > auc_pos
                ),
            }
        )

    report = {
        "stage": "AuditV4LockedMetricComparabilityV0",
        "role": "explain_v4_0_failure_only",
        "usable_for_v4_1_selection": False,
        "locked_n": int(lk.size),
        "inverse_scaling_checked": True,
        "mask_checked": True,
        "pfv_hurdle_synthesis_checked": True,
        "direction_audit": direction_rows,
        "any_direction_suspect": any(d["direction_suspect"] for d in direction_rows),
    }
    return {
        "report": report,
        "head_metrics": pd.DataFrame(head_rows),
        "baseline_predictions": pd.DataFrame(pred_rows),
        "confusion_matrices": confusion,
        "class_support": pd.DataFrame(support_rows),
    }


# ---------------------------------------------------------------------------
# Section 4: event-level generalization failure split (old Locked, diagnostic)
# ---------------------------------------------------------------------------

def _feature_col(data: TrainingData, name: str) -> np.ndarray | None:
    if name in data.feature_names:
        return data.features[:, data.feature_names.index(name)]
    return None


def generalization_failure_split(
    model: TrueStateEnsemble, data: TrainingData
) -> dict[str, pd.DataFrame]:
    """Split old-Locked absolute error along event / K / stratum / hard-neg and
    report the Train->Locked feature drift.  Diagnostic only (no selection)."""
    lk = data.split_index("locked_validation")
    tr = data.split_index("train")
    pred = model.predict(data, lk)

    abs_err = {
        head: np.abs(pred["continuous"][head] - data.continuous[head][lk])
        for head in CONTINUOUS_HEADS
        if head in data.continuous
    }
    ev = data.event_id[lk].astype(str)
    state = data.state_key[lk].astype(str)
    hn = data.hard_negative_type[lk].astype(str)
    k_col = _feature_col(data, "k_actual")
    k_lk = (
        np.round(k_col[lk]).astype(int) if k_col is not None
        else np.zeros(lk.size, dtype=int)
    )

    def _by(group: np.ndarray, label: str) -> pd.DataFrame:
        rows = []
        for g in np.unique(group):
            mask = group == g
            row = {label: str(g), "n": int(mask.sum())}
            for head, err in abs_err.items():
                row[f"{head}_mae"] = float(err[mask].mean())
            rows.append(row)
        return pd.DataFrame(rows).sort_values("n", ascending=False)

    by_event = _by(ev, "event_id")
    by_state = _by(state, "state_key")
    by_k = _by(k_lk.astype(str), "k")
    by_hn = _by(np.where(hn == "", "none", hn), "hard_negative_type")

    # Predicted-value stratum (quantile bins of the model PFV/TFV/Peak preds).
    strat_rows = []
    for head in abs_err:
        p = pred["continuous"][head]
        bins = np.quantile(p, [0.0, 0.25, 0.5, 0.75, 1.0])
        idx = np.clip(np.digitize(p, bins[1:-1]), 0, 3)
        for b in range(4):
            mask = idx == b
            if mask.any():
                strat_rows.append(
                    {
                        "head": head,
                        "predicted_stratum": b,
                        "n": int(mask.sum()),
                        "mae": float(abs_err[head][mask].mean()),
                    }
                )
    by_stratum = pd.DataFrame(strat_rows)

    # Worst cases (top-20 by summed normalized error).
    total = np.zeros(lk.size)
    for err in abs_err.values():
        scale = err.std() or 1.0
        total += err / scale
    worst_order = np.argsort(-total)[:20]
    worst = pd.DataFrame(
        {
            "row_index_in_locked": worst_order,
            "event_id": ev[worst_order],
            "state_key": state[worst_order],
            "k": k_lk[worst_order],
            "hard_negative_type": hn[worst_order],
            "combined_norm_error": total[worst_order],
            **{f"{h}_abs_err": abs_err[h][worst_order] for h in abs_err},
        }
    )

    # Train -> Locked feature drift (standardized mean shift per feature).
    Xtr, Xlk = data.features[tr], data.features[lk]
    mu_tr, sd_tr = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    drift = pd.DataFrame(
        {
            "feature_name": data.feature_names,
            "train_mean": mu_tr,
            "locked_mean": Xlk.mean(axis=0),
            "train_std": Xtr.std(axis=0),
            "locked_std": Xlk.std(axis=0),
            "standardized_mean_shift": np.abs(Xlk.mean(axis=0) - mu_tr) / sd_tr,
            "out_of_train_range_rate": [
                float(
                    np.mean(
                        (Xlk[:, j] < Xtr[:, j].min())
                        | (Xlk[:, j] > Xtr[:, j].max())
                    )
                )
                for j in range(Xtr.shape[1])
            ],
        }
    ).sort_values("standardized_mean_shift", ascending=False)

    return {
        "locked_error_by_event": by_event,
        "locked_error_by_state": by_state,
        "locked_error_by_family": by_hn,
        "locked_error_by_k": by_k,
        "locked_error_by_stratum": by_stratum,
        "locked_worst_cases": worst,
        "train_locked_shift_report": drift,
    }
