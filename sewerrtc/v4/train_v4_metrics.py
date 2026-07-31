"""Metric library for the V4 offline training chain (spec sections 6/10/14).

Pure functions only -- no I/O, no SWMM.  Every metric is JSON-friendly and
fails soft to ``None`` (never silently 0/1) when a class is absent, per the
"指标写null；记录class_support" rule.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Continuous-head metrics
# ---------------------------------------------------------------------------

def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, dead_zone: float = 0.0
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    std = float(np.std(y_true))
    nrmse = float(rmse / std) if std > 0 else None
    if len(y_true) > 1 and std > 0 and np.std(y_pred) > 0:
        rho = spearmanr(y_true, y_pred).correlation
        spearman = None if np.isnan(rho) else float(rho)
    else:
        spearman = None
    out_dz = np.abs(y_true) > dead_zone
    if out_dz.any():
        sign_acc = float(
            np.mean(np.sign(y_pred[out_dz]) == np.sign(y_true[out_dz]))
        )
        sign_support = int(out_dz.sum())
    else:
        sign_acc, sign_support = None, 0
    return {
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "spearman": spearman,
        "sign_accuracy_outside_dead_zone": sign_acc,
        "sign_support": sign_support,
        "n": int(len(y_true)),
    }


def per_event_mae(
    event_id: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    out: dict[str, float] = {}
    for ev in np.unique(event_id):
        m = event_id == ev
        out[str(ev)] = float(np.mean(np.abs(y_pred[m] - y_true[m])))
    return out


def worst_event(event_mae: dict[str, float]) -> dict[str, Any]:
    if not event_mae:
        return {"event_id": None, "mae": None}
    ev = max(event_mae, key=event_mae.get)
    return {"event_id": ev, "mae": event_mae[ev]}


# ---------------------------------------------------------------------------
# Classification metrics (null-safe)
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true: np.ndarray, p: np.ndarray, *, threshold: float = 0.5
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    support = {
        "positive": int((y_true == 1).sum()),
        "negative": int((y_true == 0).sum()),
    }
    single_class = support["positive"] == 0 or support["negative"] == 0
    yhat = (p >= threshold).astype(int)
    if single_class:
        return {
            "class_support": support,
            "balanced_accuracy": None,
            "mcc": None,
            "auroc": None,
            "average_precision": None,
            "false_safe_rate": None,
            "false_reject_rate": None,
            "brier": float(np.mean((p - y_true) ** 2)),
            "n": int(len(y_true)),
        }
    # false-safe: predicted safe (1) among truly unsafe (0).
    unsafe = y_true == 0
    safe = y_true == 1
    return {
        "class_support": support,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, yhat)),
        "mcc": float(matthews_corrcoef(y_true, yhat)),
        "auroc": float(roc_auc_score(y_true, p)),
        "average_precision": float(average_precision_score(y_true, p)),
        "false_safe_rate": float(np.mean(yhat[unsafe] == 1)),
        "false_reject_rate": float(np.mean(yhat[safe] == 0)),
        "brier": float(np.mean((p - y_true) ** 2)),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# One-sided conformal bound (conservative safety direction)
# ---------------------------------------------------------------------------

def one_sided_conformal(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    direction: str,
    coverage: float = 0.9,
) -> dict[str, Any]:
    """One-sided residual quantile.

    ``direction='overprediction'`` bounds ``y_pred - y_true`` (model claims a
    larger benefit than real -- the unsafe direction for TFV/Peak improvement
    claims); ``'underprediction'`` bounds ``y_true - y_pred`` (real harm larger
    than predicted -- unsafe for PFV increase).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if direction == "overprediction":
        resid = y_pred - y_true
    elif direction == "underprediction":
        resid = y_true - y_pred
    else:
        raise ValueError(f"unknown direction: {direction}")
    q = float(np.quantile(resid, coverage))
    achieved = float(np.mean(resid <= q))
    return {
        "direction": direction,
        "coverage_target": coverage,
        "bound": q,
        "empirical_coverage": achieved,
        "n": int(len(resid)),
    }


def empirical_coverage(
    y_true: np.ndarray, y_pred: np.ndarray, bound: float, direction: str
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = (y_pred - y_true) if direction == "overprediction" else (y_true - y_pred)
    return float(np.mean(resid <= bound))


# ---------------------------------------------------------------------------
# Decision-level metrics (per-state candidate ranking)
# ---------------------------------------------------------------------------

def decision_metrics(
    *,
    state_key: np.ndarray,
    score: np.ndarray,
    feasible_true: np.ndarray,
    regret: np.ndarray,
    predicted_safe: np.ndarray | None = None,
    abstain: np.ndarray | None = None,
    pfv_unsafe_true: np.ndarray | None = None,
    peak_unsafe_true: np.ndarray | None = None,
    tfv_degraded_true: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-state candidate decision metrics (spec section 14 decision block).

    ``score``: higher is better.  ``feasible_true``: boolean joint-feasible
    label.  ``regret``: regret_to_exact_best (>=0; negative/missing -> NaN).
    ``predicted_safe``: candidate passes the model safety screen; a state with
    no predicted-safe candidate (or fully abstained) falls back.
    """
    state_key = np.asarray(state_key)
    score = np.asarray(score, dtype=float)
    feasible_true = np.asarray(feasible_true).astype(bool)
    regret = np.asarray(regret, dtype=float)
    n = len(score)
    predicted_safe = (
        np.ones(n, dtype=bool) if predicted_safe is None else np.asarray(predicted_safe).astype(bool)
    )
    abstain = (
        np.zeros(n, dtype=bool) if abstain is None else np.asarray(abstain).astype(bool)
    )

    states = list(dict.fromkeys(state_key.tolist()))  # stable order
    topk_hits = {1: 0, 3: 0, 5: 0}
    states_with_feasible = 0
    regrets: list[float] = []
    fallbacks = 0
    sel_pfv_unsafe = []
    sel_peak_unsafe = []
    sel_tfv_degraded = []
    per_state_rows = []

    for key in states:
        idx = np.flatnonzero(state_key == key)
        order = idx[np.argsort(-score[idx], kind="stable")]
        feas = feasible_true[idx]
        true_feasible_count = int(feas.sum())
        eligible = idx[predicted_safe[idx] & ~abstain[idx]]
        if eligible.size:
            sel = eligible[np.argmax(score[eligible])]
            fallback = False
        else:
            sel = None
            fallback = True
            fallbacks += 1
        if true_feasible_count > 0:
            states_with_feasible += 1
            for k in topk_hits:
                topk = order[:k]
                if feasible_true[topk].any():
                    topk_hits[k] += 1
        finite_regret = regret[idx][np.isfinite(regret[idx]) & (regret[idx] >= 0)]
        true_best = (
            int(idx[np.nanargmin(np.where(regret[idx] >= 0, regret[idx], np.nan))])
            if (np.isfinite(regret[idx]) & (regret[idx] >= 0)).any()
            else None
        )
        sel_regret = None
        if sel is not None and np.isfinite(regret[sel]) and regret[sel] >= 0:
            sel_regret = float(regret[sel])
            regrets.append(sel_regret)
        if sel is not None:
            if pfv_unsafe_true is not None:
                sel_pfv_unsafe.append(bool(pfv_unsafe_true[sel]))
            if peak_unsafe_true is not None:
                sel_peak_unsafe.append(bool(peak_unsafe_true[sel]))
            if tfv_degraded_true is not None:
                sel_tfv_degraded.append(bool(tfv_degraded_true[sel]))
        per_state_rows.append(
            {
                "state_key": str(key),
                "true_feasible_count": true_feasible_count,
                "predicted_safe_count": int(
                    (predicted_safe[idx] & ~abstain[idx]).sum()
                ),
                "selected_index": None if sel is None else int(sel),
                "true_best_index": true_best,
                "selected_regret": sel_regret,
                "fallback": fallback,
                "_finite_regret_n": int(finite_regret.size),
            }
        )

    n_states = len(states)
    def _rate(values: list[bool]):
        return float(np.mean(values)) if values else None

    return {
        "n_states": n_states,
        "states_with_feasible": states_with_feasible,
        "top_k_feasible_recall": {
            str(k): (
                float(topk_hits[k] / states_with_feasible)
                if states_with_feasible
                else None
            )
            for k in (1, 3, 5)
        },
        "decision_regret_mean": (
            float(np.mean(regrets)) if regrets else None
        ),
        "decision_regret_n": len(regrets),
        "fallback_rate": float(fallbacks / n_states) if n_states else None,
        "selected_pfv_unsafe_rate": _rate(sel_pfv_unsafe),
        "selected_peak_unsafe_rate": _rate(sel_peak_unsafe),
        "selected_tfv_degraded_rate": _rate(sel_tfv_degraded),
        "per_state": per_state_rows,
    }


# ---------------------------------------------------------------------------
# Uncertainty / error relationship
# ---------------------------------------------------------------------------

def uncertainty_error_correlation(
    uncertainty: np.ndarray, abs_error: np.ndarray
) -> dict[str, Any]:
    uncertainty = np.asarray(uncertainty, dtype=float)
    abs_error = np.asarray(abs_error, dtype=float)
    if len(uncertainty) < 3 or np.std(uncertainty) == 0 or np.std(abs_error) == 0:
        return {"spearman": None, "positive": None, "n": int(len(uncertainty))}
    rho = spearmanr(uncertainty, abs_error).correlation
    rho = None if np.isnan(rho) else float(rho)
    return {
        "spearman": rho,
        "positive": None if rho is None else bool(rho > 0),
        "n": int(len(uncertainty)),
    }
