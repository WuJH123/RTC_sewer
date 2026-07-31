"""Extended V4 baselines with event-grouped CV (spec section 6).

Continuous heads (delta PFV / TFV / Peak): zero, train-mean, event-mean
(legal internal CV only), Ridge, ElasticNet, HistGradientBoostingRegressor.
Classification heads (PFV-active, pfv_safe, tfv_improved, peak_noninferior):
majority, LogisticRegression, HistGradientBoostingClassifier.
Ranking: random, zero-delta (all-tie), HGB predicted-utility.

Internal validation is ALWAYS event-grouped over Train events -- candidate
rows are never split randomly.  Single-class folds report ``null`` metrics
with class support, never fabricated 0/1.
"""
from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .train_v4_loader import TrainingData
from .train_v4_metrics import (
    classification_metrics,
    decision_metrics,
    per_event_mae,
    regression_metrics,
    worst_event,
)
from .train_v4_preflight import event_grouped_folds

CONTINUOUS_BASELINES = ("zero", "train_mean", "event_mean", "ridge", "elasticnet", "hgb")
CLASSIFICATION_BASELINES = ("majority", "logistic", "hgb")
RANKING_BASELINES = ("random", "zero_delta", "hgb_utility")

CLASSIFICATION_HEADS = ("pfv_active", "pfv_safe", "tfv_improved", "peak_noninferior")


def _hgb_reg(cfg, seed=0):
    return HistGradientBoostingRegressor(
        max_iter=cfg.hgb_max_iter,
        max_depth=cfg.hgb_max_depth,
        learning_rate=cfg.hgb_learning_rate,
        random_state=seed,
    )


def _hgb_cls(cfg, seed=0):
    return HistGradientBoostingClassifier(
        max_iter=cfg.hgb_max_iter,
        max_depth=cfg.hgb_max_depth,
        learning_rate=cfg.hgb_learning_rate,
        random_state=seed,
    )


def classification_labels(
    data: TrainingData, *, pfv_dead_zone: float
) -> dict[str, np.ndarray]:
    """The four baseline classification targets (incl. derived PFV-active)."""
    labels: dict[str, np.ndarray] = {}
    if "pfv" in data.continuous:
        labels["pfv_active"] = (
            np.abs(data.continuous["pfv"]) > pfv_dead_zone
        ).astype(int)
    for col in ("pfv_safe", "tfv_improved", "peak_noninferior"):
        if col in data.classification:
            labels[col] = data.classification[col]
    return labels


def run_baseline_cv(
    data: TrainingData,
    *,
    cfg,
    dead_zones: dict[str, float],
    n_folds: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Event-grouped CV over the Train split only.  Returns metrics tables and
    out-of-fold predictions (JSON/CSV-friendly)."""
    tr = data.split_index("train")
    ev_tr = data.event_id[tr]
    folds = event_grouped_folds(ev_tr, n_folds=n_folds, seed=seed)
    X_all = data.features[tr]
    cls_labels = classification_labels(data, pfv_dead_zone=dead_zones["pfv"])

    by_fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    oof = {
        ("continuous", head, name): np.full(len(tr), np.nan)
        for head in data.continuous
        for name in CONTINUOUS_BASELINES
    }
    oof.update(
        {
            ("classification", head, name): np.full(len(tr), np.nan)
            for head in cls_labels
            for name in CLASSIFICATION_BASELINES
        }
    )
    oof.update(
        {
            ("ranking", "score", name): np.full(len(tr), np.nan)
            for name in RANKING_BASELINES
        }
    )

    for fold_id, (fit_idx, val_idx) in enumerate(folds):
        scaler = StandardScaler().fit(X_all[fit_idx])
        Xf, Xv = scaler.transform(X_all[fit_idx]), scaler.transform(X_all[val_idx])

        # ---- continuous heads ----
        for head, y_full in data.continuous.items():
            y = y_full[tr]
            yf, yv = y[fit_idx], y[val_idx]
            ev_fit = ev_tr[fit_idx]
            ev_val = ev_tr[val_idx]
            event_means = {e: float(yf[ev_fit == e].mean()) for e in set(ev_fit)}
            global_mean = float(yf.mean())
            preds = {
                "zero": np.zeros_like(yv),
                "train_mean": np.full_like(yv, global_mean),
                # unseen validation events fall back to the fit-side mean
                "event_mean": np.array(
                    [event_means.get(e, global_mean) for e in ev_val]
                ),
                "ridge": Ridge(alpha=1.0).fit(Xf, yf).predict(Xv),
                "elasticnet": ElasticNet(alpha=0.01, max_iter=5000)
                .fit(Xf, yf)
                .predict(Xv),
                "hgb": _hgb_reg(cfg).fit(Xf, yf).predict(Xv),
            }
            dz = float(dead_zones.get(head, 0.0))
            for name, pred in preds.items():
                oof[("continuous", head, name)][val_idx] = pred
                m = regression_metrics(yv, pred, dead_zone=dz)
                m.update({"fold": fold_id, "head": head, "model": name, "kind": "continuous"})
                by_fold_rows.append(m)

        # ---- classification heads ----
        for head, y_full in cls_labels.items():
            y = y_full[tr]
            yf, yv = y[fit_idx], y[val_idx]
            base_rate = float(yf.mean())
            preds = {"majority": np.full(len(val_idx), base_rate)}
            if len(np.unique(yf)) > 1:
                preds["logistic"] = (
                    LogisticRegression(max_iter=2000, class_weight="balanced")
                    .fit(Xf, yf)
                    .predict_proba(Xv)[:, 1]
                )
                preds["hgb"] = _hgb_cls(cfg).fit(Xf, yf).predict_proba(Xv)[:, 1]
            for name, p in preds.items():
                oof[("classification", head, name)][val_idx] = p
                m = classification_metrics(yv, p)
                m.update({"fold": fold_id, "head": head, "model": name, "kind": "classification"})
                by_fold_rows.append(m)
            for name in CLASSIFICATION_BASELINES:
                if name not in preds:
                    # single-class fold: metrics stay null, support recorded
                    m = classification_metrics(yv, np.full(len(val_idx), base_rate))
                    m.update(
                        {
                            "fold": fold_id,
                            "head": head,
                            "model": name,
                            "kind": "classification",
                            "single_class_fit_fold": True,
                        }
                    )
                    by_fold_rows.append(m)

        # ---- ranking baselines (within-state, validation side only) ----
        rng = np.random.RandomState(1000 + fold_id)
        rank_scores = {
            "random": rng.rand(len(val_idx)),
            "zero_delta": np.zeros(len(val_idx)),
        }
        if "joint_noninferior" in data.classification:
            yj = data.classification["joint_noninferior"][tr]
            if len(np.unique(yj[fit_idx])) > 1:
                rank_scores["hgb_utility"] = (
                    _hgb_cls(cfg).fit(Xf, yj[fit_idx]).predict_proba(Xv)[:, 1]
                )
            else:
                rank_scores["hgb_utility"] = np.zeros(len(val_idx))
        for name, s in rank_scores.items():
            oof[("ranking", "score", name)][val_idx] = s

    # ---- assemble OOF prediction table ----
    state_tr = data.state_key[tr]
    for i in range(len(tr)):
        row: dict[str, Any] = {
            "row_index_in_train": i,
            "event_id": str(ev_tr[i]),
            "state_key": str(state_tr[i]),
        }
        for (kind, head, name), arr in oof.items():
            row[f"{kind}.{head}.{name}"] = arr[i]
        pred_rows.append(row)

    # ---- by-event continuous MAE + worst event ----
    by_event_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"continuous": {}, "classification": {}, "ranking": {}}
    for head in data.continuous:
        y = data.continuous[head][tr]
        head_summary = {}
        for name in CONTINUOUS_BASELINES:
            pred = oof[("continuous", head, name)]
            valid = np.isfinite(pred)
            m = regression_metrics(
                y[valid], pred[valid], dead_zone=float(dead_zones.get(head, 0.0))
            )
            ev_mae = per_event_mae(ev_tr[valid], y[valid], pred[valid])
            m["worst_event"] = worst_event(ev_mae)
            head_summary[name] = m
            for ev, mae in ev_mae.items():
                by_event_rows.append(
                    {"head": head, "model": name, "event_id": ev, "mae": mae}
                )
        summary["continuous"][head] = head_summary
    for head, y_full in cls_labels.items():
        y = y_full[tr]
        head_summary = {}
        for name in CLASSIFICATION_BASELINES:
            p = oof[("classification", head, name)]
            valid = np.isfinite(p)
            head_summary[name] = (
                classification_metrics(y[valid], p[valid])
                if valid.any()
                else {"class_support": None, "n": 0}
            )
        summary["classification"][head] = head_summary

    # ---- ranking decision metrics on OOF scores ----
    ranking_rows: list[dict[str, Any]] = []
    feasible = (
        data.classification.get("joint_noninferior", np.zeros(len(data.split)))[tr]
        .astype(bool)
    )
    regret = data.ranking.get(
        "regret_to_exact_best", np.full(len(data.split), np.nan)
    )[tr]
    for name in RANKING_BASELINES:
        s = oof[("ranking", "score", name)]
        valid = np.isfinite(s)
        dm = decision_metrics(
            state_key=state_tr[valid],
            score=s[valid],
            feasible_true=feasible[valid],
            regret=regret[valid],
        )
        dm.pop("per_state", None)
        dm["model"] = name
        ranking_rows.append(dm)
        summary["ranking"][name] = dm

    return {
        "summary": summary,
        "by_fold": by_fold_rows,
        "by_event": by_event_rows,
        "predictions": pred_rows,
        "ranking": ranking_rows,
        "n_folds": n_folds,
        "grouping": "event",
        "candidate_rows_randomly_split": False,
    }


# ---------------------------------------------------------------------------
# Full-train fitted baseline models (for Locked relative comparisons)
# ---------------------------------------------------------------------------

class BaselineModels:
    """Zero / mean / Ridge / ElasticNet / HGB fitted on the full Train split.

    Used at Locked time to compute relative improvement.  Fitting never sees
    Calibration or Locked rows.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler_: StandardScaler | None = None
        self.cont_: dict[str, dict[str, Any]] = {}
        self.train_means_: dict[str, float] = {}

    def fit(self, data: TrainingData) -> "BaselineModels":
        tr = data.split_index("train")
        self.scaler_ = StandardScaler().fit(data.features[tr])
        X = self.scaler_.transform(data.features[tr])
        for head, y_full in data.continuous.items():
            y = y_full[tr]
            self.train_means_[head] = float(y.mean())
            self.cont_[head] = {
                "ridge": Ridge(alpha=1.0).fit(X, y),
                "elasticnet": ElasticNet(alpha=0.01, max_iter=5000).fit(X, y),
                "hgb": _hgb_reg(self.cfg).fit(X, y),
            }
        return self

    def predict(self, data: TrainingData, index: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        X = self.scaler_.transform(data.features[index])
        out: dict[str, dict[str, np.ndarray]] = {}
        for head, models in self.cont_.items():
            out[head] = {
                "zero": np.zeros(len(index)),
                "train_mean": np.full(len(index), self.train_means_[head]),
                **{name: m.predict(X) for name, m in models.items()},
            }
        return out

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(blob: bytes) -> "BaselineModels":
        return pickle.loads(blob)


def baseline_stop_verdict(summary: dict[str, Any]) -> dict[str, Any]:
    """Spec section 6 stop rule: continue unless *no* head shows signal."""
    cont_beats_zero = []
    for head, models in summary.get("continuous", {}).items():
        zero_mae = models.get("zero", {}).get("mae")
        best_learned = min(
            (
                models[name]["mae"]
                for name in ("ridge", "elasticnet", "hgb")
                if name in models and models[name].get("mae") is not None
            ),
            default=None,
        )
        if zero_mae is not None and best_learned is not None:
            cont_beats_zero.append(bool(best_learned < zero_mae))
    cls_signal = []
    for head, models in summary.get("classification", {}).items():
        for name in ("logistic", "hgb"):
            auroc = models.get(name, {}).get("auroc")
            if auroc is not None:
                cls_signal.append(bool(auroc > 0.5))
    all_cont_fail = bool(cont_beats_zero) and not any(cont_beats_zero)
    all_cls_fail = bool(cls_signal) and not any(cls_signal)
    stop = all_cont_fail and all_cls_fail
    return {
        "continuous_any_beats_zero": any(cont_beats_zero) if cont_beats_zero else None,
        "classification_any_signal": any(cls_signal) if cls_signal else None,
        "stop_training": stop,
    }
