"""V4.1 Train-only learning-curve and feature-block ablation ops (spec 6-7).

Everything here uses **only** the Train split, always event-grouped (candidate
rows are never split randomly) and fits every preprocessing / model inside each
CV fold (no fit-on-all-Train-then-CV leakage).  Old Calibration and old Locked
are never read.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .train_v4_loader import TrainingData
from .train_v4_metrics import classification_metrics, regression_metrics
from .train_v4_preflight import event_grouped_folds
from .v4_compact_diag_ops import (
    CLASSIFICATION_HEADS,
    CONTINUOUS_HEADS,
    classify_feature_block,
)

CONTINUOUS_CURVE_MODELS = ("zero", "mean", "ridge", "elasticnet", "pls", "hgb")
CLASSIFICATION_CURVE_MODELS = ("majority", "logistic", "hgb")


def _reg_row(y: np.ndarray, pred: np.ndarray, dz: float) -> dict[str, Any]:
    m = regression_metrics(y, pred, dead_zone=dz)
    var = float(np.var(y)) or 1.0
    r2 = float(1.0 - np.mean((pred - y) ** 2) / var)
    return {
        "mae": m["mae"],
        "r2": r2,
        "sign_accuracy": m["sign_accuracy_outside_dead_zone"],
        "spearman": m["spearman"],
    }


def _fit_continuous(name: str, Xf, yf, cfg, seed: int):
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=seed).fit(Xf, yf)
    if name == "elasticnet":
        return ElasticNet(alpha=0.01, max_iter=5000, random_state=seed).fit(Xf, yf)
    if name == "pls":
        n_comp = int(min(10, max(1, Xf.shape[1] - 1)))
        return PLSRegression(n_components=n_comp).fit(Xf, yf)
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=cfg.hgb_max_iter,
            max_depth=cfg.hgb_max_depth,
            learning_rate=cfg.hgb_learning_rate,
            random_state=seed,
        ).fit(Xf, yf)
    raise ValueError(name)


def _predict_continuous(name: str, model, X, yf_mean: float) -> np.ndarray:
    if name == "zero":
        return np.zeros(X.shape[0])
    if name == "mean":
        return np.full(X.shape[0], yf_mean)
    pred = model.predict(X)
    return np.asarray(pred).reshape(-1)


def _grouped_cv(
    data: TrainingData,
    *,
    feat_idx: np.ndarray,
    use_pos: np.ndarray,
    cfg,
    dead_zones: dict[str, float],
    cont_models: tuple[str, ...],
    n_folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Event-grouped CV over a Train-subset ``use_pos`` and feature subset.

    Returns per-(fold, head, model, split_kind) metric rows.  ``split_kind`` is
    ``train`` (in-fold refit error) or ``cv`` (held-out event error).
    """
    tr = data.split_index("train")
    use = tr[use_pos]
    ev = data.event_id[use]
    X_all = data.features[np.ix_(use, feat_idx)]
    n_ev = len(np.unique(ev))
    folds = event_grouped_folds(ev, n_folds=min(n_folds, max(2, n_ev)), seed=seed)
    rows: list[dict[str, Any]] = []

    for fold_id, (fit_idx, val_idx) in enumerate(folds):
        scaler = StandardScaler().fit(X_all[fit_idx])
        Xf, Xv = scaler.transform(X_all[fit_idx]), scaler.transform(X_all[val_idx])
        for head in CONTINUOUS_HEADS:
            if head not in data.continuous:
                continue
            y = data.continuous[head][use]
            yf, yv = y[fit_idx], y[val_idx]
            dz = float(dead_zones.get(head, 0.0))
            for name in cont_models:
                model = None
                if name not in ("zero", "mean"):
                    model = _fit_continuous(name, Xf, yf, cfg, seed)
                for kind, Xe, ye in (("train", Xf, yf), ("cv", Xv, yv)):
                    pred = _predict_continuous(name, model, Xe, float(yf.mean()))
                    r = _reg_row(ye, pred, dz)
                    r.update(
                        {
                            "fold": fold_id,
                            "head": head,
                            "model": name,
                            "kind": "continuous",
                            "split_kind": kind,
                            "n": int(len(ye)),
                        }
                    )
                    rows.append(r)
        for head in CLASSIFICATION_HEADS:
            if head not in data.classification:
                continue
            y = data.classification[head][use].astype(int)
            yf, yv = y[fit_idx], y[val_idx]
            preds = {"majority": np.full(len(val_idx), float(yf.mean()))}
            if len(np.unique(yf)) > 1:
                preds["logistic"] = (
                    LogisticRegression(max_iter=2000, class_weight="balanced")
                    .fit(Xf, yf)
                    .predict_proba(Xv)[:, 1]
                )
                preds["hgb"] = (
                    HistGradientBoostingClassifier(
                        max_iter=cfg.hgb_max_iter,
                        max_depth=cfg.hgb_max_depth,
                        learning_rate=cfg.hgb_learning_rate,
                        random_state=seed,
                    )
                    .fit(Xf, yf)
                    .predict_proba(Xv)[:, 1]
                )
            for name in CLASSIFICATION_CURVE_MODELS:
                p = preds.get(name, np.full(len(val_idx), float(yf.mean())))
                m = classification_metrics(yv, p)
                rows.append(
                    {
                        "fold": fold_id,
                        "head": head,
                        "model": name,
                        "kind": "classification",
                        "split_kind": "cv",
                        "balanced_accuracy": m.get("balanced_accuracy"),
                        "mcc": m.get("mcc"),
                        "auroc": m.get("auroc"),
                        "average_precision": m.get("average_precision"),
                        "false_safe_rate": m.get("false_safe_rate"),
                        "single_class_fit_fold": name not in preds,
                        "n": int(len(yv)),
                    }
                )
    return rows


def _select_events(events: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    uniq = np.array(sorted(np.unique(events).tolist()))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(uniq))
    k = max(2, int(round(len(uniq) * ratio)))
    keep = set(uniq[order[:k]].tolist())
    return np.array([e in keep for e in events])


def build_learning_curves(
    data: TrainingData,
    *,
    cfg,
    dead_zones: dict[str, float],
    ratios: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    n_folds: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """Train-only event-grouped learning curves at 20/40/60/80/100% events."""
    tr = data.split_index("train")
    ev_tr = data.event_id[tr]
    n_total_events = len(np.unique(ev_tr))
    all_rows: list[dict[str, Any]] = []
    for ratio in ratios:
        mask = _select_events(ev_tr, ratio, seed)
        use_pos = np.flatnonzero(mask)
        n_ev = len(np.unique(ev_tr[mask]))
        rows = _grouped_cv(
            data,
            feat_idx=np.arange(data.features.shape[1]),
            use_pos=use_pos,
            cfg=cfg,
            dead_zones=dead_zones,
            cont_models=CONTINUOUS_CURVE_MODELS,
            n_folds=n_folds,
            seed=seed,
        )
        for r in rows:
            r.update(
                {
                    "train_ratio": ratio,
                    "n_train_events": n_ev,
                    "n_total_train_events": n_total_events,
                }
            )
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def diagnose_learning_curves(curve: pd.DataFrame) -> dict[str, Any]:
    """Spec section 6 verdicts per continuous head."""
    verdicts: dict[str, Any] = {}
    cont = curve[curve["kind"] == "continuous"]
    for head in CONTINUOUS_HEADS:
        h = cont[cont["head"] == head]
        if h.empty:
            continue
        full = h[h["train_ratio"] == 1.0]
        cv = full[full["split_kind"] == "cv"]
        train = full[full["split_kind"] == "train"]
        best_simple = cv[cv["model"].isin(["ridge", "elasticnet", "pls"])]
        deep = cv[cv["model"] == "hgb"]
        simple_mae = float(best_simple["mae"].mean()) if not best_simple.empty else np.nan
        deep_mae = float(deep["mae"].mean()) if not deep.empty else np.nan
        train_mae = float(train[train["model"] == "hgb"]["mae"].mean()) if not train.empty else np.nan
        cv_mae = float(cv[cv["model"] == "hgb"]["mae"].mean()) if not cv.empty else np.nan
        # curve slope: cv mae at 100% vs 40%
        low = cont[(cont["head"] == head) & (cont["train_ratio"] == 0.4) & (cont["split_kind"] == "cv") & (cont["model"] == "hgb")]
        low_mae = float(low["mae"].mean()) if not low.empty else np.nan
        verdicts[head] = {
            "train_hgb_mae": train_mae,
            "cv_hgb_mae": cv_mae,
            "cv_best_simple_mae": simple_mae,
            "cv_deep_mae": deep_mae,
            "simple_beats_deep": bool(simple_mae < deep_mae) if np.isfinite(simple_mae) and np.isfinite(deep_mae) else None,
            "train_cv_gap": (cv_mae - train_mae) if np.isfinite(cv_mae) and np.isfinite(train_mae) else None,
            "curve_still_improving": bool(cv_mae < low_mae) if np.isfinite(cv_mae) and np.isfinite(low_mae) else None,
        }
    return verdicts


# ---------------------------------------------------------------------------
# Section 7: feature-block ablation
# ---------------------------------------------------------------------------

# 13 block combinations (spec section 7).  Each maps to a set of block letters;
# ``None`` means "all blocks minus the listed removal set".
ABLATION_COMBOS: dict[str, dict[str, Any]] = {
    "state_only": {"blocks": {"A", "B", "C", "D", "E", "F"}},
    "action_only": {"blocks": {"H", "I", "J", "K", "M"}},
    "rainfall_only": {"blocks": {"G"}},
    "state_action": {"blocks": {"A", "B", "C", "D", "E", "F", "H", "I", "J", "K", "M"}},
    "state_action_rain": {"blocks": {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "M"}},
    "candidate_minus_di_only": {"blocks": {"I"}},
    "candidate_minus_hold_only": {"blocks": {"J", "K"}},
    "global_hydraulic": {"blocks": {"C", "D", "E", "F"}},
    "priority_local": {"blocks": {"A", "B"}},
    "full_570": {"blocks": None, "remove": set()},
    "drop_static_highdim": {"blocks": None, "remove": {"H"}},
    "drop_candidate_family_meta": {"blocks": None, "remove": {"M"}},
    "drop_absolute_event_scale": {"blocks": None, "remove": {"H", "L"}},
}


def _block_of_features(data: TrainingData) -> np.ndarray:
    return np.array([classify_feature_block(n) for n in data.feature_names])


def _combo_feature_idx(block_arr: np.ndarray, combo: dict[str, Any]) -> np.ndarray:
    if combo.get("blocks") is None:
        remove = combo.get("remove", set())
        return np.flatnonzero(~np.isin(block_arr, list(remove)))
    return np.flatnonzero(np.isin(block_arr, list(combo["blocks"])))


def run_feature_block_ablation(
    data: TrainingData,
    *,
    cfg,
    dead_zones: dict[str, float],
    seeds: tuple[int, ...] = (0, 1),
    n_folds: int = 4,
) -> pd.DataFrame:
    """Event-grouped CV per block combo; fold-local scaling; seed variance."""
    import time as _time

    block_arr = _block_of_features(data)
    rows: list[dict[str, Any]] = []
    for combo_name, combo in ABLATION_COMBOS.items():
        feat_idx = _combo_feature_idx(block_arr, combo)
        if feat_idx.size == 0:
            continue
        per_seed_cont: dict[str, list[float]] = {h: [] for h in CONTINUOUS_HEADS}
        agg_rows: list[dict[str, Any]] = []
        t0 = _time.time()
        for seed in seeds:
            cv_rows = _grouped_cv(
                data,
                feat_idx=feat_idx,
                use_pos=np.arange(data.split_index("train").size),
                cfg=cfg,
                dead_zones=dead_zones,
                cont_models=("ridge", "hgb"),
                n_folds=n_folds,
                seed=seed,
            )
            agg_rows.extend([{**r, "seed": seed} for r in cv_rows])
            for head in CONTINUOUS_HEADS:
                sub = [
                    r["mae"] for r in cv_rows
                    if r["head"] == head and r["model"] == "hgb" and r["split_kind"] == "cv"
                ]
                if sub:
                    per_seed_cont[head].append(float(np.mean(sub)))
        elapsed = _time.time() - t0
        df = pd.DataFrame(agg_rows)
        for head in CONTINUOUS_HEADS:
            hb = df[(df["head"] == head) & (df["kind"] == "continuous") & (df["split_kind"] == "cv") & (df["model"] == "hgb")]
            if hb.empty:
                continue
            rows.append(
                {
                    "combo": combo_name,
                    "head": head,
                    "n_features": int(feat_idx.size),
                    "cv_mae_mean": float(hb["mae"].mean()),
                    "cv_r2_mean": float(hb["r2"].mean()),
                    "event_worst_mae": float(hb["mae"].max()),
                    "seed_variance": float(np.var(per_seed_cont[head])) if len(per_seed_cont[head]) > 1 else 0.0,
                    "fit_seconds": float(elapsed),
                    "feature_selection_fold_local": True,
                }
            )
        # classification summary rows
        cls = df[(df["kind"] == "classification") & (df["model"] == "hgb")]
        for head in CLASSIFICATION_HEADS:
            hc = cls[cls["head"] == head]
            if hc.empty or hc["mcc"].isna().all():
                continue
            rows.append(
                {
                    "combo": combo_name,
                    "head": head,
                    "n_features": int(feat_idx.size),
                    "cv_balanced_accuracy": float(hc["balanced_accuracy"].mean()),
                    "cv_mcc": float(hc["mcc"].mean()),
                    "cv_average_precision": float(hc["average_precision"].mean()),
                    "feature_selection_fold_local": True,
                }
            )
    return pd.DataFrame(rows)
