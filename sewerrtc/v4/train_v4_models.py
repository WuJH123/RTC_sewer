"""H120 True-state V4 models: baselines + multi-head ensemble (spec 4-8).

Everything here is pure/offline (no SWMM, no closed loop).  Two families:

* ``fit_baselines`` -- zero / mean / ridge / logistic / HGB reference models.
* ``TrueStateEnsemble`` -- the deployable multi-head model:
    - 7-channel x 12-step process-residual regression;
    - Delta PFV / TFV / Peak continuous regression, PFV via a **hurdle**
      (active/inactive gate then active-only regression) so the ~46% near-zero
      PFV mass never sees a plain MSE;
    - PFV-safe / TFV-improved / Peak-noninferior / joint-noninferior
      classification;
    - within-state candidate ranking score;
    - ensemble-variance uncertainty, feature-distance OOD and an abstain flag.

Training uses only Train rows, a 5-seed ensemble, event-equal sampling weights
and class + hard-negative weights.  Peak hard negatives are always kept (never
downsampled).  Calibration (temperature / conformal / abstain thresholds) is fit
on the Calibration split only and must never read Locked.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .train1600_v4 import CONTINUOUS_HEAD_KEYS
from .train_v4_loader import (
    CLASSIFICATION_TARGET_COLUMNS,
    TrainingData,
)

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
PEAK_HARD_NEGATIVE = "Peak_hard_negative"


@dataclass
class ModelConfig:
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    hgb_max_iter: int = 200
    hgb_max_depth: int | None = 6
    hgb_learning_rate: float = 0.06
    hard_negative_weight: float = 2.0
    abstain_uncertainty_quantile: float = 0.9
    ood_quantile: float = 0.99

    def light(self) -> "ModelConfig":
        """A fast variant for tests / smoke."""
        return ModelConfig(
            seeds=(0, 1),
            hgb_max_iter=30,
            hgb_max_depth=3,
            hgb_learning_rate=0.1,
            hard_negative_weight=self.hard_negative_weight,
            abstain_uncertainty_quantile=self.abstain_uncertainty_quantile,
            ood_quantile=self.ood_quantile,
        )


# ---------------------------------------------------------------------------
# Sample weights (event-equal + class + hard-negative)
# ---------------------------------------------------------------------------

def event_equal_weights(event_id: np.ndarray) -> np.ndarray:
    """Each event contributes equal total weight regardless of candidate count."""
    weights = np.ones(len(event_id), dtype=float)
    events, inverse, counts = np.unique(
        event_id, return_inverse=True, return_counts=True
    )
    n_events = len(events)
    per_event_total = 1.0 / max(n_events, 1)
    weights = per_event_total / counts[inverse]
    # Normalise to mean 1 for numerical friendliness.
    weights *= len(event_id) / weights.sum()
    return weights


def class_balance_weights(y: np.ndarray) -> np.ndarray:
    w = np.ones(len(y), dtype=float)
    for cls in np.unique(y):
        mask = y == cls
        frac = mask.mean()
        if frac > 0:
            w[mask] = 0.5 / frac
    return w


def hard_negative_weights(
    hard_negative_type: np.ndarray, *, weight: float
) -> np.ndarray:
    w = np.ones(len(hard_negative_type), dtype=float)
    is_hard = hard_negative_type.astype(str) != ""
    w[is_hard] = weight
    return w


def combined_sample_weights(
    data: TrainingData, index: np.ndarray, y: np.ndarray | None, *, cfg: ModelConfig
) -> np.ndarray:
    ev = event_equal_weights(data.event_id[index])
    hn = hard_negative_weights(
        data.hard_negative_type[index], weight=cfg.hard_negative_weight
    )
    weights = ev * hn
    if y is not None:
        weights = weights * class_balance_weights(y)
    return weights


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    denom = float(np.var(y_true)) or 1.0
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - np.mean(err**2) / denom),
    }


def _cls_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y_true) ** 2))
    acc = float(np.mean((p >= 0.5).astype(int) == y_true))
    # AUC via rank statistic (robust to ties).
    pos = y_true == 1
    neg = ~pos
    if pos.any() and neg.any():
        order = np.argsort(p)
        ranks = np.empty(len(p), dtype=float)
        ranks[order] = np.arange(1, len(p) + 1)
        auc = float(
            (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
            / (pos.sum() * neg.sum())
        )
    else:
        auc = float("nan")
    return {"brier": brier, "accuracy": acc, "auc": auc}


def fit_baselines(
    data: TrainingData, *, cfg: ModelConfig | None = None
) -> dict[str, Any]:
    """Fit reference baselines on Train, evaluate on Train and Calibration.

    Returns a JSON-friendly report (no heavy objects).  Locked is untouched.
    """
    cfg = cfg or ModelConfig()
    tr = data.split_index("train")
    ca = data.split_index("calibration")
    scaler = StandardScaler().fit(data.features[tr])
    Xtr = scaler.transform(data.features[tr])
    Xca = scaler.transform(data.features[ca])

    report: dict[str, Any] = {"continuous": {}, "classification": {}}

    for head, col in CONTINUOUS_HEAD_KEYS.items():
        if head not in data.continuous:
            continue
        ytr = data.continuous[head][tr]
        yca = data.continuous[head][ca]
        mean_val = float(np.mean(ytr))
        preds = {
            "zero": np.zeros_like(yca),
            "mean": np.full_like(yca, mean_val),
            "ridge": Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xca),
            "hgb": HistGradientBoostingRegressor(
                max_iter=cfg.hgb_max_iter,
                max_depth=cfg.hgb_max_depth,
                learning_rate=cfg.hgb_learning_rate,
                random_state=0,
            ).fit(Xtr, ytr).predict(Xca),
        }
        report["continuous"][head] = {
            name: _reg_metrics(yca, pred) for name, pred in preds.items()
        }

    for col in CLASSIFICATION_TARGET_COLUMNS:
        if col not in data.classification:
            continue
        ytr = data.classification[col][tr]
        yca = data.classification[col][ca]
        base_rate = float(np.mean(ytr))
        preds = {"mean": np.full_like(yca, base_rate, dtype=float)}
        if len(np.unique(ytr)) > 1:
            preds["logistic"] = (
                LogisticRegression(max_iter=1000, class_weight="balanced")
                .fit(Xtr, ytr)
                .predict_proba(Xca)[:, 1]
            )
            preds["hgb"] = (
                HistGradientBoostingClassifier(
                    max_iter=cfg.hgb_max_iter,
                    max_depth=cfg.hgb_max_depth,
                    learning_rate=cfg.hgb_learning_rate,
                    random_state=0,
                )
                .fit(Xtr, ytr)
                .predict_proba(Xca)[:, 1]
            )
        report["classification"][col] = {
            name: _cls_metrics(yca, pred) for name, pred in preds.items()
        }
    return report


# ---------------------------------------------------------------------------
# True-state multi-head ensemble
# ---------------------------------------------------------------------------

@dataclass
class TrueStateEnsemble:
    cfg: ModelConfig = field(default_factory=ModelConfig)
    pfv_dead_zone: float = 1.0
    # fitted attributes
    scaler_: StandardScaler | None = None
    residual_models_: list = field(default_factory=list)
    cont_models_: dict = field(default_factory=dict)
    pfv_gate_: list = field(default_factory=list)
    pfv_active_: list = field(default_factory=list)
    cls_models_: dict = field(default_factory=dict)
    feature_names_: list[str] = field(default_factory=list)
    train_centroid_: np.ndarray | None = None
    train_scale_: np.ndarray | None = None

    # ----- fit -----
    def fit(self, data: TrainingData) -> "TrueStateEnsemble":
        cfg = self.cfg
        tr = data.split_index("train")
        if tr.size == 0:
            raise ValueError("no training rows in split 'train'")
        self.feature_names_ = list(data.feature_names)
        self.scaler_ = StandardScaler().fit(data.features[tr])
        Xtr = self.scaler_.transform(data.features[tr])
        w_event = combined_sample_weights(data, tr, None, cfg=cfg)

        # Process residual head: multi-output Ridge (native, fast, stable).
        Ytr_res = data.residuals[tr].reshape(len(tr), -1)
        self.residual_models_ = []
        for seed in cfg.seeds:
            model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=seed))
            model.fit(Xtr, Ytr_res)
            self.residual_models_.append(model)

        # Continuous heads: TFV / Peak plain regression, PFV hurdle.
        self.cont_models_ = {}
        for head in ("tfv", "peak"):
            if head not in data.continuous:
                continue
            y = data.continuous[head][tr]
            self.cont_models_[head] = [
                HistGradientBoostingRegressor(
                    max_iter=cfg.hgb_max_iter,
                    max_depth=cfg.hgb_max_depth,
                    learning_rate=cfg.hgb_learning_rate,
                    random_state=seed,
                ).fit(Xtr, y, sample_weight=w_event)
                for seed in cfg.seeds
            ]

        # PFV hurdle: gate (active vs inactive) then active-only regressor.
        self.pfv_gate_ = []
        self.pfv_active_ = []
        if "pfv" in data.continuous:
            y_pfv = data.continuous["pfv"][tr]
            active = (np.abs(y_pfv) > self.pfv_dead_zone).astype(int)
            for seed in cfg.seeds:
                if len(np.unique(active)) > 1:
                    gate = HistGradientBoostingClassifier(
                        max_iter=cfg.hgb_max_iter,
                        max_depth=cfg.hgb_max_depth,
                        learning_rate=cfg.hgb_learning_rate,
                        random_state=seed,
                    ).fit(
                        Xtr,
                        active,
                        sample_weight=w_event * class_balance_weights(active),
                    )
                else:
                    gate = _ConstantProba(float(active.mean()))
                self.pfv_gate_.append(gate)
                act_idx = np.flatnonzero(active == 1)
                if act_idx.size >= 5:
                    reg = HistGradientBoostingRegressor(
                        max_iter=cfg.hgb_max_iter,
                        max_depth=cfg.hgb_max_depth,
                        learning_rate=cfg.hgb_learning_rate,
                        random_state=seed,
                    ).fit(Xtr[act_idx], y_pfv[act_idx], sample_weight=w_event[act_idx])
                else:
                    reg = _ConstantReg(float(y_pfv.mean()))
                self.pfv_active_.append(reg)

        # Classification heads.
        self.cls_models_ = {}
        for col in CLASSIFICATION_TARGET_COLUMNS:
            if col not in data.classification:
                continue
            y = data.classification[col][tr]
            if len(np.unique(y)) < 2:
                self.cls_models_[col] = [_ConstantProba(float(y.mean()))]
                continue
            w = w_event * class_balance_weights(y)
            self.cls_models_[col] = [
                HistGradientBoostingClassifier(
                    max_iter=cfg.hgb_max_iter,
                    max_depth=cfg.hgb_max_depth,
                    learning_rate=cfg.hgb_learning_rate,
                    random_state=seed,
                ).fit(Xtr, y, sample_weight=w)
                for seed in cfg.seeds
            ]

        # OOD reference statistics (standardised feature space).
        self.train_centroid_ = Xtr.mean(axis=0)
        self.train_scale_ = Xtr.std(axis=0) + 1e-9
        return self

    # ----- predict -----
    def predict(self, data: TrainingData, index: np.ndarray) -> dict[str, Any]:
        if self.scaler_ is None:
            raise ValueError("model not fitted")
        X = self.scaler_.transform(data.features[index])
        out: dict[str, Any] = {}

        # Residuals (mean over ensemble), reshaped to (n, C, T).
        res_stack = np.stack([m.predict(X) for m in self.residual_models_])
        n = X.shape[0]
        C = len(data.residual_channels)
        T = res_stack.shape[-1] // C
        out["residuals"] = res_stack.mean(axis=0).reshape(n, C, T)
        out["residual_std"] = res_stack.std(axis=0).mean(axis=1)

        # Continuous heads.
        cont: dict[str, np.ndarray] = {}
        cont_std: dict[str, np.ndarray] = {}
        for head, models in self.cont_models_.items():
            preds = np.stack([m.predict(X) for m in models])
            cont[head] = preds.mean(axis=0)
            cont_std[head] = preds.std(axis=0)
        # PFV hurdle: E[PFV] = P(active) * active_regression.
        if self.pfv_gate_:
            gates = np.stack([_proba(g, X) for g in self.pfv_gate_])
            acts = np.stack([r.predict(X) for r in self.pfv_active_])
            p_active = gates.mean(axis=0)
            cont["pfv"] = p_active * acts.mean(axis=0)
            cont_std["pfv"] = (gates * acts).std(axis=0)
            out["pfv_active_prob"] = p_active
        out["continuous"] = cont
        out["continuous_std"] = cont_std

        # Classification heads.
        cls: dict[str, np.ndarray] = {}
        cls_std: dict[str, np.ndarray] = {}
        for col, models in self.cls_models_.items():
            preds = np.stack([_proba(m, X) for m in models])
            cls[col] = preds.mean(axis=0)
            cls_std[col] = preds.std(axis=0)
        out["classification"] = cls
        out["classification_std"] = cls_std

        # Within-state ranking score: joint-noninferior probability if present,
        # else a safety-weighted composite.
        if "joint_noninferior" in cls:
            score = cls["joint_noninferior"]
        else:
            score = np.mean([cls[c] for c in cls], axis=0) if cls else np.zeros(n)
        out["ranking_score"] = score
        out["ranking"] = self._rank_within_state(data.state_key[index], score)

        # Uncertainty: aggregate ensemble std across heads.
        unc_parts = [out["residual_std"]]
        unc_parts += [cont_std[h] for h in cont_std]
        unc_parts += [cls_std[c] for c in cls_std]
        uncertainty = np.mean(np.stack(unc_parts), axis=0) if unc_parts else np.zeros(n)
        out["uncertainty"] = uncertainty

        # OOD: standardised distance to train centroid.
        z = (X - self.train_centroid_) / self.train_scale_
        out["ood_distance"] = np.sqrt(np.mean(z**2, axis=1))
        return out

    @staticmethod
    def _rank_within_state(
        state_key: np.ndarray, score: np.ndarray
    ) -> np.ndarray:
        ranks = np.zeros(len(score), dtype=float)
        for key in np.unique(state_key):
            idx = np.flatnonzero(state_key == key)
            order = np.argsort(-score[idx])  # higher score -> rank 0 (best)
            ranks[idx[order]] = np.arange(len(idx))
        return ranks

    # ----- persistence -----
    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(blob: bytes) -> "TrueStateEnsemble":
        return pickle.loads(blob)


# ---------------------------------------------------------------------------
# Calibration (Calibration split only -- never Locked)
# ---------------------------------------------------------------------------

def _temperature_fit(p: np.ndarray, y: np.ndarray) -> float:
    """1-D temperature on logits by simple grid search minimising Brier."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    best_t, best_loss = 1.0, np.inf
    for t in np.linspace(0.4, 3.0, 27):
        q = 1.0 / (1.0 + np.exp(-logit / t))
        loss = np.mean((q - y) ** 2)
        if loss < best_loss:
            best_loss, best_t = loss, t
    return float(best_t)


def calibrate(
    model: TrueStateEnsemble, data: TrainingData, *, cfg: ModelConfig | None = None
) -> dict[str, Any]:
    """Fit temperature / conformal quantiles / abstain thresholds on Calibration.

    Raises if the Calibration split is empty.  Never reads Locked.
    """
    cfg = cfg or model.cfg
    ca = data.split_index("calibration")
    if ca.size == 0:
        raise ValueError("calibration split empty; cannot calibrate")
    pred = model.predict(data, ca)

    temperatures: dict[str, float] = {}
    for col, p in pred["classification"].items():
        y = data.classification[col][ca]
        temperatures[col] = (
            _temperature_fit(p, y) if len(np.unique(y)) > 1 else 1.0
        )

    # Split-conformal absolute-residual quantiles for continuous heads.
    conformal: dict[str, float] = {}
    for head, yhat in pred["continuous"].items():
        y = data.continuous[head][ca]
        conformal[head] = float(np.quantile(np.abs(y - yhat), 0.9))

    abstain_threshold = float(
        np.quantile(pred["uncertainty"], cfg.abstain_uncertainty_quantile)
    )
    ood_threshold = float(np.quantile(pred["ood_distance"], cfg.ood_quantile))
    return {
        "temperatures": temperatures,
        "conformal_abs_q90": conformal,
        "abstain_uncertainty_threshold": abstain_threshold,
        "ood_threshold": ood_threshold,
        "calibration_n": int(ca.size),
        "split_used": "calibration",
    }


def apply_calibration(
    pred: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    out = dict(pred)
    temps = calibration.get("temperatures", {})
    cls_cal = {}
    for col, p in pred.get("classification", {}).items():
        t = float(temps.get(col, 1.0))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
        cls_cal[col] = 1.0 / (1.0 + np.exp(-logit / t))
    out["classification_calibrated"] = cls_cal
    unc_t = float(calibration.get("abstain_uncertainty_threshold", np.inf))
    ood_t = float(calibration.get("ood_threshold", np.inf))
    out["abstain"] = (
        (pred.get("uncertainty", np.zeros(1)) > unc_t)
        | (pred.get("ood_distance", np.zeros(1)) > ood_t)
    )
    return out


# ---------------------------------------------------------------------------
# Constant fallbacks (degenerate single-class / tiny-support cases)
# ---------------------------------------------------------------------------

class _ConstantProba:
    def __init__(self, p: float):
        self.p = float(np.clip(p, 0.0, 1.0))

    def predict_proba(self, X):  # noqa: N802 (sklearn-like)
        n = X.shape[0]
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


class _ConstantReg:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X):
        return np.full(X.shape[0], self.value)


def _proba(model, X) -> np.ndarray:
    proba = model.predict_proba(X)
    return proba[:, 1] if proba.ndim == 2 else proba


# ---------------------------------------------------------------------------
# Locked one-shot evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_split(
    model: TrueStateEnsemble,
    data: TrainingData,
    split: str,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    idx = data.split_index(split)
    if idx.size == 0:
        return {"split": split, "n": 0, "empty": True}
    pred = model.predict(data, idx)
    if calibration is not None:
        pred = apply_calibration(pred, calibration)
    report: dict[str, Any] = {"split": split, "n": int(idx.size)}

    cont_report = {}
    for head, yhat in pred["continuous"].items():
        y = data.continuous[head][idx]
        cont_report[head] = _reg_metrics(y, yhat)
    report["continuous"] = cont_report

    cls_report = {}
    cls_source = pred.get("classification_calibrated", pred["classification"])
    for col, p in cls_source.items():
        y = data.classification[col][idx]
        cls_report[col] = _cls_metrics(y, p)
    report["classification"] = cls_report

    # Residual channel MAE (mean across steps).
    res_pred = pred["residuals"]
    res_true = data.residuals[idx]
    report["residual_mae"] = {
        ch: float(np.mean(np.abs(res_pred[:, c, :] - res_true[:, c, :])))
        for c, ch in enumerate(data.residual_channels)
    }
    if "abstain" in pred:
        report["abstain_rate"] = float(np.mean(pred["abstain"]))
    return report
