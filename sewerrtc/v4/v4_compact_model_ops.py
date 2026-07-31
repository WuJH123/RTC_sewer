"""V4.1 head-architecture, gradient, selection and compact-model ops (spec 8-11).

Train-only and offline.  Architectures A-D and the sequence-derived Peak head
are compared with event-grouped CV; the multitask gradient audit measures
whether the KPI heads pull a shared linear encoder in conflicting directions;
selection reads only Train-grouped evidence; the compact model trains on the
original Train 1200 with 5 deterministic seeds and event-balanced weights.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .train_v4_loader import TrainingData
from .train_v4_metrics import classification_metrics, decision_metrics, regression_metrics
from .train_v4_models import class_balance_weights, combined_sample_weights
from .train_v4_preflight import event_grouped_folds
from .v4_compact_diag_ops import (
    CLASSIFICATION_HEADS,
    CONTINUOUS_HEADS,
    classify_feature_block,
)

# Head-specific physical feature blocks (spec section 8 priorities).
HEAD_FEATURE_BLOCKS: dict[str, set[str]] = {
    "pfv": {"A", "B", "E", "G", "H", "I", "K", "M"},
    "tfv": {"C", "D", "G", "H", "I", "M"},
    "peak": {"C", "D", "E", "F", "G", "H", "I", "K"},
}
TFV_RATE_CHANNEL = "tfv_rate_residual"


def _block_arr(data: TrainingData) -> np.ndarray:
    return np.array([classify_feature_block(n) for n in data.feature_names])


def _head_idx(data: TrainingData, head: str, block_arr: np.ndarray) -> np.ndarray:
    blocks = HEAD_FEATURE_BLOCKS.get(head, set())
    idx = np.flatnonzero(np.isin(block_arr, list(blocks)))
    return idx if idx.size else np.arange(data.features.shape[1])


def _reg(y: np.ndarray, pred: np.ndarray, dz: float) -> dict[str, Any]:
    m = regression_metrics(y, pred, dead_zone=dz)
    var = float(np.var(y)) or 1.0
    return {
        "mae": m["mae"],
        "r2": float(1.0 - np.mean((pred - y) ** 2) / var),
        "sign_accuracy": m["sign_accuracy_outside_dead_zone"],
    }


def _hgb_reg(cfg, seed):
    return HistGradientBoostingRegressor(
        max_iter=cfg.hgb_max_iter, max_depth=cfg.hgb_max_depth,
        learning_rate=cfg.hgb_learning_rate, random_state=seed,
    )


# ---------------------------------------------------------------------------
# Section 8: head-architecture ablation
# ---------------------------------------------------------------------------

def _predict_arch_head(arch: str, head: str, data, fit_idx, val_idx, feat_all, block_arr, cfg, seed):
    """Return validation-side prediction for one continuous head + architecture."""
    y = data.continuous[head]
    yf = y[feat_all][fit_idx]
    Xall = data.features
    use = feat_all
    if arch == "A":  # shared full-feature encoder, per-head HGB
        sc = StandardScaler().fit(Xall[use][fit_idx])
        Xf, Xv = sc.transform(Xall[use][fit_idx]), sc.transform(Xall[use][val_idx])
        return _hgb_reg(cfg, seed).fit(Xf, yf).predict(Xv)
    if arch == "B":  # shared shallow PCA encoder + head-specific raw block
        sc = StandardScaler().fit(Xall[use][fit_idx])
        Zf, Zv = sc.transform(Xall[use][fit_idx]), sc.transform(Xall[use][val_idx])
        n_comp = int(min(20, max(2, Zf.shape[1] - 1)))
        pca = PCA(n_components=n_comp, random_state=seed).fit(Zf)
        hi = _head_idx(data, head, block_arr)
        hsc = StandardScaler().fit(Xall[use][fit_idx][:, hi])
        Ff = np.hstack([pca.transform(Zf), hsc.transform(Xall[use][fit_idx][:, hi])])
        Fv = np.hstack([pca.transform(Zv), hsc.transform(Xall[use][val_idx][:, hi])])
        return _hgb_reg(cfg, seed).fit(Ff, yf).predict(Fv)
    if arch == "C":  # fully independent compact per-head model
        hi = _head_idx(data, head, block_arr)
        sc = StandardScaler().fit(Xall[use][fit_idx][:, hi])
        Xf, Xv = sc.transform(Xall[use][fit_idx][:, hi]), sc.transform(Xall[use][val_idx][:, hi])
        return _hgb_reg(cfg, seed).fit(Xf, yf).predict(Xv)
    if arch == "D":  # simple head-specific families
        hi = _head_idx(data, head, block_arr)
        sc = StandardScaler().fit(Xall[use][fit_idx][:, hi])
        Xf, Xv = sc.transform(Xall[use][fit_idx][:, hi]), sc.transform(Xall[use][val_idx][:, hi])
        if head == "pfv":  # hurdle: gate + active ElasticNet
            active = (np.abs(yf) > 1.0).astype(int)
            if len(np.unique(active)) > 1:
                gate = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xf, active)
                pg = gate.predict_proba(Xv)[:, 1]
            else:
                pg = np.full(len(val_idx), float(active.mean()))
            act = np.flatnonzero(active == 1)
            if act.size >= 5:
                reg = ElasticNet(alpha=0.01, max_iter=5000).fit(Xf[act], yf[act])
                return pg * reg.predict(Xv)
            return pg * float(yf.mean())
        if head == "tfv":  # PLS
            n_comp = int(min(10, max(1, Xf.shape[1] - 1)))
            return PLSRegression(n_components=n_comp).fit(Xf, yf).predict(Xv).reshape(-1)
        return _hgb_reg(cfg, seed).fit(Xf, yf).predict(Xv)  # peak GBDT
    raise ValueError(arch)


def _sequence_peak_prediction(data, fit_idx, val_idx, feat_all, block_arr, cfg, seed):
    """Predict 12-step tfv-rate residual sequence, derive Peak, calibrate to
    the Peak target; return (direct, sequence, consistency) predictions."""
    use = feat_all
    hi = _head_idx(data, "peak", block_arr)
    sc = StandardScaler().fit(data.features[use][fit_idx][:, hi])
    Xf, Xv = sc.transform(data.features[use][fit_idx][:, hi]), sc.transform(data.features[use][val_idx][:, hi])
    yf = data.continuous["peak"][use][fit_idx]
    direct = _hgb_reg(cfg, seed).fit(Xf, yf).predict(Xv)
    if TFV_RATE_CHANNEL in data.residual_channels:
        ch = data.residual_channels.index(TFV_RATE_CHANNEL)
        seq_f = data.residuals[use][fit_idx][:, ch, :]  # (n, 12)
        from sklearn.multioutput import MultiOutputRegressor
        seq_model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=seed)).fit(Xf, seq_f)
        seq_pred_f = seq_model.predict(Xf)
        seq_pred_v = seq_model.predict(Xv)
        stat_f = seq_pred_f.max(axis=1, keepdims=True)
        stat_v = seq_pred_v.max(axis=1, keepdims=True)
        calib = Ridge(alpha=1.0).fit(stat_f, yf)  # map sequence stat -> peak
        sequence = calib.predict(stat_v)
    else:
        sequence = direct.copy()
    consistency = 0.5 * (direct + sequence)
    return direct, sequence, consistency


def run_head_architecture_ablation(
    data: TrainingData, *, cfg, dead_zones: dict[str, float], seed: int = 0, n_folds: int = 4
) -> pd.DataFrame:
    block_arr = _block_arr(data)
    tr = data.split_index("train")
    ev = data.event_id[tr]
    folds = event_grouped_folds(ev, n_folds=n_folds, seed=seed)
    rows: list[dict[str, Any]] = []
    for arch in ("A", "B", "C", "D"):
        for head in CONTINUOUS_HEADS:
            if head not in data.continuous:
                continue
            dz = float(dead_zones.get(head, 0.0))
            fold_maes, fold_r2 = [], []
            for fit_idx, val_idx in folds:
                pred = _predict_arch_head(arch, head, data, fit_idx, val_idx, tr, block_arr, cfg, seed)
                yv = data.continuous[head][tr][val_idx]
                r = _reg(yv, np.asarray(pred).reshape(-1), dz)
                fold_maes.append(r["mae"]); fold_r2.append(r["r2"])
            rows.append({
                "architecture": arch, "head": head,
                "cv_mae_mean": float(np.mean(fold_maes)),
                "cv_mae_worst": float(np.max(fold_maes)),
                "cv_r2_mean": float(np.mean(fold_r2)),
                "n_head_features": int(_head_idx(data, head, block_arr).size),
            })
    # sequence-derived Peak comparison
    for variant_idx, variant in enumerate(("direct", "sequence", "consistency")):
        maes, r2s = [], []
        for fit_idx, val_idx in folds:
            d, s, c = _sequence_peak_prediction(data, fit_idx, val_idx, tr, block_arr, cfg, seed)
            pred = (d, s, c)[variant_idx]
            yv = data.continuous["peak"][tr][val_idx]
            r = _reg(yv, pred, float(dead_zones.get("peak", 0.0)))
            maes.append(r["mae"]); r2s.append(r["r2"])
        rows.append({
            "architecture": f"peak_{variant}", "head": "peak",
            "cv_mae_mean": float(np.mean(maes)), "cv_mae_worst": float(np.max(maes)),
            "cv_r2_mean": float(np.mean(r2s)), "n_head_features": int(_head_idx(data, "peak", block_arr).size),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 9: multitask gradient-conflict audit (shared linear encoder)
# ---------------------------------------------------------------------------

def audit_gradient_conflict(data: TrainingData, *, seed: int = 0) -> dict[str, Any]:
    """Per-head gradient of a shared linear encoder at init; pairwise cosine.

    Standardize X and each head target so gradient magnitudes are comparable;
    the init gradient of MSE at w=0 is ``-X^T y_std / n`` -- its direction is
    the pull each head exerts on shared weights.  Conflict = cosine < 0.
    """
    tr = data.split_index("train")
    X = StandardScaler().fit_transform(data.features[tr])
    n = X.shape[0]
    grads: dict[str, np.ndarray] = {}

    def _std_target(y):
        s = y.std() or 1.0
        return (y - y.mean()) / s

    for head in CONTINUOUS_HEADS:
        if head in data.continuous:
            grads[head] = -X.T @ _std_target(data.continuous[head][tr]) / n
    # process head: mean over residual channels/steps
    proc = data.residuals[tr].reshape(n, -1).mean(axis=1)
    grads["process"] = -X.T @ _std_target(proc) / n
    if "joint_noninferior" in data.classification:
        grads["ranking"] = -X.T @ _std_target(data.classification["joint_noninferior"][tr].astype(float)) / n

    heads = list(grads)
    pair_rows = []
    conflicts = 0
    total = 0
    for i in range(len(heads)):
        for j in range(i + 1, len(heads)):
            a, b = grads[heads[i]], grads[heads[j]]
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
            cos = float(a @ b / denom)
            pair_rows.append({"head_a": heads[i], "head_b": heads[j], "cosine": cos})
            total += 1
            conflicts += int(cos < 0)
    dominant = max(heads, key=lambda h: float(np.linalg.norm(grads[h])))
    return {
        "stage": "AuditV4MultitaskGradientConflictV1",
        "heads": heads,
        "pairwise_cosine": pair_rows,
        "conflict_fraction": float(conflicts / total) if total else 0.0,
        "persistent_conflict": bool(total and conflicts / total > 0.5),
        "shared_encoder_dominant_head": dominant,
        "loss_scale": {h: float(np.linalg.norm(grads[h])) for h in heads},
        "recommend_multitask_mitigation": bool(total and conflicts / total > 0.5),
        "note": "Only recommend PCGrad/GradNorm/encoder split on persistent conflict.",
    }


# ---------------------------------------------------------------------------
# Section 10: compact-model selection (Train-grouped evidence only)
# ---------------------------------------------------------------------------

def select_compact_model(
    *,
    learning_diag: dict[str, Any],
    ablation: pd.DataFrame,
    architecture: pd.DataFrame,
    gradient: dict[str, Any],
) -> dict[str, Any]:
    """Decision-first selection.  Reads only Train-grouped artifacts."""
    # Best architecture per head by CV MAE (decision proxy = low error + parsimony).
    arch_cont = architecture[architecture["architecture"].isin(["A", "B", "C", "D"])]
    head_choice: dict[str, str] = {}
    for head in CONTINUOUS_HEADS:
        h = arch_cont[arch_cont["head"] == head]
        if h.empty:
            continue
        # prefer lowest cv_mae_mean, tie-break fewer features
        h = h.sort_values(["cv_mae_mean", "n_head_features"])
        head_choice[head] = str(h.iloc[0]["architecture"])
    # Peak head style: pick best of direct/sequence/consistency.
    peak_variants = architecture[architecture["architecture"].str.startswith("peak_")]
    peak_style = "direct"
    if not peak_variants.empty:
        peak_style = str(peak_variants.sort_values("cv_mae_mean").iloc[0]["architecture"]).replace("peak_", "")
    # Overall architecture: majority vote across heads (fallback C = independent).
    if head_choice:
        vals, counts = np.unique(list(head_choice.values()), return_counts=True)
        overall_arch = str(vals[int(np.argmax(counts))])
    else:
        overall_arch = "C"
    # Feature combo: lowest mean cv_mae with parsimony from ablation continuous rows.
    abl = ablation[ablation["head"].isin(CONTINUOUS_HEADS) & ablation["cv_mae_mean"].notna()]
    combo_choice = "state_action_rain"
    if not abl.empty:
        grp = abl.groupby("combo").agg(mae=("cv_mae_mean", "mean"), nf=("n_features", "max")).reset_index()
        grp = grp.sort_values(["mae", "nf"])
        combo_choice = str(grp.iloc[0]["combo"])
    return {
        "stage": "SelectV4CompactModelV1",
        "reads_old_calibration": False,
        "reads_old_locked": False,
        "reads_new_calibration": False,
        "reads_new_locked": False,
        "selection_basis": [
            "train_grouped_cv", "learning_curves", "feature_ablation",
            "architecture_ablation", "gradient_audit",
        ],
        "selected_architecture": overall_arch,
        "per_head_architecture": head_choice,
        "peak_head_style": peak_style,
        "selected_feature_combo": combo_choice,
        "head_feature_blocks": {h: sorted(HEAD_FEATURE_BLOCKS[h]) for h in HEAD_FEATURE_BLOCKS},
        "multitask_mitigation": gradient.get("recommend_multitask_mitigation", False),
        "learning_curve_verdict": learning_diag,
        "frozen_contract": {
            "seeds": [0, 1, 2, 3, 4],
            "pfv_hurdle": True,
            "full_event_heads_disabled": True,
            "online_disabled_k": [1, 2],
            "preprocessing": "per_head_standard_scaler",
            "uncertainty": "seed_ensemble_std",
        },
    }


# ---------------------------------------------------------------------------
# Section 11: compact head-specific model
# ---------------------------------------------------------------------------

@dataclass
class CompactHeadSpecificModel:
    """Independent per-head compact models on head-specific feature blocks."""

    cfg: Any
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    pfv_dead_zone: float = 1.0
    peak_style: str = "direct"
    feature_names_: list[str] = field(default_factory=list)
    head_idx_: dict[str, np.ndarray] = field(default_factory=dict)
    scalers_: dict[str, Any] = field(default_factory=dict)
    cont_: dict[str, list] = field(default_factory=dict)
    pfv_gate_: list = field(default_factory=list)
    pfv_active_: list = field(default_factory=list)
    cls_: dict[str, list] = field(default_factory=dict)

    def fit(self, data: TrainingData) -> "CompactHeadSpecificModel":
        tr = data.split_index("train")
        block_arr = _block_arr(data)
        self.feature_names_ = list(data.feature_names)
        w = combined_sample_weights(data, tr, None, cfg=self.cfg)
        for head in ("tfv", "peak"):
            if head not in data.continuous:
                continue
            hi = _head_idx(data, head, block_arr)
            self.head_idx_[head] = hi
            sc = StandardScaler().fit(data.features[tr][:, hi])
            self.scalers_[head] = sc
            X = sc.transform(data.features[tr][:, hi])
            y = data.continuous[head][tr]
            self.cont_[head] = [_hgb_reg(self.cfg, s).fit(X, y, sample_weight=w) for s in self.seeds]
        # PFV hurdle
        if "pfv" in data.continuous:
            hi = _head_idx(data, "pfv", block_arr)
            self.head_idx_["pfv"] = hi
            sc = StandardScaler().fit(data.features[tr][:, hi])
            self.scalers_["pfv"] = sc
            X = sc.transform(data.features[tr][:, hi])
            y = data.continuous["pfv"][tr]
            active = (np.abs(y) > self.pfv_dead_zone).astype(int)
            for s in self.seeds:
                if len(np.unique(active)) > 1:
                    gate = HistGradientBoostingClassifier(
                        max_iter=self.cfg.hgb_max_iter, max_depth=self.cfg.hgb_max_depth,
                        learning_rate=self.cfg.hgb_learning_rate, random_state=s,
                    ).fit(X, active, sample_weight=w * class_balance_weights(active))
                else:
                    from .train_v4_models import _ConstantProba
                    gate = _ConstantProba(float(active.mean()))
                self.pfv_gate_.append(gate)
                act = np.flatnonzero(active == 1)
                if act.size >= 5:
                    reg = _hgb_reg(self.cfg, s).fit(X[act], y[act], sample_weight=w[act])
                else:
                    from .train_v4_models import _ConstantReg
                    reg = _ConstantReg(float(y.mean()))
                self.pfv_active_.append(reg)
        # classification heads (shared full-feature standardization)
        self.head_idx_["_cls"] = np.arange(data.features.shape[1])
        sc = StandardScaler().fit(data.features[tr])
        self.scalers_["_cls"] = sc
        Xc = sc.transform(data.features[tr])
        for col in CLASSIFICATION_HEADS:
            if col not in data.classification:
                continue
            y = data.classification[col][tr]
            if len(np.unique(y)) < 2:
                from .train_v4_models import _ConstantProba
                self.cls_[col] = [_ConstantProba(float(y.mean()))]
                continue
            ww = w * class_balance_weights(y)
            self.cls_[col] = [
                HistGradientBoostingClassifier(
                    max_iter=self.cfg.hgb_max_iter, max_depth=self.cfg.hgb_max_depth,
                    learning_rate=self.cfg.hgb_learning_rate, random_state=s,
                ).fit(Xc, y, sample_weight=ww) for s in self.seeds
            ]
        return self

    def predict(self, data: TrainingData, index: np.ndarray) -> dict[str, Any]:
        from .train_v4_models import _proba
        out: dict[str, Any] = {"continuous": {}, "continuous_std": {}, "classification": {}}
        for head in ("tfv", "peak"):
            if head not in self.cont_:
                continue
            X = self.scalers_[head].transform(data.features[index][:, self.head_idx_[head]])
            preds = np.stack([m.predict(X) for m in self.cont_[head]])
            out["continuous"][head] = preds.mean(axis=0)
            out["continuous_std"][head] = preds.std(axis=0)
        if self.pfv_gate_:
            X = self.scalers_["pfv"].transform(data.features[index][:, self.head_idx_["pfv"]])
            gates = np.stack([_proba(g, X) for g in self.pfv_gate_])
            acts = np.stack([r.predict(X) for r in self.pfv_active_])
            out["continuous"]["pfv"] = gates.mean(axis=0) * acts.mean(axis=0)
            out["continuous_std"]["pfv"] = (gates * acts).std(axis=0)
        if self.cls_:
            Xc = self.scalers_["_cls"].transform(data.features[index])
            for col, models in self.cls_.items():
                preds = np.stack([_proba(m, Xc) for m in models])
                out["classification"][col] = preds.mean(axis=0)
        std_parts = list(out["continuous_std"].values())
        out["uncertainty"] = np.mean(np.stack(std_parts), axis=0) if std_parts else np.zeros(len(index))
        return out

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)


def compact_cv_report(
    model_factory, data: TrainingData, *, cfg, dead_zones: dict[str, float], n_folds: int = 4, seed: int = 0
) -> dict[str, Any]:
    """Event-grouped CV of the compact model on Train (predictions + metrics)."""
    tr = data.split_index("train")
    ev = data.event_id[tr]
    folds = event_grouped_folds(ev, n_folds=n_folds, seed=seed)
    n = tr.size
    oof_cont = {h: np.full(n, np.nan) for h in CONTINUOUS_HEADS if h in data.continuous}
    oof_cls = {c: np.full(n, np.nan) for c in CLASSIFICATION_HEADS if c in data.classification}
    for fit_idx, val_idx in folds:
        sub = _SubTrainingData(data, tr[fit_idx])
        model = model_factory().fit(sub)
        pred = model.predict(data, tr[val_idx])
        for h in oof_cont:
            oof_cont[h][val_idx] = pred["continuous"].get(h, np.full(len(val_idx), np.nan))
        for c in oof_cls:
            if c in pred["classification"]:
                oof_cls[c][val_idx] = pred["classification"][c]
    by_event_rows, metric_rows = [], []
    for h, p in oof_cont.items():
        y = data.continuous[h][tr]
        valid = np.isfinite(p)
        r = _reg(y[valid], p[valid], float(dead_zones.get(h, 0.0)))
        metric_rows.append({"head": h, "kind": "continuous", **r})
        for e in np.unique(ev[valid]):
            m = (ev == e) & valid
            by_event_rows.append({"head": h, "event_id": str(e), "mae": float(np.mean(np.abs(p[m] - y[m])))})
    for c, p in oof_cls.items():
        y = data.classification[c][tr].astype(int)
        valid = np.isfinite(p)
        m = classification_metrics(y[valid], p[valid])
        metric_rows.append({"head": c, "kind": "classification", "balanced_accuracy": m.get("balanced_accuracy"),
                            "mcc": m.get("mcc"), "auroc": m.get("auroc"), "average_precision": m.get("average_precision"),
                            "false_safe_rate": m.get("false_safe_rate")})
    # decision metrics on ranking score = joint prob
    dm = {}
    if "joint_noninferior" in oof_cls:
        score = oof_cls["joint_noninferior"]
        feasible = data.classification["joint_noninferior"][tr].astype(bool)
        regret = data.ranking.get("regret_to_exact_best", np.full(len(data.split), np.nan))[tr]
        valid = np.isfinite(score)
        dm = decision_metrics(state_key=data.state_key[tr][valid], score=score[valid],
                              feasible_true=feasible[valid], regret=regret[valid])
        dm.pop("per_state", None)
    pred_df = pd.DataFrame({"row_index_in_train": np.arange(n), "event_id": ev.astype(str),
                            "state_key": data.state_key[tr].astype(str),
                            **{f"pred_{h}": oof_cont[h] for h in oof_cont},
                            **{f"pred_{c}": oof_cls[c] for c in oof_cls}})
    return {"predictions": pred_df, "by_event": pd.DataFrame(by_event_rows),
            "metrics": metric_rows, "decision": dm}


class _SubTrainingData:
    """Read-only view exposing only a subset of Train rows as split='train'."""

    def __init__(self, data: TrainingData, keep_rows: np.ndarray):
        self._d = data
        self._keep = keep_rows
        self.feature_names = data.feature_names
        self.features = data.features
        self.continuous = data.continuous
        self.classification = data.classification
        self.residuals = data.residuals
        self.residual_channels = data.residual_channels
        self.ranking = data.ranking
        self.event_id = data.event_id
        self.state_key = data.state_key
        self.hard_negative_type = data.hard_negative_type
        self._split = np.array(["unused"] * len(data.split), dtype=object)
        self._split[keep_rows] = "train"

    def split_index(self, name: str) -> np.ndarray:
        return np.flatnonzero(self._split == name)

    @property
    def split(self) -> np.ndarray:
        return self._split
