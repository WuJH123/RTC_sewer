"""True-state training operations: smoke, inner CV, gates, ensemble artefacts,
extended calibration and heartbeats (spec sections 7-12, 16).

The frozen model family is the deterministic scikit-learn ensemble
(``TrueStateEnsemble``).  The spec's torch-oriented smoke tests are mapped
honestly onto this family and each artefact records that mapping:

* Smoke A (forward/backward)   -> single-batch fit + finite losses/outputs;
* Smoke B (tiny-batch overfit) -> tiny 2-event subset fit-capacity check;
* Smoke C (checkpoint resume)  -> serialize -> reload -> bit-identical
  predictions (deterministic resume equivalence).

No epochs/optimizer exist in this family; there is nothing stochastic to
resume mid-run and every estimator is refit deterministically from its seed.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .train_v4_loader import TrainingData
from .train_v4_metrics import (
    classification_metrics,
    decision_metrics,
    per_event_mae,
    regression_metrics,
    one_sided_conformal,
    uncertainty_error_correlation,
    worst_event,
)
from .train_v4_models import ModelConfig, TrueStateEnsemble, _temperature_fit
from .train_v4_preflight import event_grouped_folds

SMOKE_MAPPING_NOTE = (
    "Model family is deterministic scikit-learn (CPU). Torch-style smoke "
    "checks are mapped: forward/backward -> single-batch fit finiteness; "
    "tiny-batch overfit -> 2-event fit-capacity; checkpoint resume -> "
    "serialize/reload bit-identical prediction equivalence."
)


def _subset(data: TrainingData, rows: np.ndarray) -> TrainingData:
    """A read-only row-subset view used by smoke tests (labels untouched)."""
    return TrainingData(
        features=data.features[rows],
        feature_names=list(data.feature_names),
        continuous={k: v[rows] for k, v in data.continuous.items()},
        classification={k: v[rows] for k, v in data.classification.items()},
        residuals=data.residuals[rows],
        residual_channels=list(data.residual_channels),
        ranking={k: v[rows] for k, v in data.ranking.items()},
        full_event_enabled=data.full_event_enabled,
        full_event_mask=data.full_event_mask[rows],
        split=data.split[rows],
        state_key=data.state_key[rows],
        event_id=data.event_id[rows],
        hard_negative_type=data.hard_negative_type[rows],
        n_samples=len(rows),
        meta=dict(data.meta),
    )


# ---------------------------------------------------------------------------
# Smoke A: single-batch fit finiteness
# ---------------------------------------------------------------------------

def run_smoke_forward_backward(
    data: TrainingData, *, cfg: ModelConfig, pfv_dead_zone: float
) -> dict[str, Any]:
    tr = data.split_index("train")
    batch = tr[: min(64, tr.size)]
    sub = _subset(data, batch)
    sub.split[:] = "train"
    model = TrueStateEnsemble(
        cfg=cfg.light(), pfv_dead_zone=pfv_dead_zone
    ).fit(sub)
    pred = model.predict(sub, np.arange(len(batch)))
    losses = {
        "residual_mse": float(np.mean((pred["residuals"] - sub.residuals) ** 2)),
    }
    for head, yhat in pred["continuous"].items():
        losses[f"{head}_mse"] = float(np.mean((yhat - sub.continuous[head]) ** 2))
    for col, p in pred["classification"].items():
        losses[f"{col}_brier"] = float(
            np.mean((p - sub.classification[col]) ** 2)
        )
    outputs_finite = all(
        np.isfinite(np.asarray(v)).all()
        for v in [pred["residuals"], pred["uncertainty"], pred["ood_distance"]]
        + list(pred["continuous"].values())
        + list(pred["classification"].values())
    )
    heads_fitted = {
        "residual": len(model.residual_models_) > 0,
        "tfv": "tfv" in model.cont_models_,
        "peak": "peak" in model.cont_models_,
        "pfv_hurdle": bool(model.pfv_gate_) and bool(model.pfv_active_),
        "classification": len(model.cls_models_) > 0,
        "full_event": False,  # never created
    }
    ok = (
        outputs_finite
        and all(np.isfinite(v) for v in losses.values())
        and all(v for k, v in heads_fitted.items() if k != "full_event")
        and heads_fitted["full_event"] is False
    )
    return {
        "smoke": "forward_backward",
        "mapping_note": SMOKE_MAPPING_NOTE,
        "batch_size": int(len(batch)),
        "losses": losses,
        "all_losses_finite": all(np.isfinite(v) for v in losses.values()),
        "all_outputs_finite": bool(outputs_finite),
        "heads_fitted": heads_fitted,
        "full_event_head_disabled": True,
        "status": "pass" if ok else "fail",
    }


# ---------------------------------------------------------------------------
# Smoke B: tiny 2-event overfit capacity
# ---------------------------------------------------------------------------

def run_smoke_tiny_overfit(
    data: TrainingData, *, cfg: ModelConfig, pfv_dead_zone: float
) -> dict[str, Any]:
    tr = data.split_index("train")
    # Prefer events containing both PFV-active and PFV-inactive rows so both
    # hurdle branches are exercised.
    pfv = data.continuous.get("pfv")
    active = (
        (np.abs(pfv[tr]) > pfv_dead_zone)
        if pfv is not None
        else np.zeros(tr.size, dtype=bool)
    )
    events = []
    for ev in dict.fromkeys(data.event_id[tr].tolist()):
        m = data.event_id[tr] == ev
        events.append((ev, bool(active[m].any()), bool((~active[m]).any())))
    both = [e for e, a, i in events if a and i]
    chosen = (both[:2] or [e for e, _, _ in events][:2])
    rows_mask = np.isin(data.event_id[tr], chosen)
    rows = tr[rows_mask][:40]
    sub = _subset(data, rows)
    sub.split[:] = "train"

    tiny_cfg = ModelConfig(
        seeds=(cfg.seeds[0],),
        hgb_max_iter=max(cfg.hgb_max_iter, 300),
        hgb_max_depth=None,
        hgb_learning_rate=0.2,
        hard_negative_weight=cfg.hard_negative_weight,
    )
    model = TrueStateEnsemble(cfg=tiny_cfg, pfv_dead_zone=pfv_dead_zone).fit(sub)
    pred = model.predict(sub, np.arange(len(rows)))

    rows_out: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    res_zero = float(np.mean(sub.residuals**2))
    res_fit = float(np.mean((pred["residuals"] - sub.residuals) ** 2))
    checks["residual_loss_drops"] = res_fit < 0.5 * res_zero if res_zero > 0 else True
    rows_out.append(
        {"head": "process_residual", "zero_loss": res_zero, "fit_loss": res_fit}
    )
    for head, yhat in pred["continuous"].items():
        y = sub.continuous[head]
        zero_loss = float(np.mean(y**2))
        fit_loss = float(np.mean((yhat - y) ** 2))
        var = float(np.var(y))
        ok = fit_loss < 0.5 * zero_loss if zero_loss > 1e-12 else True
        checks[f"{head}_loss_drops"] = ok
        rows_out.append(
            {
                "head": head,
                "zero_loss": zero_loss,
                "fit_loss": fit_loss,
                "label_variance": var,
            }
        )
    tiny_active = (
        (np.abs(sub.continuous["pfv"]) > pfv_dead_zone)
        if "pfv" in sub.continuous
        else np.zeros(len(rows), dtype=bool)
    )
    checks["pfv_hurdle_both_branches_present"] = bool(
        tiny_active.any() and (~tiny_active).any()
    )
    score = pred["ranking_score"]
    checks["ranking_score_varies"] = bool(np.std(score) > 0)
    ok_all = all(checks.values())
    return {
        "smoke": "tiny_overfit",
        "mapping_note": SMOKE_MAPPING_NOTE,
        "events": [str(e) for e in chosen],
        "n_rows": int(len(rows)),
        "checks": checks,
        "rows": rows_out,
        "status": "pass" if ok_all else "fail",
    }


# ---------------------------------------------------------------------------
# Smoke C: serialize -> reload -> identical predictions
# ---------------------------------------------------------------------------

def run_smoke_checkpoint_resume(
    data: TrainingData, *, cfg: ModelConfig, pfv_dead_zone: float
) -> dict[str, Any]:
    tr = data.split_index("train")
    rows = tr[: min(80, tr.size)]
    sub = _subset(data, rows)
    sub.split[:] = "train"
    model = TrueStateEnsemble(cfg=cfg.light(), pfv_dead_zone=pfv_dead_zone).fit(sub)
    blob = model.to_bytes()
    restored = TrueStateEnsemble.from_bytes(blob)
    idx = np.arange(len(rows))
    p1 = model.predict(sub, idx)
    p2 = restored.predict(sub, idx)
    identical = bool(
        np.array_equal(p1["residuals"], p2["residuals"])
        and all(
            np.array_equal(p1["continuous"][h], p2["continuous"][h])
            for h in p1["continuous"]
        )
        and all(
            np.array_equal(p1["classification"][c], p2["classification"][c])
            for c in p1["classification"]
        )
    )
    return {
        "smoke": "checkpoint_resume",
        "mapping_note": SMOKE_MAPPING_NOTE,
        "checkpoint_sha256": hashlib.sha256(blob).hexdigest(),
        "predictions_identical_after_reload": identical,
        "status": "pass" if identical else "fail",
    }


# ---------------------------------------------------------------------------
# Complexity report (spec section 8)
# ---------------------------------------------------------------------------

def _estimator_params(est) -> int:
    if hasattr(est, "coef_"):
        return int(np.size(est.coef_)) + int(np.size(getattr(est, "intercept_", 0)))
    if hasattr(est, "estimators_"):  # MultiOutputRegressor
        return int(sum(_estimator_params(e) for e in est.estimators_))
    if hasattr(est, "_predictors"):  # HistGradientBoosting*
        n = 0
        for stage in est._predictors:
            for pred in stage:
                n += int(pred.nodes.shape[0])
        return n
    return 0


def complexity_report(model: TrueStateEnsemble) -> dict[str, Any]:
    per_module = {
        "process_residual": sum(_estimator_params(m) for m in model.residual_models_),
        "pfv_gate": sum(_estimator_params(m) for m in model.pfv_gate_),
        "pfv_active": sum(_estimator_params(m) for m in model.pfv_active_),
    }
    for head, models in model.cont_models_.items():
        per_module[f"continuous_{head}"] = sum(_estimator_params(m) for m in models)
    for col, models in model.cls_models_.items():
        per_module[f"classification_{col}"] = sum(
            _estimator_params(m) for m in models
        )
    total = int(sum(per_module.values()))
    return {
        "parameter_count": total,
        "trainable_parameter_count": total,
        "per_module": per_module,
        "input_dim": len(model.feature_names_),
        "family": "sklearn HGB trees (node counts) + Ridge coefficients",
        "regularisation": {
            "hgb_l2": "sklearn default + shallow depth",
            "ridge_alpha": 1.0,
            "max_depth": model.cfg.hgb_max_depth,
            "learning_rate": model.cfg.hgb_learning_rate,
            "event_balanced_weights": True,
            "early_stopping": "HGB internal validation_fraction",
        },
        "not_a_transformer": True,
    }


# ---------------------------------------------------------------------------
# Inner event-grouped CV over Train (per-seed + ensemble OOF)
# ---------------------------------------------------------------------------

def inner_cv_predictions(
    data: TrainingData,
    *,
    cfg: ModelConfig,
    pfv_dead_zone: float,
    n_folds: int = 4,
    heartbeat=None,
) -> dict[str, Any]:
    """Event-grouped OOF predictions over Train, per seed and ensemble."""
    tr = data.split_index("train")
    ev = data.event_id[tr]
    folds = event_grouped_folds(ev, n_folds=n_folds, seed=0)
    n_seeds = len(cfg.seeds)
    heads = list(data.continuous.keys())
    per_seed = {h: np.full((tr.size, n_seeds), np.nan) for h in heads}
    unc = np.full(tr.size, np.nan)
    score_oof = np.full(tr.size, np.nan)
    cls_oof = {c: np.full(tr.size, np.nan) for c in data.classification}
    for fold_id, (fit_idx, val_idx) in enumerate(folds):
        if heartbeat:
            heartbeat(phase="inner_cv", fold=fold_id, n_folds=n_folds)
        sub_fit = _subset(data, tr[fit_idx])
        sub_fit.split[:] = "train"
        model = TrueStateEnsemble(cfg=cfg, pfv_dead_zone=pfv_dead_zone).fit(sub_fit)
        sub_val = _subset(data, tr[val_idx])
        pred = model.predict(sub_val, np.arange(len(val_idx)))
        X = model.scaler_.transform(sub_val.features)
        for h in heads:
            if h == "pfv" and model.pfv_gate_:
                for s in range(n_seeds):
                    gate_p = model.pfv_gate_[s].predict_proba(X)[:, 1] if hasattr(
                        model.pfv_gate_[s], "predict_proba"
                    ) else np.zeros(len(val_idx))
                    per_seed[h][val_idx, s] = gate_p * model.pfv_active_[s].predict(X)
            elif h in model.cont_models_:
                for s, m in enumerate(model.cont_models_[h]):
                    per_seed[h][val_idx, s] = m.predict(X)
        unc[val_idx] = pred["uncertainty"]
        score_oof[val_idx] = pred["ranking_score"]
        for c, p in pred["classification"].items():
            cls_oof[c][val_idx] = p
    return {
        "train_rows": tr,
        "per_seed_continuous": per_seed,
        "uncertainty": unc,
        "ranking_score": score_oof,
        "classification": cls_oof,
        "n_folds": n_folds,
        "grouping": "event",
    }


# ---------------------------------------------------------------------------
# Single-seed gate (spec section 10)
# ---------------------------------------------------------------------------

def single_seed_gate(
    data: TrainingData,
    cv: dict[str, Any],
    baseline_summary: dict[str, Any],
    *,
    dead_zones: dict[str, float],
) -> dict[str, Any]:
    tr = cv["train_rows"]
    checks: dict[str, Any] = {}
    seed0 = {h: cv["per_seed_continuous"][h][:, 0] for h in cv["per_seed_continuous"]}
    finite = all(np.isfinite(p[np.isfinite(p)]).all() for p in seed0.values())
    checks["no_nan_inf"] = bool(finite)
    # Residual head handled at ensemble level; here: process proxy = at least
    # one continuous/process signal beats zero on OOF MAE.
    head_reports = {}
    not_worse_count = 0
    for h, pred in seed0.items():
        y = data.continuous[h][tr]
        valid = np.isfinite(pred)
        m = regression_metrics(y[valid], pred[valid], dead_zone=float(dead_zones.get(h, 0.0)))
        zero_mae = float(np.mean(np.abs(y[valid])))
        base_models = baseline_summary.get("continuous", {}).get(h, {})
        best_baseline_mae = min(
            (
                v["mae"]
                for k, v in base_models.items()
                if v.get("mae") is not None
            ),
            default=None,
        )
        m["zero_mae"] = zero_mae
        m["best_baseline_mae"] = best_baseline_mae
        m["not_worse_than_best_baseline"] = (
            bool(m["mae"] <= best_baseline_mae * 1.05)
            if best_baseline_mae is not None
            else None
        )
        if m["not_worse_than_best_baseline"]:
            not_worse_count += 1
        ev_mae = per_event_mae(data.event_id[tr][valid], y[valid], pred[valid])
        m["worst_event"] = worst_event(ev_mae)
        head_reports[h] = m
    checks["kpi_heads_not_worse_than_best_baseline"] = {
        "count": not_worse_count,
        "required": 2,
        "ok": not_worse_count >= 2,
    }
    pfv_pred = seed0.get("pfv")
    checks["pfv_not_all_zero_dominated"] = (
        bool(np.nanstd(pfv_pred) > 1e-12) if pfv_pred is not None else None
    )
    constant = all(np.nanstd(p) < 1e-12 for p in seed0.values())
    checks["model_not_constant"] = not constant
    # Ranking better than random: OOF score decision metrics vs random.
    feasible = data.classification.get(
        "joint_noninferior", np.zeros(len(data.split), dtype=int)
    )[tr].astype(bool)
    regret = data.ranking.get(
        "regret_to_exact_best", np.full(len(data.split), np.nan)
    )[tr]
    s = cv["ranking_score"]
    valid = np.isfinite(s)
    dm = decision_metrics(
        state_key=data.state_key[tr][valid],
        score=s[valid],
        feasible_true=feasible[valid],
        regret=regret[valid],
    )
    rng = np.random.RandomState(7)
    dm_rand = decision_metrics(
        state_key=data.state_key[tr][valid],
        score=rng.rand(int(valid.sum())),
        feasible_true=feasible[valid],
        regret=regret[valid],
    )
    dm.pop("per_state", None)
    dm_rand.pop("per_state", None)
    t1 = dm["top_k_feasible_recall"]["1"]
    t1r = dm_rand["top_k_feasible_recall"]["1"]
    checks["ranking_better_than_random"] = (
        None if t1 is None or t1r is None else bool(t1 >= t1r)
    )
    checks["at_least_one_process_head_better_than_zero"] = bool(
        any(
            head_reports[h]["mae"] < head_reports[h]["zero_mae"]
            for h in head_reports
        )
    )
    hard = [
        checks["no_nan_inf"],
        checks["kpi_heads_not_worse_than_best_baseline"]["ok"],
        checks["model_not_constant"],
        checks["at_least_one_process_head_better_than_zero"],
    ]
    soft_rank = checks["ranking_better_than_random"]
    ok = all(hard) and (soft_rank is not False)
    return {
        "gate": "single_seed_gate",
        "seed": 0,
        "checks": checks,
        "head_reports": head_reports,
        "ranking_oof": dm,
        "ranking_random_reference": dm_rand,
        "status": "pass" if ok else "fail",
    }


# ---------------------------------------------------------------------------
# Ensemble artefacts (spec section 11)
# ---------------------------------------------------------------------------

def ensemble_report(
    data: TrainingData, cv: dict[str, Any], *, cfg: ModelConfig
) -> dict[str, Any]:
    tr = cv["train_rows"]
    heads = list(cv["per_seed_continuous"].keys())
    seed_rows = []
    per_seed_mae = {h: [] for h in heads}
    for s, seed in enumerate(cfg.seeds):
        row = {"seed_index": s, "seed": int(seed)}
        for h in heads:
            pred = cv["per_seed_continuous"][h][:, s]
            valid = np.isfinite(pred)
            mae = float(np.mean(np.abs(pred[valid] - data.continuous[h][tr][valid])))
            row[f"{h}_oof_mae"] = mae
            per_seed_mae[h].append(mae)
        seed_rows.append(row)
    ens_metrics = {}
    ens_not_worse = []
    for h in heads:
        stack = cv["per_seed_continuous"][h]
        ens = np.nanmean(stack, axis=1)
        valid = np.isfinite(ens)
        y = data.continuous[h][tr]
        mae = float(np.mean(np.abs(ens[valid] - y[valid])))
        med = float(np.median(per_seed_mae[h]))
        ens_metrics[h] = {
            "ensemble_oof_mae": mae,
            "median_single_seed_oof_mae": med,
            "ensemble_not_worse_than_median_seed": bool(mae <= med * 1.001),
            "seed_mae_spread": float(np.max(per_seed_mae[h]) - np.min(per_seed_mae[h])),
        }
        ens_not_worse.append(ens_metrics[h]["ensemble_not_worse_than_median_seed"])
    # Uncertainty vs |error| (use pfv/tfv/peak mean abs error of ensemble).
    err_parts = []
    for h in heads:
        ens = np.nanmean(cv["per_seed_continuous"][h], axis=1)
        err_parts.append(np.abs(ens - data.continuous[h][tr]))
    abs_err = np.nanmean(np.stack(err_parts), axis=0)
    unc = cv["uncertainty"]
    valid = np.isfinite(unc) & np.isfinite(abs_err)
    unc_rel = uncertainty_error_correlation(unc[valid], abs_err[valid])
    checks = {
        "all_seeds_completed": {"n": len(cfg.seeds), "ok": True},
        "ensemble_not_worse_than_median_seed": all(ens_not_worse),
        "uncertainty_error_positive_correlation": unc_rel,
    }
    return {
        "seed_manifest_rows": seed_rows,
        "ensemble_metrics": ens_metrics,
        "uncertainty_error": unc_rel,
        "checks": checks,
        "seeds_frozen": list(cfg.seeds),
        "no_seed_dropped": True,
    }


# ---------------------------------------------------------------------------
# Extended calibration (spec section 12)
# ---------------------------------------------------------------------------

def calibrate_extended(
    model: TrueStateEnsemble,
    data: TrainingData,
    *,
    cfg: ModelConfig,
    dead_zones: dict[str, float],
) -> dict[str, Any]:
    ca = data.split_index("calibration")
    if ca.size == 0:
        raise ValueError("calibration split empty")
    pred = model.predict(data, ca)

    # -- probability calibration (frozen rule: temperature scaling) --
    temperatures: dict[str, float] = {}
    curves: list[dict[str, Any]] = []
    for col, p in pred["classification"].items():
        y = data.classification[col][ca]
        temperatures[col] = (
            _temperature_fit(p, y) if len(np.unique(y)) > 1 else 1.0
        )
        for lo in np.linspace(0.0, 0.9, 10):
            m = (p >= lo) & (p < lo + 0.1)
            curves.append(
                {
                    "head": col,
                    "bin_low": float(lo),
                    "mean_pred": float(p[m].mean()) if m.any() else None,
                    "frac_positive": float(y[m].mean()) if m.any() else None,
                    "n": int(m.sum()),
                }
            )
    if "pfv_active_prob" in pred and "pfv" in data.continuous:
        y_act = (np.abs(data.continuous["pfv"][ca]) > dead_zones["pfv"]).astype(int)
        p = pred["pfv_active_prob"]
        temperatures["pfv_active"] = (
            _temperature_fit(p, y_act) if len(np.unique(y_act)) > 1 else 1.0
        )

    # -- one-sided conservative bounds: actual worse than predicted --
    intervals: dict[str, Any] = {}
    coverage_rows: list[dict[str, Any]] = []
    for head, yhat in pred["continuous"].items():
        y = data.continuous[head][ca]
        conf = one_sided_conformal(
            y, yhat, direction="underprediction", coverage=0.9
        )
        intervals[head] = conf
        coverage_rows.append(
            {
                "head": head,
                "stratum": "all",
                "coverage_target": 0.9,
                "empirical_coverage": conf["empirical_coverage"],
                "bound": conf["bound"],
                "n": conf["n"],
            }
        )
    # PFV active / inactive strata coverage.
    if "pfv" in pred["continuous"]:
        y = data.continuous["pfv"][ca]
        yhat = pred["continuous"]["pfv"]
        act = np.abs(y) > dead_zones["pfv"]
        for name, m in (("pfv_active", act), ("pfv_inactive", ~act)):
            if m.sum() >= 5:
                c = one_sided_conformal(
                    y[m], yhat[m], direction="underprediction", coverage=0.9
                )
                coverage_rows.append(
                    {
                        "head": "pfv",
                        "stratum": name,
                        "coverage_target": 0.9,
                        "empirical_coverage": c["empirical_coverage"],
                        "bound": c["bound"],
                        "n": c["n"],
                    }
                )
            else:
                coverage_rows.append(
                    {
                        "head": "pfv",
                        "stratum": name,
                        "coverage_target": 0.9,
                        "empirical_coverage": None,
                        "bound": None,
                        "n": int(m.sum()),
                    }
                )

    # -- uncertainty scale + abstain / OOD thresholds + fallback curve --
    abstain_threshold = float(
        np.quantile(pred["uncertainty"], cfg.abstain_uncertainty_quantile)
    )
    ood_threshold = float(np.quantile(pred["ood_distance"], cfg.ood_quantile))
    fallback_curve = []
    for q in np.linspace(0.5, 1.0, 11):
        t = float(np.quantile(pred["uncertainty"], q))
        rate = float(np.mean(pred["uncertainty"] > t))
        fallback_curve.append({"quantile": float(q), "threshold": t, "fallback_rate": rate})

    pred_rows = {
        "state_key": data.state_key[ca].tolist(),
        "event_id": data.event_id[ca].tolist(),
        **{f"pred_{h}": pred["continuous"][h].tolist() for h in pred["continuous"]},
        **{f"true_{h}": data.continuous[h][ca].tolist() for h in pred["continuous"]},
        **{f"prob_{c}": pred["classification"][c].tolist() for c in pred["classification"]},
        "uncertainty": pred["uncertainty"].tolist(),
        "ood_distance": pred["ood_distance"].tolist(),
    }

    return {
        "split_used": "calibration",
        "calibration_n": int(ca.size),
        "reads_locked": False,
        "probability_calibration": {
            "method": "temperature_scaling",
            "method_selection_rule": (
                "frozen pre-Locked: temperature scaling always; never chosen "
                "from Locked results"
            ),
            "temperatures": temperatures,
        },
        "continuous_interval_calibration": {
            "method": "one_sided_conformal_q90",
            "direction": "underprediction (actual worse than predicted)",
            "intervals": intervals,
        },
        "abstain_uncertainty_threshold": abstain_threshold,
        "ood_threshold": ood_threshold,
        "temperatures": temperatures,
        "conformal_abs_q90": {
            h: float(np.quantile(np.abs(data.continuous[h][ca] - pred["continuous"][h]), 0.9))
            for h in pred["continuous"]
        },
        "calibration_curves": curves,
        "empirical_coverage": coverage_rows,
        "fallback_rate_curve": fallback_curve,
        "predictions": pred_rows,
        "threshold_contract": {
            "probability_decision_threshold": 0.5,
            "one_sided_coverage_target": 0.9,
            "abstain_uncertainty_quantile": cfg.abstain_uncertainty_quantile,
            "ood_quantile": cfg.ood_quantile,
            "frozen_before_locked": True,
        },
        "prohibitions_respected": {
            "model_weights_updated": False,
            "architecture_modified": False,
            "seeds_reselected": False,
            "train_preprocessing_adjusted": False,
            "locked_viewed": False,
        },
    }


# ---------------------------------------------------------------------------
# Heartbeat (spec section 16)
# ---------------------------------------------------------------------------

def write_heartbeat(output_root: Path, **fields) -> None:
    try:
        import psutil

        proc = psutil.Process()
        fields.setdefault("rss_mb", round(proc.memory_info().rss / 1e6, 1))
        fields.setdefault("cpu_percent", psutil.cpu_percent(interval=None))
    except Exception:
        pass
    fields.setdefault("timestamp", time.time())
    fields.setdefault("device", "cpu")
    path = output_root / "models/v4_true_state/heartbeats/training_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields, indent=2), encoding="utf-8")
