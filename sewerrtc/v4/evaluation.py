from __future__ import annotations

import hashlib
import json

import pandas as pd
import numpy as np


def build_policy_lock(**components: str) -> dict:
    required = {
        "model_sha",
        "candidate_sha",
        "threshold_sha",
        "uncertainty_sha",
        "fallback_sha",
        "reference_sha",
        "kpi_sha",
        "event_split_sha",
        "code_sha",
        "config_sha",
    }
    missing = required - set(components)
    if missing:
        raise ValueError(f"policy lock missing: {sorted(missing)}")
    canonical = json.dumps(components, sort_keys=True).encode("utf-8")
    return {
        "status": "locked",
        "components": components,
        "policy_lock_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def audit_formal_blind_inventory(inventory: pd.DataFrame) -> dict:
    required = {
        "event_id",
        "rainfall_sha256",
        "historically_used",
        "revealed",
    }
    missing = required - set(inventory)
    if missing:
        return {"status": "blocked", "missing_columns": sorted(missing)}
    checks = {
        "events_at_least_24": len(inventory) >= 24,
        "event_ids_unique": not inventory["event_id"].duplicated().any(),
        "rainfall_sha_unique": not inventory["rainfall_sha256"].duplicated().any(),
        "never_used": not inventory["historically_used"].astype(bool).any(),
        "never_revealed": not inventory["revealed"].astype(bool).any(),
    }
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
    }


def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import spearmanr

    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    correlation = spearmanr(truth, prediction).statistic
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
    }


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    result = {
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, prediction)
        ),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "mcc": float(matthews_corrcoef(truth, prediction)),
        "false_safe": int(((prediction == 1) & (truth == 0)).sum()),
        "false_reject": int(((prediction == 0) & (truth == 1)).sum()),
    }
    if y_score is not None and len(np.unique(truth)) == 2:
        score = np.asarray(y_score, dtype=float)
        result["auroc"] = float(roc_auc_score(truth, score))
        result["auprc"] = float(average_precision_score(truth, score))
    return result


def bootstrap_event_interval(
    frame: pd.DataFrame,
    value_column: str,
    *,
    iterations: int = 2000,
    seed: int = 20260727,
) -> dict:
    if "event_id" not in frame or value_column not in frame:
        raise ValueError("event_id and value column are required")
    event_values = frame.groupby("event_id")[value_column].mean().to_numpy(float)
    if not len(event_values):
        raise ValueError("no event values")
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [
            rng.choice(event_values, size=len(event_values), replace=True).mean()
            for _ in range(int(iterations))
        ]
    )
    return {
        "mean": float(event_values.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "events": int(len(event_values)),
    }
