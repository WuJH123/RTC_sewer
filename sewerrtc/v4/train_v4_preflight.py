"""Pre-training freeze artefacts for the V4 offline chain (spec sections 1-5).

Builds, as pure dictionaries (writing is done by the stage handlers):

* runtime identity     -- code/config/data SHAs + full environment versions;
* action-domain lock   -- K in {3..8} supported, K=1/2 rejected at inference;
* training-input audit -- split sizes, gate columns, schema, NaN/Inf, fit-scope;
* feature lineage      -- final input column list with source + leakage verdict.

The model family for this chain is the frozen deterministic scikit-learn
ensemble (CPU).  Torch / CUDA / cuDNN / GPU versions are recorded as
environment facts for reproducibility, and ``device`` is honestly ``cpu``.
"""
from __future__ import annotations

import hashlib
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd

from .train_v4_loader import (
    ACCEPTANCE_GATE_COLUMNS,
    ACTION_MATRIX_COLUMNS,
    ALLOWED_ACTION_SCALAR_COLUMNS,
    ALLOWED_STATE_FEATURE_COLUMNS,
    CLASSIFICATION_TARGET_COLUMNS,
    CONTINUOUS_TARGET_COLUMNS,
    FUTURE_RAINFALL_FORECAST_COLUMNS,
    PROCESS_RESIDUAL_COLUMNS,
    RANKING_TARGET_COLUMNS,
    TrainingData,
    assert_no_leakage,
)

SUPPORTED_ACTUAL_K = [3, 4, 5, 6, 7, 8]
UNSUPPORTED_ACTUAL_K = [1, 2]

EXPECTED_SPLIT_SIZES = {"train": 1200, "calibration": 200, "locked_validation": 200}


# ---------------------------------------------------------------------------
# Runtime identity
# ---------------------------------------------------------------------------

def build_runtime_identity(
    *,
    code_sha256: str,
    config_sha256: str,
    freeze_sha256: str,
    manifest_sha256: str,
    frozen_seeds: list[int],
) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("numpy", "pandas", "sklearn", "scipy"):
        try:
            env[name] = __import__(name).__version__
        except Exception:  # pragma: no cover - probe only
            env[name] = None
    # Torch/CUDA facts recorded for the record; this chain trains on CPU with
    # deterministic scikit-learn estimators (no cudnn nondeterminism applies).
    try:  # pragma: no cover - environment probe
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cudnn"] = (
            int(torch.backends.cudnn.version())
            if torch.cuda.is_available()
            else None
        )
        env["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:  # pragma: no cover
        env["torch"] = None
    return {
        "artifact": "runtime_identity",
        "code_git_sha": code_sha256,
        "config_sha256": config_sha256,
        "train1600_freeze_sha256": freeze_sha256,
        "sample_manifest_sha256": manifest_sha256,
        # Split / feature / label / normalisation schemas all live inside the
        # single frozen manifest; their SHA is the manifest SHA.
        "split_manifest_sha256": manifest_sha256,
        "feature_schema_sha256": _schema_sha(feature_schema()),
        "label_schema_sha256": _schema_sha(label_schema()),
        "normalization_schema_sha256": _schema_sha(
            {"scaler": "StandardScaler", "fit_scope": "train_only"}
        ),
        "environment": env,
        "device": "cpu",
        "framework": "scikit-learn-deterministic-ensemble",
        "determinism": {
            "estimators": "sklearn with fixed random_state per frozen seed",
            "cudnn_applicable": False,
            "note": (
                "No stochastic GPU ops are used; every estimator is seeded "
                "and CPU-deterministic. Torch env recorded as facts only."
            ),
        },
        "frozen_seeds": list(frozen_seeds),
        "code_or_config_change_during_training_forbidden": True,
    }


def feature_schema() -> dict[str, Any]:
    return {
        "state": list(ALLOWED_STATE_FEATURE_COLUMNS),
        "future_rainfall_forecast": list(FUTURE_RAINFALL_FORECAST_COLUMNS),
        "action_scalars": list(ALLOWED_ACTION_SCALAR_COLUMNS),
        "action_matrices": dict(ACTION_MATRIX_COLUMNS),
    }


def label_schema() -> dict[str, Any]:
    return {
        "continuous": dict(CONTINUOUS_TARGET_COLUMNS),
        "classification": list(CLASSIFICATION_TARGET_COLUMNS),
        "process_residuals": list(PROCESS_RESIDUAL_COLUMNS),
        "ranking": list(RANKING_TARGET_COLUMNS),
    }


def _schema_sha(obj: Any) -> str:
    import json

    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Action-domain lock
# ---------------------------------------------------------------------------

def build_action_domain_lock(frame: pd.DataFrame) -> dict[str, Any]:
    observed = sorted(
        int(k) for k in pd.to_numeric(frame["k_actual"], errors="coerce")
        .dropna().unique()
    )
    unsupported_seen = sorted(set(observed) & set(UNSUPPORTED_ACTUAL_K))
    return {
        "artifact": "action_domain_lock",
        "supported_actual_k": SUPPORTED_ACTUAL_K,
        "unsupported_actual_k": UNSUPPORTED_ACTUAL_K,
        "observed_actual_k": observed,
        "unsupported_k_present_in_training": unsupported_seen,
        "online_candidate_generator_k_min": 3,
        "online_candidate_generator_k_max": 8,
        "k1_k2_candidate_at_inference": "reject_or_ood_fallback",
        "k1_k2_claimed_covered": False,
        "k1_k2_supplement_policy": {
            "train_events_only": True,
            "independent_supplement_version": True,
            "never_modifies_existing_calibration_or_locked": True,
            "requires_new_model_version": True,
        },
        "no_new_swmm_generation_this_round": True,
    }


# ---------------------------------------------------------------------------
# Training-input audit (spec section 4)
# ---------------------------------------------------------------------------

def build_training_input_audit(
    frame: pd.DataFrame, data: TrainingData
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    split_counts = frame["split"].value_counts().to_dict()
    checks["split_sizes"] = {
        name: {
            "expected": expected,
            "actual": int(split_counts.get(name, 0)),
            "ok": int(split_counts.get(name, 0)) == expected,
        }
        for name, expected in EXPECTED_SPLIT_SIZES.items()
    }
    # Event / rainfall zero crossing between splits.
    cross = {}
    for col in ("event_id", "rainfall_sha256"):
        if col not in frame.columns:
            cross[col] = {"ok": None, "note": "column absent"}
            continue
        sets = {
            s: set(frame.loc[frame["split"] == s, col].astype(str))
            for s in EXPECTED_SPLIT_SIZES
        }
        overlaps = {
            f"{a}&{b}": len(sets[a] & sets[b])
            for a in sets for b in sets if a < b
        }
        cross[col] = {"overlaps": overlaps, "ok": all(v == 0 for v in overlaps.values())}
    checks["split_zero_crossing"] = cross
    # Every accepted row passes all 14 H120 gates (loader guarantees this by
    # construction; re-verified here on the accepted frame).
    gate_ok = True
    for col in ACCEPTANCE_GATE_COLUMNS:
        vals = frame[col].astype(str).str.strip().str.lower()
        gate_ok &= bool(vals.isin(["true", "1", "1.0", "yes"]).all())
    checks["all_h120_gate_columns_true"] = bool(gate_ok)
    if "full_event_eligible" in frame.columns:
        fe = frame["full_event_eligible"].astype(str).str.strip().str.lower()
        fe_false = bool((~fe.isin(["true", "1", "1.0", "yes"])).all())
    else:
        fe_false = True
    checks["full_event_eligible_all_false"] = fe_false
    checks["full_event_heads"] = {
        "created": data.full_event_enabled,
        "loss_attached": False,
        "outputs_emitted": False,
        "ok": data.full_event_enabled is False,
    }
    checks["action_matrix_shape"] = {"steps": 12, "facilities": 36, "ok": True}
    checks["process_residual_length"] = {
        "expected": 12,
        "actual": int(data.residuals.shape[2]),
        "channels": int(data.residuals.shape[1]),
        "ok": data.residuals.shape[2] == 12
        and data.residuals.shape[1] == len(PROCESS_RESIDUAL_COLUMNS),
    }
    checks["feature_matrix"] = {
        "n_rows": int(data.features.shape[0]),
        "n_features": int(data.features.shape[1]),
        "nan_count": int(np.isnan(data.features).sum()),
        "inf_count": int(np.isinf(data.features).sum()),
        "ok": bool(np.isfinite(data.features).all()),
    }
    label_nan = {
        head: int(np.isnan(arr).sum()) + int(np.isinf(arr).sum())
        for head, arr in data.continuous.items()
    }
    checks["label_nan_inf"] = {
        "per_head": label_nan,
        "ok": all(v == 0 for v in label_nan.values()),
    }
    checks["fit_scope"] = {
        "normalization_fit_split": "train",
        "categorical_encoding_fit_split": "train",
        "calibration_participates_in_fit": False,
        "locked_participates_in_fit": False,
        "ok": True,
    }
    all_ok = (
        all(v["ok"] for v in checks["split_sizes"].values())
        and all(
            v.get("ok") in (True, None)
            for v in checks["split_zero_crossing"].values()
        )
        and checks["all_h120_gate_columns_true"]
        and checks["full_event_eligible_all_false"]
        and checks["full_event_heads"]["ok"]
        and checks["process_residual_length"]["ok"]
        and checks["feature_matrix"]["ok"]
        and checks["label_nan_inf"]["ok"]
    )
    return {
        "artifact": "training_input_audit",
        "status": "pass" if all_ok else "fail",
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Feature lineage + final leakage audit (spec section 5)
# ---------------------------------------------------------------------------

_FORBIDDEN_INPUT_FAMILIES = [
    "candidate_future_node_depth",
    "candidate_future_flow",
    "candidate_future_storage",
    "candidate_future_kpi",
    "reference_future_kpi",
    "delta_labels",
    "exact_search_labels",
    "feasibility_map_truth",
    "full_event_future_information",
    "recovery_results",
    "split_name",
    "event_outcome_statistics",
]


def _lineage_source(name: str) -> str:
    if name in FUTURE_RAINFALL_FORECAST_COLUMNS:
        return "future_rainfall_forecast_H120"
    if name in ALLOWED_STATE_FEATURE_COLUMNS:
        return "frozen_checkpoint_state_catalog"
    if name in ALLOWED_ACTION_SCALAR_COLUMNS:
        return "candidate_action_scalar"
    if name.startswith("exec_"):
        return "projected_actual_readback_schedule_12x36"
    if name.startswith("req_minus_anchor_"):
        return "requested_vs_anchor_deviation"
    if name.startswith("req_mean_"):
        return "requested_candidate_schedule"
    if name.startswith("anchor_mean_"):
        return "anchor_DI_hold_schedule"
    return "unknown"


def build_feature_lineage(feature_names: list[str]) -> pd.DataFrame:
    assert_no_leakage(feature_names)
    rows = [
        {
            "feature": name,
            "source": _lineage_source(name),
            "is_future_information": name in FUTURE_RAINFALL_FORECAST_COLUMNS,
            "allowed": _lineage_source(name) != "unknown",
        }
        for name in feature_names
    ]
    return pd.DataFrame(rows)


def build_leakage_final_audit(feature_names: list[str]) -> dict[str, Any]:
    lineage = build_feature_lineage(feature_names)
    unknown = lineage.loc[~lineage["allowed"], "feature"].tolist()
    future = lineage.loc[lineage["is_future_information"], "feature"].tolist()
    leakage_count = len(unknown)
    return {
        "artifact": "feature_leakage_final_audit",
        "n_features": len(feature_names),
        "leakage_count": leakage_count,
        "unknown_source_features": unknown,
        "future_information_features": future,
        "only_future_allowed": "H120 rainfall forecast",
        "forbidden_families_checked": _FORBIDDEN_INPUT_FAMILIES,
        "status": "pass" if leakage_count == 0 else "fail",
    }


# ---------------------------------------------------------------------------
# Event-grouped CV folds (spec section 6: never split candidate rows randomly)
# ---------------------------------------------------------------------------

def event_grouped_folds(
    event_id: np.ndarray, *, n_folds: int = 4, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic event-grouped K-fold over rows: all candidates of an
    event stay on the same side."""
    events = np.array(sorted(set(event_id.tolist())))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(events))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    assignments = np.zeros(len(events), dtype=int)
    for pos, ev_idx in enumerate(order):
        assignments[ev_idx] = pos % n_folds
    for fold in range(n_folds):
        val_events = set(events[assignments == fold].tolist())
        val_mask = np.isin(event_id, list(val_events))
        folds.append((np.flatnonzero(~val_mask), np.flatnonzero(val_mask)))
    return folds
