"""Read-only Train1600 V3 training-data loader (spec section 4-9, defect 2).

This module is the *only* sanctioned entry point that turns the frozen
Train1600 V3 sample manifest into model inputs / targets.  It enforces the
acceptance and anti-leakage contract from the user specification:

* Acceptance is decided by the twelve/fourteen boolean gate columns -- **never**
  by ``status`` (which is the historical *planning* state and is uniformly
  ``planned`` for every one of the 1600 accepted rows).
* Exactly 1600 accepted H120 training samples are expected.
* Only whitelisted state / action / future-rainfall-forecast fields may enter
  the feature matrix.  Every KPI label, process residual, rank / regret column,
  future SWMM state and future true flow is forbidden as an input.
* The only *future* information allowed as input is the rainfall forecast.
* When ``full_event_eligible`` is false the full-event head is disabled -- its
  labels are masked, never imputed.

Nothing in this module mutates the manifest, labels, splits, margins,
dead-zones or the Locked data; it only reads.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .train1600_v4 import (
    CONTINUOUS_HEAD_KEYS,
    HORIZON_STEPS,
    PROHIBITED_INPUT_COLUMNS,
)

# ---------------------------------------------------------------------------
# Acceptance contract (spec: acceptance is the AND of these gate columns)
# ---------------------------------------------------------------------------

ACCEPTANCE_GATE_COLUMNS: tuple[str, ...] = (
    "h120_eligible",
    "label_validity_pfv",
    "label_validity_tfv",
    "label_validity_peak",
    "same_state_ok",
    "physical_sha_ok",
    "rainfall_sha_ok",
    "prefix_sha_ok",
    "readback_ok",
    "no_hotstart",
    "k_le_8",
    "actuator_semantics_ok",
    "h120_window_complete",
    "kpi_recompute_ok",
)

EXPECTED_ACCEPTED_COUNT = 1600

# The manifest ``status`` column is the historical planning state and must
# never be used to filter accepted training rows.
FORBIDDEN_FILTER_COLUMN = "status"

# ---------------------------------------------------------------------------
# Target columns (outputs -- never inputs)
# ---------------------------------------------------------------------------

CONTINUOUS_TARGET_COLUMNS: dict[str, str] = dict(CONTINUOUS_HEAD_KEYS)

CLASSIFICATION_TARGET_COLUMNS: tuple[str, ...] = (
    "pfv_safe",
    "tfv_improved",
    "peak_noninferior",
    "joint_noninferior",
)

# Full-event heads are gated by ``full_event_eligible``.  In the frozen V3
# domain every row has ``full_event_eligible == False`` so these heads are
# disabled (labels masked, never imputed).
FULL_EVENT_TARGET_COLUMNS: tuple[str, ...] = (
    "label_validity_full",
)

RANKING_TARGET_COLUMNS: tuple[str, ...] = (
    "feasible_rank",
    "regret_to_exact_best",
)

PROCESS_RESIDUAL_COLUMNS: tuple[str, ...] = (
    "priority_depth_residual",
    "sentinel_depth_residual",
    "active_link_flow_residual",
    "storage_volume_residual",
    "tfv_rate_residual",
    "system_stored_volume_residual",
    "conduit_fullness_summary_residual",
)

# ---------------------------------------------------------------------------
# Allowed inputs
# ---------------------------------------------------------------------------

# Current-state / static hydraulic signals sourced from the frozen checkpoint
# catalog.  These summarise the frozen GAT / true-state history at the decision
# checkpoint.  ``forecast_rain_*`` are the *only* future signals allowed.
ALLOWED_STATE_FEATURE_COLUMNS: tuple[str, ...] = (
    "elapsed_min",
    "opportunity_score",
    "active_flow_signal",
    "flood_signal",
    "storage_signal",
    "facility_head_difference_signal",
    "downstream_capacity_signal",
    "inflow_outflow_imbalance_signal",
    "native_switch_signal",
    "rainfall_signal",
    "hydraulic_driver",
    "forecast_rain_depth_120min_mm",
    "forecast_rain_peak_120min_mm_h",
)

FUTURE_RAINFALL_FORECAST_COLUMNS: tuple[str, ...] = (
    "forecast_rain_depth_120min_mm",
    "forecast_rain_peak_120min_mm_h",
)

# Action-only scalar features (candidate / executed action descriptors; no KPI,
# no benefit, no rank).
ALLOWED_ACTION_SCALAR_COLUMNS: tuple[str, ...] = (
    "k_actual",
    "k_target",
    "is_noop",
    "action_cost",
    "actual_action_distance",
)

# 12x36 action matrices (JSON): requested candidate, projected actual/readback,
# anchor fallback.  These are inputs (the action being evaluated), never labels.
ACTION_MATRIX_COLUMNS: dict[str, str] = {
    "requested": "requested_schedule_json",
    "projected": "projected_schedule_json",
    "anchor": "anchor_schedule_json",
}

N_FACILITIES = 36
STATE_KEY_COLUMNS = ("event_id", "checkpoint_id")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LeakageError(ValueError):
    """Raised when a prohibited column is offered as a model input."""


class AcceptanceError(ValueError):
    """Raised when the accepted-sample contract is violated."""


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class TrainingData:
    """Immutable-by-convention view of accepted training data."""

    features: np.ndarray
    feature_names: list[str]
    continuous: dict[str, np.ndarray]
    classification: dict[str, np.ndarray]
    residuals: np.ndarray  # shape (n, 7, HORIZON_STEPS)
    residual_channels: list[str]
    ranking: dict[str, np.ndarray]
    full_event_enabled: bool
    full_event_mask: np.ndarray
    split: np.ndarray
    state_key: np.ndarray
    event_id: np.ndarray
    hard_negative_type: np.ndarray
    n_samples: int
    meta: dict[str, Any] = field(default_factory=dict)

    def split_index(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.split == name)


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------

def _coerce_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy()
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0", "yes"])
        .to_numpy()
    )


def compute_acceptance(manifest: pd.DataFrame) -> np.ndarray:
    """Return the boolean ``accepted_for_h120_training`` mask.

    Acceptance is the logical AND of the gate columns only.  ``status`` is
    never consulted.
    """
    missing = [c for c in ACCEPTANCE_GATE_COLUMNS if c not in manifest.columns]
    if missing:
        raise AcceptanceError(f"manifest missing gate columns: {missing}")
    mask = np.ones(len(manifest), dtype=bool)
    for column in ACCEPTANCE_GATE_COLUMNS:
        mask &= _coerce_bool(manifest[column])
    return mask


def load_accepted_frame(
    manifest: pd.DataFrame, *, require_count: int | None = EXPECTED_ACCEPTED_COUNT
) -> pd.DataFrame:
    """Select accepted rows using the gate contract (never ``status``).

    Guarantees exactly ``require_count`` accepted rows when provided.
    """
    mask = compute_acceptance(manifest)
    accepted = manifest.loc[mask].reset_index(drop=True).copy()
    accepted["accepted_for_h120_training"] = True
    if require_count is not None and len(accepted) != require_count:
        raise AcceptanceError(
            f"expected {require_count} accepted samples, got {len(accepted)}"
        )
    return accepted


def full_event_heads_enabled(frame: pd.DataFrame) -> bool:
    """Full-event heads are enabled only if any row is full_event_eligible."""
    if "full_event_eligible" not in frame.columns:
        return False
    return bool(_coerce_bool(frame["full_event_eligible"]).any())


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _parse_matrix(cell: Any) -> np.ndarray:
    """Parse a 12x36 schedule JSON into a float array, tolerant of ragged."""
    if isinstance(cell, str):
        data = json.loads(cell)
    else:
        data = cell
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, N_FACILITIES)
    # Pad / trim to (HORIZON_STEPS, N_FACILITIES).
    out = np.zeros((HORIZON_STEPS, N_FACILITIES), dtype=float)
    rows = min(arr.shape[0], HORIZON_STEPS)
    cols = min(arr.shape[1], N_FACILITIES)
    out[:rows, :cols] = arr[:rows, :cols]
    return out


def _action_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Engineered, leakage-free features from the 12x36 action matrices.

    The full executed (projected) matrix is flattened; requested and anchor are
    summarised per facility, plus the requested-minus-anchor deviation.  All
    derive purely from action inputs.
    """
    n = len(frame)
    proj = np.stack([_parse_matrix(v) for v in frame[ACTION_MATRIX_COLUMNS["projected"]]])
    req = np.stack([_parse_matrix(v) for v in frame[ACTION_MATRIX_COLUMNS["requested"]]])
    anc = np.stack([_parse_matrix(v) for v in frame[ACTION_MATRIX_COLUMNS["anchor"]]])

    blocks: list[np.ndarray] = []
    names: list[str] = []

    # Full executed (actual/readback) 12x36 flatten.
    blocks.append(proj.reshape(n, -1))
    names += [
        f"exec_t{t}_f{f}" for t in range(HORIZON_STEPS) for f in range(N_FACILITIES)
    ]
    # Requested candidate per-facility horizon mean.
    blocks.append(req.mean(axis=1))
    names += [f"req_mean_f{f}" for f in range(N_FACILITIES)]
    # Anchor per-facility horizon mean.
    blocks.append(anc.mean(axis=1))
    names += [f"anchor_mean_f{f}" for f in range(N_FACILITIES)]
    # Requested-minus-anchor per-facility deviation.
    blocks.append((req - anc).mean(axis=1))
    names += [f"req_minus_anchor_f{f}" for f in range(N_FACILITIES)]
    # Per-step active count of the executed schedule.
    blocks.append((proj > 0.5).sum(axis=2).astype(float))
    names += [f"exec_active_count_t{t}" for t in range(HORIZON_STEPS)]

    return np.concatenate(blocks, axis=1), names


def build_feature_matrix(
    frame: pd.DataFrame, catalog: pd.DataFrame | None = None
) -> tuple[np.ndarray, list[str]]:
    """Assemble the numeric input matrix from whitelisted fields only."""
    blocks: list[np.ndarray] = []
    names: list[str] = []

    # State / forecast features (joined from the checkpoint catalog when the
    # columns are not already present on the frame).
    state_cols = [c for c in ALLOWED_STATE_FEATURE_COLUMNS if c in frame.columns]
    joined = frame
    if catalog is not None:
        catalog_cols = [
            c
            for c in ALLOWED_STATE_FEATURE_COLUMNS
            if c in catalog.columns and c not in frame.columns
        ]
        if catalog_cols:
            joined = frame.merge(
                catalog[["checkpoint_id", *catalog_cols]].drop_duplicates(
                    "checkpoint_id"
                ),
                on="checkpoint_id",
                how="left",
            )
            state_cols = [
                c for c in ALLOWED_STATE_FEATURE_COLUMNS if c in joined.columns
            ]
    if state_cols:
        block = joined[state_cols].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        blocks.append(block)
        names += list(state_cols)

    # Action scalar features.
    action_cols = [
        c for c in ALLOWED_ACTION_SCALAR_COLUMNS if c in frame.columns
    ]
    if action_cols:
        block = frame[action_cols].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        blocks.append(block)
        names += list(action_cols)

    # Action matrix features.
    action_block, action_names = _action_features(frame)
    blocks.append(action_block)
    names += action_names

    features = np.concatenate(blocks, axis=1)
    assert_no_leakage(names)
    return features, names


def assert_no_leakage(feature_names: list[str]) -> None:
    """Raise LeakageError if any prohibited/label/residual column is present."""
    prohibited = set(PROHIBITED_INPUT_COLUMNS)
    prohibited |= set(CONTINUOUS_TARGET_COLUMNS.values())
    prohibited |= set(CLASSIFICATION_TARGET_COLUMNS)
    prohibited |= set(PROCESS_RESIDUAL_COLUMNS)
    prohibited |= set(RANKING_TARGET_COLUMNS)
    hit = sorted(set(feature_names) & prohibited)
    if hit:
        raise LeakageError(f"prohibited input columns present: {hit}")
    # Guard against future SWMM state / future true flow leaking via naming.
    forbidden_suffixes = ("_kpi", "_delta", "_residual", "_future", "_label")
    named = [
        name
        for name in feature_names
        if name.endswith(forbidden_suffixes) and name not in FUTURE_RAINFALL_FORECAST_COLUMNS
    ]
    if named:
        raise LeakageError(f"suspicious future/label-suffixed features: {named}")


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def _parse_residual_series(cell: Any) -> np.ndarray:
    if isinstance(cell, str):
        data = json.loads(cell)
    else:
        data = cell
    arr = np.asarray(data, dtype=float).reshape(-1)
    out = np.zeros(HORIZON_STEPS, dtype=float)
    k = min(arr.size, HORIZON_STEPS)
    out[:k] = arr[:k]
    return out


def build_targets(frame: pd.DataFrame) -> dict[str, Any]:
    n = len(frame)
    continuous = {
        head: pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(float)
        for head, col in CONTINUOUS_TARGET_COLUMNS.items()
        if col in frame.columns
    }
    classification = {
        col: _coerce_bool(frame[col]).astype(int)
        for col in CLASSIFICATION_TARGET_COLUMNS
        if col in frame.columns
    }
    residuals = np.zeros((n, len(PROCESS_RESIDUAL_COLUMNS), HORIZON_STEPS))
    for c, col in enumerate(PROCESS_RESIDUAL_COLUMNS):
        if col in frame.columns:
            residuals[:, c, :] = np.stack(
                [_parse_residual_series(v) for v in frame[col]]
            )
    ranking = {
        col: pd.to_numeric(frame[col], errors="coerce").fillna(-1.0).to_numpy(float)
        for col in RANKING_TARGET_COLUMNS
        if col in frame.columns
    }
    return {
        "continuous": continuous,
        "classification": classification,
        "residuals": residuals,
        "ranking": ranking,
    }


# ---------------------------------------------------------------------------
# Top-level bundle builder
# ---------------------------------------------------------------------------

def build_training_data(
    manifest: pd.DataFrame,
    catalog: pd.DataFrame | None = None,
    *,
    require_count: int | None = EXPECTED_ACCEPTED_COUNT,
) -> TrainingData:
    """Full accepted -> features/targets pipeline (read-only)."""
    frame = load_accepted_frame(manifest, require_count=require_count)
    features, feature_names = build_feature_matrix(frame, catalog)
    targets = build_targets(frame)
    fe_enabled = full_event_heads_enabled(frame)
    fe_mask = (
        _coerce_bool(frame["full_event_eligible"])
        if "full_event_eligible" in frame.columns
        else np.zeros(len(frame), dtype=bool)
    )
    state_key = (
        frame["event_id"].astype(str) + "::" + frame["checkpoint_id"].astype(str)
    ).to_numpy()
    hard_neg = (
        frame["hard_negative_type"].fillna("").astype(str).to_numpy()
        if "hard_negative_type" in frame.columns
        else np.array([""] * len(frame))
    )
    return TrainingData(
        features=features,
        feature_names=feature_names,
        continuous=targets["continuous"],
        classification=targets["classification"],
        residuals=targets["residuals"],
        residual_channels=list(PROCESS_RESIDUAL_COLUMNS),
        ranking=targets["ranking"],
        full_event_enabled=fe_enabled,
        full_event_mask=fe_mask,
        split=frame["split"].astype(str).to_numpy(),
        state_key=state_key,
        event_id=frame["event_id"].astype(str).to_numpy(),
        hard_negative_type=hard_neg,
        n_samples=len(frame),
        meta={
            "n_features": features.shape[1],
            "full_event_enabled": fe_enabled,
            "accepted_count": len(frame),
        },
    )
