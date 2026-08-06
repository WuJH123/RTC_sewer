"""Small shared metrics/feasibility primitives for V4.2 direct SWMM audits."""
from __future__ import annotations

import hashlib
import json
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DT_SEC = 600.0


def action_sha256(sequence: np.ndarray) -> str:
    value = np.asarray(sequence, dtype=np.float32)
    return hashlib.sha256(np.round(value, 6).tobytes()).hexdigest()


def h3_prefix_is_valid(
    sequence: np.ndarray,
    current_action: np.ndarray,
    *,
    binary_indices: Sequence[int] = (),
    prefix_steps: int = 3,
) -> bool:
    value = np.asarray(sequence, dtype=float)
    current = np.asarray(current_action, dtype=float).reshape(-1)
    if value.ndim != 2 or value.shape[1] != current.size or value.shape[0] < int(prefix_steps):
        return False
    if not np.isfinite(value).all() or not np.isfinite(current).all():
        return False
    if np.any(value < 0.0) or np.any(value > 1.0):
        return False
    if not np.allclose(value[int(prefix_steps):], current[None, :], rtol=0.0, atol=1.0e-6):
        return False
    for index in binary_indices:
        if int(index) < 0 or int(index) >= value.shape[1]:
            return False
        if not np.isin(value[:, int(index)], (0.0, 1.0)).all():
            return False
    return True


def trajectory_metrics(
    flood: np.ndarray,
    priority_indices: Sequence[int],
    *,
    dt_sec: float = DT_SEC,
) -> dict[str, float]:
    values = np.asarray(flood, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("flood trajectory must be a finite 2D array")
    priority = [int(index) for index in priority_indices]
    if any(index < 0 or index >= values.shape[1] for index in priority):
        raise ValueError("priority index is outside the flood trajectory")
    total_rate = values.sum(axis=1)
    priority_rate = values[:, priority].sum(axis=1) if priority else np.zeros(len(values))
    return {
        "PFV": float(priority_rate.sum() * float(dt_sec)),
        "TFV": float(total_rate.sum() * float(dt_sec)),
        "peak_TFV_rate": float(total_rate.max()) if len(total_rate) else 0.0,
    }


def _flood_columns(detail: pd.DataFrame) -> list[str]:
    return [str(column) for column in detail.columns if str(column).startswith("flood:")]


def _horizon(detail: pd.DataFrame, checkpoint_min: float, steps: int) -> pd.DataFrame:
    if "elapsed_min" not in detail.columns:
        raise ValueError("detail is missing elapsed_min")
    times = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    # Frozen Round2 manifests store future H120 at t+10, ..., t+120.
    targets = float(checkpoint_min) + 10.0 * np.arange(1, int(steps) + 1, dtype=float)
    values = detail.copy()
    values["_elapsed_numeric"] = times
    rows = []
    for target in targets:
        matches = values[(values["_elapsed_numeric"] - target).abs() <= 1.0e-6]
        if len(matches) != 1:
            raise ValueError(f"detail does not have one exact 10-minute row at {target:g} min")
        rows.append(matches.iloc[0])
    future = pd.DataFrame(rows).drop(columns=["_elapsed_numeric"], errors="ignore")
    if len(future) != int(steps):
        raise ValueError(f"detail has {len(future)} rows after checkpoint; expected {steps}")
    if future["elapsed_min"].duplicated().any():
        raise ValueError("detail has duplicate elapsed_min in H120")
    return future


def detail_horizon_metrics(
    detail: pd.DataFrame,
    priority_nodes: Iterable[str],
    *,
    checkpoint_min: float,
    steps: int = 12,
    dt_sec: float = DT_SEC,
) -> dict[str, float]:
    horizon = _horizon(detail, checkpoint_min, steps)
    flood_columns = _flood_columns(horizon)
    if not flood_columns:
        raise ValueError("detail has no flood columns")
    node_ids = [column.split(":", 1)[1] for column in flood_columns]
    priority_indices = [node_ids.index(str(node)) for node in priority_nodes if str(node) in node_ids]
    return trajectory_metrics(horizon[flood_columns].to_numpy(float), priority_indices, dt_sec=dt_sec)


def _prefix(detail: pd.DataFrame, checkpoint_min: float) -> pd.DataFrame:
    if "elapsed_min" not in detail.columns:
        raise ValueError("detail is missing elapsed_min")
    times = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    checkpoint = float(checkpoint_min)
    delta = (times - checkpoint) / 10.0
    on_control_grid = np.isclose(delta, np.round(delta), rtol=0.0, atol=1.0e-6)
    mask = (times < checkpoint - 1.0e-6) & on_control_grid
    return detail[mask].sort_values("elapsed_min", kind="stable").copy()


def _detail_metrics(detail: pd.DataFrame, priority_nodes: Iterable[str], *, dt_sec: float) -> dict[str, float]:
    flood_columns = _flood_columns(detail)
    if not flood_columns:
        raise ValueError("detail has no flood columns")
    node_ids = [column.split(":", 1)[1] for column in flood_columns]
    priority_indices = [node_ids.index(str(node)) for node in priority_nodes if str(node) in node_ids]
    return trajectory_metrics(detail[flood_columns].to_numpy(float), priority_indices, dt_sec=dt_sec)


def realised_prefix_budget_metric(
    candidate_prefix: pd.DataFrame,
    no_control_prefix: pd.DataFrame,
    *,
    priority_nodes: Iterable[str],
    relative_margin: float,
    dt_sec: float = DT_SEC,
) -> float:
    candidate = _detail_metrics(candidate_prefix, priority_nodes, dt_sec=dt_sec) if len(candidate_prefix) else {"PFV": 0.0}
    reference = _detail_metrics(no_control_prefix, priority_nodes, dt_sec=dt_sec) if len(no_control_prefix) else {"PFV": 0.0}
    return float(candidate["PFV"] - (1.0 + float(relative_margin)) * reference["PFV"])


def rolling_pfv_budget_metric(
    candidate_detail: pd.DataFrame,
    no_control_detail: pd.DataFrame,
    *,
    priority_nodes: Iterable[str],
    checkpoint_min: float,
    relative_margin: float,
    steps: int = 12,
    dt_sec: float = DT_SEC,
) -> float:
    candidate_prefix = _prefix(candidate_detail, checkpoint_min)
    reference_prefix = _prefix(no_control_detail, checkpoint_min)
    prefix_metric = realised_prefix_budget_metric(
        candidate_prefix,
        reference_prefix,
        priority_nodes=priority_nodes,
        relative_margin=relative_margin,
        dt_sec=dt_sec,
    )
    candidate_future = detail_horizon_metrics(
        candidate_detail, priority_nodes, checkpoint_min=checkpoint_min, steps=steps, dt_sec=dt_sec
    )
    reference_future = detail_horizon_metrics(
        no_control_detail, priority_nodes, checkpoint_min=checkpoint_min, steps=steps, dt_sec=dt_sec
    )
    return float(
        prefix_metric
        + candidate_future["PFV"]
        - (1.0 + float(relative_margin)) * reference_future["PFV"]
    )


def pfv_budget_metric(pfv_candidate: float, pfv_no_control: float, *, relative_margin: float) -> float:
    return float(pfv_candidate - (1.0 + float(relative_margin)) * pfv_no_control)


def pfv_feasible_first_score(
    *,
    tfv: float,
    budget_metric: float,
    absolute_margin: float,
    reference_tfv: float,
) -> float:
    if not np.isfinite([tfv, budget_metric, absolute_margin, reference_tfv]).all():
        return float("inf")
    if float(budget_metric) <= float(absolute_margin) + 1.0e-9:
        return float(tfv)
    penalty = max(1.0e12, abs(float(reference_tfv)) * 1.0e9 + 1.0)
    return float(penalty + max(0.0, float(budget_metric) - float(absolute_margin)))


def causal_prefix_matches(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    checkpoint_min: float,
    tolerance: float = 1.0e-6,
) -> dict[str, float | bool]:
    c = _prefix(candidate, checkpoint_min)
    r = _prefix(reference, checkpoint_min)
    if c.empty or r.empty:
        return {"prefix_match": False, "common_rows": 0, "max_action_error": float("inf"), "max_rainfall_error": float("inf")}
    if c["elapsed_min"].duplicated().any() or r["elapsed_min"].duplicated().any():
        return {"prefix_match": False, "common_rows": 0, "max_action_error": float("inf"), "max_rainfall_error": float("inf")}
    common = sorted(set(np.round(c["elapsed_min"], 6)) & set(np.round(r["elapsed_min"], 6)))
    if not common:
        return {"prefix_match": False, "common_rows": 0, "max_action_error": float("inf"), "max_rainfall_error": float("inf")}
    c = c.set_index(c["elapsed_min"].round(6)).loc[common]
    r = r.set_index(r["elapsed_min"].round(6)).loc[common]
    action_errors = []
    for column in [str(x) for x in c.columns if str(x).startswith("a:") and str(x) in r.columns]:
        action_errors.append(float(np.max(np.abs(c[column].to_numpy(float) - r[column].to_numpy(float)))))
    rain_error = float(np.max(np.abs(c["rainfall_mm_h"].to_numpy(float) - r["rainfall_mm_h"].to_numpy(float)))) if "rainfall_mm_h" in c and "rainfall_mm_h" in r else float("inf")
    max_action = max(action_errors or [float("inf")])
    return {
        "prefix_match": bool(max_action <= tolerance and rain_error <= tolerance),
        "common_rows": int(len(common)),
        "max_action_error": max_action,
        "max_rainfall_error": rain_error,
    }
