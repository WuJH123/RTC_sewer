from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def compute_kpis(detail: pd.DataFrame, priority_nodes: Iterable[str], dt_sec: int = 300) -> dict:
    priority_nodes = set(priority_nodes)
    flood_cols = [c for c in detail.columns if c.startswith("flood:")]
    pr_cols = [c for c in flood_cols if c.split(":", 1)[1] in priority_nodes]
    if not flood_cols:
        return {"TFV": 0.0, "PFV": 0.0, "peak_TFV_rate": 0.0, "flood_duration_min": 0.0}
    flood = detail[flood_cols].fillna(0.0).to_numpy(float)
    total_rate = flood.sum(axis=1)
    tfv = float(total_rate.sum() * dt_sec)
    peak = float(total_rate.max()) if len(total_rate) else 0.0
    flood_duration = float((total_rate > 1e-9).sum() * dt_sec / 60.0)
    if pr_cols:
        pfv = float(detail[pr_cols].fillna(0.0).to_numpy(float).sum() * dt_sec)
        pr_rate = detail[pr_cols].fillna(0.0).to_numpy(float).sum(axis=1)
        pr_duration = float((pr_rate > 1e-9).sum() * dt_sec / 60.0)
    else:
        pfv = 0.0
        pr_duration = 0.0
    action_cols = [c for c in detail.columns if c.startswith("a:")]
    action_changes = 0.0
    if action_cols and len(detail) > 1:
        a = detail[action_cols].fillna(0.0).to_numpy(float)
        action_changes = float((np.abs(np.diff(a, axis=0)) > 1e-6).sum())
    return {
        "TFV": tfv,
        "PFV": pfv,
        "peak_TFV_rate": peak,
        "flood_duration_min": flood_duration,
        "priority_flood_duration_min": pr_duration,
        "action_changes": action_changes,
    }


def compute_window_kpis(
    detail: pd.DataFrame,
    priority_nodes: Iterable[str],
    start_min: float,
    horizon_min: float,
    dt_sec: int = 300,
) -> dict:
    """Compute authoritative KPIs on ``(start_min, start_min + horizon_min]``.

    PySWMM ``Node.flooding`` values stored in ``flood:*`` columns are rates in
    m3/s.  Volumes therefore require multiplication by the sampling interval,
    while peak TFV rate is the maximum summed rate and must not be divided by
    the sampling interval.
    """
    if int(dt_sec) <= 0:
        raise ValueError("dt_sec must be positive")
    if float(horizon_min) <= 0:
        raise ValueError("horizon_min must be positive")
    if "elapsed_min" not in detail.columns:
        raise ValueError("detail is missing elapsed_min")

    end_min = float(start_min) + float(horizon_min)
    window = detail[
        (pd.to_numeric(detail["elapsed_min"], errors="coerce") > float(start_min))
        & (pd.to_numeric(detail["elapsed_min"], errors="coerce") <= end_min)
    ].copy()
    result = compute_kpis(window, priority_nodes, dt_sec=int(dt_sec))
    result["steps"] = int(len(window))
    result["window_start_min"] = float(start_min)
    result["window_end_min"] = float(end_min)
    return result
