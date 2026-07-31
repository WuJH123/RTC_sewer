"""Shared KPI implementation for Project6 PFV-first dual-fallback V3.

This module is intentionally small and dependency-light so scripts, model
builders, MPC gates, and Formal evaluation can call the same KPI code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class KPIContract:
    version: str
    default_dt_sec: float | None
    priority_nodes: tuple[str, ...]
    all_nodes: tuple[str, ...] = ()
    priority_nodes_hash: str | None = None
    dry_flooding_rate_m3s: float = 0.0
    near_zero_volume_m3: float = 1.0


def load_kpi_contract(path: str | Path, priority_nodes: Iterable[str]) -> KPIContract:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"KPI contract not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    raw_dt = payload.get("time_step", {}).get("default_dt_sec", None)
    dt = None if raw_dt is None else float(raw_dt)
    dry = float(payload.get("dry_threshold", {}).get("node_flooding_rate_m3s", 0.0))
    return KPIContract(
        version=str(payload.get("version", "unknown")),
        default_dt_sec=dt,
        priority_nodes=tuple(priority_nodes),
        dry_flooding_rate_m3s=dry,
    )


def _as_2d(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape((-1, 1))
    if arr.ndim != 2:
        raise ValueError(f"{name} must be [time,node], got shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def dt_seconds_from_timestamps(timestamps: np.ndarray | list[object]) -> np.ndarray:
    """Return timestamp interval lengths from a monotonic timestamp vector.

    Boundary timestamps produce `len(timestamps)-1` intervals. Formal V3 KPI
    arrays require n rate rows and n+1 boundary timestamps.
    """

    arr = np.asarray(timestamps)
    if arr.ndim != 1:
        raise ValueError(f"timestamps must be 1D, got shape={arr.shape}")
    if arr.size < 2:
        raise ValueError("at least two timestamps are required unless explicit dt_sec is provided")
    if np.issubdtype(arr.dtype, np.datetime64):
        diffs = np.diff(arr.astype("datetime64[ns]").astype("int64")) / 1e9
    else:
        # Accept numeric seconds or pandas-like values convertible to datetime64.
        try:
            dt64 = arr.astype("datetime64[ns]")
            diffs = np.diff(dt64.astype("int64")) / 1e9
        except (TypeError, ValueError):
            diffs = np.diff(arr.astype(float))
    if np.any(~np.isfinite(diffs)) or np.any(diffs <= 0):
        raise ValueError("timestamps must be strictly increasing and finite")
    return diffs.astype(float)


def _dt_vector(n: int, *, dt_sec: float | np.ndarray | None = None, timestamps: np.ndarray | list[object] | None = None) -> np.ndarray:
    if timestamps is not None:
        diffs = dt_seconds_from_timestamps(timestamps)
        if diffs.shape == (n,):
            dt = diffs
        else:
            dt = diffs
    elif dt_sec is not None:
        dt = np.asarray(dt_sec, dtype=float)
        if dt.ndim == 0:
            dt = np.full(n, float(dt), dtype=float)
    else:
        raise ValueError("KPI integration requires timestamps or explicit dt_sec; do not assume control interval")
    if dt.shape != (n,):
        raise ValueError(f"dt length must equal time rows ({n}); formal timestamp input must provide n+1 boundaries, got {dt.shape}")
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("dt values must be positive and finite")
    return dt


def integrate_volume_from_rates(
    rates: np.ndarray,
    dt_sec: float | np.ndarray | None = None,
    dry_threshold: float = 0.0,
    timestamps: np.ndarray | list[object] | None = None,
) -> float:
    arr = _as_2d(rates, name="rates")
    positive = np.maximum(arr - float(dry_threshold), 0.0)
    dt = _dt_vector(positive.shape[0], dt_sec=dt_sec, timestamps=timestamps)
    return float((positive.sum(axis=1) * dt).sum())


def peak_total_rate(rates: np.ndarray, dry_threshold: float = 0.0) -> float:
    arr = _as_2d(rates, name="rates")
    positive = np.maximum(arr - float(dry_threshold), 0.0)
    if positive.shape[0] == 0:
        return 0.0
    return float(positive.sum(axis=1).max())


def compute_kpis(
    *,
    all_flooding_rates: np.ndarray,
    priority_flooding_rates: np.ndarray,
    dt_sec: float | np.ndarray | None = None,
    timestamps: np.ndarray | list[object] | None = None,
    dry_threshold: float = 0.0,
) -> dict[str, float]:
    """Compute PFV, TFV, and peak_TFV_rate from flooding-rate arrays.

    Arrays must be indexed as `[time, node]`. Units are m3/s for rates and m3
    for integrated volumes.
    """

    return {
        "PFV": integrate_volume_from_rates(priority_flooding_rates, dt_sec, dry_threshold, timestamps=timestamps),
        "TFV": integrate_volume_from_rates(all_flooding_rates, dt_sec, dry_threshold, timestamps=timestamps),
        "peak_TFV_rate": peak_total_rate(all_flooding_rates, dry_threshold),
    }


def add_near_zero_flags(kpis: Mapping[str, float], near_zero_volume_m3: float = 1.0) -> dict[str, object]:
    out: dict[str, object] = dict(kpis)
    out["PFV_near_zero"] = abs(float(kpis.get("PFV", 0.0))) <= float(near_zero_volume_m3)
    out["TFV_near_zero"] = abs(float(kpis.get("TFV", 0.0))) <= float(near_zero_volume_m3)
    return out
