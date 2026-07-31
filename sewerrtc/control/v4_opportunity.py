"""Online-feature-only control-opportunity scoring for V4 Gate 5R."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _row_signal(detail: pd.DataFrame, columns: list[str], mode: str = "sum") -> np.ndarray:
    present = [column for column in columns if column in detail.columns]
    if not present:
        return np.zeros(len(detail), dtype=float)
    values = (
        detail[present]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    if mode == "max":
        return np.max(np.abs(values), axis=1)
    return np.sum(np.abs(values), axis=1)


def _bounded_signal(values: np.ndarray, scale: float) -> np.ndarray:
    """Map current observations to [0, 1] without event-future normalisation."""
    values = np.nan_to_num(np.abs(np.asarray(values, dtype=float)), nan=0.0)
    scale = max(float(scale), 1e-12)
    return np.clip(values / (values + scale), 0.0, 1.0)


def _future_rainfall_descriptors(
    elapsed_min: np.ndarray,
    rainfall_mm_h: np.ndarray,
    horizon_min: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return forecast depth and peak using rainfall only, never hydraulics."""
    times = np.asarray(elapsed_min, dtype=float)
    rain = np.nan_to_num(np.asarray(rainfall_mm_h, dtype=float), nan=0.0)
    positive_steps = np.diff(times)
    positive_steps = positive_steps[positive_steps > 0]
    sample_min = float(np.median(positive_steps)) if positive_steps.size else 5.0
    depth = np.zeros(len(times), dtype=float)
    peak = np.zeros(len(times), dtype=float)
    for index, current in enumerate(times):
        mask = (times >= current) & (times < current + float(horizon_min))
        if np.any(mask):
            depth[index] = float(np.sum(rain[mask]) * sample_min / 60.0)
            peak[index] = float(np.max(rain[mask]))
    return depth, peak


def scan_control_opportunities(
    detail: pd.DataFrame,
    facility_ids: list[str],
    facility_nodes: dict[str, tuple[str, str]] | None = None,
    responsive_threshold: float = 0.25,
    weak_threshold: float = 0.05,
) -> pd.DataFrame:
    """Score checkpoints without reading any realised future target fields."""
    if "elapsed_min" not in detail.columns:
        raise ValueError("detail is missing elapsed_min")
    flow = _row_signal(detail, [f"flow:{fid}" for fid in facility_ids], mode="max")
    flood = _row_signal(
        detail, [column for column in detail.columns if column.startswith("flood:")]
    )
    storage = _row_signal(
        detail,
        [column for column in detail.columns if column.startswith("storage_volume:")],
        mode="max",
    )
    facility_head_differences: list[np.ndarray] = []
    for facility_id in facility_ids:
        upstream, downstream = (facility_nodes or {}).get(
            facility_id, ("", "")
        )
        upstream_column = next(
            (
                column
                for column in (f"head:{upstream}", f"h:{upstream}")
                if upstream and column in detail.columns
            ),
            None,
        )
        downstream_column = next(
            (
                column
                for column in (f"head:{downstream}", f"h:{downstream}")
                if downstream and column in detail.columns
            ),
            None,
        )
        if upstream_column and downstream_column:
            upstream_values = pd.to_numeric(
                detail[upstream_column], errors="coerce"
            ).fillna(0.0).to_numpy(float)
            downstream_values = pd.to_numeric(
                detail[downstream_column], errors="coerce"
            ).fillna(0.0).to_numpy(float)
            facility_head_differences.append(
                np.abs(upstream_values - downstream_values)
            )
    facility_head_difference = (
        np.max(np.vstack(facility_head_differences), axis=0)
        if facility_head_differences
        else np.zeros(len(detail), dtype=float)
    )
    excess_fullness = (
        pd.to_numeric(detail["excess_fullness_p95"], errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
        if "excess_fullness_p95" in detail.columns
        else np.zeros(len(detail), dtype=float)
    )
    downstream_capacity = np.clip(1.0 - excess_fullness, 0.0, 1.0)
    inflow = (
        pd.to_numeric(detail["system_inflow_m3s"], errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
        if "system_inflow_m3s" in detail.columns
        else np.zeros(len(detail), dtype=float)
    )
    outflow = (
        pd.to_numeric(detail["total_outfall_flow_m3s"], errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
        if "total_outfall_flow_m3s" in detail.columns
        else np.zeros(len(detail), dtype=float)
    )
    imbalance = np.maximum(inflow - outflow, 0.0)
    action_columns = [f"a:{fid}" for fid in facility_ids if f"a:{fid}" in detail.columns]
    if action_columns:
        action = (
            detail[action_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(float)
        )
        switching = np.zeros(len(detail), dtype=float)
        if len(action) > 1:
            switching[1:] = np.max(np.abs(np.diff(action, axis=0)), axis=1)
    else:
        switching = np.zeros(len(detail), dtype=float)
    rainfall = (
        pd.to_numeric(detail.get("rainfall_mm_h", 0.0), errors="coerce")
        if "rainfall_mm_h" in detail.columns
        else pd.Series(np.zeros(len(detail)))
    ).fillna(0.0).to_numpy(float)
    elapsed_min = pd.to_numeric(
        detail["elapsed_min"], errors="coerce"
    ).to_numpy(float)
    forecast_depth, forecast_peak = _future_rainfall_descriptors(
        elapsed_min, rainfall
    )

    flow_signal = _bounded_signal(flow, 0.05)
    flood_risk = _bounded_signal(flood, 0.01)
    storage_signal = _bounded_signal(storage, 1000.0)
    switch_signal = np.clip(switching, 0.0, 1.0)
    rainfall_signal = _bounded_signal(rainfall, 10.0)
    forecast_depth_signal = _bounded_signal(forecast_depth, 5.0)
    forecast_peak_signal = _bounded_signal(forecast_peak, 20.0)
    forecast_signal = np.maximum(forecast_depth_signal, forecast_peak_signal)
    head_signal = _bounded_signal(facility_head_difference, 0.5)
    imbalance_signal = _bounded_signal(imbalance, 0.05)
    # V3 uses local controllability evidence. Real facility flow and local head
    # difference remain informative during dry-weather recession; absence of
    # rain or flooding must not force them to zero.
    current_driver = np.maximum.reduce(
        [flow_signal, head_signal, flood_risk, imbalance_signal, switch_signal]
    )
    capacity_opportunity = (
        downstream_capacity * np.maximum(flow_signal, head_signal)
    )
    score = (
        0.25 * flow_signal
        + 0.20 * flood_risk
        + 0.10 * storage_signal * np.maximum(flow_signal, head_signal)
        + 0.10 * switch_signal
        + 0.10 * rainfall_signal
        + 0.10 * head_signal
        + 0.10 * capacity_opportunity
        + 0.05 * imbalance_signal
        + 0.05 * forecast_signal
    )
    if not 0.0 <= float(weak_threshold) <= float(responsive_threshold) <= 1.0:
        raise ValueError("opportunity thresholds must satisfy 0 <= weak <= responsive <= 1")
    classes = np.where(
        score >= float(responsive_threshold),
        "responsive",
        np.where(score >= float(weak_threshold), "weakly_responsive", "flat"),
    )
    return pd.DataFrame(
        {
            "elapsed_min": pd.to_numeric(
                detail["elapsed_min"], errors="coerce"
            ).to_numpy(float),
            "opportunity_score": score,
            "opportunity_class": classes,
            "active_flow_signal": flow,
            "flood_signal": flood,
            "storage_signal": storage,
            "facility_head_difference_signal": facility_head_difference,
            "downstream_capacity_signal": downstream_capacity,
            "inflow_outflow_imbalance_signal": imbalance,
            "native_switch_signal": switching,
            "rainfall_signal": rainfall,
            "forecast_rain_depth_120min_mm": forecast_depth,
            "forecast_rain_peak_120min_mm_h": forecast_peak,
            "hydraulic_driver": current_driver,
        }
    )
