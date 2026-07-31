from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .horizon_action_features import ACTION_FEATURE_COLUMNS, build_action_feature_map


BASE_DETAIL_COLUMNS = {"event_id", "policy_id", "elapsed_min", "datetime", "rainfall_mm_h", "phase"}


def _detail_usecols(path: Path, priority_nodes: list[str]) -> list[str]:
    """Return only columns needed for horizon labels/features.

    Detail CSVs in this project can exceed 2k columns. Reading everything and
    then repeatedly scanning column names dominated the horizon dataset build.
    We still keep all hydraulic depth/flood/action columns needed for the
    published targets, but avoid unrelated runtime/debug columns.
    """
    header = pd.read_csv(path, nrows=0).columns.tolist()
    priority_h = {f"h:{n}" for n in priority_nodes}
    priority_flood = {f"flood:{n}" for n in priority_nodes}
    keep: list[str] = []
    for c in header:
        if c in BASE_DETAIL_COLUMNS:
            keep.append(c)
        elif c.startswith("h:") or c.startswith("flood:") or c.startswith("a:"):
            keep.append(c)
        elif c in priority_h or c in priority_flood:
            keep.append(c)
    return keep


def _read_detail_for_horizon(path: Path, priority_nodes: list[str]) -> pd.DataFrame:
    usecols = _detail_usecols(path, priority_nodes)
    numeric_prefixes = ("h:", "flood:", "a:")
    dtype = {
        c: "float32"
        for c in usecols
        if c.startswith(numeric_prefixes) or c in {"elapsed_min", "rainfall_mm_h"}
    }
    return pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False, memory_map=True)


def _window_sum(values: np.ndarray, starts: np.ndarray, first_offset: int, horizon_steps: int) -> np.ndarray:
    csum = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    lo = starts + int(first_offset)
    hi = lo + int(horizon_steps)
    return csum[hi] - csum[lo]


def _window_max(values: np.ndarray, starts: np.ndarray, first_offset: int, horizon_steps: int) -> np.ndarray:
    if len(starts) == 0:
        return np.asarray([], dtype=np.float32)
    stacked = np.vstack([values[starts + int(first_offset) + k] for k in range(int(horizon_steps))])
    return np.nanmax(stacked, axis=0)


def _window_matrix_stat(
    matrix: np.ndarray,
    starts: np.ndarray,
    first_offset: int,
    horizon_steps: int,
    stat: str,
    quantile: float = 0.95,
) -> np.ndarray:
    if len(starts) == 0 or matrix.shape[1] == 0:
        return np.zeros(len(starts), dtype=np.float32)
    stacked = np.stack([matrix[starts + int(first_offset) + k] for k in range(int(horizon_steps))], axis=0)
    if stat == "mean":
        return np.nanmean(stacked, axis=(0, 2)).astype(np.float32)
    if stat == "max":
        return np.nanmax(stacked, axis=(0, 2)).astype(np.float32)
    if stat == "quantile":
        return np.nanquantile(stacked, float(quantile), axis=(0, 2)).astype(np.float32)
    raise ValueError(f"Unknown window matrix stat: {stat}")


def horizon_labels(
    detail: pd.DataFrame,
    start_idx: int,
    horizon_steps: int,
    priority_nodes: list[str],
    dt_sec: int,
) -> dict[str, float]:
    window = detail.iloc[start_idx + 1 : start_idx + 1 + int(horizon_steps)].copy()
    if window.empty:
        return {
            "PFV_H": 0.0,
            "TFV_H": 0.0,
            "peak_TFV_rate_H": 0.0,
            "priority_peak_depth_H": 0.0,
            "high_risk_exposure_time_H": 0.0,
            "full_depth_mean_H": 0.0,
            "full_depth_p95_H": 0.0,
            "full_depth_max_H": 0.0,
            "priority_depth_mean_H": 0.0,
            "priority_depth_p95_H": 0.0,
        }
    flood_cols = [c for c in window.columns if c.startswith("flood:")]
    h_cols = [c for c in window.columns if c.startswith("h:")]
    priority_flood_cols = [f"flood:{n}" for n in priority_nodes if f"flood:{n}" in window.columns]
    priority_h_cols = [f"h:{n}" for n in priority_nodes if f"h:{n}" in window.columns]
    dt_min = float(dt_sec) / 60.0
    tfv_rate = window[flood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1) if flood_cols else pd.Series(0.0, index=window.index)
    pfv_rate = (
        window[priority_flood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        if priority_flood_cols
        else pd.Series(0.0, index=window.index)
    )
    priority_depth = (
        window[priority_h_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).max(axis=1)
        if priority_h_cols
        else pd.Series(0.0, index=window.index)
    )
    full_depth_frame = window[h_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0) if h_cols else pd.DataFrame()
    priority_depth_frame = window[priority_h_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0) if priority_h_cols else pd.DataFrame()
    return {
        "PFV_H": float(pfv_rate.sum() * dt_sec),
        "TFV_H": float(tfv_rate.sum() * dt_sec),
        "peak_TFV_rate_H": float(tfv_rate.max()),
        "priority_peak_depth_H": float(priority_depth.max()),
        "high_risk_exposure_time_H": float((priority_depth > 0.20).sum() * dt_min),
        "full_depth_mean_H": float(full_depth_frame.to_numpy(float).mean()) if not full_depth_frame.empty else 0.0,
        "full_depth_p95_H": float(np.nanquantile(full_depth_frame.to_numpy(float), 0.95)) if not full_depth_frame.empty else 0.0,
        "full_depth_max_H": float(full_depth_frame.to_numpy(float).max()) if not full_depth_frame.empty else 0.0,
        "priority_depth_mean_H": float(priority_depth_frame.to_numpy(float).mean()) if not priority_depth_frame.empty else 0.0,
        "priority_depth_p95_H": float(np.nanquantile(priority_depth_frame.to_numpy(float), 0.95)) if not priority_depth_frame.empty else 0.0,
    }


def horizon_features(
    detail: pd.DataFrame,
    start_idx: int,
    horizon_steps: int,
    history_steps: int,
    priority_nodes: list[str],
    actuators: pd.DataFrame | None = None,
    priority_to_actuators: pd.DataFrame | None = None,
) -> dict[str, float]:
    row = detail.iloc[start_idx]
    hist = detail.iloc[max(0, start_idx - int(history_steps) + 1) : start_idx + 1]
    future = detail.iloc[start_idx : start_idx + int(horizon_steps)]
    hcols = [c for c in detail.columns if c.startswith("h:")]
    priority_h_cols = [f"h:{n}" for n in priority_nodes if f"h:{n}" in detail.columns]
    action_cols = [c for c in detail.columns if c.startswith("a:")]
    rain = pd.to_numeric(future.get("rainfall_mm_h", pd.Series(0.0, index=future.index)), errors="coerce").fillna(0.0)
    current_depths = pd.to_numeric(row[hcols], errors="coerce").fillna(0.0) if hcols else pd.Series(dtype=float)
    priority_now = pd.to_numeric(row[priority_h_cols], errors="coerce").fillna(0.0) if priority_h_cols else pd.Series(dtype=float)
    priority_hist = hist[priority_h_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0) if priority_h_cols else pd.DataFrame()
    actions = pd.to_numeric(row[action_cols], errors="coerce").fillna(1.0) if action_cols else pd.Series(dtype=float)
    action_ids = [c.split(":", 1)[1] for c in action_cols]
    previous_actions = (
        pd.to_numeric(detail.iloc[max(0, start_idx - 1)][action_cols], errors="coerce").fillna(1.0).to_numpy(float)
        if action_cols
        else np.asarray([], dtype=float)
    )
    action_sequence = (
        detail.iloc[start_idx : start_idx + int(horizon_steps)][action_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(1.0)
        .to_numpy(float)
        if action_cols
        else np.empty((0, 0), dtype=float)
    )
    feats = {
        "current_depth_mean": float(current_depths.mean()) if len(current_depths) else 0.0,
        "current_depth_p95": float(current_depths.quantile(0.95)) if len(current_depths) else 0.0,
        "current_depth_max": float(current_depths.max()) if len(current_depths) else 0.0,
        "priority_depth_mean": float(priority_now.mean()) if len(priority_now) else 0.0,
        "priority_depth_max": float(priority_now.max()) if len(priority_now) else 0.0,
        "priority_depth_trend": 0.0,
        "rain_now": float(row.get("rainfall_mm_h", 0.0) or 0.0),
        "rain_forecast_mean": float(rain.mean()) if len(rain) else 0.0,
        "rain_forecast_max": float(rain.max()) if len(rain) else 0.0,
        "action_mean": float(actions.mean()) if len(actions) else 1.0,
        "action_min": float(actions.min()) if len(actions) else 1.0,
        "action_max": float(actions.max()) if len(actions) else 1.0,
        "action_std": float(actions.std()) if len(actions) > 1 else 0.0,
    }
    feats.update(
        build_action_feature_map(
            action_ids,
            actions.to_numpy(float) if len(actions) else np.asarray([], dtype=float),
            sequence=action_sequence,
            reference_action=previous_actions,
            actuators=actuators,
            priority_to_actuators=priority_to_actuators,
        )
    )
    if not priority_hist.empty and len(priority_hist) >= 2:
        feats["priority_depth_trend"] = float(priority_hist.max(axis=1).iloc[-1] - priority_hist.max(axis=1).iloc[0])
    return feats


def build_horizon_samples_from_detail(
    detail_path: str | Path,
    priority_nodes: list[str],
    horizon_steps: int,
    history_steps: int,
    dt_sec: int,
    stride: int = 1,
    actuators: pd.DataFrame | None = None,
    priority_to_actuators: pd.DataFrame | None = None,
    reference_detail_path: str | Path | None = None,
) -> pd.DataFrame:
    path = Path(detail_path)
    detail = _read_detail_for_horizon(path, priority_nodes)
    reference = None
    reference_path = Path(reference_detail_path) if reference_detail_path else None
    if reference_path is not None:
        reference = _read_detail_for_horizon(reference_path, priority_nodes)
        if len(reference) != len(detail):
            raise ValueError(
                f"Candidate/reference row mismatch: candidate={len(detail)} reference={len(reference)} "
                f"candidate_path={path} reference_path={reference_path}"
            )
        candidate_time = pd.to_numeric(detail.get("elapsed_min"), errors="coerce").to_numpy(float)
        reference_time = pd.to_numeric(reference.get("elapsed_min"), errors="coerce").to_numpy(float)
        if not np.allclose(candidate_time, reference_time, atol=1.0e-6, rtol=0.0, equal_nan=False):
            raise ValueError(f"Candidate/reference elapsed_min mismatch: {path} vs {reference_path}")
    if len(detail) <= horizon_steps + 1:
        return pd.DataFrame()

    n = len(detail)
    h = int(horizon_steps)
    starts = np.arange(0, n - h - 1, max(1, int(stride)), dtype=np.int64)
    if starts.size == 0:
        return pd.DataFrame()

    hcols = [c for c in detail.columns if c.startswith("h:")]
    flood_cols = [c for c in detail.columns if c.startswith("flood:")]
    action_cols = [c for c in detail.columns if c.startswith("a:")]
    action_ids = [c.split(":", 1)[1] for c in action_cols]
    priority_h_cols = [f"h:{node}" for node in priority_nodes if f"h:{node}" in detail.columns]
    priority_flood_cols = [f"flood:{node}" for node in priority_nodes if f"flood:{node}" in detail.columns]

    h_mat = detail[hcols].to_numpy(dtype=np.float32, copy=False) if hcols else np.empty((n, 0), dtype=np.float32)
    priority_h_mat = (
        detail[priority_h_cols].to_numpy(dtype=np.float32, copy=False)
        if priority_h_cols
        else np.empty((n, 0), dtype=np.float32)
    )
    action_mat = (
        detail[action_cols].to_numpy(dtype=np.float32, copy=False)
        if action_cols
        else np.empty((n, 0), dtype=np.float32)
    )
    reference_action_mat = action_mat
    if reference is not None and action_cols:
        missing_actions = [c for c in action_cols if c not in reference]
        if missing_actions:
            raise ValueError(f"No-control reference is missing action columns: {missing_actions[:5]}")
        reference_action_mat = reference[action_cols].to_numpy(dtype=np.float32, copy=False)
    flood_rate = (
        np.nan_to_num(detail[flood_cols].to_numpy(dtype=np.float32, copy=False), nan=0.0).sum(axis=1)
        if flood_cols
        else np.zeros(n, dtype=np.float32)
    )
    priority_flood_rate = (
        np.nan_to_num(detail[priority_flood_cols].to_numpy(dtype=np.float32, copy=False), nan=0.0).sum(axis=1)
        if priority_flood_cols
        else np.zeros(n, dtype=np.float32)
    )
    priority_depth = (
        np.nanmax(priority_h_mat, axis=1)
        if priority_h_mat.shape[1]
        else np.zeros(n, dtype=np.float32)
    )
    rainfall = (
        pd.to_numeric(detail.get("rainfall_mm_h", pd.Series(0.0, index=detail.index)), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    elapsed = (
        pd.to_numeric(detail.get("elapsed_min", pd.Series(np.arange(n) * float(dt_sec) / 60.0)), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    # Candidate outcomes remain the supervised target, but the controller
    # state and action delta are anchored to the same-time No-control twin.
    # This removes policy-history state leakage from the effect model input.
    state_detail = reference if reference is not None else detail
    state_h_mat = (
        state_detail[hcols].to_numpy(dtype=np.float32, copy=False)
        if hcols
        else np.empty((n, 0), dtype=np.float32)
    )
    state_priority_h_mat = (
        state_detail[priority_h_cols].to_numpy(dtype=np.float32, copy=False)
        if priority_h_cols
        else np.empty((n, 0), dtype=np.float32)
    )
    state_priority_depth = (
        np.nanmax(state_priority_h_mat, axis=1)
        if state_priority_h_mat.shape[1]
        else np.zeros(n, dtype=np.float32)
    )

    dt_min = float(dt_sec) / 60.0
    current_depths = state_h_mat[starts] if state_h_mat.shape[1] else np.empty((len(starts), 0), dtype=np.float32)
    priority_now = state_priority_h_mat[starts] if state_priority_h_mat.shape[1] else np.empty((len(starts), 0), dtype=np.float32)
    actions = action_mat[starts] if action_mat.shape[1] else np.empty((len(starts), 0), dtype=np.float32)
    history_start = np.maximum(0, starts - int(history_steps) + 1)

    rain_window = np.vstack([rainfall[starts + k] for k in range(h)])
    rows = {
        "event_id": str(detail.get("event_id", pd.Series([path.name.split("__")[0]])).iloc[0]),
        "policy_id": str(detail.get("policy_id", pd.Series(["unknown"])).iloc[0]),
        "detail_file": str(path),
        "row_index": starts,
        "elapsed_min": elapsed[starts],
        "phase": (
            detail["phase"].astype(str).iloc[starts].to_numpy()
            if "phase" in detail.columns
            else np.asarray(["unknown"] * len(starts), dtype=object)
        ),
        "current_depth_mean": np.nanmean(current_depths, axis=1) if current_depths.shape[1] else np.zeros(len(starts)),
        "current_depth_p95": np.nanquantile(current_depths, 0.95, axis=1) if current_depths.shape[1] else np.zeros(len(starts)),
        "current_depth_max": np.nanmax(current_depths, axis=1) if current_depths.shape[1] else np.zeros(len(starts)),
        "priority_depth_mean": np.nanmean(priority_now, axis=1) if priority_now.shape[1] else np.zeros(len(starts)),
        "priority_depth_max": np.nanmax(priority_now, axis=1) if priority_now.shape[1] else np.zeros(len(starts)),
        "priority_depth_trend": state_priority_depth[starts] - state_priority_depth[history_start],
        "rain_now": rainfall[starts],
        "rain_forecast_mean": np.nanmean(rain_window, axis=0),
        "rain_forecast_max": np.nanmax(rain_window, axis=0),
        "PFV_H": _window_sum(priority_flood_rate, starts, 1, h) * dt_sec,
        "TFV_H": _window_sum(flood_rate, starts, 1, h) * dt_sec,
        "peak_TFV_rate_H": _window_max(flood_rate, starts, 1, h),
        "priority_peak_depth_H": _window_max(priority_depth, starts, 1, h),
        "full_depth_mean_H": _window_matrix_stat(h_mat, starts, 1, h, "mean"),
        "full_depth_p95_H": _window_matrix_stat(h_mat, starts, 1, h, "quantile", 0.95),
        "full_depth_max_H": _window_matrix_stat(h_mat, starts, 1, h, "max"),
        "priority_depth_mean_H": _window_matrix_stat(priority_h_mat, starts, 1, h, "mean"),
        "priority_depth_p95_H": _window_matrix_stat(priority_h_mat, starts, 1, h, "quantile", 0.95),
    }
    rows["high_risk_exposure_time_H"] = (
        np.vstack([(priority_depth[starts + 1 + k] > 0.20).astype(np.float32) for k in range(h)]).sum(axis=0) * dt_min
    )
    target_columns = [
        "PFV_H", "TFV_H", "peak_TFV_rate_H", "priority_peak_depth_H",
        "high_risk_exposure_time_H", "full_depth_mean_H", "full_depth_p95_H",
        "full_depth_max_H", "priority_depth_mean_H", "priority_depth_p95_H",
    ]
    if reference is not None:
        reference_samples = build_horizon_samples_from_detail(
            reference_path,
            priority_nodes,
            horizon_steps=horizon_steps,
            history_steps=history_steps,
            dt_sec=dt_sec,
            stride=stride,
            actuators=actuators,
            priority_to_actuators=priority_to_actuators,
            reference_detail_path=None,
        )
        if not np.array_equal(
            reference_samples["row_index"].to_numpy(np.int64), starts
        ):
            raise ValueError(f"No-control reference horizon rows are not aligned for {path}")
        for col in target_columns:
            rows[f"reference_{col}"] = reference_samples[col].to_numpy(np.float32)
            rows[f"effect_{col}"] = np.asarray(rows[col], dtype=np.float32) - rows[f"reference_{col}"]
        rows["reference_detail_file"] = str(reference_path)
        rows["action_semantics"] = "absolute_from_no_control_reference"
        rows["effect_label_mode"] = "paired_no_control_same_time"
    action_feature_rows = []
    for start in starts:
        if action_mat.shape[1]:
            current_action = action_mat[start]
            sequence = action_mat[start : start + h]
            reference_sequence = reference_action_mat[start : start + h]
        else:
            current_action = np.asarray([], dtype=np.float32)
            sequence = np.empty((0, 0), dtype=np.float32)
            reference_sequence = np.empty((0, 0), dtype=np.float32)
        action_feature_rows.append(
            build_action_feature_map(
                action_ids,
                current_action,
                sequence=sequence,
                reference_action=reference_sequence,
                actuators=actuators,
                priority_to_actuators=priority_to_actuators,
            )
        )
    if action_feature_rows:
        for col in ACTION_FEATURE_COLUMNS:
            rows[col] = np.asarray([r.get(col, 0.0) for r in action_feature_rows], dtype=np.float32)
    return pd.DataFrame(rows)
