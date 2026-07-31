from __future__ import annotations

"""Project6 V4 causal dual-reference dataset, model and readiness stages.

V4 is intentionally a two-layer model:

1. a *causal reference-envelope model* predicts the No-control/Passive PFV
   safety envelope and the Internal PFV/TFV/Peak envelope from only the current
   reconstructed state, causal history summaries and the operational rainfall
   forecast;
2. an *action residual model* predicts the candidate effect relative to those
   metric-specific references.

The module never copies an older model, never fabricates runtime labels and
never uses a pre-run future SWMM trajectory as an online feature.
"""

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

from sewerrtc.prompt3.action_effect_mpc import _fit_ridge
from sewerrtc.simulation.runtime_contracts import sha256_file, utc_now, write_csv, write_json

CONTEXT_FEATURE_NAMES = (
    "elapsed_fraction",
    "rain_current",
    "rain_sum_120",
    "rain_max_120",
    "rain_mean_120",
    "rain_trend_120",
    "state_mean",
    "state_std",
    "state_p50",
    "state_p90",
    "state_max",
    "action_mean",
    "action_std",
    "action_min",
    "action_max",
    "phase_rising",
    "phase_peak",
    "phase_recession",
)

ACTION_FEATURE_NAMES = (
    "active_count",
    "active_fraction",
    "increase_fraction",
    "decrease_fraction",
    "mean_absolute_delta",
    "max_absolute_delta",
    "binary_off_to_on",
    "binary_on_to_off",
    "fallback_is_internal",
)

REFERENCE_LABELS = (
    "no_control_PFV_H120",
    "passive_PFV_H120",
    "internal_PFV_H120",
    "internal_TFV_H120",
    "internal_peak_H120",
    "no_control_PFV_full",
    "passive_PFV_full",
    "internal_PFV_full",
)

RESIDUAL_LABELS = (
    "delta_PFV_H120_vs_no_control",
    "delta_PFV_H120_vs_passive",
    "delta_TFV_H120_vs_internal",
    "delta_peak_H120_vs_internal",
    "delta_PFV_full_vs_no_control",
    "delta_PFV_full_vs_passive",
)

CAUSAL_FEATURE_NAMES = (
    "cumulative_PFV_before_checkpoint",
    "cumulative_TFV_before_checkpoint",
    "cumulative_priority_duration_before_checkpoint",
    "rainfall_elapsed_total",
    "operational_forecast_remaining_rainfall_total",
    "operational_forecast_remaining_peak",
    "operational_forecast_time_to_peak",
    "elapsed_fraction",
    "estimated_remaining_event_fraction",
    "current_priority_depth_mean",
    "current_priority_depth_max",
    "current_storage_volume_mean",
    "storage_headroom_mean",
    "downstream_headroom_mean",
    "previous_60min_action_variation",
    "previous_60min_candidate_fallback_switches",
    "previous_executed_setting_mean",
    "controller_memory_override_active",
)

V4_LABELS = REFERENCE_LABELS + RESIDUAL_LABELS


class FutureHydraulicLeakageError(RuntimeError):
    """Raised when a causal feature builder is asked to read a post-checkpoint row.

    The full-event PFV heads must learn a path-dependent, causally-available
    signal. Any hydraulic quantity used as a feature has to come from a
    ``elapsed_min <= checkpoint`` row. Future rainfall is only allowed from the
    frozen design hyetograph, which is an operational forecast rather than a
    realised hydraulic truth.
    """


def causal_context_features(
    prefix_rows: Sequence[dict[str, Any]],
    *,
    checkpoint_elapsed_min: float,
    event_duration_min: float,
    rainfall_forecast: Sequence[tuple[float, float]],
    priority_nodes: Sequence[str],
    storage_nodes: Sequence[str] = (),
    downstream_nodes: Sequence[str] = (),
    controller_memory: dict[str, Any] | None = None,
    storage_capacity: dict[str, float] | None = None,
    node_freeboard: dict[str, float] | None = None,
) -> np.ndarray:
    """Build the Aug1 causal path-dependent context vector (leakage-free).

    ``prefix_rows`` must contain only rows at or before the checkpoint. Passing
    any row with ``elapsed_min`` beyond ``checkpoint_elapsed_min`` is a contract
    violation and raises :class:`FutureHydraulicLeakageError`; this is what the
    Section 八 leakage test relies on. ``rainfall_forecast`` is the frozen design
    hyetograph ``[(elapsed_min, intensity_mm_h), ...]`` used as the operational
    forecast; only its post-checkpoint tail feeds the "remaining" features.
    """
    checkpoint = float(checkpoint_elapsed_min)
    tol = 1.0e-6
    rows: list[dict[str, Any]] = []
    for row in prefix_rows:
        try:
            elapsed = float(row.get("elapsed_min", 0.0) or 0.0)
        except Exception:
            elapsed = 0.0
        if elapsed > checkpoint + tol:
            raise FutureHydraulicLeakageError(
                f"causal feature builder received a post-checkpoint row: "
                f"elapsed_min={elapsed} > checkpoint={checkpoint}"
            )
        rows.append(row)

    priority = {str(n) for n in priority_nodes}
    storage = {str(n) for n in storage_nodes}
    downstream = {str(n) for n in downstream_nodes}
    capacity = {str(k): float(v) for k, v in (storage_capacity or {}).items()}
    freeboard = {str(k): float(v) for k, v in (node_freeboard or {}).items()}

    dt_min = 10.0
    if len(rows) >= 2:
        try:
            dt_min = max(1.0e-6, float(rows[1].get("elapsed_min", 10.0)) - float(rows[0].get("elapsed_min", 0.0)))
        except Exception:
            dt_min = 10.0
    dt_hours = dt_min / 60.0

    cum_pfv = 0.0
    cum_tfv = 0.0
    cum_priority_duration = 0.0
    rain_elapsed_total = 0.0
    for row in rows:
        step_priority_flood = 0.0
        step_total_flood = 0.0
        for key, value in row.items():
            if not key.startswith("flood:"):
                continue
            try:
                flood = float(value or 0.0)
            except Exception:
                flood = 0.0
            if flood <= 0.0:
                continue
            step_total_flood += flood
            node = key.split(":", 1)[1]
            if node in priority:
                step_priority_flood += flood
        cum_tfv += step_total_flood * dt_min * 60.0
        cum_pfv += step_priority_flood * dt_min * 60.0
        if step_priority_flood > 0.0:
            cum_priority_duration += dt_min
        try:
            rain_elapsed_total += max(0.0, float(row.get("rainfall_mm_h", 0.0) or 0.0)) * dt_hours
        except Exception:
            pass

    remaining = [
        (float(t), max(0.0, float(i)))
        for t, i in rainfall_forecast
        if float(t) > checkpoint + tol
    ]
    remaining_total = sum(i for _, i in remaining) * dt_hours
    remaining_peak = max((i for _, i in remaining), default=0.0)
    if remaining and remaining_peak > 0.0:
        peak_time = min(t for t, i in remaining if i >= remaining_peak - tol)
        time_to_peak = max(0.0, peak_time - checkpoint)
    else:
        time_to_peak = 0.0

    duration = max(1.0e-6, float(event_duration_min))
    elapsed_fraction = min(max(checkpoint / duration, 0.0), 2.0)
    remaining_fraction = max(0.0, 1.0 - elapsed_fraction)

    last = rows[-1] if rows else {}
    priority_depths: list[float] = []
    storage_volumes: list[float] = []
    storage_headrooms: list[float] = []
    downstream_headrooms: list[float] = []
    for key, value in last.items():
        if not key.startswith("h:"):
            continue
        try:
            depth = float(value or 0.0)
        except Exception:
            continue
        node = key.split(":", 1)[1]
        if node in priority:
            priority_depths.append(depth)
        if node in storage:
            storage_volumes.append(depth)
            cap = capacity.get(node)
            if cap is not None:
                storage_headrooms.append(max(0.0, cap - depth))
        if node in downstream:
            fb = freeboard.get(node)
            if fb is not None:
                downstream_headrooms.append(max(0.0, fb - depth))

    window_rows = [
        row for row in rows
        if float(row.get("elapsed_min", 0.0) or 0.0) >= checkpoint - 60.0 - tol
    ]
    variation = 0.0
    fallback_switches = 0.0
    executed_settings: list[float] = []
    prev_settings: dict[str, float] | None = None
    prev_fallback: str | None = None
    for row in window_rows:
        settings: dict[str, float] = {}
        for key, value in row.items():
            if key.startswith("setting:") or key.startswith("a:"):
                try:
                    settings[key] = float(value or 0.0)
                except Exception:
                    pass
        if settings:
            executed_settings.extend(settings.values())
        if prev_settings is not None:
            shared = set(settings) & set(prev_settings)
            variation += sum(abs(settings[k] - prev_settings[k]) for k in shared)
        prev_settings = settings or prev_settings
        fallback = str(row.get("selected_fallback", row.get("fallback_id", "")) or "")
        if prev_fallback is not None and fallback and fallback != prev_fallback:
            fallback_switches += 1.0
        if fallback:
            prev_fallback = fallback

    memory = controller_memory or {}
    override_active = 1.0 if _truth(memory.get("override_active")) else 0.0

    return np.asarray([
        cum_pfv, cum_tfv, cum_priority_duration,
        rain_elapsed_total, remaining_total, remaining_peak, time_to_peak,
        elapsed_fraction, remaining_fraction,
        float(np.mean(priority_depths)) if priority_depths else 0.0,
        float(np.max(priority_depths)) if priority_depths else 0.0,
        float(np.mean(storage_volumes)) if storage_volumes else 0.0,
        float(np.mean(storage_headrooms)) if storage_headrooms else 0.0,
        float(np.mean(downstream_headrooms)) if downstream_headrooms else 0.0,
        variation,
        fallback_switches,
        float(np.mean(executed_settings)) if executed_settings else 0.0,
        override_active,
    ], dtype=float)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _project_root(config: str | Path) -> Path:
    cfg = _load_yaml(config)
    root_raw = (cfg.get("project", {}) or {}).get("root", ".")
    root = Path(root_raw)
    if root.is_absolute():
        return root
    # .git-based detection, then config-relative fallback
    config_path = Path(config).resolve()
    cur = config_path
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            return parent
    return config_path.parent.parent / root


def _output_root(config: str | Path) -> Path:
    cfg = _load_yaml(config)
    root = _project_root(config)
    raw = Path((cfg.get("project", {}) or {}).get("output_root", "outputs/project6_dual_reference_v4"))
    return raw if raw.is_absolute() else root / raw


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = row.get(key, "")
        if value in {"", None}:
            return None
        out = float(value)
        return out if np.isfinite(out) else None
    except Exception:
        return None


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "completed"}


def _selected_candidate_prefix(row: dict[str, Any]) -> str:
    explicit = str(row.get("candidate_selected_branch_id") or "").strip()
    if explicit:
        return explicit
    selected = str(row.get("selected_fallback") or row.get("selected_fallback_branch_id") or row.get("fallback_id") or "").lower()
    return "candidate_then_internal" if "internal" in selected else "candidate_then_passive"


def _metric(row: dict[str, Any], branch: str, metric: str, horizon: str) -> float | None:
    aliases = [
        f"{branch}_{metric}_{horizon}",
        f"{branch}_{metric}_{horizon.lower()}",
    ]
    if branch == "executable_passive":
        aliases += [f"passive_anchor_{metric}_{horizon}", f"passive_{metric}_{horizon}"]
    if branch == "internal_rules":
        aliases += [f"internal_{metric}_{horizon}"]
    if branch == "no_control":
        aliases += [f"no_control_{metric}_{horizon}"]
    for key in aliases:
        value = _float(row, key)
        if value is not None:
            return value
    return None


def _phase_features(phase: str) -> tuple[float, float, float]:
    text = str(phase or "").lower()
    return (
        1.0 if any(token in text for token in ("rise", "rising", "pre_peak", "pre-rain")) else 0.0,
        1.0 if "peak" in text and "pre" not in text else 0.0,
        1.0 if any(token in text for token in ("recession", "recovery", "release", "recovered")) else 0.0,
    )


def _summary(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        float(arr.mean()), float(arr.std()), float(np.quantile(arr, 0.50)),
        float(np.quantile(arr, 0.90)), float(arr.max()),
    )


def runtime_context_features(context: dict[str, Any]) -> np.ndarray:
    """Create the causal context vector used by both training and deployment."""
    state = np.asarray(context.get("reconstructed_state", []), dtype=float).reshape(-1)
    state = state[np.isfinite(state)]
    rain = np.asarray(context.get("rainfall_window", []), dtype=float).reshape(-1)
    rain = np.nan_to_num(rain, nan=0.0, posinf=0.0, neginf=0.0)
    action = np.asarray(context.get("current_action", []), dtype=float).reshape(-1)
    action = action[np.isfinite(action)]
    state_mean, state_std, state_p50, state_p90, state_max = _summary(state)
    action_mean, action_std, _, _, action_max = _summary(action)
    action_min = float(action.min()) if action.size else 0.0
    rain_current = float(rain[0]) if rain.size else 0.0
    rain_sum = float(np.maximum(rain, 0.0).sum()) if rain.size else 0.0
    rain_max = float(np.maximum(rain, 0.0).max()) if rain.size else 0.0
    rain_mean = float(np.maximum(rain, 0.0).mean()) if rain.size else 0.0
    rain_trend = float(rain[-1] - rain[0]) if rain.size > 1 else 0.0
    phase_rising, phase_peak, phase_recession = _phase_features(str(context.get("phase", "")))
    return np.asarray([
        min(max(float(context.get("elapsed_min", 0.0)) / 720.0, 0.0), 2.0),
        rain_current, rain_sum, rain_max, rain_mean, rain_trend,
        state_mean, state_std, state_p50, state_p90, state_max,
        action_mean, action_std, action_min, action_max,
        phase_rising, phase_peak, phase_recession,
    ], dtype=float)


def runtime_action_features(sequence: np.ndarray, context: dict[str, Any]) -> np.ndarray:
    seq = np.asarray(sequence, dtype=float)
    action = seq[0] if seq.ndim == 2 and seq.shape[0] else np.asarray([], dtype=float)
    reference = np.asarray(context.get("reference_action_sequence", action), dtype=float)
    reference_first = reference[0] if reference.ndim == 2 and reference.shape[0] else np.asarray(context.get("current_action", action), dtype=float)
    action = action.reshape(-1)
    reference_first = reference_first.reshape(-1)
    n = min(action.size, reference_first.size)
    delta = action[:n] - reference_first[:n] if n else np.asarray([], dtype=float)
    active = np.abs(delta) > 1.0e-6
    active_count = int(active.sum())
    denom = max(1, n)
    inc = float(np.sum(delta[active] > 0.0) / max(1, active_count)) if active_count else 0.0
    dec = float(np.sum(delta[active] < 0.0) / max(1, active_count)) if active_count else 0.0
    mean_abs = float(np.mean(np.abs(delta[active]))) if active_count else 0.0
    max_abs = float(np.max(np.abs(delta[active]))) if active_count else 0.0
    label = str(context.get("label", "")).lower()
    fallback = str(context.get("selected_fallback_id", "")).lower()
    return np.asarray([
        float(active_count), float(active_count / denom), inc, dec, mean_abs, max_abs,
        1.0 if "off_to_on" in label or "off->on" in label else 0.0,
        1.0 if "on_to_off" in label or "on->off" in label else 0.0,
        1.0 if "internal" in fallback else 0.0,
    ], dtype=float)


def _resolve_detail_path(row: dict[str, Any], project_root: Path) -> Path | None:
    raw = str(row.get("candidate_detail_file") or row.get("detail_file") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    return path if path.exists() and path.is_file() else None


def _context_from_detail(path: Path, checkpoint_elapsed_min: float, phase: str) -> np.ndarray | None:
    """Extract only causal state and rainfall forcing; never hydraulic future."""
    best_row: dict[str, str] | None = None
    best_distance = float("inf")
    rainfall: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for idx, row in enumerate(reader):
            try:
                elapsed = float(row.get("elapsed_min", idx * 10.0) or idx * 10.0)
            except Exception:
                elapsed = float(idx * 10.0)
            distance = abs(elapsed - float(checkpoint_elapsed_min))
            if distance < best_distance:
                best_distance = distance
                best_row = row
            try:
                rain = float(row.get("rainfall_mm_h", 0.0) or 0.0)
            except Exception:
                rain = 0.0
            rainfall.append((elapsed, rain))
    if best_row is None:
        return None
    try:
        current_elapsed = float(best_row.get("elapsed_min", checkpoint_elapsed_min) or checkpoint_elapsed_min)
    except Exception:
        current_elapsed = float(checkpoint_elapsed_min)
    h_values = []
    action_values = []
    for key, value in best_row.items():
        if key.startswith("h:"):
            try:
                h_values.append(float(value))
            except Exception:
                pass
        elif key.startswith("setting:"):
            try:
                action_values.append(float(value))
            except Exception:
                pass
    if not action_values:
        for key, value in best_row.items():
            if key.startswith("a:"):
                try:
                    action_values.append(float(value))
                except Exception:
                    pass
    if not h_values:
        return None
    rain_window = [value for elapsed, value in rainfall if current_elapsed - 1.0e-6 <= elapsed <= current_elapsed + 120.0 + 1.0e-6]
    context = {
        "elapsed_min": current_elapsed,
        "phase": phase or best_row.get("phase", ""),
        "reconstructed_state": h_values,
        "rainfall_window": rain_window or [float(best_row.get("rainfall_mm_h", 0.0) or 0.0)],
        "current_action": action_values,
    }
    return runtime_context_features(context)


def _training_action_features(row: dict[str, Any]) -> np.ndarray:
    try:
        active = max(0.0, float(row.get("k_value", row.get("concurrency", 0)) or 0.0))
    except Exception:
        active = 0.0
    direction = str(row.get("action_direction", row.get("action_directions", ""))).lower()
    magnitude = str(row.get("action_magnitude", "")).lower()
    mag = {"small": 0.05, "medium": 0.15, "large": 0.40, "boundary": 1.0, "binary": 1.0}.get(magnitude, 0.0)
    fallback = str(row.get("selected_fallback", row.get("selected_fallback_branch_id", ""))).lower()
    return np.asarray([
        active, active / 36.0,
        1.0 if "increase" in direction or "off_to_on" in direction else 0.0,
        1.0 if "decrease" in direction or "on_to_off" in direction else 0.0,
        mag, mag,
        1.0 if "off_to_on" in direction else 0.0,
        1.0 if "on_to_off" in direction else 0.0,
        1.0 if "internal" in fallback else 0.0,
    ], dtype=float)


def _precomputed_context(row: dict[str, Any]) -> np.ndarray | None:
    values = []
    for name in CONTEXT_FEATURE_NAMES:
        value = _float(row, f"v4_ctx_{name}")
        if value is None:
            return None
        values.append(value)
    return np.asarray(values, dtype=float)


def materialize_v4_row(
    row: dict[str, Any], *, project_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return one leakage-free V4 training row, or ``None`` when unsupported."""
    if str(row.get("truth_future_leakage", row.get("truth_leakage", "0"))).strip().lower() not in {"", "0", "0.0", "false"}:
        return None
    if row.get("runtime_executed") not in {None, ""} and not _truth(row.get("runtime_executed")):
        return None
    candidate = _selected_candidate_prefix(row)
    nc_h = _metric(row, "no_control", "PFV", "H120")
    pa_h = _metric(row, "executable_passive", "PFV", "H120")
    in_pfv_h = _metric(row, "internal_rules", "PFV", "H120")
    in_tfv_h = _metric(row, "internal_rules", "TFV", "H120")
    in_peak_h = _metric(row, "internal_rules", "peak_TFV_rate", "H120")
    cand_pfv_h = _metric(row, candidate, "PFV", "H120")
    cand_tfv_h = _metric(row, candidate, "TFV", "H120")
    cand_peak_h = _metric(row, candidate, "peak_TFV_rate", "H120")
    nc_full = _metric(row, "no_control", "PFV", "full_recovery")
    pa_full = _metric(row, "executable_passive", "PFV", "full_recovery")
    in_pfv_full = _metric(row, "internal_rules", "PFV", "full_recovery")
    cand_pfv_full = _metric(row, candidate, "PFV", "full_recovery")
    required = [nc_h, pa_h, in_pfv_h, in_tfv_h, in_peak_h, cand_pfv_h, cand_tfv_h, cand_peak_h, nc_full, pa_full, in_pfv_full, cand_pfv_full]
    if any(value is None for value in required):
        return None
    context = _precomputed_context(row)
    if context is None:
        root = Path(project_root) if project_root is not None else Path.cwd()
        detail = _resolve_detail_path(row, root)
        if detail is None:
            return None
        context = _context_from_detail(detail, float(row.get("checkpoint_elapsed_min", 0.0) or 0.0), str(row.get("phase", "")))
    if context is None or context.shape != (len(CONTEXT_FEATURE_NAMES),):
        return None
    action = _training_action_features(row)
    out = dict(row)
    values = {
        "no_control_PFV_H120": nc_h,
        "passive_PFV_H120": pa_h,
        "internal_PFV_H120": in_pfv_h,
        "internal_TFV_H120": in_tfv_h,
        "internal_peak_H120": in_peak_h,
        "no_control_PFV_full": nc_full,
        "passive_PFV_full": pa_full,
        "internal_PFV_full": in_pfv_full,
        "delta_PFV_H120_vs_no_control": cand_pfv_h - nc_h,
        "delta_PFV_H120_vs_passive": cand_pfv_h - pa_h,
        "delta_TFV_H120_vs_internal": cand_tfv_h - in_tfv_h,
        "delta_peak_H120_vs_internal": cand_peak_h - in_peak_h,
        "delta_PFV_full_vs_no_control": cand_pfv_full - nc_full,
        "delta_PFV_full_vs_passive": cand_pfv_full - pa_full,
    }
    out.update({key: float(value) for key, value in values.items()})
    out.update({f"v4_ctx_{name}": float(value) for name, value in zip(CONTEXT_FEATURE_NAMES, context)})
    out.update({f"v4_act_{name}": float(value) for name, value in zip(ACTION_FEATURE_NAMES, action)})
    out["v4_candidate_branch"] = candidate
    out["v4_label_contract"] = "causal_reference_envelope_plus_metric_aligned_action_residual"
    out["v4_authoritative_label"] = "true"
    out["v4_online_future_hydraulics_used"] = "false"
    return out


def _dataset_sources(config: str | Path) -> list[Path]:
    cfg = _load_yaml(config)
    root = _project_root(config)
    configured = list(((cfg.get("v4", {}) or {}).get("training", {}) or {}).get("dataset_manifests", []) or [])
    if configured:
        return [p if (p := Path(item)).is_absolute() else root / p for item in configured]
    return [
        root / "outputs/project6_pfvfirst_dualfallback_10min_v3/action_effect_dataset/action_effect_dataset_manifest.csv",
        root / "outputs/project6_pfvfirst_dualfallback_10min_v3_1/round3_dataset/round3_dataset_manifest.csv",
    ]


def _event_balanced_sample(
    frame: pd.DataFrame,
    *,
    event_column: str,
    max_samples: int,
    minimum_events: int,
    random_seed: int,
) -> pd.DataFrame:
    """Deterministically sample rows while preserving event diversity.

    Rows are grouped by ``event_column``; selection then round-robins across
    the shuffled groups so a single large event cannot fill the whole Smoke
    budget. When the number of available events is below ``minimum_events``
    the function returns the plain head slice and never fabricates diversity;
    the downstream audit must block in that case.
    """

    if frame.empty:
        return frame.copy()

    if event_column not in frame.columns:
        raise KeyError(
            f"Event column {event_column!r} is missing from the V4 dataset."
        )

    working = frame.copy().reset_index(drop=True)
    working[event_column] = (
        working[event_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    working = working.loc[
        working[event_column] != ""
    ].copy()

    if max_samples <= 0 or len(working) <= max_samples:
        return working.reset_index(drop=True)

    grouped = [
        (str(event_id), group.reset_index(drop=True))
        for event_id, group in working.groupby(
            event_column,
            sort=True,
            dropna=False,
        )
        if not group.empty
    ]

    if len(grouped) < minimum_events:
        # Do not fabricate diversity. The downstream audit must block.
        return working.iloc[:max_samples].reset_index(drop=True)

    rng = random.Random(random_seed)
    rng.shuffle(grouped)

    shuffled_groups: list[pd.DataFrame] = []

    for group_index, (_, group) in enumerate(grouped):
        shuffled = group.sample(
            frac=1.0,
            replace=False,
            random_state=random_seed + group_index,
        ).reset_index(drop=True)

        shuffled_groups.append(shuffled)

    cursors = [0 for _ in shuffled_groups]
    selected_rows: list[pd.DataFrame] = []

    while len(selected_rows) < max_samples:
        made_progress = False

        for group_index, group in enumerate(shuffled_groups):
            cursor = cursors[group_index]

            if cursor >= len(group):
                continue

            selected_rows.append(group.iloc[[cursor]])
            cursors[group_index] += 1
            made_progress = True

            if len(selected_rows) >= max_samples:
                break

        if not made_progress:
            break

    if not selected_rows:
        return working.iloc[0:0].copy()

    sampled = pd.concat(
        selected_rows,
        axis=0,
        ignore_index=True,
    )

    return sampled.iloc[:max_samples].reset_index(drop=True)


def build_v4_dataset(config: str | Path, *, smoke: bool = False, max_samples: int = 0) -> tuple[int, dict[str, Path]]:
    out_root = _output_root(config)
    out_dir = out_root / "action_effect_dataset_v4"
    project_root = _project_root(config)
    cfg = _load_yaml(config)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in _dataset_sources(config):
        for raw in _read_csv(source):
            case_id = str(raw.get("candidate_id") or raw.get("case_id") or raw.get("sample_id") or "")
            identity = hashlib.sha256(json.dumps({
                "state": raw.get("state_hash", raw.get("hydraulic_state_hash", raw.get("checkpoint_id", ""))),
                "action": raw.get("actual_action_hash", raw.get("action_hash", case_id)),
                "event": raw.get("event_id", ""),
            }, sort_keys=True).encode()).hexdigest()
            if identity in seen:
                rejected.append({"source": str(source), "case_id": case_id, "reason": "duplicate_state_action_identity"})
                continue
            seen.add(identity)
            row = materialize_v4_row(raw, project_root=project_root)
            if row is None:
                rejected.append({"source": str(source), "case_id": case_id, "reason": "missing_authoritative_H120_full_or_causal_context"})
                continue
            row["v4_source_manifest"] = str(source)
            row["v4_sample_identity_sha256"] = identity
            rows.append(row)
    rows.sort(key=lambda item: (str(item.get("event_id", "")), str(item.get("checkpoint_id", "")), str(item.get("candidate_id", ""))))
    # Never head-truncate the sorted rows: that concentrates the Smoke budget in
    # the lexicographically first event. Sample with event balancing instead so
    # the audit's minimum_unique_events contract can actually be met whenever
    # the source manifests contain enough labelled events.
    limit = int(max_samples) if max_samples > 0 else (64 if smoke else 0)
    if limit > 0 and len(rows) > limit:
        seed_cfg = list((((cfg.get("v4", {}) or {}).get("training", {}) or {})).get("seeds", []) or [])
        random_seed = int(seed_cfg[0]) if seed_cfg else 20260723
        rows = _event_balanced_sample(
            pd.DataFrame(rows),
            event_column="event_id",
            max_samples=limit,
            minimum_events=2 if smoke else 12,
            random_seed=random_seed,
        ).to_dict(orient="records")
    manifest = write_csv(out_dir / ("v4_dataset_smoke_manifest.csv" if smoke else "v4_dataset_manifest.csv"), rows)
    rejected_path = write_csv(out_dir / ("v4_dataset_smoke_rejected.csv" if smoke else "v4_dataset_rejected.csv"), rejected)
    required = 8 if smoke else int((((cfg.get("v4", {}) or {}).get("training", {}) or {}).get("required_min_samples", 3000)))
    unique_events = len({str(row.get("event_id", "")) for row in rows if str(row.get("event_id", ""))})
    status = "pass" if len(rows) >= required and unique_events >= (2 if smoke else 12) else "blocked"
    report = write_json(out_dir / ("v4_dataset_smoke_audit.json" if smoke else "v4_dataset_audit.json"), {
        "status": status,
        "sample_count": len(rows),
        "required_min": required,
        "unique_event_count": unique_events,
        "minimum_unique_events": 2 if smoke else 12,
        "reference_heads": list(REFERENCE_LABELS),
        "residual_heads": list(RESIDUAL_LABELS),
        "context_features": list(CONTEXT_FEATURE_NAMES),
        "action_features": list(ACTION_FEATURE_NAMES),
        "source_manifests": [str(p) for p in _dataset_sources(config)],
        "truth_future_leakage_count": sum(_truth(row.get("v4_online_future_hydraulics_used")) for row in rows),
        "rejected_count": len(rejected),
        "manifest_sha256": sha256_file(manifest),
        "created_at": utc_now(),
    })
    return (0 if status == "pass" else 3), {"manifest": manifest, "rejected": rejected_path, "audit": report}


def _arrays(rows: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x_context: list[list[float]] = []
    x_residual: list[list[float]] = []
    y_reference: list[list[float]] = []
    y_residual: list[list[float]] = []
    used: list[dict[str, Any]] = []
    for row in rows:
        context = [_float(row, f"v4_ctx_{name}") for name in CONTEXT_FEATURE_NAMES]
        action = [_float(row, f"v4_act_{name}") for name in ACTION_FEATURE_NAMES]
        reference = [_float(row, label) for label in REFERENCE_LABELS]
        residual = [_float(row, label) for label in RESIDUAL_LABELS]
        if any(value is None for value in context + action + reference + residual):
            continue
        x_context.append([float(value) for value in context])
        x_residual.append([float(value) for value in context + action])
        y_reference.append([float(value) for value in reference])
        y_residual.append([float(value) for value in residual])
        used.append(row)
    return np.asarray(x_context), np.asarray(x_residual), np.asarray(y_reference), np.asarray(y_residual), used


def _split_by_event(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    event_ids = [str(row.get("event_id", "")) for row in rows]
    validation_events = {
        event for event in sorted(set(event_ids))
        if int(hashlib.sha256(event.encode()).hexdigest()[:8], 16) % 5 == 0
    }
    if not validation_events and event_ids:
        validation_events = {sorted(set(event_ids))[-1]}
    val = np.asarray([event in validation_events for event in event_ids], dtype=bool)
    train = ~val
    if not train.any() and len(train):
        train[:] = True
        val[:] = False
    return train, val


def _ensemble_predict(weights: np.ndarray, mean: np.ndarray, scale: np.ndarray, x: np.ndarray) -> np.ndarray:
    preds = []
    for member in range(weights.shape[0]):
        design = np.column_stack([np.ones(len(x)), (x - mean[member]) / scale[member]])
        preds.append(design @ weights[member])
    return np.asarray(preds, dtype=float)


def train_v4_ensemble(
    config: str | Path, *, smoke: bool = False, ensemble_size: int = 5,
    seeds: Sequence[int] | None = None,
) -> tuple[int, dict[str, Path]]:
    out_root = _output_root(config)
    ds_dir = out_root / "action_effect_dataset_v4"
    model_dir = out_root / "action_effect_models_v4"
    rows = _read_csv(ds_dir / ("v4_dataset_smoke_manifest.csv" if smoke else "v4_dataset_manifest.csv"))
    x_ctx, x_res, y_ref, y_res, used = _arrays(rows)
    cfg = _load_yaml(config)
    training_cfg = (((cfg.get("v4", {}) or {}).get("training", {}) or {}))
    minimum = 8 if smoke else int(training_cfg.get("required_min_samples", 3000))
    if len(used) < minimum:
        report = write_json(model_dir / ("v4_model_smoke_report.json" if smoke else "v4_model_report.json"), {
            "status": "blocked", "sample_count": len(used), "required_min": minimum,
            "reason": "insufficient_authoritative_causal_dual_reference_samples",
        })
        return 3, {"report": report}
    train_mask, val_mask = _split_by_event(used)
    size = max(2 if smoke else 5, int(ensemble_size))
    seed_values = list(seeds or training_cfg.get("seeds", []) or [20260723 + i for i in range(size)])[:size]
    while len(seed_values) < size:
        seed_values.append(20260723 + len(seed_values))
    ref_members, res_members = [], []
    for seed in seed_values:
        rng = np.random.default_rng(int(seed))
        pool = np.flatnonzero(train_mask)
        idx = rng.choice(pool, size=len(pool), replace=True)
        ref_w, ref_mean, ref_scale, _ = _fit_ridge(x_ctx[idx], y_ref[idx])
        res_w, res_mean, res_scale, _ = _fit_ridge(x_res[idx], y_res[idx])
        ref_members.append((ref_w, ref_mean, ref_scale))
        res_members.append((res_w, res_mean, res_scale))
    ref_weights = np.asarray([item[0] for item in ref_members])
    ref_mean = np.asarray([item[1] for item in ref_members])
    ref_scale = np.asarray([item[2] for item in ref_members])
    res_weights = np.asarray([item[0] for item in res_members])
    res_mean = np.asarray([item[1] for item in res_members])
    res_scale = np.asarray([item[2] for item in res_members])
    eval_mask = val_mask if val_mask.any() else train_mask
    ref_pred_members = _ensemble_predict(ref_weights, ref_mean, ref_scale, x_ctx[eval_mask])
    res_pred_members = _ensemble_predict(res_weights, res_mean, res_scale, x_res[eval_mask])
    ref_pred = ref_pred_members.mean(axis=0)
    res_pred = res_pred_members.mean(axis=0)
    dual_cfg = (((cfg.get("v4", {}) or {}).get("dual_reference", {}) or {}))
    quantile = float(dual_cfg.get("pfv_event_quantile", dual_cfg.get("pfv_quantile", 0.95)))
    quantile = min(max(quantile, 0.50), 0.999)
    ref_abs_error = np.abs(ref_pred - y_ref[eval_mask])
    res_abs_error = np.abs(res_pred - y_res[eval_mask])
    ref_conformal = np.quantile(ref_abs_error, quantile, axis=0) if len(ref_abs_error) else np.zeros(len(REFERENCE_LABELS))
    res_conformal = np.quantile(res_abs_error, quantile, axis=0) if len(res_abs_error) else np.zeros(len(RESIDUAL_LABELS))
    metrics: dict[str, Any] = {
        "validation_row_count": int(eval_mask.sum()),
        "validation_event_count": len({str(row.get("event_id", "")) for row, flag in zip(used, eval_mask) if flag}),
    }
    for label, idx in zip(REFERENCE_LABELS, range(len(REFERENCE_LABELS))):
        metrics[f"rmse_{label}"] = float(np.sqrt(np.mean((ref_pred[:, idx] - y_ref[eval_mask, idx]) ** 2)))
    for label, idx in zip(RESIDUAL_LABELS, range(len(RESIDUAL_LABELS))):
        truth = y_res[eval_mask, idx]
        pred = res_pred[:, idx]
        tolerance = 1.0e-9
        direction = np.where(np.abs(truth) <= tolerance, 0.0, np.sign(truth))
        pred_direction = np.where(np.abs(pred) <= tolerance, 0.0, np.sign(pred))
        metrics[f"rmse_{label}"] = float(np.sqrt(np.mean((pred - truth) ** 2)))
        metrics[f"direction_accuracy_{label}"] = float(np.mean(direction == pred_direction))
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / ("action_effect_dual_reference_v4_smoke.npz" if smoke else "action_effect_dual_reference_v4.npz")
    np.savez(
        model_path,
        reference_weights=ref_weights,
        reference_feature_mean=ref_mean,
        reference_feature_scale=ref_scale,
        residual_weights=res_weights,
        residual_feature_mean=res_mean,
        residual_feature_scale=res_scale,
        reference_labels=np.asarray(REFERENCE_LABELS),
        residual_labels=np.asarray(RESIDUAL_LABELS),
        context_feature_names=np.asarray(CONTEXT_FEATURE_NAMES),
        action_feature_names=np.asarray(ACTION_FEATURE_NAMES),
        reference_conformal=np.asarray(ref_conformal),
        residual_conformal=np.asarray(res_conformal),
        quantile=np.asarray([quantile]),
        seeds=np.asarray(seed_values),
        contract_version=np.asarray(["project6_v4_causal_dual_reference_v2"]),
    )
    metrics_path = write_csv(model_dir / ("v4_model_smoke_metrics.csv" if smoke else "v4_model_metrics.csv"), [metrics])
    dataset_path = ds_dir / ("v4_dataset_smoke_manifest.csv" if smoke else "v4_dataset_manifest.csv")
    report = write_json(model_dir / ("v4_model_smoke_report.json" if smoke else "v4_model_report.json"), {
        "status": "pass",
        "sample_count": len(used),
        "train_row_count": int(train_mask.sum()),
        "validation_row_count": int(eval_mask.sum()),
        "ensemble_size": len(seed_values),
        "reference_labels": list(REFERENCE_LABELS),
        "residual_labels": list(RESIDUAL_LABELS),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "copied_from_previous_version": False,
        "online_future_hydraulic_reference_forbidden": True,
        "validation_metrics": metrics,
        "created_at": utc_now(),
    })
    return 0, {"model": model_path, "metrics": metrics_path, "report": report}


def evaluate_v4_model_gate(config: str | Path, *, smoke: bool = False) -> tuple[int, dict[str, Path]]:
    root = _output_root(config)
    model_dir = root / "action_effect_models_v4"
    report_path = model_dir / ("v4_model_smoke_report.json" if smoke else "v4_model_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    cfg = _load_yaml(config)
    gate_cfg = (((cfg.get("v4", {}) or {}).get("model_gate", {}) or {}))
    failures: list[str] = []
    if report.get("status") != "pass":
        failures.append("v4_model_not_trained")
    if report.get("copied_from_previous_version") is not False:
        failures.append("v4_model_was_copied")
    if report.get("online_future_hydraulic_reference_forbidden") is not True:
        failures.append("online_future_hydraulic_reference_contract_missing")
    metrics = dict(report.get("validation_metrics", {}) or {})
    residual_thresholds = dict(gate_cfg.get("residual_direction_accuracy_min", {}) or {})
    thresholds = {
        "delta_PFV_H120_vs_no_control": float(gate_cfg.get("min_pfv_direction_accuracy", residual_thresholds.get("PFV", 0.70))),
        "delta_PFV_H120_vs_passive": float(gate_cfg.get("min_pfv_direction_accuracy", residual_thresholds.get("PFV", 0.70))),
        "delta_TFV_H120_vs_internal": float(gate_cfg.get("min_tfv_direction_accuracy", residual_thresholds.get("TFV", 0.70))),
        "delta_peak_H120_vs_internal": float(gate_cfg.get("min_peak_direction_accuracy", residual_thresholds.get("peak", 0.80))),
        "delta_PFV_full_vs_no_control": float(gate_cfg.get("min_pfv_direction_accuracy", residual_thresholds.get("PFV", 0.70))),
        "delta_PFV_full_vs_passive": float(gate_cfg.get("min_pfv_direction_accuracy", residual_thresholds.get("PFV", 0.70))),
    }
    if not smoke:
        for label, threshold in thresholds.items():
            value = float(metrics.get(f"direction_accuracy_{label}", -1.0))
            if value < threshold:
                failures.append(f"direction_accuracy_below_gate:{label}:{value:.6f}<{threshold:.6f}")
    status = "pass" if not failures else "failed_gate"
    path = write_json(model_dir / ("v4_model_smoke_gate.json" if smoke else "v4_model_gate.json"), {
        "status": status, "failures": failures, "thresholds": thresholds,
        "validation_metrics": metrics, "created_at": utc_now(),
    })
    return (0 if status == "pass" else 5), {"gate": path}


def _sign_counts(values: np.ndarray, near_zero_abs: float) -> dict[str, int]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    near = np.abs(arr) <= near_zero_abs
    return {
        "positive": int(np.sum((arr > 0.0) & ~near)),
        "negative": int(np.sum((arr < 0.0) & ~near)),
        "near_zero": int(np.sum(near)),
        "total": int(arr.size),
    }


def _phase_bucket(text: str) -> str:
    token = str(text or "").strip().lower()
    if not token:
        return "unknown"
    if "near_peak" in token or "near-peak" in token or "pre_peak" in token or "pre-peak" in token:
        return "near_peak"
    if "peak" in token:
        return "peak"
    if "rise" in token or "rising" in token:
        return "rising"
    if "recession" in token:
        return "recession"
    if any(t in token for t in ("early_recovery", "late_recovery", "recovery", "recovered", "release")):
        return "recovery"
    return token


def diagnose_v4_full_event_pfv_gate(
    config: str | Path, *, smoke: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Section 一 diagnosis of why the full-event PFV direction heads fail.

    Reads the current V4 dataset manifest, the trained model, and the model
    gate; reconstructs the same event-isolated validation split used at
    training time; and reports per-event/phase/action label balance and the
    real-vs-predicted direction of the two full-event PFV residual heads. It
    never retrains and never touches the base manifest.
    """
    out_root = _output_root(config)
    ds_dir = out_root / "action_effect_dataset_v4"
    model_dir = out_root / "action_effect_models_v4"
    diag_dir = out_root / "diagnostics"
    manifest = ds_dir / ("v4_dataset_smoke_manifest.csv" if smoke else "v4_dataset_manifest.csv")
    rejected = ds_dir / ("v4_dataset_smoke_rejected.csv" if smoke else "v4_dataset_rejected.csv")
    model_path = model_dir / ("action_effect_dual_reference_v4_smoke.npz" if smoke else "action_effect_dual_reference_v4.npz")
    gate_path = model_dir / ("v4_model_smoke_gate.json" if smoke else "v4_model_gate.json")

    rows = _read_csv(manifest)
    reject_rows = _read_csv(rejected)
    x_ctx, x_res, y_ref, y_res, used = _arrays(rows)
    if not used or not model_path.exists():
        report = write_json(diag_dir / "v4_full_event_pfv_gate_diagnosis.json", {
            "status": "blocked",
            "reason": "missing_manifest_rows_or_trained_model",
            "manifest": str(manifest),
            "manifest_exists": manifest.exists(),
            "model": str(model_path),
            "model_exists": model_path.exists(),
            "materialized_rows": len(used),
            "created_at": utc_now(),
        })
        return 3, {"diagnosis": report}

    train_mask, val_mask = _split_by_event(used)
    eval_mask = val_mask if val_mask.any() else train_mask
    with np.load(model_path, allow_pickle=True) as data:
        res_pred = _ensemble_predict(
            data["residual_weights"], data["residual_feature_mean"],
            data["residual_feature_scale"], x_res[eval_mask],
        ).mean(axis=0)

    near_zero_abs = 1.0e-9  # matches the gate's direction definition
    full_idx = {label: RESIDUAL_LABELS.index(label) for label in (
        "delta_PFV_full_vs_no_control", "delta_PFV_full_vs_passive",
    )}
    h120_idx = {label: RESIDUAL_LABELS.index(label) for label in (
        "delta_PFV_H120_vs_no_control", "delta_PFV_H120_vs_passive",
    )}

    used_events = [str(r.get("event_id", "")) for r in used]
    train_events = {e for e, m in zip(used_events, train_mask) if m}
    val_events = {e for e, m in zip(used_events, val_mask) if m}
    eval_rows = [r for r, m in zip(used, eval_mask) if m]
    y_res_eval = y_res[eval_mask]

    def _dir(v: np.ndarray) -> np.ndarray:
        return np.where(np.abs(v) <= near_zero_abs, 0.0, np.sign(v))

    # ---- Per-event metrics (worst-event families) --------------------------
    event_metric_records: list[dict[str, Any]] = []
    per_event_eval: dict[str, list[int]] = {}
    for i, r in enumerate(eval_rows):
        per_event_eval.setdefault(str(r.get("event_id", "")), []).append(i)
    for event, idxs in sorted(per_event_eval.items()):
        rec: dict[str, Any] = {"event_id": event, "eval_sample_count": len(idxs)}
        arr_idx = np.asarray(idxs)
        for label, col in {**full_idx, **h120_idx}.items():
            truth = _dir(y_res_eval[arr_idx, col])
            pred = _dir(res_pred[arr_idx, col])
            rec[f"direction_accuracy_{label}"] = float(np.mean(truth == pred)) if len(idxs) else 0.0
        for label, col in full_idx.items():
            rec[f"near_zero_fraction_{label}"] = float(np.mean(np.abs(y_res_eval[arr_idx, col]) <= near_zero_abs))
        event_metric_records.append(rec)
    event_metric_records.sort(key=lambda d: d.get("direction_accuracy_delta_PFV_full_vs_no_control", 0.0))
    write_csv(diag_dir / "v4_full_event_pfv_event_metrics.csv", event_metric_records)

    # ---- Label balance by phase and action magnitude -----------------------
    balance_records: list[dict[str, Any]] = []
    def _group_balance(group_key: str, getter) -> None:
        buckets: dict[str, list[int]] = {}
        for i, r in enumerate(used):
            buckets.setdefault(str(getter(r)), []).append(i)
        for name, idxs in sorted(buckets.items()):
            rec = {"dimension": group_key, "bucket": name, "sample_count": len(idxs)}
            arr_idx = np.asarray(idxs)
            for label, col in full_idx.items():
                sc = _sign_counts(y_res[arr_idx, col], near_zero_abs)
                rec[f"{label}_pos"] = sc["positive"]
                rec[f"{label}_neg"] = sc["negative"]
                rec[f"{label}_near_zero"] = sc["near_zero"]
            balance_records.append(rec)
    _group_balance("phase", lambda r: _phase_bucket(r.get("phase", "")))
    _group_balance("action_magnitude", lambda r: r.get("action_magnitude", ""))
    _group_balance("k_value", lambda r: r.get("k_value", ""))
    _group_balance("candidate_branch", lambda r: r.get("v4_candidate_branch", r.get("candidate_selected_branch_id", "")))
    write_csv(diag_dir / "v4_full_event_pfv_label_balance.csv", balance_records)

    # ---- Failure rows (validation full-event direction predicted wrong) ----
    failure_records: list[dict[str, Any]] = []
    for i, r in enumerate(eval_rows):
        for label, col in full_idx.items():
            truth = _dir(np.asarray([y_res_eval[i, col]]))[0]
            pred = _dir(np.asarray([res_pred[i, col]]))[0]
            if truth != pred:
                failure_records.append({
                    "event_id": r.get("event_id", ""),
                    "checkpoint_id": r.get("checkpoint_id", ""),
                    "candidate_id": r.get("candidate_id", ""),
                    "phase": r.get("phase", ""),
                    "k_value": r.get("k_value", ""),
                    "action_magnitude": r.get("action_magnitude", ""),
                    "candidate_branch": r.get("v4_candidate_branch", ""),
                    "head": label,
                    "true_delta": float(y_res_eval[i, col]),
                    "pred_delta": float(res_pred[i, col]),
                    "true_direction": float(truth),
                    "pred_direction": float(pred),
                })
    write_csv(diag_dir / "v4_full_event_pfv_failure_rows.csv", failure_records)

    # ---- Feature coverage / variance (shows local features lack signal) ----
    coverage_records: list[dict[str, Any]] = []
    for j, name in enumerate(CONTEXT_FEATURE_NAMES):
        col = x_ctx[:, j]
        coverage_records.append({
            "feature_group": "context", "feature": name,
            "mean": float(np.mean(col)), "std": float(np.std(col)),
            "min": float(np.min(col)), "max": float(np.max(col)),
            "nonzero_fraction": float(np.mean(np.abs(col) > 1.0e-12)),
        })
    action_offset = len(CONTEXT_FEATURE_NAMES)
    for j, name in enumerate(ACTION_FEATURE_NAMES):
        col = x_res[:, action_offset + j]
        coverage_records.append({
            "feature_group": "action", "feature": name,
            "mean": float(np.mean(col)), "std": float(np.std(col)),
            "min": float(np.min(col)), "max": float(np.max(col)),
            "nonzero_fraction": float(np.mean(np.abs(col) > 1.0e-12)),
        })
    write_csv(diag_dir / "v4_full_event_pfv_feature_coverage.csv", coverage_records)

    # ---- Pairing / censoring / leakage checks ------------------------------
    checkpoint_branches: dict[str, set[str]] = {}
    for r in used:
        checkpoint_branches.setdefault(str(r.get("checkpoint_id", "")), set()).add(str(r.get("v4_candidate_branch", "")))
    censored = sum(_truth(r.get("recovery_censored")) or _truth(r.get("censored_mask")) for r in used)
    full_recovery_incomplete = sum(
        1 for r in used
        if str(r.get("full_recovery_label_status", "")).strip().lower() not in {"", "ok", "complete", "completed", "pass", "labeled"}
    )
    leaked_events = sorted(train_events & val_events)

    reject_reason_counts: dict[str, int] = {}
    for r in reject_rows:
        reason = str(r.get("reason", "unknown"))
        reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + 1

    def _dist(getter) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in used:
            out[str(getter(r))] = out.get(str(getter(r)), 0) + 1
        return dict(sorted(out.items()))

    val_dir_full = {
        label: float(np.mean(_dir(y_res_eval[:, col]) == _dir(res_pred[:, col])))
        for label, col in full_idx.items()
    }

    diagnosis = {
        "status": "diagnosed",
        "manifest": str(manifest),
        "model": str(model_path),
        "model_gate": str(gate_path),
        "near_zero_abs": near_zero_abs,
        # 1-2
        "full_event_train_sample_count": int(train_mask.sum()),
        "full_event_val_sample_count": int(val_mask.sum()),
        "eval_sample_count": int(eval_mask.sum()),
        "unique_event_count_total": len(set(used_events)),
        "unique_event_count_train": len(train_events),
        "unique_event_count_val": len(val_events),
        # 3
        "per_event_sample_count": _dist(lambda r: r.get("event_id", "")),
        # 4
        "phase_sample_counts": _dist(lambda r: _phase_bucket(r.get("phase", ""))),
        # 5-6
        "delta_PFV_full_vs_no_control_sign_counts": _sign_counts(y_res[:, full_idx["delta_PFV_full_vs_no_control"]], near_zero_abs),
        "delta_PFV_full_vs_passive_sign_counts": _sign_counts(y_res[:, full_idx["delta_PFV_full_vs_passive"]], near_zero_abs),
        # 7
        "k_value_distribution": _dist(lambda r: r.get("k_value", "")),
        # 8
        "action_magnitude_distribution": _dist(lambda r: r.get("action_magnitude", "")),
        # 9
        "candidate_branch_distribution": _dist(lambda r: r.get("v4_candidate_branch", "")),
        "selected_fallback_distribution": _dist(lambda r: r.get("selected_fallback_branch_id", "")),
        "anchor_type_distribution": _dist(lambda r: r.get("anchor_type", "")),
        # 10
        "reject_reason_counts": reject_reason_counts,
        # 11
        "checkpoints_with_incomplete_branch_pairing": int(sum(1 for b in checkpoint_branches.values() if len(b) < 2)),
        "checkpoint_count": len(checkpoint_branches),
        # 12
        "full_recovery_incomplete_count": int(full_recovery_incomplete),
        "recovery_censored_count": int(censored),
        # 13-14
        "train_val_event_isolated": len(leaked_events) == 0,
        "events_leaked_into_both": leaked_events,
        # 15
        "worst_event_families": event_metric_records[:8],
        # validation-set direction accuracy (reproduces gate)
        "validation_direction_accuracy_full_heads": val_dir_full,
        "created_at": utc_now(),
    }
    report = write_json(diag_dir / "v4_full_event_pfv_gate_diagnosis.json", diagnosis)
    return 0, {
        "diagnosis": report,
        "event_metrics": diag_dir / "v4_full_event_pfv_event_metrics.csv",
        "label_balance": diag_dir / "v4_full_event_pfv_label_balance.csv",
        "failure_rows": diag_dir / "v4_full_event_pfv_failure_rows.csv",
        "feature_coverage": diag_dir / "v4_full_event_pfv_feature_coverage.csv",
    }


def audit_v4_readiness(config: str | Path) -> tuple[int, dict[str, Path]]:
    root = _output_root(config)
    model_report = root / "action_effect_models_v4/v4_model_report.json"
    model_gate = root / "action_effect_models_v4/v4_model_gate.json"
    dataset_audit = root / "action_effect_dataset_v4/v4_dataset_audit.json"
    cfg = _load_yaml(config)
    def _status(path: Path) -> str:
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("status", ""))
        except Exception:
            return ""
    checks = {
        "network_exists": (_project_root(config) / (cfg.get("project", {}) or {}).get("inp", "data/wuhan_v8_storage_retrofit.inp")).exists(),
        "dataset_audit_pass": _status(dataset_audit) == "pass",
        "model_report_pass": _status(model_report) == "pass",
        "model_gate_pass": _status(model_gate) == "pass",
        "model_not_copied": bool(model_report.exists() and json.loads(model_report.read_text(encoding="utf-8")).get("copied_from_previous_version") is False),
        "online_future_hydraulics_forbidden": bool(model_report.exists() and json.loads(model_report.read_text(encoding="utf-8")).get("online_future_hydraulic_reference_forbidden") is True),
        "dual_reference_config_present": bool((cfg.get("v4", {}) or {}).get("dual_reference")),
        "runtime_limits_present": bool((cfg.get("runtime_limits", {}) or {})),
    }
    status = "pass" if all(checks.values()) else "blocked"
    report = write_json(root / "audit/v4_readiness.json", {"status": status, "checks": checks, "created_at": utc_now()})
    return (0 if status == "pass" else 3), {"report": report}
