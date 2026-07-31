from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.io.safe_paths import atomic_write_json, ensure_within_budget, mkdir_parent


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    return atomic_write_json(path, payload)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    rows = list(rows)
    path = mkdir_parent(ensure_within_budget(path))
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def parse_swmm_time_options(inp_path: Path) -> dict[str, Any]:
    options: dict[str, str] = {}
    section = ""
    if not inp_path.exists():
        return {"status": "missing_inp", "inp_path": str(inp_path)}
    for raw in inp_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").upper()
            continue
        if section == "OPTIONS":
            parts = line.split()
            if len(parts) >= 2:
                options[parts[0].upper()] = " ".join(parts[1:])
    return {
        "status": "parsed",
        "routing_step": options.get("ROUTING_STEP"),
        "report_step": options.get("REPORT_STEP"),
        "wet_step": options.get("WET_STEP"),
        "rule_step": options.get("RULE_STEP"),
        "variable_step": options.get("VARIABLE_STEP"),
        "dynamic_wave_internal_step_assumed_equal_visible_step": False,
    }


def checkpoint_targets(duration_min: int, simulation_duration_min: int, step_min: float = 5.0) -> list[dict[str, Any]]:
    candidates = [
        ("rising", max(60.0, float(duration_min) * 0.5)),
        ("near_peak", float(duration_min)),
        ("recession", float(duration_min) + 60.0),
    ]
    out: list[dict[str, Any]] = []
    for phase, elapsed in candidates:
        snapped = round(elapsed / float(step_min)) * float(step_min)
        if snapped >= 60.0 and snapped + 120.0 <= float(simulation_duration_min):
            out.append({"phase": phase, "elapsed_min": snapped})
    return out


def analyze_recovery(
    detail: pd.DataFrame,
    *,
    event_id: str,
    policy_id: str,
    trajectory_id: str,
    duration_min: int,
    minimum_tail_min: int = 180,
    max_tail_min: int = 720,
    priority_nodes: list[str] | None = None,
) -> dict[str, Any]:
    priority_nodes = priority_nodes or []
    elapsed = pd.to_numeric(detail.get("elapsed_min", pd.Series(dtype=float)), errors="coerce")
    max_elapsed = float(elapsed.max()) if not elapsed.empty and elapsed.notna().any() else 0.0
    flood_cols = [col for col in detail.columns if str(col).startswith("flood:")]
    priority_flood_cols = [f"flood:{node}" for node in priority_nodes if f"flood:{node}" in detail.columns]
    flood_sum = detail[flood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1) if flood_cols else pd.Series([0.0] * len(detail))
    priority_flood_sum = detail[priority_flood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1) if priority_flood_cols else pd.Series([0.0] * len(detail))
    last_flood_time = float(elapsed[flood_sum.gt(0.0)].max()) if flood_sum.gt(0.0).any() else None
    last_priority_flood_time = float(elapsed[priority_flood_sum.gt(0.0)].max()) if priority_flood_sum.gt(0.0).any() else None
    actual_tail = max(0.0, max_elapsed - float(duration_min))
    restored_rows = detail[(elapsed >= float(duration_min) + float(minimum_tail_min)) & flood_sum.eq(0.0) & priority_flood_sum.eq(0.0)].copy()
    recovery_criteria_met = False
    recovery_start = None
    if not restored_rows.empty:
        values = pd.to_numeric(restored_rows["elapsed_min"], errors="coerce").dropna().tolist()
        run_start = None
        previous = None
        for value in values:
            if run_start is None or previous is None or value - previous > 5.0001:
                run_start = value
            previous = value
            if value - float(run_start) >= 60.0:
                recovery_criteria_met = True
                recovery_start = float(run_start)
                break
    censored = actual_tail >= float(max_tail_min) and not recovery_criteria_met
    return {
        "trajectory_id": trajectory_id,
        "event_id": event_id,
        "policy_id": policy_id,
        "rain_start_min": 0,
        "rain_end_min": int(duration_min),
        "simulation_start_min": 0,
        "simulation_end_min": max_elapsed,
        "rain_duration_min": int(duration_min),
        "minimum_tail_min": int(minimum_tail_min),
        "actual_tail_min": actual_tail,
        "last_flood_time_min": "" if last_flood_time is None else last_flood_time,
        "last_priority_flood_time_min": "" if last_priority_flood_time is None else last_priority_flood_time,
        "storage_recovery_time_min": "",
        "storage_recovery_status": "not_materialized_in_baseline_detail",
        "tail_termination_reason": "recovery_criteria_met" if recovery_criteria_met else ("max_tail_censored" if censored else "fixed_tail_without_60min_recovery"),
        "recovery_status": "recovered" if recovery_criteria_met else "censored" if censored else "not_recovered",
        "recovery_censored": censored,
        "recovery_criteria_met": recovery_criteria_met,
        "continuous_60min_recovery_start_min": "" if recovery_start is None else recovery_start,
    }


def controller_memory_payload(
    *,
    trajectory_id: str,
    event_id: str,
    policy_id: str,
    elapsed_min: float,
    row: dict[str, Any],
    actuator_ids: list[str],
    phase: str,
) -> dict[str, Any]:
    settings = {}
    native_target: dict[str, Any] = {}
    anchor: dict[str, Any] = {}
    requested_action: dict[str, Any] = {}
    projected_action: dict[str, Any] = {}
    target_action: dict[str, Any] = {}
    actual_action: dict[str, Any] = {}
    override_ttl: dict[str, int] = {}
    binary_pump_states: dict[str, Any] = {}
    for aid in actuator_ids:
        target = row.get(f"a:{aid}")
        actual = row.get(f"setting:{aid}", target)
        native_target[aid] = target
        anchor[aid] = target
        requested_action[aid] = target
        projected_action[aid] = target
        target_action[aid] = target
        actual_action[aid] = actual
        override_ttl[aid] = 0
        if aid in {"ADD301.2", "ADD301.3"}:
            binary_pump_states[aid] = actual
        settings[aid] = {
            "anchor_setting": target,
            "native_target_setting": target,
            "requested_setting": target,
            "projected_setting": target,
            "target_setting": target,
            "actual_current_setting": actual,
            "override_ttl": 0,
            "release": True,
            "minimum_on_remaining": None,
            "minimum_off_remaining": None,
            "dwell_remaining": None,
        }
    return {
        "schema_version": "project6_v3_controller_memory_v1",
        "trajectory_id": trajectory_id,
        "event_id": event_id,
        "policy_id": policy_id,
        "checkpoint_elapsed_min": elapsed_min,
        "phase": phase,
        "selected_fallback": policy_id if policy_id in {"internal_rules", "executable_passive"} else "",
        "fallback_mode": policy_id if policy_id in {"internal_rules", "executable_passive"} else "none",
        "forecast_issue": None,
        "forecast_issue_id": None,
        "last_measurement_time": row.get("datetime"),
        "last_decision_time": row.get("datetime"),
        "decision_time": row.get("datetime"),
        "next_rtc_decision_time": None,
        "continuation_policy": "hold_same_baseline_policy",
        "continuation_policy_id": "hold_same_baseline_policy",
        "gat_state_hash": None,
        "data_quality_state": "baseline_truth_observed_current_and_past_only",
        "facility_order": list(actuator_ids),
        "facility_order_hash": hashlib.sha256(json.dumps(list(actuator_ids), ensure_ascii=False).encode("utf-8")).hexdigest(),
        "native_target": native_target,
        "anchor": anchor,
        "requested_action": requested_action,
        "projected_action": projected_action,
        "target_action": target_action,
        "actual_action": actual_action,
        "override_ttl": override_ttl,
        "release": {aid: True for aid in actuator_ids},
        "override_mask": {aid: False for aid in actuator_ids},
        "pump_on_duration": {aid: None for aid in actuator_ids},
        "pump_off_duration": {aid: None for aid in actuator_ids},
        "minimum_on_remaining": {aid: None for aid in actuator_ids},
        "minimum_off_remaining": {aid: None for aid in actuator_ids},
        "dwell_remaining": {aid: None for aid in actuator_ids},
        "add350_speed_actual": actual_action.get("add350.1"),
        "add350_speed_target": target_action.get("add350.1"),
        "binary_pump_states": binary_pump_states,
        "facility_settings": settings,
    }


def try_save_hotstart(sim: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_fn = getattr(sim, "save_hotstart", None) or getattr(sim, "save_hot_start", None)
    if save_fn is None:
        return {"status": "not_supported_by_pyswmm_object", "path": "", "sha256": None}
    try:
        save_fn(str(path))
    except Exception as exc:  # pragma: no cover - depends on PySWMM runtime
        return {"status": "failed", "path": str(path), "error": str(exc), "sha256": None}
    return {
        "status": "saved" if path.exists() else "missing_after_save",
        "path": str(path) if path.exists() else "",
        "sha256": sha256_file(path),
    }


def try_use_hotstart(sim: Any, path: Path) -> dict[str, Any]:
    """Load a PySWMM hot-start file through the available local API."""
    if not path.exists() or not path.is_file():
        return {"status": "missing", "path": str(path), "sha256": None}
    load_fn = (
        getattr(sim, "use_hotstart", None)
        or getattr(sim, "use_hot_start", None)
        or getattr(sim, "load_hotstart", None)
        or getattr(sim, "load_hot_start", None)
    )
    if load_fn is None:
        return {"status": "not_supported_by_pyswmm_object", "path": str(path), "sha256": sha256_file(path)}
    try:
        load_fn(str(path))
    except Exception as exc:  # pragma: no cover - depends on PySWMM runtime
        return {"status": "failed", "path": str(path), "error": str(exc), "sha256": sha256_file(path)}
    return {"status": "loaded", "path": str(path), "sha256": sha256_file(path)}
