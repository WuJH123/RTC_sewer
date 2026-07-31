from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.contracts.prompt3a import OUT_ROOT, read_csv, sha256_file, write_csv, write_json
from sewerrtc.simulation.action_policies import GenericActionPolicy, PolicyContext, phase_from_time
from sewerrtc.simulation.controller_state import REQUIRED_CONTROLLER_MEMORY_FIELDS, validate_controller_memory
from sewerrtc.simulation.kpi_metrics import compute_kpis
from sewerrtc.simulation.runtime_contracts import parse_swmm_time_options, try_use_hotstart


CLONE_METRICS = [
    "node_depth",
    "node_head",
    "link_flow",
    "storage_volume",
    "actual_setting",
    "PFV",
    "TFV",
    "peak",
    "recovery_end_time",
]

REAL_SWMM_CLONE_REQUIRED_REASON = "requires_real_swmm_hotstart_equivalence_run"
REAL_SWMM_CLONE_DETAIL = "SWMM hot-start and controller memory restore comparison is required before same-state PASS"

BINARY_PUMPS = {"ADD301.2", "ADD301.3"}
VARIABLE_SPEED_PUMPS = {"add350.1"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {"_json_error": "decode_failed"}


def _as_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _load_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _merge_checkpoint_sources(
    checkpoint_catalog: Path,
    baseline_checkpoint_audit: Path,
) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for source in (checkpoint_catalog, baseline_checkpoint_audit):
        for row in read_csv(source):
            checkpoint_id = row.get("checkpoint_id", "")
            if not checkpoint_id:
                continue
            rows.setdefault(checkpoint_id, {})
            rows[checkpoint_id].update({k: v for k, v in row.items() if v not in (None, "")})
    return rows


def prepare_state_clone_checkpoints(
    checkpoint_catalog: str | Path = OUT_ROOT / "checkpoint_catalog" / "checkpoint_catalog.csv",
    out_dir: str | Path = OUT_ROOT / "state_clone",
    baseline_checkpoint_audit: str | Path = OUT_ROOT / "baseline_trajectories" / "baseline_checkpoint_audit.csv",
) -> tuple[int, dict[str, Path]]:
    checkpoint_catalog = Path(checkpoint_catalog)
    baseline_checkpoint_audit = Path(baseline_checkpoint_audit)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readiness = out_dir / "state_clone_checkpoint_readiness.csv"
    report_path = out_dir / "state_clone_checkpoint_readiness_report.json"
    if not checkpoint_catalog.exists() or not baseline_checkpoint_audit.exists():
        report = {
            "status": "blocked",
            "reason": "checkpoint_catalog_or_baseline_checkpoint_audit_missing",
            "checkpoint_catalog": str(checkpoint_catalog),
            "baseline_checkpoint_audit": str(baseline_checkpoint_audit),
            "completion_marker": None,
        }
        _atomic_json(report_path, report)
        return 3, {"readiness": readiness, "report": report_path}

    merged = _merge_checkpoint_sources(checkpoint_catalog, baseline_checkpoint_audit)
    rows: list[dict[str, Any]] = []
    for row in merged.values():
        hotstart = Path(row.get("hotstart_path", ""))
        memory = Path(row.get("controller_memory_path", ""))
        detail = Path(row.get("detail_file", ""))
        network = Path(row.get("network_path", row.get("state_clone_source", "")))
        ready = (
            hotstart.exists()
            and memory.exists()
            and detail.exists()
            and network.exists()
            and str(row.get("eligible_for_state_clone", row.get("runtime_clone_eligible", ""))).lower() == "true"
        )
        rows.append(
            {
                "checkpoint_id": row.get("checkpoint_id", ""),
                "trajectory_id": row.get("trajectory_id", ""),
                "event_id": row.get("event_id", ""),
                "policy_id": row.get("policy_id", ""),
                "phase": row.get("phase", ""),
                "checkpoint_elapsed_min": row.get("checkpoint_elapsed_min", row.get("event_time_min", "")),
                "hotstart_path": str(hotstart) if hotstart.exists() else "",
                "hotstart_sha256": sha256_file(hotstart) if hotstart.exists() else "",
                "controller_memory_path": str(memory) if memory.exists() else "",
                "controller_memory_sha256": sha256_file(memory) if memory.exists() else "",
                "detail_file": str(detail) if detail.exists() else "",
                "detail_file_sha256": sha256_file(detail) if detail.exists() else "",
                "network_path": str(network) if network.exists() else "",
                "network_sha256": sha256_file(network) if network.exists() else "",
                "rainfall_state_hash": row.get("rainfall_state_hash", row.get("rainfall_sha256", "")),
                "history_60min_available": row.get("history_60min_available", ""),
                "future_120min_available": row.get("future_120min_available", ""),
                "ready_for_clone": str(ready).lower(),
                "exclusion_reason": "" if ready else "missing_hotstart_controller_memory_detail_network_or_temporal_support",
            }
        )
    ready_count = sum(1 for row in rows if row["ready_for_clone"] == "true")
    write_csv(readiness, rows)
    report = {
        "status": "completed" if ready_count else "blocked",
        "created_at": _utc_now(),
        "checkpoint_catalog": str(checkpoint_catalog),
        "checkpoint_catalog_sha256": sha256_file(checkpoint_catalog),
        "baseline_checkpoint_audit": str(baseline_checkpoint_audit),
        "baseline_checkpoint_audit_sha256": sha256_file(baseline_checkpoint_audit),
        "ready_checkpoint_count": ready_count,
        "completion_marker": None if ready_count == 0 else "allowed_after_marker_write",
    }
    _atomic_json(report_path, report)
    return (0 if ready_count else 3), {"readiness": readiness, "report": report_path}


def _select_checkpoints(rows: list[dict[str, str]], mode: str, max_checkpoints: int) -> list[dict[str, str]]:
    ready = [row for row in rows if str(row.get("ready_for_clone", "")).lower() == "true"]
    if mode == "smoke":
        chosen: list[dict[str, str]] = []
        for phase in ("rising", "near_peak", "recession"):
            item = next((row for row in ready if row.get("phase") == phase and row not in chosen), None)
            if item is not None:
                chosen.append(item)
        for row in ready:
            if len(chosen) >= max(1, max_checkpoints or 3):
                break
            if row not in chosen:
                chosen.append(row)
        return chosen[: max(1, max_checkpoints or 3)]
    if max_checkpoints > 0:
        return ready[:max_checkpoints]
    return ready


def _memory_audit_row(row: dict[str, str], memory: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    missing = validate_controller_memory(memory)
    facility_order = list(memory.get("facility_order") or memory.get("facility_settings", {}).keys())
    actual = memory.get("actual_action") or {}
    target = memory.get("target_action") or {}
    binary_errors: list[str] = []
    for pump_id in BINARY_PUMPS:
        value = actual.get(pump_id, memory.get("binary_pump_states", {}).get(pump_id))
        if value in (None, ""):
            binary_errors.append(f"{pump_id}:missing")
            continue
        number = _as_float(value)
        if number not in (0.0, 1.0):
            binary_errors.append(f"{pump_id}:{value}")
    add350_mode = "continuous" if "add350.1" in facility_order else "missing"
    reasons = list(missing)
    if len(facility_order) != 36:
        reasons.append(f"facility_order_count_{len(facility_order)}_not_36")
    if binary_errors:
        reasons.append("binary_pump_state_invalid")
    audit = {
        "checkpoint_id": row.get("checkpoint_id", ""),
        "controller_memory_path": row.get("controller_memory_path", ""),
        "required_fields_present": str(not missing).lower(),
        "missing_required_fields": ";".join(missing),
        "facility_order_count": len(facility_order),
        "facility_order_hash": memory.get("facility_order_hash", ""),
        "target_setting_count": len(target) if isinstance(target, dict) else 0,
        "actual_setting_count": len(actual) if isinstance(actual, dict) else 0,
        "binary_pump_status": "pass" if not binary_errors else "fail",
        "binary_pump_errors": ";".join(binary_errors),
        "add350_semantics": add350_mode,
        "ttl_dwell_present": str("override_ttl" in memory and "dwell_remaining" in memory).lower(),
        "selected_fallback": memory.get("selected_fallback", ""),
        "status": "pass" if not reasons else "blocked",
    }
    return audit, reasons


def _reference_schema_reasons(detail: pd.DataFrame, memory: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    required_base = {"elapsed_min", "datetime", "event_id", "policy_id", "phase"}
    missing_base = sorted(required_base - set(detail.columns))
    if missing_base:
        reasons.append("reference_detail_missing_base_columns:" + ",".join(missing_base))
    h_cols = [col for col in detail.columns if col.startswith("h:")]
    head_cols = [col for col in detail.columns if col.startswith("head:")]
    flood_cols = [col for col in detail.columns if col.startswith("flood:")]
    if len(h_cols) < 900:
        reasons.append(f"reference_detail_node_depth_count_{len(h_cols)}_lt_900")
    if len(head_cols) < len(h_cols):
        reasons.append("reference_detail_missing_node_head_columns")
    if not flood_cols:
        reasons.append("reference_detail_missing_flood_columns")
    facility_order = list(memory.get("facility_order") or memory.get("facility_settings", {}).keys())
    for prefix in ("a:", "setting:", "flow:"):
        missing = [aid for aid in facility_order if f"{prefix}{aid}" not in detail.columns]
        if missing:
            reasons.append(f"reference_detail_missing_{prefix.rstrip(':')}_columns:{len(missing)}")
    storage_cols = [col for col in detail.columns if col.startswith("storage_volume:")]
    if not storage_cols:
        reasons.append("reference_detail_missing_storage_volume_columns")
    return reasons


def _patch_option_line(line: str, key: str, value: str) -> str:
    prefix = line[: len(line) - len(line.lstrip())]
    return f"{prefix}{key:<24} {value}"


def _infer_visible_step_minutes(reference: pd.DataFrame) -> float:
    if "datetime" in reference:
        times = pd.to_datetime(reference["datetime"], errors="coerce").dropna()
        if len(times) >= 2:
            diffs = times.sort_values().diff().dropna().dt.total_seconds() / 60.0
            positive = diffs[diffs.gt(0.0)]
            if not positive.empty:
                return float(positive.median())
    if "elapsed_min" in reference:
        elapsed = pd.to_numeric(reference["elapsed_min"], errors="coerce").dropna().sort_values()
        if len(elapsed) >= 2:
            diffs = elapsed.diff().dropna()
            positive = diffs[diffs.gt(0.0)]
            if not positive.empty:
                return float(positive.median())
    return 5.0


def _write_clone_inp(source: Path, target: Path, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[str] = []
    section = ""
    replacements = {
        "START_DATE": start_time.strftime("%m/%d/%Y"),
        "START_TIME": start_time.strftime("%H:%M:%S"),
        "REPORT_START_DATE": start_time.strftime("%m/%d/%Y"),
        "REPORT_START_TIME": start_time.strftime("%H:%M:%S"),
        "END_DATE": end_time.strftime("%m/%d/%Y"),
        "END_TIME": end_time.strftime("%H:%M:%S"),
    }
    seen: set[str] = set()
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if section == "OPTIONS":
                for key, value in replacements.items():
                    if key not in seen:
                        out.append(_patch_option_line("", key, value))
            section = stripped.strip("[]").upper()
            out.append(raw)
            continue
        if section == "OPTIONS" and stripped and not stripped.startswith(";"):
            key = stripped.split()[0].upper()
            if key in replacements:
                out.append(_patch_option_line(raw, key, replacements[key]))
                seen.add(key)
                continue
        out.append(raw)
    if section == "OPTIONS":
        for key, value in replacements.items():
            if key not in seen:
                out.append(_patch_option_line("", key, value))
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    return target


def _run_restored_continuation(
    row: dict[str, str],
    memory: dict[str, Any],
    reference: pd.DataFrame,
    out_csv: Path,
    work_dir: Path,
    checkpoint_time: pd.Timestamp,
) -> dict[str, Any]:
    from pyswmm import Links, Nodes, RainGages, Simulation  # type: ignore

    inp_path = Path(row["network_path"])
    hotstart_path = Path(row["hotstart_path"])
    checkpoint_elapsed = _as_float(row.get("checkpoint_elapsed_min"), 0.0)
    start_time = checkpoint_time
    visible_step_min = _infer_visible_step_minutes(reference)
    # SWMM's stride iterator stops before END_TIME.  Set END_TIME one visible
    # output step after the last reference row so the restored run materializes
    # the same final comparison timestamp.
    end_time = pd.to_datetime(reference.iloc[-1]["datetime"]) + pd.Timedelta(minutes=visible_step_min)
    clone_inp = _write_clone_inp(inp_path, work_dir / "clone.inp", start_time, end_time)
    facility_order = list(memory.get("facility_order") or memory.get("facility_settings", {}).keys())
    policy_id = row.get("policy_id", "")
    duration_min = int(max(1, float(reference["elapsed_min"].max()) - float(reference["elapsed_min"].min())))
    records: list[dict[str, Any]] = []
    hotstart_status: dict[str, Any] = {}
    with Simulation(str(clone_inp)) as sim:
        hotstart_status = try_use_hotstart(sim, hotstart_path)
        if hotstart_status.get("status") != "loaded":
            raise RuntimeError(f"hotstart_load_failed:{hotstart_status}")
        sim.step_advance(300)
        nodes = Nodes(sim)
        links = Links(sim)
        gages = RainGages(sim)
        node_ids = [str(x) for x in getattr(nodes, "nodeid", [])] if hasattr(nodes, "nodeid") else []
        if not node_ids:
            node_ids = [col.split(":", 1)[1] for col in reference.columns if col.startswith("h:")]
        node_objs = {}
        for nid in node_ids:
            try:
                node_objs[nid] = nodes[nid]
            except Exception:
                pass
        link_objs = {}
        for aid in facility_order:
            try:
                link_objs[aid] = links[aid]
            except Exception:
                pass
        rain_ids = []
        try:
            rain_ids = [str(x) for x in getattr(gages, "raingageid", [])]
        except Exception:
            rain_ids = []
        rain_obj = gages[rain_ids[0]] if rain_ids else None
        policy = GenericActionPolicy(policy_id, pd.DataFrame({"actuator_id": facility_order, "link_type": [""] * len(facility_order)}))
        previous = np.asarray([_as_float(memory.get("actual_action", {}).get(aid), 1.0) for aid in facility_order], dtype=np.float32)
        for aid, value in zip(facility_order, previous):
            if aid in link_objs and policy_id != "internal_rules":
                try:
                    link_objs[aid].target_setting = float(np.clip(value, 0.0, 1.0))
                except Exception:
                    pass
        for _ in sim:
            clone_elapsed = (sim.current_time - sim.start_time).total_seconds() / 60.0
            elapsed_min = checkpoint_elapsed + clone_elapsed
            rain = _as_float(getattr(rain_obj, "rainfall", 0.0), 0.0) if rain_obj is not None else 0.0
            phase = phase_from_time(elapsed_min, duration_min)
            if policy_id not in {"internal_rules", "no_control"}:
                ctx = PolicyContext(elapsed_min, duration_min, rain, phase, previous)
                action = policy.action(ctx)[: len(facility_order)]
                for aid, value in zip(facility_order, action):
                    if aid in link_objs:
                        try:
                            link_objs[aid].target_setting = float(np.clip(value, 0.0, 1.0))
                        except Exception:
                            pass
                previous = np.asarray(action, dtype=np.float32)
            row_out: dict[str, Any] = {
                "event_id": row.get("event_id", ""),
                "policy_id": policy_id,
                "elapsed_min": elapsed_min,
                "datetime": str(sim.current_time),
                "rainfall_mm_h": rain,
                "phase": phase,
            }
            for nid, obj in node_objs.items():
                row_out[f"h:{nid}"] = _as_float(getattr(obj, "depth", np.nan), np.nan)
                row_out[f"head:{nid}"] = _as_float(getattr(obj, "head", np.nan), np.nan)
                row_out[f"storage_volume:{nid}"] = _as_float(getattr(obj, "volume", np.nan), np.nan)
                row_out[f"flood:{nid}"] = _as_float(getattr(obj, "flooding", 0.0), 0.0)
            for aid in facility_order:
                obj = link_objs.get(aid)
                row_out[f"a:{aid}"] = _as_float(previous[facility_order.index(aid)] if aid in facility_order else np.nan, np.nan)
                row_out[f"setting:{aid}"] = _as_float(getattr(obj, "current_setting", np.nan), np.nan) if obj is not None else np.nan
                row_out[f"flow:{aid}"] = _as_float(getattr(obj, "flow", np.nan), np.nan) if obj is not None else np.nan
            records.append(row_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_csv, index=False)
    return {
        "clone_inp": str(clone_inp),
        "clone_inp_sha256": sha256_file(clone_inp),
        "hotstart_status": hotstart_status,
        "visible_step_min": visible_step_min,
        "clone_start_time": str(start_time),
        "clone_end_time": str(end_time),
    }


def _reference_after_checkpoint(detail: pd.DataFrame, checkpoint_elapsed: float) -> pd.DataFrame:
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    return detail[elapsed > float(checkpoint_elapsed) + 1.0e-6].copy().reset_index(drop=True)


def _checkpoint_timestamp(detail: pd.DataFrame, checkpoint_elapsed: float) -> pd.Timestamp:
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    idx = (elapsed - float(checkpoint_elapsed)).abs().idxmin()
    return pd.to_datetime(detail.loc[idx, "datetime"])


def _numeric_diff(reference: pd.DataFrame, restored: pd.DataFrame, columns: list[str]) -> tuple[float, float, int]:
    if not columns:
        return math.nan, math.nan, 0
    ref = reference[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    res = restored[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = min(ref.shape[0], res.shape[0])
    if n == 0:
        return math.nan, math.nan, 0
    ref = ref[:n]
    res = res[:n]
    diff = np.abs(ref - res)
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return math.nan, math.nan, n
    denom = np.maximum(np.abs(ref[np.isfinite(diff)]), 1.0e-12)
    return float(np.nanmax(finite)), float(np.nanmax(finite / denom)), int(n)


def _metric_columns(reference: pd.DataFrame, metric: str) -> list[str]:
    if metric == "node_depth":
        return [col for col in reference.columns if col.startswith("h:")]
    if metric == "node_head":
        return [col for col in reference.columns if col.startswith("head:")]
    if metric == "link_flow":
        return [col for col in reference.columns if col.startswith("flow:")]
    if metric == "actual_setting":
        return [col for col in reference.columns if col.startswith("setting:")]
    if metric == "storage_volume":
        return [col for col in reference.columns if col.startswith("storage_volume:")]
    return []


def _load_tolerances(out_dir: Path) -> dict[str, dict[str, Any]]:
    data = _read_json(out_dir / "state_clone_numerical_noise.json")
    items = data.get("tolerances", {})
    out: dict[str, dict[str, Any]] = {}
    for metric in CLONE_METRICS:
        raw = items.get(metric, {}) if isinstance(items, dict) else {}
        if isinstance(raw, dict):
            out[metric] = raw
        else:
            out[metric] = {"tolerance_abs": raw, "tolerance_rel": 1.0e-9}
    return out


def _write_runtime_blocked(out_dir: Path, reason: str, reasons: list[str], mode: str) -> tuple[int, dict[str, Path]]:
    equivalence = out_dir / "state_clone_equivalence.csv"
    memory_audit = out_dir / "state_clone_controller_memory_audit.csv"
    timeline = out_dir / "state_clone_timeline_audit.csv"
    report_path = out_dir / "state_clone_report.json"
    write_csv(equivalence, [])
    write_csv(memory_audit, [])
    write_csv(timeline, [])
    report = {
        "status": "blocked",
        "mode": mode,
        "created_at": _utc_now(),
        "runtime_executed": False,
        "reason": reason,
        "blocking_reasons": reasons,
        "hotstart_equivalence_status": "not_run",
        "controller_memory_restore_status": "not_run",
        "formal_same_state_unlock_allowed": False,
        "completion_marker": None,
    }
    _atomic_json(report_path, report)
    return 3, {"equivalence": equivalence, "memory_audit": memory_audit, "timeline": timeline, "report": report_path}


def run_state_clone_equivalence(
    out_dir: str | Path = OUT_ROOT / "state_clone",
    *,
    mode: str = "full",
    max_checkpoints: int = 0,
    workers: int = 1,
    resume: bool = False,
) -> tuple[int, dict[str, Path]]:
    del workers  # independent PySWMM process workers are orchestrated by the runner in future extensions.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readiness = out_dir / "state_clone_checkpoint_readiness.csv"
    if mode not in {"smoke", "full"}:
        return _write_runtime_blocked(out_dir, "invalid_state_clone_mode", [mode], mode)
    rows = read_csv(readiness)
    selected = _select_checkpoints(rows, mode, max_checkpoints)
    if not selected:
        return _write_runtime_blocked(out_dir, "no_ready_state_clone_checkpoints", ["state_clone_checkpoint_readiness_missing_or_empty"], mode)
    if mode == "full":
        noise = _read_json(out_dir / "state_clone_numerical_noise.json")
        if not bool(noise.get("empirically_measured")):
            return _write_runtime_blocked(out_dir, "numerical_noise_not_empirically_measured", ["run_EstimateStateCloneNumericalNoise_first"], mode)

    equivalence = out_dir / ("state_clone_equivalence_smoke.csv" if mode == "smoke" else "state_clone_equivalence.csv")
    memory_audit = out_dir / ("state_clone_controller_memory_audit_smoke.csv" if mode == "smoke" else "state_clone_controller_memory_audit.csv")
    timeline = out_dir / ("state_clone_timeline_audit_smoke.csv" if mode == "smoke" else "state_clone_timeline_audit.csv")
    report_path = out_dir / ("state_clone_report_smoke.json" if mode == "smoke" else "state_clone_report.json")
    tolerances = _load_tolerances(out_dir)
    eq_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    executed = 0
    passed = 0
    failed = 0
    blocked = 0
    blocking_reasons: list[str] = []

    for row in selected:
        checkpoint_id = row.get("checkpoint_id", "")
        cp_dir = out_dir / ("smoke_runs" if mode == "smoke" else "full_runs") / checkpoint_id
        reference_csv = cp_dir / "reference_continuation.csv"
        restored_csv = cp_dir / "restored_continuation.csv"
        memory = _read_json(Path(row.get("controller_memory_path", "")))
        memory_row, memory_reasons = _memory_audit_row(row, memory)
        memory_rows.append(memory_row)
        detail_path = Path(row.get("detail_file", ""))
        local_reasons = list(memory_reasons)
        if not detail_path.exists():
            local_reasons.append("reference_detail_missing")
        try:
            detail = _load_dataframe(detail_path) if detail_path.exists() else pd.DataFrame()
        except Exception as exc:
            detail = pd.DataFrame()
            local_reasons.append(f"reference_detail_read_failed:{exc}")
        if not detail.empty:
            local_reasons.extend(_reference_schema_reasons(detail, memory))
        checkpoint_elapsed = _as_float(row.get("checkpoint_elapsed_min"), math.nan)
        reference = _reference_after_checkpoint(detail, checkpoint_elapsed) if not detail.empty and math.isfinite(checkpoint_elapsed) else pd.DataFrame()
        if reference.empty:
            local_reasons.append("reference_continuation_empty")
        if local_reasons:
            blocked += 1
            blocking_reasons.extend([f"{checkpoint_id}:{reason}" for reason in local_reasons])
            for metric in CLONE_METRICS:
                eq_rows.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "event_id": row.get("event_id", ""),
                        "policy_id": row.get("policy_id", ""),
                        "phase": row.get("phase", ""),
                        "metric": metric,
                        "max_abs_diff": "",
                        "max_rel_diff": "",
                        "tolerance_abs": "",
                        "tolerance_rel": "",
                        "timestamp_count": 0,
                        "status": "blocked",
                        "evidence_path": "",
                        "evidence_sha256": "",
                        "blocking_reason": ";".join(local_reasons),
                    }
                )
            continue
        cp_dir.mkdir(parents=True, exist_ok=True)
        runtime_info: dict[str, Any] = {}
        if reference_csv.exists() and restored_csv.exists() and resume:
            restored = _load_dataframe(restored_csv)
            resume_timestamp_match = list(reference.get("datetime", [])) == list(restored.get("datetime", []))
            if not resume_timestamp_match:
                restored = pd.DataFrame()
        else:
            restored = pd.DataFrame()
        if restored.empty:
            reference.to_csv(reference_csv, index=False)
            try:
                checkpoint_time = _checkpoint_timestamp(detail, checkpoint_elapsed)
                runtime_info = _run_restored_continuation(row, memory, reference, restored_csv, cp_dir, checkpoint_time)
            except Exception as exc:  # pragma: no cover - requires PySWMM runtime
                failed += 1
                blocking_reasons.append(f"{checkpoint_id}:restored_continuation_failed:{exc}")
                for metric in CLONE_METRICS:
                    eq_rows.append(
                        {
                            "checkpoint_id": checkpoint_id,
                            "event_id": row.get("event_id", ""),
                            "policy_id": row.get("policy_id", ""),
                            "phase": row.get("phase", ""),
                            "metric": metric,
                            "status": "failed_runtime",
                            "blocking_reason": str(exc),
                        }
                    )
                continue
            restored = _load_dataframe(restored_csv)
        executed += 1
        timestamp_match = list(reference.get("datetime", [])) == list(restored.get("datetime", []))
        timeline_rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "reference_start_time": str(reference.iloc[0].get("datetime", "")),
                "restored_start_time": str(restored.iloc[0].get("datetime", "")) if not restored.empty else "",
                "reference_end_time": str(reference.iloc[-1].get("datetime", "")),
                "restored_end_time": str(restored.iloc[-1].get("datetime", "")) if not restored.empty else "",
                "rainfall_path_hash": row.get("rainfall_state_hash", ""),
                "timestamp_match": str(timestamp_match).lower(),
                "row_count_match": str(len(reference) == len(restored)).lower(),
                "missing_timestamp_count": 0 if timestamp_match else abs(len(reference) - len(restored)),
                "duplicate_timestamp_count": int(pd.Series(restored.get("datetime", [])).duplicated().sum()) if not restored.empty else "",
                "decision_clock_match": str(timestamp_match).lower(),
                "hotstart_load_status": runtime_info.get("hotstart_status", {}).get("status", "reused_existing_restored_output" if resume else ""),
                "hotstart_sha256": runtime_info.get("hotstart_status", {}).get("sha256", ""),
                "clone_inp": runtime_info.get("clone_inp", ""),
                "clone_inp_sha256": runtime_info.get("clone_inp_sha256", ""),
                "visible_step_min": runtime_info.get("visible_step_min", ""),
                "status": "pass" if timestamp_match else "failed_gate",
            }
        )
        checkpoint_pass = bool(timestamp_match)
        for metric in CLONE_METRICS:
            tol = tolerances.get(metric, {})
            tolerance_abs = _as_float(tol.get("tolerance_abs", tol.get("abs", 1.0e-6)), 1.0e-6)
            tolerance_rel = _as_float(tol.get("tolerance_rel", tol.get("rel", 1.0e-9)), 1.0e-9)
            if metric in {"PFV", "TFV", "peak"}:
                ref_kpi = compute_kpis(reference, [], dt_sec=300)
                res_kpi = compute_kpis(restored, [], dt_sec=300)
                key = {"PFV": "PFV", "TFV": "TFV", "peak": "peak_TFV_rate"}[metric]
                max_abs = abs(_as_float(ref_kpi.get(key), 0.0) - _as_float(res_kpi.get(key), 0.0))
                max_rel = max_abs / max(abs(_as_float(ref_kpi.get(key), 0.0)), 1.0e-12)
                count = len(reference)
            elif metric == "recovery_end_time":
                max_abs = abs(_as_float(reference["elapsed_min"].max(), 0.0) - _as_float(restored["elapsed_min"].max(), 0.0))
                max_rel = 0.0
                count = len(reference)
            else:
                cols = [col for col in _metric_columns(reference, metric) if col in restored.columns]
                max_abs, max_rel, count = _numeric_diff(reference, restored, cols)
            status = "pass" if math.isfinite(max_abs) and max_abs <= tolerance_abs and (not math.isfinite(max_rel) or max_rel <= max(tolerance_rel, 1.0e-12)) and timestamp_match else "failed_gate"
            checkpoint_pass = checkpoint_pass and status == "pass"
            if status != "pass":
                blocking_reasons.append(
                    f"{checkpoint_id}:{metric}_exceeds_tolerance:"
                    f"abs={max_abs}:tol_abs={tolerance_abs}:rel={max_rel}:tol_rel={tolerance_rel}"
                )
            eq_rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "event_id": row.get("event_id", ""),
                    "policy_id": row.get("policy_id", ""),
                    "phase": row.get("phase", ""),
                    "metric": metric,
                    "max_abs_diff": max_abs,
                    "max_rel_diff": max_rel,
                    "tolerance_abs": tolerance_abs,
                    "tolerance_rel": tolerance_rel,
                    "timestamp_count": count,
                    "status": status,
                    "evidence_path": str(restored_csv),
                    "evidence_sha256": sha256_file(restored_csv) or "",
                    "blocking_reason": "" if status == "pass" else "metric_exceeds_tolerance_or_timeline_mismatch",
                }
            )
        if checkpoint_pass:
            passed += 1
        else:
            failed += 1
    write_csv(equivalence, eq_rows)
    write_csv(memory_audit, memory_rows)
    write_csv(timeline, timeline_rows)
    full_unlock = mode == "full" and len(selected) == 18 and passed == 18 and failed == 0 and blocked == 0
    status = "pass" if (mode == "smoke" and executed == len(selected) and failed == 0 and blocked == 0) or full_unlock else ("failed_gate" if executed and failed else "blocked")
    memory_pass = bool(memory_rows) and all(row.get("status") == "pass" for row in memory_rows)
    timeline_pass = bool(timeline_rows) and all(row.get("status") == "pass" for row in timeline_rows)
    report = {
        "status": status,
        "mode": mode,
        "created_at": _utc_now(),
        "runtime_executed": executed > 0,
        "eligible_checkpoint_count": len([r for r in rows if str(r.get("ready_for_clone", "")).lower() == "true"]),
        "selected_checkpoint_count": len(selected),
        "executed_checkpoint_count": executed,
        "passed_checkpoint_count": passed,
        "failed_checkpoint_count": failed,
        "blocked_checkpoint_count": blocked,
        "blocking_reasons": blocking_reasons[:200],
        "hotstart_equivalence_status": "pass" if status == "pass" else ("failed" if executed else "not_run"),
        "controller_memory_restore_status": "pass" if memory_pass else ("failed" if memory_rows else "not_run"),
        "timeline_status": "pass" if timeline_pass else "not_run" if not timeline_rows else "failed",
        "hydraulic_state_status": "pass" if status == "pass" else "not_run" if not executed else "failed",
        "facility_state_status": "pass" if status == "pass" else "not_run" if not executed else "failed",
        "kpi_status": "pass" if status == "pass" else "not_run" if not executed else "failed",
        "formal_same_state_unlock_allowed": full_unlock,
        "completion_marker": "allowed_after_marker_write" if full_unlock else None,
    }
    _atomic_json(report_path, report)
    if status == "pass":
        return 0, {"equivalence": equivalence, "memory_audit": memory_audit, "timeline": timeline, "report": report_path}
    if executed and failed:
        return 5, {"equivalence": equivalence, "memory_audit": memory_audit, "timeline": timeline, "report": report_path}
    return 3, {"equivalence": equivalence, "memory_audit": memory_audit, "timeline": timeline, "report": report_path}


def estimate_state_clone_numerical_noise(
    out_dir: str | Path = OUT_ROOT / "state_clone",
    *,
    max_checkpoints: int = 3,
    workers: int = 1,
) -> tuple[int, dict[str, Path]]:
    del workers
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tolerance_path = out_dir / "state_clone_numerical_noise.json"
    readiness = read_csv(out_dir / "state_clone_checkpoint_readiness.csv")
    selected = _select_checkpoints(readiness, "smoke", max_checkpoints or 3)
    if not selected:
        report = {
            "status": "blocked",
            "created_at": _utc_now(),
            "empirically_measured": False,
            "blocking_reasons": ["no_ready_checkpoints_for_noise_estimation"],
            "completion_marker": None,
        }
        _atomic_json(tolerance_path, report)
        return 3, {"noise": tolerance_path}
    blocking_reasons: list[str] = []
    for row in selected:
        checkpoint_id = row.get("checkpoint_id", "")
        memory = _read_json(Path(row.get("controller_memory_path", "")))
        _, memory_reasons = _memory_audit_row(row, memory)
        detail_path = Path(row.get("detail_file", ""))
        if not detail_path.exists():
            blocking_reasons.append(f"{checkpoint_id}:reference_detail_missing")
            continue
        try:
            detail = _load_dataframe(detail_path)
        except Exception as exc:
            blocking_reasons.append(f"{checkpoint_id}:reference_detail_read_failed:{exc}")
            continue
        schema_reasons = _reference_schema_reasons(detail, memory)
        if memory_reasons or schema_reasons:
            blocking_reasons.extend([f"{checkpoint_id}:{reason}" for reason in memory_reasons + schema_reasons])
            continue
        checkpoint_elapsed = _as_float(row.get("checkpoint_elapsed_min"), math.nan)
        reference = _reference_after_checkpoint(detail, checkpoint_elapsed) if math.isfinite(checkpoint_elapsed) else pd.DataFrame()
        if reference.empty:
            blocking_reasons.append(f"{checkpoint_id}:reference_continuation_empty")
            continue
        checkpoint_time = _checkpoint_timestamp(detail, checkpoint_elapsed)
        noise_dir = out_dir / "noise_runs" / checkpoint_id
        try:
            _run_restored_continuation(row, memory, reference, noise_dir / f"{checkpoint_id}__repeat_a.csv", noise_dir / "repeat_a", checkpoint_time)
            _run_restored_continuation(row, memory, reference, noise_dir / f"{checkpoint_id}__repeat_b.csv", noise_dir / "repeat_b", checkpoint_time)
        except Exception as exc:  # pragma: no cover - requires PySWMM runtime
            blocking_reasons.append(f"{checkpoint_id}:duplicate_restore_failed:{exc}")
    duplicate_pairs = list((out_dir / "noise_runs").glob("*/*__repeat_*.csv"))
    if blocking_reasons:
        report = {
            "status": "blocked",
            "created_at": _utc_now(),
            "method": "duplicate_restored_continuation",
            "empirically_measured": False,
            "selected_checkpoint_count": len(selected),
            "blocking_reasons": blocking_reasons[:200],
            "tolerances": {},
            "completion_marker": None,
        }
        _atomic_json(tolerance_path, report)
        return 3, {"noise": tolerance_path}
    rows: list[dict[str, Any]] = []
    tolerances: dict[str, dict[str, Any]] = {}
    for metric in CLONE_METRICS:
        diffs: list[float] = []
        for first in duplicate_pairs:
            if "__repeat_a" not in first.name:
                continue
            second = first.with_name(first.name.replace("__repeat_a", "__repeat_b"))
            if not second.exists():
                continue
            a = _load_dataframe(first)
            b = _load_dataframe(second)
            cols = _metric_columns(a, metric)
            max_abs, max_rel, count = _numeric_diff(a, b, [c for c in cols if c in b.columns])
            if math.isfinite(max_abs):
                diffs.append(max_abs)
            rows.append(
                {
                    "metric": metric,
                    "max_abs_diff": max_abs,
                    "max_rel_diff": max_rel,
                    "sample_count": count,
                    "checkpoint_pair": first.stem.replace("__repeat_a", ""),
                    "empirically_measured": str(math.isfinite(max_abs)).lower(),
                }
            )
        if diffs:
            arr = np.asarray(diffs, dtype=float)
            tol = float(np.nanmax(arr) + max(1.0e-9, np.nanpercentile(arr, 99) * 0.05))
            tolerances[metric] = {"tolerance_abs": tol, "tolerance_rel": 1.0e-9, "empirically_measured": True}
    csv_path = out_dir / "state_clone_numerical_noise_audit.csv"
    write_csv(csv_path, rows)
    measured = bool(tolerances) and all(metric in tolerances for metric in ["node_depth", "actual_setting", "PFV", "TFV", "peak"])
    report = {
        "status": "completed" if measured else "blocked",
        "created_at": _utc_now(),
        "method": "duplicate_restored_continuation",
        "empirically_measured": measured,
        "checkpoint_count": len(selected),
        "audit_csv": str(csv_path),
        "audit_csv_sha256": sha256_file(csv_path),
        "tolerances": tolerances,
        "completion_marker": "allowed_after_marker_write" if measured else None,
    }
    _atomic_json(tolerance_path, report)
    return (0 if measured else 3), {"noise": tolerance_path}


def evaluate_state_clone_gate(out_dir: str | Path = OUT_ROOT / "state_clone") -> tuple[int, dict[str, Path]]:
    out_dir = Path(out_dir)
    gate_path = out_dir / "state_clone_gate.json"
    equivalence = out_dir / "state_clone_equivalence.csv"
    memory_audit = out_dir / "state_clone_controller_memory_audit.csv"
    timeline = out_dir / "state_clone_timeline_audit.csv"
    report_path = out_dir / "state_clone_report.json"
    report = _read_json(report_path)
    rows = read_csv(equivalence)
    memory_rows = read_csv(memory_audit)
    timeline_rows = read_csv(timeline)
    runtime_executed = bool(report.get("runtime_executed"))
    all_pass = bool(rows) and all(row.get("status") == "pass" for row in rows)
    memory_pass = bool(memory_rows) and all(row.get("status") == "pass" for row in memory_rows)
    timeline_pass = bool(timeline_rows) and all(row.get("status") == "pass" for row in timeline_rows)
    noise = _read_json(out_dir / "state_clone_numerical_noise.json")
    noise_measured = bool(noise.get("empirically_measured"))
    full_support = int(report.get("eligible_checkpoint_count") or 0) == 18 and int(report.get("executed_checkpoint_count") or 0) == 18
    gate_pass = runtime_executed and all_pass and memory_pass and timeline_pass and noise_measured and full_support
    status = "pass" if gate_pass else ("failed_gate" if runtime_executed and rows else "blocked")
    gate = {
        "status": status,
        "created_at": _utc_now(),
        "eligible_checkpoint_count": report.get("eligible_checkpoint_count", 0),
        "executed_checkpoint_count": report.get("executed_checkpoint_count", 0),
        "passed_checkpoint_count": report.get("passed_checkpoint_count", 0),
        "failed_checkpoint_count": report.get("failed_checkpoint_count", 0),
        "blocked_checkpoint_count": report.get("blocked_checkpoint_count", 0),
        "phase_support": sorted({row.get("phase", "") for row in rows if row.get("phase", "")}),
        "policy_support": sorted({row.get("policy_id", "") for row in rows if row.get("policy_id", "")}),
        "noise_empirically_measured": noise_measured,
        "timeline_pass": timeline_pass,
        "controller_memory_pass": memory_pass,
        "hydraulic_state_pass": all_pass,
        "facility_state_pass": all_pass,
        "KPI_pass": all_pass,
        "runtime_executed": runtime_executed,
        "formal_same_state_unlock_allowed": gate_pass,
        "blocking_reasons": [] if gate_pass else report.get("blocking_reasons", []) + [
            reason
            for reason, ok in {
                "runtime_not_executed": runtime_executed,
                "equivalence_metrics_not_all_pass": all_pass,
                "controller_memory_audit_not_pass": memory_pass,
                "timeline_audit_not_pass": timeline_pass,
                "numerical_noise_not_empirically_measured": noise_measured,
                "full_18_checkpoint_support_missing": full_support,
            }.items()
            if not ok
        ],
        "evidence": {
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
            "equivalence": str(equivalence),
            "equivalence_sha256": sha256_file(equivalence),
            "memory_audit": str(memory_audit),
            "memory_audit_sha256": sha256_file(memory_audit),
            "timeline": str(timeline),
            "timeline_sha256": sha256_file(timeline),
        },
        "completion_marker": "allowed_after_marker_write" if gate_pass else None,
    }
    _atomic_json(gate_path, gate)
    if gate_pass:
        return 0, {"gate": gate_path}
    return (5 if runtime_executed and rows else 3), {"gate": gate_path}


def prepare_state_clone_outputs(out_dir: str | Path = OUT_ROOT / "state_clone") -> list[Path]:
    _, prepared = prepare_state_clone_checkpoints(out_dir=out_dir)
    _, noise = estimate_state_clone_numerical_noise(out_dir=out_dir)
    _, plan = run_state_clone_equivalence(out_dir=out_dir, mode="smoke", max_checkpoints=3)
    return list(prepared.values()) + list(noise.values()) + list(plan.values())
