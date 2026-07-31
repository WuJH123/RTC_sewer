#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Project6 V4 event-level Oracle/Pareto feasibility analysis.

This script builds a *declared-neighbourhood engineering oracle* for the Wuhan
urban drainage RTC study. It does not claim a mathematical global optimum over
all 36 actuators and all time steps. Instead, it evaluates a broad, reproducible
library of full-event action schedules with authoritative SWMM, extracts the
non-dominated Pareto set, and determines whether the V4 scientific contract is
physically achievable within the searched action neighbourhood.

Scientific contract
-------------------
PFV safety reference:
    min(no_control, executable_passive)
TFV and peak performance reference:
    internal_rules
A schedule is strictly feasible when all three event-level inequalities pass.

Stages
------
references : create event INPs and run no_control/internal/passive references
plan       : build deterministic and random-sparse candidate schedules
run        : execute planned candidate schedules with SWMM (resume-safe)
analyze    : build Pareto fronts, feasibility classes and plots
all        : references -> plan -> run -> analyze

The script is intended to be placed at:
    E:/RTC_sewer/Project6/scripts/206_oracle_pareto_v4.py

Required project modules
------------------------
- sewerrtc.io.project_paths.load_config
- sewerrtc.io.swmm_mutation.mutate_inp_for_event
- sewerrtc.simulation.pyswmm_runner.run_swmm_trajectory

Example
-------
python scripts/206_oracle_pareto_v4.py \
  --config configs/wuhan_project6_dual_reference_v4.yaml \
  --engineering-config configs/wuhan_project6_engineering36.yaml \
  --actuators-csv outputs/closed_loop_paired_no_controls/formal/\
project6_no_control_repair_formal_30_v8/control_actuator_table.csv \
  --stage all --event-limit 3 --workers 4 --resume

Authoritative full development run should use 12-20 development events that are
strictly disjoint from Calibration, Validation and Formal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def _project_root_from_script() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "sewerrtc").exists() and (parent / "scripts").exists():
            return parent
    return Path.cwd().resolve()


PROJECT_ROOT = _project_root_from_script()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def short_hash(obj: Any, n: int = 12) -> str:
    return sha256_json(obj)[:n]


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, p)


def now_utc_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def resolve_path(root: Path, raw: str | Path | None) -> Path | None:
    if raw is None or str(raw).strip() == "":
        return None
    p = Path(raw)
    return p if p.is_absolute() else (root / p).resolve()


def nested_get(data: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    obj: Any = data
    for key in dotted.split("."):
        if not isinstance(obj, Mapping) or key not in obj:
            return default
        obj = obj[key]
    return obj


def first_existing(paths: Iterable[Path | None]) -> Path | None:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def load_yaml_with_inheritance(path: str | Path) -> dict[str, Any]:
    """Use Project6 loader when available; otherwise perform a local merge."""
    p = Path(path).resolve()
    try:
        from sewerrtc.io.project_paths import load_config

        return dict(load_config(p))
    except Exception:
        def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
            out = dict(a)
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = _deep_merge(out[k], v)
                else:
                    out[k] = v
            return out

        obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if "_inherits" in obj:
            parent = Path(obj["_inherits"])
            if not parent.is_absolute():
                parent = p.parent.parent / parent
            obj = _deep_merge(
                load_yaml_with_inheritance(parent),
                {k: v for k, v in obj.items() if k != "_inherits"},
            )
        root = Path(obj.get("project_root", obj.get("project", {}).get("root", ".")))
        if not root.is_absolute():
            root = p.parent.parent / root
        obj["project_root"] = str(root.resolve())
        obj["_config_path"] = str(p)
        return obj


# ---------------------------------------------------------------------------
# Contracts and configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    rainfall_csv: str
    duration_min: int
    simulation_duration_min: int
    recession_min: int
    split: str = "development"
    rainfall_sha256: str = ""


@dataclass(frozen=True)
class CandidateMeta:
    case_id: str
    event_id: str
    policy_id: str
    label: str
    family: str
    source_anchor: str
    constraint_mode: str
    schedule_csv: str
    schedule_sha256: str
    candidate_rank: int
    seed: int
    notes: str = ""


@dataclass
class OracleSettings:
    control_step_sec: int = 600
    recession_min: int = 180
    seed: int = 20260723
    delta: float = 0.05
    pulse_steps: int = 2
    max_cases_per_event: int = 450
    random_candidates_per_event: int = 48
    allowed_k: tuple[int, ...] = (1, 2, 4, 6, 8)
    single_pulse_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    delay_minutes: tuple[int, ...] = (10, 20)
    min_hold_minutes: tuple[int, ...] = (20, 30)
    include_relaxed: bool = True
    path_budget_chars: int = 235
    pfv_abs_margin_m3: float = 0.0
    pfv_rel_margin: float = 0.0
    tfv_abs_margin_m3: float = 0.0
    tfv_rel_margin: float = 0.0
    peak_abs_margin: float = 0.0
    peak_rel_margin: float = 0.0
    nonpriority_abs_margin_m3: float | None = None
    convergence_min_candidates: int = 100
    convergence_tail_fraction: float = 0.20
    convergence_hv_relative_tol: float = 0.01


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------


def _normalise_event_table(frame: pd.DataFrame, root: Path, recession_min: int) -> pd.DataFrame:
    aliases = {
        "event_id": ["event_id", "rainfall_event_id", "event", "id"],
        "rainfall_csv": ["rainfall_csv", "rain_csv", "csv_path", "path"],
        "duration_min": ["duration_min", "rain_duration_min", "storm_duration_min"],
        "simulation_duration_min": ["simulation_duration_min", "sim_duration_min"],
        "split": ["split", "event_split", "dataset_split"],
    }
    rename: dict[str, str] = {}
    for target, options in aliases.items():
        for col in options:
            if col in frame.columns:
                rename[col] = target
                break
    frame = frame.rename(columns=rename).copy()
    required = ["event_id", "rainfall_csv", "duration_min"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Rainfall event table is missing columns: {missing}; columns={list(frame.columns)}")
    frame["event_id"] = frame["event_id"].astype(str).str.strip()
    frame["duration_min"] = pd.to_numeric(frame["duration_min"], errors="raise").astype(int)
    if "simulation_duration_min" not in frame:
        frame["simulation_duration_min"] = frame["duration_min"] + int(recession_min)
    frame["simulation_duration_min"] = pd.to_numeric(
        frame["simulation_duration_min"], errors="coerce"
    ).fillna(frame["duration_min"] + int(recession_min)).astype(int)
    if "split" not in frame:
        frame["split"] = "development"
    frame["split"] = frame["split"].fillna("development").astype(str)
    frame["rainfall_csv"] = frame["rainfall_csv"].map(
        lambda x: str(resolve_path(root, str(x)))
    )
    frame = frame.drop_duplicates(subset=["event_id"], keep="first").reset_index(drop=True)
    return frame


def discover_event_table(
    cfg: Mapping[str, Any],
    engineering_cfg: Mapping[str, Any],
    explicit: str | None,
) -> Path:
    root = Path(str(cfg.get("project_root", PROJECT_ROOT))).resolve()
    eroot = Path(str(engineering_cfg.get("project_root", root))).resolve()
    candidates = [
        resolve_path(root, explicit),
        resolve_path(eroot, nested_get(engineering_cfg, "engineering36.rainfall_event_table")),
        resolve_path(eroot, nested_get(engineering_cfg, "outputs.rainfall")) / "rainfall_event_table.csv"
        if nested_get(engineering_cfg, "outputs.rainfall")
        else None,
        resolve_path(root, nested_get(cfg, "outputs.rainfall")) / "rainfall_event_table.csv"
        if nested_get(cfg, "outputs.rainfall")
        else None,
        root / "outputs/rainfall_library_v8_storage_36/rainfall_event_table.csv",
    ]
    found = first_existing(candidates)
    if found is None:
        raise FileNotFoundError(
            "Cannot find rainfall_event_table.csv. Pass --events-csv explicitly. Tried: "
            + ", ".join(str(x) for x in candidates if x is not None)
        )
    return found


def discover_base_inp(cfg: Mapping[str, Any], engineering_cfg: Mapping[str, Any], explicit: str | None) -> Path:
    root = Path(str(cfg.get("project_root", PROJECT_ROOT))).resolve()
    eroot = Path(str(engineering_cfg.get("project_root", root))).resolve()
    candidates = [
        resolve_path(root, explicit),
        resolve_path(root, nested_get(cfg, "project.inp")),
        resolve_path(root, nested_get(cfg, "network.inp")),
        resolve_path(eroot, nested_get(engineering_cfg, "network.inp")),
        root / "data/wuhan_v8_storage_retrofit.inp",
    ]
    found = first_existing(candidates)
    if found is None:
        raise FileNotFoundError("Cannot find base SWMM INP; pass --inp explicitly")
    return found


def discover_actuator_csv(cfg: Mapping[str, Any], explicit: str | None) -> Path:
    root = Path(str(cfg.get("project_root", PROJECT_ROOT))).resolve()
    candidates = [
        resolve_path(root, explicit),
        resolve_path(root, nested_get(cfg, "oracle_pareto.actuators_csv")),
        resolve_path(root, nested_get(cfg, "storage_retrofit.original_v8_control_table")),
        # V4 dual-reference formal outputs
        root / "outputs/project6_dual_reference_v4/cl/formal/p6v4_dev_3/control_actuator_table.csv",
        # V3 formal directories that do exist
        root / "outputs/closed_loop_paired_no_controls/formal/project6_no_control_repair_formal_30_v8/control_actuator_table.csv",
        # Generic audit actuator table
        root / "outputs/audit_v8_storage_variablepump/actuator_table.csv",
    ]
    found = first_existing(candidates)
    if found is not None:
        return found
    # Last-resort glob: any control_actuator_table.csv under outputs/
    glob_matches = sorted(root.glob("outputs/**/control_actuator_table.csv"))
    if glob_matches:
        return glob_matches[0]
    raise FileNotFoundError(
        "Cannot find control actuator table; pass --actuators-csv explicitly"
    )


def load_priority_nodes(
    cfg: Mapping[str, Any],
    engineering_cfg: Mapping[str, Any],
    explicit: str | None,
) -> list[str]:
    root = Path(str(cfg.get("project_root", PROJECT_ROOT))).resolve()
    if explicit:
        p = resolve_path(root, explicit)
        if p and p.exists():
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
                for col in ("node_id", "priority_node", "node", "id"):
                    if col in df:
                        return df[col].dropna().astype(str).tolist()
                return df.iloc[:, 0].dropna().astype(str).tolist()
            return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        return [x.strip() for x in explicit.split(",") if x.strip()]
    core = list(nested_get(engineering_cfg, "engineering36.priority_core_nodes", []) or [])
    sentinel = list(nested_get(engineering_cfg, "engineering36.sentinel_nodes", []) or [])
    nodes = list(dict.fromkeys(str(x) for x in [*core, *sentinel] if str(x).strip()))
    if not nodes:
        raise ValueError("No priority nodes found; pass --priority-nodes")
    return nodes


def select_events(
    table: pd.DataFrame,
    event_ids: str | None,
    event_limit: int,
    allowed_splits: Sequence[str],
) -> list[EventSpec]:
    work = table.copy()
    if allowed_splits:
        allowed = {s.lower() for s in allowed_splits}
        work = work[work["split"].str.lower().isin(allowed)]
    if event_ids:
        selected = {x.strip() for x in event_ids.split(",") if x.strip()}
        work = work[work["event_id"].isin(selected)]
        missing = selected - set(work["event_id"])
        if missing:
            raise ValueError(f"Requested event IDs not found after split filtering: {sorted(missing)}")
    work = work.sort_values("event_id").reset_index(drop=True)
    if event_limit > 0:
        work = work.head(event_limit)
    specs: list[EventSpec] = []
    for _, row in work.iterrows():
        rain = Path(str(row["rainfall_csv"]))
        if not rain.exists():
            raise FileNotFoundError(f"Missing rainfall CSV for {row['event_id']}: {rain}")
        duration = int(row["duration_min"])
        simulation_duration = int(row["simulation_duration_min"])
        specs.append(
            EventSpec(
                event_id=str(row["event_id"]),
                rainfall_csv=str(rain),
                duration_min=duration,
                simulation_duration_min=simulation_duration,
                recession_min=max(0, simulation_duration - duration),
                split=str(row.get("split", "development")),
                rainfall_sha256=sha256_file(rain),
            )
        )
    if not specs:
        raise ValueError("No development events selected")
    return specs


# ---------------------------------------------------------------------------
# Actuator semantics and schedule helpers
# ---------------------------------------------------------------------------


def _actuator_ids(actuators: pd.DataFrame) -> list[str]:
    if "actuator_id" not in actuators:
        raise ValueError("Actuator table must contain actuator_id")
    ids = actuators["actuator_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Actuator IDs are not unique")
    return ids


def _column_numeric(frame: pd.DataFrame, names: Sequence[str], default: float) -> np.ndarray:
    for name in names:
        if name in frame:
            vals = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
            if np.isfinite(vals).any():
                return np.where(np.isfinite(vals), vals, default)
    return np.full(len(frame), default, dtype=float)


def passive_vector(actuators: pd.DataFrame) -> np.ndarray:
    return np.clip(
        _column_numeric(
            actuators,
            (
                "executable_passive_setting",
                "passive_setting",
                "fail_safe_setting",
                "default_setting",
                "initial_setting",
            ),
            1.0,
        ),
        0.0,
        1.0,
    )


def binary_pump_ids(actuators: pd.DataFrame, cfg: Mapping[str, Any], engineering_cfg: Mapping[str, Any]) -> set[str]:
    variable = set(
        str(x)
        for x in (
            nested_get(engineering_cfg, "engineering36.variable_speed_pump_ids", [])
            or nested_get(cfg, "controller.variable_speed_pump_ids", [])
            or []
        )
    )
    explicit = set(
        str(x)
        for x in (nested_get(engineering_cfg, "engineering36.binary_pump_ids", []) or [])
    )
    if "link_type" in actuators:
        all_pumps = set(
            actuators.loc[
                actuators["link_type"].astype(str).str.lower().eq("pump"), "actuator_id"
            ].astype(str)
        )
    else:
        all_pumps = set()
    return (all_pumps | explicit) - variable


def variable_speed_ids(cfg: Mapping[str, Any], engineering_cfg: Mapping[str, Any]) -> set[str]:
    return set(
        str(x)
        for x in (
            nested_get(engineering_cfg, "engineering36.variable_speed_pump_ids", [])
            or nested_get(cfg, "controller.variable_speed_pump_ids", [])
            or []
        )
    )


def time_grid(event: EventSpec, control_step_sec: int) -> np.ndarray:
    step_min = control_step_sec / 60.0
    n = int(math.ceil(event.simulation_duration_min / step_min)) + 1
    return np.arange(n, dtype=float) * step_min


def schedule_frame(times_min: np.ndarray, matrix: np.ndarray, actuator_ids: Sequence[str]) -> pd.DataFrame:
    if matrix.shape != (len(times_min), len(actuator_ids)):
        raise ValueError(
            f"Schedule shape {matrix.shape} != {(len(times_min), len(actuator_ids))}"
        )
    out = pd.DataFrame(matrix, columns=list(actuator_ids))
    out.insert(0, "simtime (hr)", times_min / 60.0)
    return out


def read_action_matrix(
    detail_csv: str | Path,
    times_min: np.ndarray,
    actuator_ids: Sequence[str],
    fallback: np.ndarray,
) -> np.ndarray:
    detail = pd.read_csv(detail_csv)
    if "elapsed_min" not in detail:
        raise ValueError(f"Detail file lacks elapsed_min: {detail_csv}")
    source_cols: dict[str, str | None] = {}
    for aid in actuator_ids:
        candidates = (f"setting:{aid}", f"a:{aid}", aid)
        source_cols[aid] = next((c for c in candidates if c in detail), None)
    src_t = pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(float)
    valid_t = np.isfinite(src_t)
    src_t = src_t[valid_t]
    out = np.empty((len(times_min), len(actuator_ids)), dtype=float)
    for j, aid in enumerate(actuator_ids):
        col = source_cols[aid]
        if col is None:
            out[:, j] = fallback[j]
            continue
        vals = pd.to_numeric(detail.loc[valid_t, col], errors="coerce").to_numpy(float)
        series = pd.Series(vals).ffill().bfill().fillna(float(fallback[j])).to_numpy(float)
        idx = np.searchsorted(src_t, times_min, side="right") - 1
        idx = np.clip(idx, 0, len(src_t) - 1)
        out[:, j] = series[idx]
    return np.clip(out, 0.0, 1.0)


def write_schedule(path: Path, times: np.ndarray, matrix: np.ndarray, ids: Sequence[str]) -> str:
    frame = schedule_frame(times, matrix, ids)
    atomic_write_csv(path, frame)
    return sha256_file(path)


def _step_limit_vector(actuators: pd.DataFrame, cfg: Mapping[str, Any]) -> np.ndarray:
    ids = _actuator_ids(actuators)
    mapping = nested_get(cfg, "controller.per_actuator_max_delta", {}) or {}
    default = float(nested_get(cfg, "controller.max_first_step_delta", 1.0) or 1.0)
    return np.asarray([float(mapping.get(aid, default)) for aid in ids], dtype=float)


def _min_hold_vector(actuators: pd.DataFrame, cfg: Mapping[str, Any]) -> np.ndarray:
    ids = _actuator_ids(actuators)
    mapping = nested_get(cfg, "controller.min_hold_steps_by_actuator", {}) or {}
    return np.asarray([int(mapping.get(aid, 1)) for aid in ids], dtype=int)


def _storage_groups(actuators: pd.DataFrame) -> dict[str, dict[str, list[int]]]:
    rows = actuators.reset_index(drop=True)
    groups: dict[str, dict[str, list[int]]] = {}
    roles = rows.get("storage_control_type", pd.Series("", index=rows.index)).fillna("").astype(str)
    for i, row in rows.iterrows():
        role = roles.iloc[i]
        if role not in {"storage_inlet", "storage_outlet"}:
            continue
        key = ""
        for col in ("storage_node", "efd_reference_node", "to_node", "from_node"):
            value = str(row.get(col, "")).strip()
            if value:
                key = value
                break
        if not key:
            continue
        groups.setdefault(key, {"storage_inlet": [], "storage_outlet": []})[role].append(i)
    return groups


def project_schedule(
    matrix: np.ndarray,
    *,
    anchor: np.ndarray,
    actuators: pd.DataFrame,
    cfg: Mapping[str, Any],
    engineering_cfg: Mapping[str, Any],
    constraint_mode: str,
    max_k: int | None,
) -> np.ndarray:
    """Project a schedule onto the declared engineering action set.

    relaxed: only [0,1] bounds and binary pump semantics.
    constrained: additionally applies rate limits, binary dwell, top-K and a
    conservative storage inlet/outlet deviation interlock.
    """
    ids = _actuator_ids(actuators)
    out = np.clip(np.asarray(matrix, dtype=float).copy(), 0.0, 1.0)
    binary = binary_pump_ids(actuators, cfg, engineering_cfg)
    binary_idx = [ids.index(aid) for aid in binary if aid in ids]
    if binary_idx:
        out[:, binary_idx] = (out[:, binary_idx] >= 0.5).astype(float)
    if constraint_mode == "relaxed":
        return out

    limits = _step_limit_vector(actuators, cfg)
    hold = _min_hold_vector(actuators, cfg)
    last_change = np.full(len(ids), -10_000, dtype=int)
    previous = np.asarray(anchor[0], dtype=float).copy()
    for t in range(len(out)):
        requested = out[t].copy()
        delta = np.clip(requested - previous, -limits, limits)
        current = np.clip(previous + delta, 0.0, 1.0)
        for j in binary_idx:
            current[j] = float(current[j] >= 0.5)
            if current[j] != previous[j]:
                if t - last_change[j] < max(1, int(hold[j])):
                    current[j] = previous[j]
                else:
                    last_change[j] = t
        if max_k is not None and max_k >= 0:
            dev = np.abs(current - anchor[t])
            changed = np.flatnonzero(dev > 1e-9)
            if len(changed) > max_k:
                keep = changed[np.argsort(dev[changed])[-max_k:]]
                reset = np.setdiff1d(changed, keep, assume_unique=False)
                current[reset] = anchor[t, reset]
        out[t] = current
        previous = current

    if bool(nested_get(cfg, "controller.storage_retrofit.inlet_outlet_incompatible_action_constraint", True)):
        for _, group in _storage_groups(actuators).items():
            for t in range(len(out)):
                inlet = group["storage_inlet"]
                outlet = group["storage_outlet"]
                if not inlet or not outlet:
                    continue
                inlet_dev = max(abs(out[t, j] - anchor[t, j]) for j in inlet)
                outlet_dev = max(abs(out[t, j] - anchor[t, j]) for j in outlet)
                if inlet_dev > 1e-9 and outlet_dev > 1e-9:
                    reset = outlet if inlet_dev >= outlet_dev else inlet
                    for j in reset:
                        out[t, j] = anchor[t, j]
    return np.clip(out, 0.0, 1.0)


def topk_deviation(schedule: np.ndarray, anchor: np.ndarray, k: int) -> np.ndarray:
    out = anchor.copy()
    for t in range(len(schedule)):
        dev = np.abs(schedule[t] - anchor[t])
        if k <= 0:
            continue
        idx = np.argsort(dev)[-min(k, len(dev)):]
        out[t, idx] = schedule[t, idx]
    return out


def delay_schedule(schedule: np.ndarray, anchor: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0:
        return schedule.copy()
    out = anchor.copy()
    if steps < len(schedule):
        out[steps:] = schedule[:-steps]
    return out


def block_hold(schedule: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 1:
        return schedule.copy()
    out = schedule.copy()
    for start in range(0, len(schedule), steps):
        out[start : start + steps] = schedule[start]
    return out


def remove_reversals(schedule: np.ndarray, deadband: float = 0.02) -> np.ndarray:
    out = schedule.copy()
    if len(out) < 3:
        return out
    last_sign = np.zeros(out.shape[1], dtype=int)
    for t in range(1, len(out)):
        d = out[t] - out[t - 1]
        sign = np.where(np.abs(d) <= deadband, 0, np.sign(d)).astype(int)
        reversal = (sign != 0) & (last_sign != 0) & (sign != last_sign)
        out[t, reversal] = out[t - 1, reversal]
        changed = sign != 0
        last_sign[changed] = sign[changed]
    return out


def storage_preserving_schedule(
    base: np.ndarray,
    passive: np.ndarray,
    actuators: pd.DataFrame,
    times: np.ndarray,
    duration_min: int,
) -> np.ndarray:
    out = base.copy()
    roles = actuators.get("storage_control_type", pd.Series("", index=actuators.index)).fillna("").astype(str)
    types = actuators.get("link_type", pd.Series("", index=actuators.index)).fillna("").astype(str).str.lower()
    before_end = times <= float(duration_min)
    after_end = ~before_end
    for j in range(len(actuators)):
        role = roles.iloc[j]
        typ = types.iloc[j]
        if role == "storage_inlet":
            out[before_end, j] = np.maximum(out[before_end, j], 0.85)
            out[after_end, j] = passive[after_end, j]
        elif role == "storage_outlet":
            out[before_end, j] = np.minimum(out[before_end, j], 0.15)
            out[after_end, j] = np.maximum(out[after_end, j], 0.85)
        elif typ == "pump":
            out[before_end, j] = np.minimum(out[before_end, j], 0.25)
            out[after_end, j] = np.maximum(out[after_end, j], 0.90)
    return out


def recession_release_schedule(
    passive: np.ndarray,
    actuators: pd.DataFrame,
    times: np.ndarray,
    duration_min: int,
) -> np.ndarray:
    out = passive.copy()
    roles = actuators.get("storage_control_type", pd.Series("", index=actuators.index)).fillna("").astype(str)
    types = actuators.get("link_type", pd.Series("", index=actuators.index)).fillna("").astype(str).str.lower()
    after = times > float(duration_min)
    for j in range(len(actuators)):
        if roles.iloc[j] == "storage_outlet" or types.iloc[j] == "pump":
            out[after, j] = 1.0
    return out


# ---------------------------------------------------------------------------
# SWMM execution and KPI extraction
# ---------------------------------------------------------------------------


def _import_project_runtime():
    from sewerrtc.io.swmm_mutation import mutate_inp_for_event

    try:
        from sewerrtc.simulation.action_policies import attach_reference_nodes
    except Exception:
        attach_reference_nodes = None
    return mutate_inp_for_event, attach_reference_nodes


def _attach_reference_nodes_if_available(actuators: pd.DataFrame, inp_path: Path) -> pd.DataFrame:
    _, attach = _import_project_runtime()
    if attach is None:
        return actuators
    try:
        return attach(actuators, inp_path)
    except TypeError:
        return attach(actuators=actuators, inp_path=inp_path)


def _load_schedule_for_runtime(
    actuators: pd.DataFrame,
    policy_id: str,
    event: EventSpec,
    settings: OracleSettings,
) -> tuple[np.ndarray, np.ndarray] | None:
    ids = _actuator_ids(actuators)
    times = time_grid(event, settings.control_step_sec)
    schedule_col = f"{policy_id}_schedule_csv"
    if schedule_col in actuators:
        paths = actuators[schedule_col].dropna().astype(str)
        paths = paths[paths.str.strip().ne("")]
        if not paths.empty:
            path = Path(paths.iloc[0])
            frame = pd.read_csv(path)
            time_col = "simtime (hr)" if "simtime (hr)" in frame else "elapsed_min"
            if time_col not in frame:
                raise ValueError(f"Schedule lacks time column: {path}")
            src_t = pd.to_numeric(frame[time_col], errors="raise").to_numpy(float)
            if time_col == "simtime (hr)":
                src_t = src_t * 60.0
            matrix = np.empty((len(times), len(ids)), dtype=float)
            fallback = passive_vector(actuators)
            for j, aid in enumerate(ids):
                if aid not in frame:
                    matrix[:, j] = fallback[j]
                    continue
                vals = pd.to_numeric(frame[aid], errors="coerce").ffill().bfill().fillna(fallback[j]).to_numpy(float)
                idx = np.searchsorted(src_t, times, side="right") - 1
                idx = np.clip(idx, 0, len(src_t) - 1)
                matrix[:, j] = vals[idx]
            return times, np.clip(matrix, 0.0, 1.0)
    if policy_id == "executable_passive":
        matrix = np.tile(passive_vector(actuators), (len(times), 1))
        return times, matrix
    return None


def _run_authoritative_pyswmm(
    *,
    inp_path: Path,
    policy_id: str,
    actuators: pd.DataFrame,
    priority_nodes: list[str],
    detail_path: Path,
    event: EventSpec,
    settings: OracleSettings,
) -> dict[str, Any]:
    """Run a complete event directly with PySWMM.

    Native/no-control branches perform no Python writes. A custom schedule is
    written through Link.target_setting at each control step and actual
    Link.current_setting is recorded for engineering/readback auditing.
    """
    from pyswmm import Links, Nodes, Simulation

    ids = _actuator_ids(actuators)
    schedule = _load_schedule_for_runtime(actuators, policy_id, event, settings)
    records: list[dict[str, Any]] = []
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Simulation(str(inp_path)) as sim:
        sim.step_advance(int(settings.control_step_sec))
        nodes = Nodes(sim)
        links = Links(sim)
        node_objs: dict[str, Any] = {}
        for obj in nodes:
            nid = str(getattr(obj, "nodeid", getattr(obj, "node_id", "")))
            if nid:
                node_objs[nid] = obj
        link_objs: dict[str, Any] = {}
        missing: list[str] = []
        for aid in ids:
            try:
                link_objs[aid] = links[aid]
            except Exception:
                missing.append(aid)
        if missing:
            raise KeyError(f"Actuators not present in SWMM INP: {missing}")

        for _ in sim:
            elapsed_min = (sim.current_time - sim.start_time).total_seconds() / 60.0
            requested: np.ndarray | None = None
            if schedule is not None:
                sched_t, sched_x = schedule
                idx = int(np.searchsorted(sched_t, elapsed_min, side="right") - 1)
                idx = int(np.clip(idx, 0, len(sched_t) - 1))
                requested = sched_x[idx]
                for j, aid in enumerate(ids):
                    link_objs[aid].target_setting = float(np.clip(requested[j], 0.0, 1.0))

            row: dict[str, Any] = {
                "event_id": event.event_id,
                "policy_id": policy_id,
                "elapsed_min": float(elapsed_min),
                "datetime": str(sim.current_time),
            }
            for nid, obj in node_objs.items():
                try:
                    row[f"h:{nid}"] = float(obj.depth)
                except Exception:
                    row[f"h:{nid}"] = np.nan
                try:
                    row[f"flood:{nid}"] = float(obj.flooding)
                except Exception:
                    row[f"flood:{nid}"] = 0.0
                try:
                    row[f"storage_volume:{nid}"] = float(obj.volume)
                except Exception:
                    row[f"storage_volume:{nid}"] = np.nan
            for j, aid in enumerate(ids):
                link = link_objs[aid]
                row[f"a:{aid}"] = float(requested[j]) if requested is not None else np.nan
                try:
                    row[f"setting:{aid}"] = float(link.current_setting)
                except Exception:
                    row[f"setting:{aid}"] = np.nan
                try:
                    row[f"target:{aid}"] = float(link.target_setting)
                except Exception:
                    row[f"target:{aid}"] = np.nan
                try:
                    row[f"flow:{aid}"] = float(link.flow)
                except Exception:
                    row[f"flow:{aid}"] = np.nan
            records.append(row)

    detail = pd.DataFrame(records)
    atomic_write_csv(detail_path, detail)
    kpis = extended_kpis(detail_path, priority_nodes, settings.control_step_sec)
    kpis.update(
        {
            "event_id": event.event_id,
            "policy_id": policy_id,
            "duration_min": event.duration_min,
            "simulation_duration_min": event.simulation_duration_min,
            "rows": len(detail),
            "detail_file": str(detail_path),
            "wall_time_sec": time.time() - t0,
        }
    )
    return kpis

def extended_kpis(
    detail_path: str | Path,
    priority_nodes: Sequence[str],
    control_step_sec: int,
    passive_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    detail = pd.read_csv(detail_path)
    flood_cols = [c for c in detail if c.startswith("flood:")]
    priority = set(priority_nodes)
    pr_cols = [c for c in flood_cols if c.split(":", 1)[1] in priority]
    non_cols = [c for c in flood_cols if c not in pr_cols]
    dt = float(control_step_sec)
    flood = detail[flood_cols].fillna(0.0).to_numpy(float) if flood_cols else np.zeros((len(detail), 0))
    pr = detail[pr_cols].fillna(0.0).to_numpy(float) if pr_cols else np.zeros((len(detail), 0))
    non = detail[non_cols].fillna(0.0).to_numpy(float) if non_cols else np.zeros((len(detail), 0))
    total_rate = flood.sum(axis=1) if flood.size else np.zeros(len(detail))
    pr_rate = pr.sum(axis=1) if pr.size else np.zeros(len(detail))
    non_rate = non.sum(axis=1) if non.size else np.zeros(len(detail))
    node_peaks = flood.max(axis=0) if flood.size else np.zeros(0)
    action_cols = [c for c in detail if c.startswith("setting:")]
    if not action_cols:
        action_cols = [c for c in detail if c.startswith("a:")]
    action = detail[action_cols].ffill().fillna(0.0).to_numpy(float) if action_cols else np.zeros((len(detail), 0))
    diff = np.diff(action, axis=0) if len(action) > 1 else np.zeros((0, action.shape[1]))
    signs = np.sign(np.where(np.abs(diff) > 1e-6, diff, 0.0))
    reversals = 0
    if len(signs) > 1:
        reversals = int(((signs[1:] * signs[:-1]) < 0).sum())
    max_simultaneous = 0
    if passive_matrix is not None and action.size:
        n = min(len(action), len(passive_matrix))
        m = min(action.shape[1], passive_matrix.shape[1])
        max_simultaneous = int((np.abs(action[:n, :m] - passive_matrix[:n, :m]) > 1e-6).sum(axis=1).max(initial=0))
    return {
        "PFV": float(pr_rate.sum() * dt),
        "TFV": float(total_rate.sum() * dt),
        "peak_TFV_rate": float(total_rate.max(initial=0.0)),
        "priority_any_flood_duration_min": float((pr_rate > 1e-9).sum() * dt / 60.0),
        "flood_duration_min": float((total_rate > 1e-9).sum() * dt / 60.0),
        "non_priority_flooding_volume_m3": float(non_rate.sum() * dt),
        "flooded_node_count": int((node_peaks > 1e-9).sum()),
        "max_node_flooding_rate": float(node_peaks.max(initial=0.0)),
        "total_action_changes": int((np.abs(diff) > 1e-6).sum()),
        "unique_acted_facilities": int((np.abs(diff) > 1e-6).any(axis=0).sum()) if diff.size else 0,
        "total_absolute_setting_variation": float(np.abs(diff).sum()),
        "reversals": reversals,
        "max_simultaneous_deviations": max_simultaneous,
    }


def _make_event_inps(base_inp: Path, event: EventSpec, event_dir: Path) -> tuple[Path, Path]:
    mutate_inp_for_event, _ = _import_project_runtime()
    inp_dir = event_dir / "inp"
    inp_dir.mkdir(parents=True, exist_ok=True)
    clean = inp_dir / f"{event.event_id}__clean.inp"
    native = inp_dir / f"{event.event_id}__native.inp"
    if not clean.exists():
        mutate_inp_for_event(
            base_inp,
            event.rainfall_csv,
            clean,
            event.simulation_duration_min,
            strip_controls=True,
        )
    if not native.exists():
        mutate_inp_for_event(
            base_inp,
            event.rainfall_csv,
            native,
            event.simulation_duration_min,
            strip_controls=False,
        )
    return clean, native


def _run_policy(
    *,
    inp_path: Path,
    policy_id: str,
    actuators: pd.DataFrame,
    priority_nodes: list[str],
    detail_path: Path,
    event: EventSpec,
    settings: OracleSettings,
    cfg: Mapping[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    del cfg, runtime_root
    return _run_authoritative_pyswmm(
        inp_path=inp_path,
        policy_id=policy_id,
        actuators=actuators,
        priority_nodes=priority_nodes,
        detail_path=detail_path,
        event=event,
        settings=settings,
    )

def run_references_for_event(job: dict[str, Any]) -> list[dict[str, Any]]:
    event = EventSpec(**job["event"])
    settings = OracleSettings(**job["settings"])
    cfg = job["cfg"]
    root = Path(job["output_root"])
    base_inp = Path(job["base_inp"])
    priority_nodes = list(job["priority_nodes"])
    actuators = pd.read_csv(job["actuators_csv"])
    event_dir = root / "events" / event.event_id
    clean, native = _make_event_inps(base_inp, event, event_dir)
    actuators = _attach_reference_nodes_if_available(actuators, clean)
    rows: list[dict[str, Any]] = []

    references = [
        ("no_control", clean, actuators),
        ("internal_rules", native, actuators),
    ]
    passive_act = actuators.copy()
    passive_act["executable_passive_setting"] = passive_vector(actuators)
    references.append(("executable_passive", clean, passive_act))

    for policy, inp, act in references:
        policy_dir = event_dir / "references" / policy
        detail = policy_dir / f"{event.event_id}__{policy}_detail.csv"
        result_json = policy_dir / "result.json"
        if job["resume"] and detail.exists() and result_json.exists():
            payload = json.loads(result_json.read_text(encoding="utf-8"))
            rows.append(payload)
            continue
        policy_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            kpis = _run_policy(
                inp_path=inp,
                policy_id=policy,
                actuators=act,
                priority_nodes=priority_nodes,
                detail_path=detail,
                event=event,
                settings=settings,
                cfg=cfg,
                runtime_root=policy_dir,
            )
            payload = {
                **kpis,
                "status": "success",
                "event_id": event.event_id,
                "candidate_label": policy,
                "candidate_family": "reference",
                "constraint_mode": "reference",
                "detail_file": str(detail),
                "inp_path": str(inp),
                "inp_sha256": sha256_file(inp),
                "rainfall_sha256": event.rainfall_sha256,
                "wall_time_sec": time.time() - t0,
            }
        except Exception as exc:
            payload = {
                "status": "failed",
                "event_id": event.event_id,
                "candidate_label": policy,
                "candidate_family": "reference",
                "constraint_mode": "reference",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "wall_time_sec": time.time() - t0,
            }
        atomic_write_json(result_json, payload)
        rows.append(payload)
    return rows


# ---------------------------------------------------------------------------
# Candidate planning
# ---------------------------------------------------------------------------


def _candidate_descriptor(
    event: EventSpec,
    label: str,
    family: str,
    source_anchor: str,
    constraint_mode: str,
    schedule_sha: str,
    seed: int,
) -> tuple[str, str]:
    payload = {
        "event_id": event.event_id,
        "label": label,
        "family": family,
        "source_anchor": source_anchor,
        "constraint_mode": constraint_mode,
        "schedule_sha256": schedule_sha,
        "seed": seed,
    }
    case_id = f"orc_{short_hash(payload, 16)}"
    policy_id = f"orc_{short_hash(payload, 10)}"
    return case_id, policy_id


def _add_candidate(
    *,
    candidates: list[CandidateMeta],
    seen_schedule_hashes: set[tuple[str, str]],
    event: EventSpec,
    root: Path,
    times: np.ndarray,
    matrix: np.ndarray,
    actuator_ids: Sequence[str],
    label: str,
    family: str,
    source_anchor: str,
    constraint_mode: str,
    seed: int,
    notes: str = "",
) -> None:
    tmp_payload = {
        "event": event.event_id,
        "matrix": np.round(matrix, 8).tolist(),
        "constraint_mode": constraint_mode,
    }
    matrix_hash = sha256_json(tmp_payload)
    dedup_key = (constraint_mode, matrix_hash)
    if dedup_key in seen_schedule_hashes:
        return
    seen_schedule_hashes.add(dedup_key)
    case_id, policy_id = _candidate_descriptor(
        event, label, family, source_anchor, constraint_mode, matrix_hash, seed
    )
    schedule_path = root / "events" / event.event_id / "schedules" / f"{case_id}.csv"
    schedule_sha = write_schedule(schedule_path, times, matrix, actuator_ids)
    candidates.append(
        CandidateMeta(
            case_id=case_id,
            event_id=event.event_id,
            policy_id=policy_id,
            label=label,
            family=family,
            source_anchor=source_anchor,
            constraint_mode=constraint_mode,
            schedule_csv=str(schedule_path),
            schedule_sha256=schedule_sha,
            candidate_rank=len(candidates),
            seed=seed,
            notes=notes,
        )
    )


def plan_candidates_for_event(
    *,
    event: EventSpec,
    root: Path,
    actuators: pd.DataFrame,
    cfg: Mapping[str, Any],
    engineering_cfg: Mapping[str, Any],
    settings: OracleSettings,
    proposed_detail_root: Path | None,
) -> list[CandidateMeta]:
    ids = _actuator_ids(actuators)
    times = time_grid(event, settings.control_step_sec)
    fallback = passive_vector(actuators)
    passive = np.tile(fallback, (len(times), 1))
    ref_dir = root / "events" / event.event_id / "references"
    no_detail = ref_dir / "no_control" / f"{event.event_id}__no_control_detail.csv"
    internal_detail = ref_dir / "internal_rules" / f"{event.event_id}__internal_rules_detail.csv"
    passive_detail = ref_dir / "executable_passive" / f"{event.event_id}__executable_passive_detail.csv"
    for p in (no_detail, internal_detail, passive_detail):
        if not p.exists():
            raise FileNotFoundError(f"Reference detail missing; run --stage references first: {p}")
    no_control = read_action_matrix(no_detail, times, ids, fallback)
    internal = read_action_matrix(internal_detail, times, ids, fallback)
    passive = read_action_matrix(passive_detail, times, ids, fallback)

    proposed: np.ndarray | None = None
    proposed_path: Path | None = None
    if proposed_detail_root is not None:
        patterns = [
            f"**/{event.event_id}__*proposed*detail.csv",
            f"**/{event.event_id}__proposed_dual_reference_v4_detail.csv",
            f"**/{event.event_id}__proposed_gat_mpc_detail.csv",
        ]
        found: list[Path] = []
        for pattern in patterns:
            found.extend(proposed_detail_root.glob(pattern))
        if found:
            proposed_path = sorted(set(found), key=lambda p: (len(str(p)), str(p)))[0]
            proposed = read_action_matrix(proposed_path, times, ids, fallback)

    candidates: list[CandidateMeta] = []
    seen: set[tuple[str, str]] = set()
    step_min = settings.control_step_sec / 60.0
    binary = binary_pump_ids(actuators, cfg, engineering_cfg)
    rng = np.random.default_rng(settings.seed + int(short_hash(event.event_id, 6), 16))

    raw_specs: list[tuple[str, str, str, np.ndarray, str]] = []
    raw_specs.extend(
        [
            ("hold_passive", "anchor", "passive", passive.copy(), "constant/executable passive trajectory"),
            ("internal_schedule", "anchor", "internal", internal.copy(), "native rules action trajectory replayed on clean INP"),
            ("no_control_schedule", "anchor", "no_control", no_control.copy(), "actual no-control settings replayed on clean INP"),
            ("half_internal_to_passive", "amplitude", "internal", passive + 0.5 * (internal - passive), "50% internal deviation"),
            ("quarter_internal_to_passive", "amplitude", "internal", passive + 0.25 * (internal - passive), "25% internal deviation"),
        ]
    )
    for k in (2, 4, 6, 8):
        raw_specs.append((f"internal_top{k}", "topk", "internal", topk_deviation(internal, passive, k), f"top-{k} deviations"))
    for delay_min in settings.delay_minutes:
        steps = max(1, int(round(delay_min / step_min)))
        raw_specs.append((f"internal_delay_{delay_min}m", "delay", "internal", delay_schedule(internal, passive, steps), "delayed internal action"))
    for hold_min in settings.min_hold_minutes:
        steps = max(1, int(round(hold_min / step_min)))
        raw_specs.append((f"internal_hold_{hold_min}m", "hold", "internal", block_hold(internal, steps), "piecewise held internal action"))
    raw_specs.append(("internal_no_reversal", "smooth", "internal", remove_reversals(internal), "remove immediate reversals"))
    raw_specs.append(("storage_preserving", "hydraulic", "passive", storage_preserving_schedule(passive, passive, actuators, times, event.duration_min), "retain during rainfall, release after"))
    raw_specs.append(("recession_release", "hydraulic", "passive", recession_release_schedule(passive, actuators, times, event.duration_min), "release only after rainfall"))

    if proposed is not None:
        raw_specs.extend(
            [
                ("proposed_replay", "proposed", "proposed", proposed.copy(), f"source={proposed_path}"),
                ("proposed_half", "amplitude", "proposed", passive + 0.5 * (proposed - passive), "50% proposed deviation"),
                ("proposed_top2", "topk", "proposed", topk_deviation(proposed, passive, 2), "proposed top-2"),
                ("proposed_top4", "topk", "proposed", topk_deviation(proposed, passive, 4), "proposed top-4"),
                ("proposed_no_reversal", "smooth", "proposed", remove_reversals(proposed), "proposed without reversal"),
            ]
        )
        for delay_min in settings.delay_minutes:
            steps = max(1, int(round(delay_min / step_min)))
            raw_specs.append((f"proposed_delay_{delay_min}m", "delay", "proposed", delay_schedule(proposed, passive, steps), "delayed proposed"))

    # Single-facility local pulses at four event phases, both directions.
    for frac in settings.single_pulse_fractions:
        center_min = float(frac) * event.duration_min
        start = int(np.argmin(np.abs(times - center_min)))
        stop = min(len(times), start + max(1, settings.pulse_steps))
        for j, aid in enumerate(ids):
            for direction in (-1.0, 1.0):
                mat = passive.copy()
                if aid in binary:
                    mat[start:stop, j] = 0.0 if direction < 0 else 1.0
                else:
                    mat[start:stop, j] = np.clip(mat[start:stop, j] + direction * settings.delta, 0.0, 1.0)
                raw_specs.append(
                    (
                        f"pulse_{aid}_{'down' if direction < 0 else 'up'}_{int(round(center_min))}m",
                        "single_pulse",
                        "passive",
                        mat,
                        f"facility={aid};direction={direction:+.0f};start={start};stop={stop}",
                    )
                )

    # Random sparse, phase-block candidates. This increases interaction coverage.
    phase_blocks = [
        (0.10, 0.35, "rising"),
        (0.35, 0.75, "peak"),
        (0.75, 1.00, "late_rain"),
        (1.00, min(1.0 + event.recession_min / max(event.duration_min, 1), 2.0), "recession"),
    ]
    for r in range(settings.random_candidates_per_event):
        k = int(rng.choice(settings.allowed_k))
        chosen = rng.choice(len(ids), size=min(k, len(ids)), replace=False)
        start_frac, end_frac, phase_name = phase_blocks[r % len(phase_blocks)]
        start_min = start_frac * event.duration_min
        end_min = end_frac * event.duration_min
        mask_t = (times >= start_min) & (times < end_min)
        mat = passive.copy()
        for j in chosen:
            aid = ids[j]
            direction = float(rng.choice([-1.0, 1.0]))
            magnitude = float(rng.choice([settings.delta, 2 * settings.delta]))
            if aid in binary:
                mat[mask_t, j] = 0.0 if direction < 0 else 1.0
            else:
                mat[mask_t, j] = np.clip(mat[mask_t, j] + direction * magnitude, 0.0, 1.0)
        raw_specs.append((f"random_sparse_{r:03d}", "random_sparse", "passive", mat, f"k={k};phase={phase_name}"))

    modes = ["constrained", "relaxed"] if settings.include_relaxed else ["constrained"]
    for label, family, source, raw, notes in raw_specs:
        for mode in modes:
            max_k = max(settings.allowed_k) if mode == "constrained" else None
            projected = project_schedule(
                raw,
                anchor=passive,
                actuators=actuators,
                cfg=cfg,
                engineering_cfg=engineering_cfg,
                constraint_mode=mode,
                max_k=max_k,
            )
            _add_candidate(
                candidates=candidates,
                seen_schedule_hashes=seen,
                event=event,
                root=root,
                times=times,
                matrix=projected,
                actuator_ids=ids,
                label=label,
                family=family,
                source_anchor=source,
                constraint_mode=mode,
                seed=settings.seed,
                notes=notes,
            )

    # Keep deterministic broad coverage when a hard cap is required.
    if len(candidates) > settings.max_cases_per_event:
        frame = pd.DataFrame([asdict(x) for x in candidates])
        mandatory_families = {"anchor", "amplitude", "topk", "delay", "hold", "smooth", "hydraulic", "proposed"}
        mandatory = frame[frame["family"].isin(mandatory_families)]
        remaining = frame.drop(mandatory.index)
        slots = max(0, settings.max_cases_per_event - len(mandatory))
        if slots < len(remaining):
            # Balanced deterministic sampling by family and constraint mode.
            parts: list[pd.DataFrame] = []
            groups = list(remaining.groupby(["family", "constraint_mode"], sort=True))
            cursor = {key: 0 for key, _ in groups}
            while sum(len(p) for p in parts) < slots:
                progressed = False
                for key, group in groups:
                    i = cursor[key]
                    if i < len(group):
                        parts.append(group.iloc[[i]])
                        cursor[key] += 1
                        progressed = True
                        if sum(len(p) for p in parts) >= slots:
                            break
                if not progressed:
                    break
            sampled = pd.concat(parts, ignore_index=False) if parts else remaining.iloc[0:0]
        else:
            sampled = remaining
        frame = pd.concat([mandatory, sampled], ignore_index=True).head(settings.max_cases_per_event)
        candidates = [CandidateMeta(**row) for row in frame.to_dict(orient="records")]
    return candidates


# ---------------------------------------------------------------------------
# Candidate execution
# ---------------------------------------------------------------------------


def run_candidate_job(job: dict[str, Any]) -> dict[str, Any]:
    event = EventSpec(**job["event"])
    meta = CandidateMeta(**job["candidate"])
    settings = OracleSettings(**job["settings"])
    cfg = job["cfg"]
    engineering_cfg = job["engineering_cfg"]
    root = Path(job["output_root"])
    base_inp = Path(job["base_inp"])
    priority_nodes = list(job["priority_nodes"])
    actuators = pd.read_csv(job["actuators_csv"])
    event_dir = root / "events" / event.event_id
    clean, _ = _make_event_inps(base_inp, event, event_dir)
    actuators = _attach_reference_nodes_if_available(actuators, clean)
    case_dir = event_dir / "cases" / meta.case_id
    detail = case_dir / "detail.csv"
    result_json = case_dir / "result.json"
    if job["resume"] and result_json.exists() and detail.exists():
        return json.loads(result_json.read_text(encoding="utf-8"))
    case_dir.mkdir(parents=True, exist_ok=True)
    act = actuators.copy()
    schedule_col = f"{meta.policy_id}_schedule_csv"
    act[schedule_col] = meta.schedule_csv
    t0 = time.time()
    try:
        kpis = _run_policy(
            inp_path=clean,
            policy_id=meta.policy_id,
            actuators=act,
            priority_nodes=priority_nodes,
            detail_path=detail,
            event=event,
            settings=settings,
            cfg=cfg,
            runtime_root=case_dir,
        )
        ids = _actuator_ids(actuators)
        times = time_grid(event, settings.control_step_sec)
        passive = np.tile(passive_vector(actuators), (len(times), 1))
        ext = extended_kpis(detail, priority_nodes, settings.control_step_sec, passive)
        kpis.update(ext)
        payload = {
            **asdict(meta),
            **kpis,
            "status": "success",
            "detail_file": str(detail),
            "inp_path": str(clean),
            "inp_sha256": sha256_file(clean),
            "rainfall_sha256": event.rainfall_sha256,
            "runtime_executed": True,
            "authoritative_swmm": True,
            "wall_time_sec": time.time() - t0,
            "created_at": now_utc_iso(),
        }
    except Exception as exc:
        payload = {
            **asdict(meta),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_executed": False,
            "authoritative_swmm": False,
            "wall_time_sec": time.time() - t0,
            "created_at": now_utc_iso(),
        }
    atomic_write_json(result_json, payload)
    return payload


# ---------------------------------------------------------------------------
# Pareto and feasibility analysis
# ---------------------------------------------------------------------------


def nondominated_mask(values: np.ndarray) -> np.ndarray:
    """Return exact minimisation non-dominated mask for a finite evaluated set."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates_i = np.all(x <= x[i], axis=1) & np.any(x < x[i], axis=1)
        if np.any(dominates_i):
            keep[i] = False
            continue
        dominated_by_i = np.all(x[i] <= x, axis=1) & np.any(x[i] < x, axis=1)
        dominated_by_i[i] = False
        keep[dominated_by_i] = False
    return keep


def _margin(reference: float, abs_margin: float, rel_margin: float) -> float:
    return max(float(abs_margin), abs(float(reference)) * float(rel_margin))


def _normalise_objectives(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    arr = frame[list(columns)].to_numpy(float)
    lo = np.nanmin(arr, axis=0)
    hi = np.nanmax(arr, axis=0)
    scale = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (arr - lo) / scale


def approximate_hypervolume_3d(values: np.ndarray, samples: int = 50_000, seed: int = 20260723) -> float:
    """Monte-Carlo dominated hypervolume after min-max normalisation.

    Used only for convergence diagnostics; Pareto membership itself is exact.
    """
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return 0.0
    lo = np.nanmin(x, axis=0)
    hi = np.nanmax(x, axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    z = np.clip((x - lo) / span, 0.0, 1.0)
    ref = np.full(z.shape[1], 1.05)
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, ref, size=(samples, z.shape[1]))
    dominated = np.zeros(samples, dtype=bool)
    for row in z:
        dominated |= np.all(points >= row, axis=1)
    return float(dominated.mean() * np.prod(ref))


def classify_event(frame: pd.DataFrame) -> str:
    if bool(frame["strict_feasible"].any()):
        return "feasible_found"
    pfv_ok = frame["pfv_feasible"].any()
    perf_ok = (frame["tfv_feasible"] & frame["peak_feasible"]).any()
    relaxed_ok = bool(frame.loc[frame["constraint_mode"].eq("relaxed"), "strict_feasible"].any())
    constrained_ok = bool(frame.loc[frame["constraint_mode"].eq("constrained"), "strict_feasible"].any())
    if relaxed_ok and not constrained_ok:
        return "operational_constraints_block_feasibility"
    if pfv_ok and not perf_ok:
        return "pfv_safe_but_internal_performance_unreachable"
    if perf_ok and not pfv_ok:
        return "internal_performance_reachable_but_pfv_unsafe"
    if pfv_ok and perf_ok:
        return "objectives_reachable_separately_not_jointly"
    return "no_feasible_neighbourhood_solution"


def analyze_event(
    event_id: str,
    result_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    settings: OracleSettings,
    event_out: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    refs = reference_frame[reference_frame["event_id"].eq(event_id)].set_index("candidate_label")
    required = {"no_control", "internal_rules", "executable_passive"}
    if not required.issubset(set(refs.index)):
        raise ValueError(f"Missing references for {event_id}: {sorted(required - set(refs.index))}")
    no = refs.loc["no_control"]
    internal = refs.loc["internal_rules"]
    passive = refs.loc["executable_passive"]
    pfv_safe_ref = min(float(no["PFV"]), float(passive["PFV"]))
    pfv_limit = pfv_safe_ref + _margin(pfv_safe_ref, settings.pfv_abs_margin_m3, settings.pfv_rel_margin)
    tfv_ref = float(internal["TFV"])
    tfv_limit = tfv_ref + _margin(tfv_ref, settings.tfv_abs_margin_m3, settings.tfv_rel_margin)
    peak_ref = float(internal["peak_TFV_rate"])
    peak_limit = peak_ref + _margin(peak_ref, settings.peak_abs_margin, settings.peak_rel_margin)

    f = result_frame[(result_frame["event_id"].eq(event_id)) & result_frame["status"].eq("success")].copy()
    if f.empty:
        raise ValueError(f"No successful candidates for {event_id}")
    f["pfv_safety_reference"] = pfv_safe_ref
    f["pfv_limit"] = pfv_limit
    f["tfv_internal_reference"] = tfv_ref
    f["tfv_limit"] = tfv_limit
    f["peak_internal_reference"] = peak_ref
    f["peak_limit"] = peak_limit
    f["delta_PFV_vs_safety_envelope"] = f["PFV"] - pfv_safe_ref
    f["delta_TFV_vs_internal"] = f["TFV"] - tfv_ref
    f["delta_peak_vs_internal"] = f["peak_TFV_rate"] - peak_ref
    f["pfv_feasible"] = f["PFV"] <= pfv_limit + 1e-9
    f["tfv_feasible"] = f["TFV"] <= tfv_limit + 1e-9
    f["peak_feasible"] = f["peak_TFV_rate"] <= peak_limit + 1e-9
    if settings.nonpriority_abs_margin_m3 is not None:
        non_ref = float(internal.get("non_priority_flooding_volume_m3", np.nan))
        f["nonpriority_feasible"] = f["non_priority_flooding_volume_m3"] <= non_ref + settings.nonpriority_abs_margin_m3
    else:
        f["nonpriority_feasible"] = True
    f["strict_feasible"] = f[["pfv_feasible", "tfv_feasible", "peak_feasible", "nonpriority_feasible"]].all(axis=1)

    denom_pfv = max(abs(pfv_safe_ref), 1.0)
    denom_tfv = max(abs(tfv_ref), 1.0)
    denom_peak = max(abs(peak_ref), 1e-6)
    f["normalised_constraint_violation"] = (
        np.maximum(0.0, f["PFV"] - pfv_limit) / denom_pfv
        + np.maximum(0.0, f["TFV"] - tfv_limit) / denom_tfv
        + np.maximum(0.0, f["peak_TFV_rate"] - peak_limit) / denom_peak
    )
    objectives3 = ["PFV", "TFV", "peak_TFV_rate"]
    objectives4 = [*objectives3, "total_absolute_setting_variation"]
    f["pareto_3d"] = nondominated_mask(f[objectives3].to_numpy(float))
    f["pareto_4d"] = nondominated_mask(f[objectives4].to_numpy(float))
    norm = _normalise_objectives(f, objectives4)
    f["distance_to_ideal"] = np.sqrt((norm**2).sum(axis=1))

    feasible = f[f["strict_feasible"]].copy()
    if not feasible.empty:
        best_idx = feasible.sort_values(
            ["TFV", "peak_TFV_rate", "PFV", "total_absolute_setting_variation"]
        ).index[0]
        knee_idx = feasible.sort_values("distance_to_ideal").index[0]
    else:
        best_idx = f.sort_values(
            ["normalised_constraint_violation", "distance_to_ideal"]
        ).index[0]
        knee_idx = best_idx
    f["selected_constrained_oracle"] = False
    f.loc[best_idx, "selected_constrained_oracle"] = True
    f["selected_knee"] = False
    f.loc[knee_idx, "selected_knee"] = True

    # Convergence diagnostic by deterministic candidate rank.
    ordered = f.sort_values("candidate_rank").reset_index(drop=True)
    checkpoints: list[dict[str, Any]] = []
    step = max(10, len(ordered) // 10)
    for n in sorted(set([*range(step, len(ordered) + 1, step), len(ordered)])):
        sub = ordered.head(n)
        front = sub[nondominated_mask(sub[objectives3].to_numpy(float))]
        hv = approximate_hypervolume_3d(front[objectives3].to_numpy(float), samples=10_000, seed=settings.seed + n)
        checkpoints.append(
            {
                "candidate_count": n,
                "pareto_count": len(front),
                "strict_feasible_count": int(sub["strict_feasible"].sum()),
                "hypervolume_approx": hv,
            }
        )
    conv = pd.DataFrame(checkpoints)
    if len(conv) >= 2:
        tail_start = max(0, int(math.floor(len(conv) * (1.0 - settings.convergence_tail_fraction))) - 1)
        hv0 = float(conv.iloc[tail_start]["hypervolume_approx"])
        hv1 = float(conv.iloc[-1]["hypervolume_approx"])
        rel_improvement = (hv1 - hv0) / max(abs(hv0), 1e-9)
    else:
        rel_improvement = float("inf")
    converged = bool(
        len(f) >= settings.convergence_min_candidates
        and rel_improvement <= settings.convergence_hv_relative_tol
    )

    event_class = classify_event(f)
    selected = f.loc[best_idx]
    summary = {
        "event_id": event_id,
        "candidate_count": int(len(f)),
        "constrained_candidate_count": int(f["constraint_mode"].eq("constrained").sum()),
        "relaxed_candidate_count": int(f["constraint_mode"].eq("relaxed").sum()),
        "pareto_3d_count": int(f["pareto_3d"].sum()),
        "pareto_4d_count": int(f["pareto_4d"].sum()),
        "strict_feasible_count": int(f["strict_feasible"].sum()),
        "strict_feasible_constrained_count": int((f["strict_feasible"] & f["constraint_mode"].eq("constrained")).sum()),
        "strict_feasible_relaxed_count": int((f["strict_feasible"] & f["constraint_mode"].eq("relaxed")).sum()),
        "event_feasibility_class": event_class,
        "search_converged": converged,
        "hypervolume_tail_relative_improvement": rel_improvement,
        "PFV_no_control": float(no["PFV"]),
        "PFV_passive": float(passive["PFV"]),
        "PFV_safety_reference": pfv_safe_ref,
        "PFV_limit": pfv_limit,
        "TFV_internal": tfv_ref,
        "TFV_limit": tfv_limit,
        "peak_internal": peak_ref,
        "peak_limit": peak_limit,
        "oracle_case_id": str(selected["case_id"]),
        "oracle_label": str(selected["label"]),
        "oracle_family": str(selected["family"]),
        "oracle_constraint_mode": str(selected["constraint_mode"]),
        "oracle_PFV": float(selected["PFV"]),
        "oracle_TFV": float(selected["TFV"]),
        "oracle_peak": float(selected["peak_TFV_rate"]),
        "oracle_action_variation": float(selected["total_absolute_setting_variation"]),
        "oracle_normalised_violation": float(selected["normalised_constraint_violation"]),
        "created_at": now_utc_iso(),
    }

    event_out.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(event_out / "all_candidates_with_feasibility.csv", f.sort_values(["pareto_3d", "strict_feasible", "distance_to_ideal"], ascending=[False, False, True]))
    atomic_write_csv(event_out / "pareto_3d.csv", f[f["pareto_3d"]].sort_values("PFV"))
    atomic_write_csv(event_out / "pareto_4d.csv", f[f["pareto_4d"]].sort_values("PFV"))
    atomic_write_csv(event_out / "convergence.csv", conv)
    atomic_write_json(event_out / "event_feasibility_summary.json", summary)
    make_event_plots(f, summary, event_out)
    return f, summary


def make_event_plots(frame: pd.DataFrame, summary: Mapping[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feasible = frame["strict_feasible"].to_numpy(bool)
    pareto = frame["pareto_3d"].to_numpy(bool)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(frame["PFV"], frame["TFV"], c=frame["peak_TFV_rate"], s=np.where(pareto, 65, 25), alpha=0.75)
    if feasible.any():
        ax.scatter(frame.loc[feasible, "PFV"], frame.loc[feasible, "TFV"], facecolors="none", edgecolors="black", s=100, linewidths=1.2, label="Strict feasible")
    ax.axvline(float(summary["PFV_limit"]), linestyle="--", linewidth=1.2, label="PFV limit")
    ax.axhline(float(summary["TFV_limit"]), linestyle="--", linewidth=1.2, label="TFV limit")
    ax.set_xlabel("PFV (m³)")
    ax.set_ylabel("TFV (m³)")
    ax.set_title(str(summary["event_id"]))
    fig.colorbar(sc, ax=ax, label="Peak TFV rate")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_pfv_tfv.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(frame["PFV"], frame["peak_TFV_rate"], c=frame["TFV"], s=np.where(pareto, 65, 25), alpha=0.75)
    if feasible.any():
        ax.scatter(frame.loc[feasible, "PFV"], frame.loc[feasible, "peak_TFV_rate"], facecolors="none", edgecolors="black", s=100, linewidths=1.2, label="Strict feasible")
    ax.axvline(float(summary["PFV_limit"]), linestyle="--", linewidth=1.2, label="PFV limit")
    ax.axhline(float(summary["peak_limit"]), linestyle="--", linewidth=1.2, label="Peak limit")
    ax.set_xlabel("PFV (m³)")
    ax.set_ylabel("Peak TFV rate")
    ax.set_title(str(summary["event_id"]))
    fig.colorbar(sc, ax=ax, label="TFV (m³)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_pfv_peak.png", dpi=220)
    plt.close(fig)


def _bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int = 5000) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate_analysis(summaries: list[dict[str, Any]], root: Path, settings: OracleSettings) -> dict[str, Any]:
    frame = pd.DataFrame(summaries)
    frame["oracle_delta_PFV_vs_safe"] = frame["oracle_PFV"] - frame["PFV_safety_reference"]
    frame["oracle_delta_TFV_vs_internal"] = frame["oracle_TFV"] - frame["TFV_internal"]
    frame["oracle_delta_peak_vs_internal"] = frame["oracle_peak"] - frame["peak_internal"]
    feasible = frame["strict_feasible_count"] > 0
    constrained = frame["strict_feasible_constrained_count"] > 0
    relaxed = frame["strict_feasible_relaxed_count"] > 0
    ci_pfv = _bootstrap_mean_ci(frame["oracle_delta_PFV_vs_safe"].to_numpy(float), settings.seed)
    ci_tfv = _bootstrap_mean_ci(frame["oracle_delta_TFV_vs_internal"].to_numpy(float), settings.seed + 1)
    ci_peak = _bootstrap_mean_ci(frame["oracle_delta_peak_vs_internal"].to_numpy(float), settings.seed + 2)
    report = {
        "status": "pass" if len(frame) > 0 else "blocked",
        "event_count": int(len(frame)),
        "any_feasible_event_count": int(feasible.sum()),
        "any_feasible_event_fraction": float(feasible.mean()),
        "constrained_feasible_event_count": int(constrained.sum()),
        "constrained_feasible_event_fraction": float(constrained.mean()),
        "relaxed_feasible_event_count": int(relaxed.sum()),
        "relaxed_feasible_event_fraction": float(relaxed.mean()),
        "search_converged_event_count": int(frame["search_converged"].sum()),
        "search_converged_event_fraction": float(frame["search_converged"].mean()),
        "event_class_counts": frame["event_feasibility_class"].value_counts().to_dict(),
        "mean_oracle_delta_PFV_vs_safe": float(frame["oracle_delta_PFV_vs_safe"].mean()),
        "mean_oracle_delta_TFV_vs_internal": float(frame["oracle_delta_TFV_vs_internal"].mean()),
        "mean_oracle_delta_peak_vs_internal": float(frame["oracle_delta_peak_vs_internal"].mean()),
        "bootstrap95_mean_delta_PFV_vs_safe": list(ci_pfv),
        "bootstrap95_mean_delta_TFV_vs_internal": list(ci_tfv),
        "bootstrap95_mean_delta_peak_vs_internal": list(ci_peak),
        "interpretation": {
            "controller_or_model_problem": "constrained oracle feasible but Proposed is not feasible",
            "operational_constraint_problem": "relaxed oracle feasible but constrained oracle is not",
            "actuator_or_objective_conflict": "neither relaxed nor constrained oracle feasible after converged broad search",
            "inconclusive": "no feasible solution and search convergence/coverage is insufficient",
        },
        "created_at": now_utc_iso(),
    }
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(out / "event_feasibility_summary.csv", frame)
    atomic_write_json(out / "aggregate_feasibility_report.json", report)
    make_aggregate_plot(frame, out)
    return report


def make_aggregate_plot(frame: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = frame["event_feasibility_class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Feasibility class")
    ax.set_ylabel("Event count")
    ax.set_title("Oracle/Pareto event feasibility classification")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_feasibility_classes.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage orchestration
# ---------------------------------------------------------------------------


def collect_reference_results(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in root.glob("events/*/references/*/result.json"):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return pd.DataFrame(rows)


def collect_candidate_results(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in root.glob("events/*/cases/*/result.json"):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return pd.DataFrame(rows)


def run_jobs(jobs: list[dict[str, Any]], worker, workers: int, label: str) -> list[Any]:
    if workers <= 1:
        out = []
        for i, job in enumerate(jobs, 1):
            print(f"[{label}] {i}/{len(jobs)}")
            out.append(worker(job))
        return out
    out: list[Any] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            out.append(result)
            print(f"[{label}] completed {i}/{len(jobs)}")
    return out


def stage_references(
    *,
    events: list[EventSpec],
    root: Path,
    base_inp: Path,
    actuators_csv: Path,
    priority_nodes: list[str],
    settings: OracleSettings,
    cfg: dict[str, Any],
    workers: int,
    resume: bool,
) -> int:
    jobs = [
        {
            "event": asdict(event),
            "settings": asdict(settings),
            "cfg": cfg,
            "output_root": str(root),
            "base_inp": str(base_inp),
            "actuators_csv": str(actuators_csv),
            "priority_nodes": priority_nodes,
            "resume": resume,
        }
        for event in events
    ]
    nested = run_jobs(jobs, run_references_for_event, workers, "references")
    rows = [x for group in nested for x in group]
    frame = pd.DataFrame(rows)
    atomic_write_csv(root / "reference_results.csv", frame)
    failed = frame[~frame["status"].eq("success")] if not frame.empty else frame
    report = {
        "status": "pass" if failed.empty and len(frame) == 3 * len(events) else "failed",
        "event_count": len(events),
        "expected_reference_rows": 3 * len(events),
        "actual_reference_rows": len(frame),
        "failed_rows": len(failed),
        "created_at": now_utc_iso(),
    }
    atomic_write_json(root / "reference_audit.json", report)
    return 0 if report["status"] == "pass" else 4


def stage_plan(
    *,
    events: list[EventSpec],
    root: Path,
    actuators_csv: Path,
    cfg: dict[str, Any],
    engineering_cfg: dict[str, Any],
    settings: OracleSettings,
    proposed_detail_root: Path | None,
) -> int:
    actuators = pd.read_csv(actuators_csv)
    all_candidates: list[CandidateMeta] = []
    event_counts: dict[str, int] = {}
    for event in events:
        planned = plan_candidates_for_event(
            event=event,
            root=root,
            actuators=actuators,
            cfg=cfg,
            engineering_cfg=engineering_cfg,
            settings=settings,
            proposed_detail_root=proposed_detail_root,
        )
        all_candidates.extend(planned)
        event_counts[event.event_id] = len(planned)
    frame = pd.DataFrame([asdict(x) for x in all_candidates])
    plan_path = root / "oracle_case_plan.csv"
    atomic_write_csv(plan_path, frame)
    audit = {
        "status": "pass" if len(frame) > 0 and frame["case_id"].is_unique else "blocked",
        "event_count": len(events),
        "planned_case_count": len(frame),
        "cases_per_event": event_counts,
        "family_counts": frame["family"].value_counts().to_dict() if not frame.empty else {},
        "constraint_mode_counts": frame["constraint_mode"].value_counts().to_dict() if not frame.empty else {},
        "unique_case_ids": int(frame["case_id"].nunique()) if not frame.empty else 0,
        "unique_schedule_hashes": int(frame["schedule_sha256"].nunique()) if not frame.empty else 0,
        "actuator_count": len(actuators),
        "created_at": now_utc_iso(),
    }
    atomic_write_json(root / "oracle_case_plan_audit.json", audit)
    return 0 if audit["status"] == "pass" else 3


def stage_run(
    *,
    events: list[EventSpec],
    root: Path,
    base_inp: Path,
    actuators_csv: Path,
    priority_nodes: list[str],
    settings: OracleSettings,
    cfg: dict[str, Any],
    engineering_cfg: dict[str, Any],
    workers: int,
    resume: bool,
) -> int:
    plan_path = root / "oracle_case_plan.csv"
    if not plan_path.exists():
        raise FileNotFoundError("Missing oracle_case_plan.csv; run --stage plan")
    plan = pd.read_csv(plan_path)
    event_map = {e.event_id: e for e in events}
    plan = plan[plan["event_id"].isin(event_map)].copy()
    jobs = []
    for row in plan.to_dict(orient="records"):
        event = event_map[str(row["event_id"])]
        jobs.append(
            {
                "event": asdict(event),
                "candidate": row,
                "settings": asdict(settings),
                "cfg": cfg,
                "engineering_cfg": engineering_cfg,
                "output_root": str(root),
                "base_inp": str(base_inp),
                "actuators_csv": str(actuators_csv),
                "priority_nodes": priority_nodes,
                "resume": resume,
            }
        )
    results = run_jobs(jobs, run_candidate_job, workers, "oracle-run")
    frame = pd.DataFrame(results)
    atomic_write_csv(root / "oracle_case_results.csv", frame)
    failed = frame[~frame["status"].eq("success")] if not frame.empty else frame
    audit = {
        "status": "pass" if len(frame) == len(plan) and failed.empty else "partial",
        "planned": len(plan),
        "completed": len(frame),
        "success": int(frame["status"].eq("success").sum()) if not frame.empty else 0,
        "failed": len(failed),
        "failure_reasons": failed.get("error", pd.Series(dtype=str)).value_counts().to_dict(),
        "created_at": now_utc_iso(),
    }
    atomic_write_json(root / "oracle_generation_audit.json", audit)
    return 0 if audit["status"] == "pass" else 4


def stage_analyze(root: Path, events: list[EventSpec], settings: OracleSettings) -> int:
    refs = collect_reference_results(root)
    results = collect_candidate_results(root)
    if refs.empty or results.empty:
        raise RuntimeError("Reference or candidate results are empty")
    refs = refs[refs["status"].eq("success")].copy()
    summaries: list[dict[str, Any]] = []
    enriched: list[pd.DataFrame] = []
    for event in events:
        event_out = root / "analysis" / "events" / event.event_id
        f, summary = analyze_event(event.event_id, results, refs, settings, event_out)
        enriched.append(f)
        summaries.append(summary)
    all_enriched = pd.concat(enriched, ignore_index=True)
    atomic_write_csv(root / "analysis" / "all_event_candidate_feasibility.csv", all_enriched)
    report = aggregate_analysis(summaries, root, settings)
    return 0 if report["status"] == "pass" else 5


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="V4 dual-reference YAML")
    p.add_argument("--engineering-config", default="configs/wuhan_project6_engineering36.yaml")
    p.add_argument("--stage", choices=["references", "plan", "run", "analyze", "all"], default="all")
    p.add_argument("--events-csv", default="")
    p.add_argument("--event-ids", default="")
    p.add_argument("--event-limit", type=int, default=0)
    p.add_argument("--allowed-splits", default="development,development_v4,train")
    p.add_argument("--inp", default="")
    p.add_argument("--actuators-csv", default="")
    p.add_argument("--priority-nodes", default="")
    p.add_argument("--proposed-detail-root", default="")
    p.add_argument("--output-root", default="outputs/project6_dual_reference_v4/oracle_pareto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=20260723)
    p.add_argument("--control-step-sec", type=int, default=600)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--pulse-steps", type=int, default=2)
    p.add_argument("--max-cases-per-event", type=int, default=450)
    p.add_argument("--random-candidates-per-event", type=int, default=48)
    p.add_argument("--no-relaxed", action="store_true")
    p.add_argument("--clean-output", action="store_true", help="Delete oracle output before running; incompatible with --resume")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml_with_inheritance(args.config)
    engineering_cfg = load_yaml_with_inheritance(args.engineering_config)
    root = Path(str(cfg.get("project_root", PROJECT_ROOT))).resolve()
    output_root = resolve_path(root, args.output_root)
    assert output_root is not None
    if args.clean_output and args.resume:
        raise ValueError("--clean-output and --resume cannot be used together")
    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    settings = OracleSettings(
        control_step_sec=int(args.control_step_sec),
        recession_min=int(nested_get(engineering_cfg, "experiment.recession_min", 180) or 180),
        seed=int(args.seed),
        delta=float(args.delta),
        pulse_steps=int(args.pulse_steps),
        max_cases_per_event=int(args.max_cases_per_event),
        random_candidates_per_event=int(args.random_candidates_per_event),
        include_relaxed=not bool(args.no_relaxed),
        path_budget_chars=int(nested_get(cfg, "runtime_limits.path_budget_chars", 235) or 235),
        pfv_abs_margin_m3=float(nested_get(cfg, "v4.dual_reference.pfv_abs_margin_m3", 0.0) or 0.0),
        pfv_rel_margin=float(nested_get(cfg, "v4.dual_reference.pfv_rel_margin", 0.0) or 0.0),
        tfv_abs_margin_m3=float(nested_get(cfg, "v4.dual_reference.tfv_abs_margin_m3", 0.0) or 0.0),
        tfv_rel_margin=float(nested_get(cfg, "v4.dual_reference.tfv_rel_margin", 0.0) or 0.0),
        peak_abs_margin=float(nested_get(cfg, "v4.dual_reference.peak_abs_margin", 0.0) or 0.0),
        peak_rel_margin=float(nested_get(cfg, "v4.dual_reference.peak_rel_margin", 0.0) or 0.0),
    )
    events_path = discover_event_table(cfg, engineering_cfg, args.events_csv or None)
    event_table = _normalise_event_table(pd.read_csv(events_path), root, settings.recession_min)
    events = select_events(
        event_table,
        args.event_ids or None,
        int(args.event_limit),
        [x.strip() for x in args.allowed_splits.split(",") if x.strip()],
    )
    base_inp = discover_base_inp(cfg, engineering_cfg, args.inp or None)
    actuators_csv = discover_actuator_csv(cfg, args.actuators_csv or None)
    priority_nodes = load_priority_nodes(cfg, engineering_cfg, args.priority_nodes or None)
    proposed_root = resolve_path(root, args.proposed_detail_root) if args.proposed_detail_root else None

    run_manifest = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(Path(args.config).resolve()),
        "engineering_config": str(Path(args.engineering_config).resolve()),
        "engineering_config_sha256": sha256_file(Path(args.engineering_config).resolve()),
        "base_inp": str(base_inp),
        "base_inp_sha256": sha256_file(base_inp),
        "actuators_csv": str(actuators_csv),
        "actuators_sha256": sha256_file(actuators_csv),
        "events_csv": str(events_path),
        "events_csv_sha256": sha256_file(events_path),
        "events": [asdict(x) for x in events],
        "priority_nodes": priority_nodes,
        "settings": asdict(settings),
        "stage": args.stage,
        "workers": args.workers,
        "resume": args.resume,
        "created_at": now_utc_iso(),
    }
    atomic_write_json(output_root / "oracle_run_manifest.json", run_manifest)

    stages = [args.stage] if args.stage != "all" else ["references", "plan", "run", "analyze"]
    for stage in stages:
        print(f"\n[oracle-pareto] stage={stage}")
        if stage == "references":
            code = stage_references(
                events=events,
                root=output_root,
                base_inp=base_inp,
                actuators_csv=actuators_csv,
                priority_nodes=priority_nodes,
                settings=settings,
                cfg=cfg,
                workers=max(1, int(args.workers)),
                resume=args.resume,
            )
        elif stage == "plan":
            code = stage_plan(
                events=events,
                root=output_root,
                actuators_csv=actuators_csv,
                cfg=cfg,
                engineering_cfg=engineering_cfg,
                settings=settings,
                proposed_detail_root=proposed_root,
            )
        elif stage == "run":
            code = stage_run(
                events=events,
                root=output_root,
                base_inp=base_inp,
                actuators_csv=actuators_csv,
                priority_nodes=priority_nodes,
                settings=settings,
                cfg=cfg,
                engineering_cfg=engineering_cfg,
                workers=max(1, int(args.workers)),
                resume=args.resume,
            )
        elif stage == "analyze":
            code = stage_analyze(output_root, events, settings)
        else:
            raise AssertionError(stage)
        print(f"[oracle-pareto] stage={stage} exit={code}")
        if code != 0:
            return int(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
