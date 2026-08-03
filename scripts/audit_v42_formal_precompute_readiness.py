"""Bounded-memory precompute audit for the existing Formal F2 artifacts.

This script reads manifests first and detail CSVs one at a time.  It never
trains a model, runs SWMM, or treats missing hydraulic targets as zeros.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.formal_f2 import sha256_file, text, yes
from sewerrtc.v4.v42_fast_feasibility import _kpis
from sewerrtc.v4.v42_step1_dataset import _detail_extract_window, _build_usecols
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology, _parse_inp_topology
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices


ROLES = ("candidate", "no_control", "dynamic_internal", "hold_previous")
RAW_GATE_COLUMNS = (
    "training_admission_authorized",
    "raw_independent_oracle_all_pass",
    "same_state_raw_verified",
    "same_forcing_raw_verified",
    "actual_readback_verified",
    "h120_window_complete",
    "kpi_recompute_ok",
)
LABEL_COLUMNS = ("pfv_delta", "tfv_delta", "peak_delta")
TARGET_NAMES = (
    "node_depth",
    "node_flooding_rate",
    "storage_volume",
    "managed_facility_flow",
    "outfall_flow",
)


def _nonempty(value: Any) -> str:
    return text(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _read_parquet(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    cols = list(columns) if columns is not None else None
    if cols is not None:
        cols = [c for c in cols if c in pf.schema.names]
    return pf.read(columns=cols).to_pandas()


def _stats(values: Iterable[float]) -> dict[str, Any]:
    a = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=float)
    if not a.size:
        return {"count": 0}
    q = np.percentile(a, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    mean, std = float(a.mean()), float(a.std())
    return {
        "count": int(a.size),
        "min": float(a.min()),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p10": float(q[2]),
        "p25": float(q[3]),
        "p50": float(q[4]),
        "p75": float(q[5]),
        "p90": float(q[6]),
        "p95": float(q[7]),
        "p99": float(q[8]),
        "max": float(a.max()),
        "mean": mean,
        "std": std,
        "cv": float(std / abs(mean)) if abs(mean) > 1e-15 else None,
    }


def _weighted_group_stats(frame: pd.DataFrame, group: str, weight: str) -> dict[str, Any]:
    counts = frame.groupby(group, dropna=False).size().astype(float)
    if counts.empty:
        return {"groups": 0, "effective_group_count": 0.0}
    total = float(counts.sum())
    neff = total * total / float((counts * counts).sum())
    ordered = counts.sort_values(ascending=False)
    top_n = max(1, int(math.ceil(len(ordered) * 0.10)))
    return {
        "groups": int(len(counts)),
        "weight_stats": _stats(counts.tolist()),
        "effective_group_count": float(neff),
        "max_single_group_share": float(ordered.iloc[0] / total),
        "top_10_percent_group_share": float(ordered.iloc[:top_n].sum() / total),
        "weight_column": weight,
    }


def _hash_json_array(raw: Any) -> str:
    try:
        a = np.asarray(json.loads(str(raw)), dtype=np.float64)
        if a.ndim == 1 and a.size % 36 == 0:
            a = a.reshape(-1, 36)
        if a.ndim != 2:
            return ""
        return hashlib.sha256(np.ascontiguousarray(a[:3]).tobytes()).hexdigest()
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""


def _array(raw: Any) -> np.ndarray:
    return np.asarray(json.loads(str(raw)), dtype=np.float64)


def _detail_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [str(x).strip() for x in next(csv.reader(handle))]


def _read_detail(path: Path, columns: list[str], chunksize: int | None = None) -> pd.DataFrame:
    header = _detail_header(path)
    missing = [c for c in columns if c not in set(header)]
    if missing:
        raise KeyError(f"missing required columns: {missing[:10]}")
    return pd.read_csv(path, usecols=columns, low_memory=False, chunksize=chunksize)  # type: ignore[return-value]


def _target_columns(node_ids: list[str], storage_ids: list[str], facility_ids: list[str], outfall_ids: list[str]) -> dict[str, list[str]]:
    return {
        "node_depth": [f"h:{x}" for x in node_ids],
        "node_flooding_rate": [f"flood:{x}" for x in node_ids],
        "storage_volume": [f"storage_volume:{x}" for x in storage_ids],
        "managed_facility_flow": [f"flow:{x}" for x in facility_ids],
        "outfall_flow": [f"outfall_flow:{x}" for x in outfall_ids],
    }


def audit_identity(root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    tables: dict[str, pd.DataFrame] = {}
    tables["step1"] = _read_parquet(paths["step1"], ["rainfall_sha256", "rainfall_fingerprint", "rainfall_group_key", "split_group_key", "event_id"])
    tables["raw"] = _read_parquet(paths["raw"], ["rainfall_sha256", "rainfall_fingerprint", "rainfall_group_key", "split_group_key", "event_id"])
    tables["ledger"] = pd.read_csv(paths["ledger"], low_memory=False)
    conflicts = []
    authority_cols = ["rainfall_sha256", "rainfall_fingerprint", "rainfall_group_key", "split_group_key"]
    for name, frame in tables.items():
        present = [c for c in authority_cols if c in frame.columns]
        for i, row in frame.iterrows():
            vals = {c: _nonempty(row.get(c, "")) for c in present if _nonempty(row.get(c, ""))}
            if len(set(vals.values())) > 1:
                conflicts.append({"table": name, "row": int(i), "values": vals})

    event_to_rain: defaultdict[str, set[str]] = defaultdict(set)
    rain_to_event: defaultdict[str, set[str]] = defaultdict(set)
    for _, row in tables["ledger"].iterrows():
        rain = _nonempty(row.get("rainfall_sha256", row.get("rainfall_group_key", "")))
        raw_ids = [row.get("inventory_event_id", "")]
        try:
            raw_ids.extend(json.loads(str(row.get("historical_event_ids", "[]"))))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        for event in {_nonempty(x) for x in raw_ids if _nonempty(x)}:
            event_to_rain[event].add(rain)
            rain_to_event[rain].add(event)
    roles = {}
    for role in ("train", "calibration", "locked_validation", "challenge", "formal_blind"):
        roles[role] = set(
            tables["ledger"].loc[
                tables["ledger"].get("formal_f2_role", "").astype(str).eq(role), "rainfall_group_key"
            ].astype(str)
        )
    overlaps = {}
    role_names = list(roles)
    for i, left in enumerate(role_names):
        for right in role_names[i + 1 :]:
            overlaps[f"{left}__{right}"] = len(roles[left] & roles[right])
    return {
        "authority_columns_present": {k: [c for c in authority_cols if c in v.columns] for k, v in tables.items()},
        "missing_authority_columns": {k: [c for c in authority_cols if c not in v.columns] for k, v in tables.items()},
        "identity_conflicts": conflicts[:200],
        "identity_conflict_count": len(conflicts),
        "rainfall_sha_to_event_ids_count": {k: len(v) for k, v in rain_to_event.items() if k},
        "event_id_to_rainfall_sha_count": {k: len(v) for k, v in event_to_rain.items() if k},
        "event_rainfall_collision_count": sum(len(v) > 1 for v in event_to_rain.values()),
        "split_group_counts": {k: len(v) for k, v in roles.items()},
        "split_overlap": overlaps,
        "evaluation_counts": {k: len(roles[k]) for k in ("calibration", "locked_validation", "challenge", "formal_blind")},
        "reserved_event_ids_without_rainfall_group": int(json.loads(paths["prepare_audit"].read_text(encoding="utf-8")).get("reserved_audit", {}).get("reserved_event_ids_without_rainfall_group", -1)),
    }


def audit_step1(root: Path, paths: dict[str, Path], max_semantic_files: int) -> tuple[dict[str, Any], dict[str, Any]]:
    cols = ["physical_identity_sha256", "detail_path", "event_id", "rainfall_sha256", "split_group_key", "anchor_min", "history_start_min", "history_end_min", "frame_count", "frame_interval_min", "formal_split", "step1_domain_role", "source_dataset", "future_hydraulic_truth_in_input"]
    frame = _read_parquet(paths["step1"], cols)
    graph = _load_graph_topology(root)
    usecols = _build_usecols([str(x) for x in graph["node_ids"]], [str(x) for x in graph["facility_ids"]])
    required_set = set(usecols)
    role_summary = {}
    for role in ("train", "validation", "auxiliary"):
        sub = frame[frame["formal_split"].astype(str).eq(role)]
        role_summary[role] = {"rows": len(sub), "physical_runs": int(sub["physical_identity_sha256"].nunique()), "rainfall_groups": int(sub["rainfall_sha256"].nunique())}
    failed_windows = []
    files_seen = files_ok = 0
    file_semantic: dict[str, str] = {}
    detail_stats = []
    grouped = frame.groupby("detail_path", sort=False)
    for files_seen, (raw_path, sub) in enumerate(grouped, start=1):
        path = Path(str(raw_path))
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            data = _read_detail(path, usecols)
            if data.empty:
                raise ValueError("empty detail")
            elapsed = pd.to_numeric(data["elapsed_min"], errors="coerce").to_numpy(float)
            if not np.isfinite(elapsed).all():
                raise ValueError("elapsed_min contains non-finite values")
            if len(np.unique(elapsed)) != len(elapsed):
                raise ValueError("duplicate elapsed_min values")
            numeric = data[usecols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            if not np.isfinite(numeric).all():
                raise ValueError("required Step1 values are non-finite")
            time_index = {round(float(t), 6): i for i, t in enumerate(elapsed)}
            for row_i, row in sub.iterrows():
                anchor = float(row["anchor_min"])
                expected = [anchor - 60.0 + 5.0 * i for i in range(13)]
                indices = [time_index.get(round(t, 6)) for t in expected]
                if any(x is None for x in indices) or len(set(indices)) != 13:
                    failed_windows.append({"row": int(row_i), "detail_path": str(path), "anchor_min": anchor, "reason": "not exactly 13 unique 5-min frames"})
                    if len(failed_windows) >= 200:
                        break
            files_ok += 1
            if len(file_semantic) < max_semantic_files:
                values = np.ascontiguousarray(np.round(numeric[:, : min(numeric.shape[1], 128)], 7), dtype=np.float64)
                file_semantic[str(path)] = hashlib.sha256(values.tobytes()).hexdigest()
            detail_stats.append({"path": str(path), "rows": len(data), "elapsed_min_min": float(elapsed.min()), "elapsed_min_max": float(elapsed.max())})
        except Exception as exc:
            failed_windows.append({"detail_path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
        if files_seen % 100 == 0:
            print(json.dumps({"phase": "STEP1_DETAIL", "files": files_seen, "total": int(frame["detail_path"].nunique()), "files_ok": files_ok, "failed_windows": len(failed_windows)}, ensure_ascii=False), flush=True)
    group_stats = _weighted_group_stats(frame[frame["formal_split"].astype(str).eq("train")], "rainfall_sha256", "windows")
    run_stats = _stats(frame.groupby("physical_identity_sha256").size().tolist())
    duplicate_key = ["physical_identity_sha256", "anchor_min", "rainfall_sha256"]
    duplicate_rows = int(frame.duplicated(duplicate_key, keep=False).sum())
    semantic_counts = Counter(file_semantic.values())
    semantic_duplicate_groups = int(sum(v > 1 for v in semantic_counts.values()))
    audit = {
        "status": "pass" if not failed_windows else "fail",
        "rows": len(frame),
        "physical_runs": int(frame["physical_identity_sha256"].nunique()),
        "rainfall_groups": int(frame["rainfall_sha256"].nunique()),
        "role_summary": role_summary,
        "detail_files": int(frame["detail_path"].nunique()),
        "detail_files_read": files_seen,
        "detail_files_ok": files_ok,
        "failed_window_count": len(failed_windows),
        "failed_window_examples": failed_windows[:200],
        "required_columns_count": len(required_set),
        "future_hydraulic_truth_in_input_count": int(frame["future_hydraulic_truth_in_input"].astype(bool).sum()),
        "windows_per_rainfall_group": group_stats,
        "windows_per_physical_run": run_stats,
        "physical_runs_per_rainfall_group": _stats(frame.groupby("rainfall_sha256")["physical_identity_sha256"].nunique().tolist()),
        "physical_identity_anchor_rainfall_duplicate_rows": duplicate_rows,
        "semantic_signature_sample_files": len(file_semantic),
        "semantic_signature_duplicate_groups_sample": semantic_duplicate_groups,
    }
    return audit, {"frame": frame, "graph": graph, "usecols": usecols}


def _raw_columns() -> list[str]:
    return [
        "case_uid", "state_key", "event_id", "rainfall_sha256", "split_group_key", "checkpoint_min", "candidate_action_sha256", "actual_k",
        *LABEL_COLUMNS, *RAW_GATE_COLUMNS,
        *(f"action_{role}_readback" for role in ROLES),
        *(f"trajectory_depth_{role}" for role in ROLES),
        *(f"trajectory_flood_{role}" for role in ROLES),
        *(f"source_detail_path_{role}" for role in ROLES),
    ]


def audit_raw(paths: dict[str, Path], graph: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = _read_parquet(paths["raw"], _raw_columns())
    gate_results = {c: bool(raw[c].map(yes).all()) if c in raw else False for c in RAW_GATE_COLUMNS}
    parse_errors = []
    recomputed = {k: [] for k in LABEL_COLUMNS}
    stored_error = {k: [] for k in LABEL_COLUMNS}
    collisions = Counter()
    action_hashes = [None] * len(raw)
    priority = get_pfv_core_node_indices([str(x) for x in graph["node_ids"]])
    for i, row in raw.iterrows():
        arrays = {}
        try:
            for role in ROLES:
                arrays[role] = {"depth": _array(row[f"trajectory_depth_{role}"]), "flood": _array(row[f"trajectory_flood_{role}"]), "action": _array(row[f"action_{role}_readback"])}
            pfv, tfv, peak = _kpis(arrays, priority)
            for key, value in zip(LABEL_COLUMNS, (pfv, tfv, peak)):
                recomputed[key].append(float(value))
                stored_error[key].append(abs(float(row[key]) - float(value)))
            hashes = {role: _hash_json_array(row[f"action_{role}_readback"]) for role in ROLES}
            action_hashes[int(i)] = hashes
            for left, right in (("candidate", "hold_previous"), ("candidate", "no_control"), ("candidate", "dynamic_internal"), ("dynamic_internal", "hold_previous"), ("no_control", "hold_previous")):
                if hashes[left] and hashes[left] == hashes[right]:
                    collisions[f"{left}=={right}"] += 1
        except Exception as exc:
            parse_errors.append({"row": int(i), "error": f"{type(exc).__name__}: {exc}"})
    raw = raw.copy()
    raw["_candidate_h3_sha"] = [x.get("candidate", "") if x else "" for x in action_hashes]
    state_counts = raw.groupby("state_key").size()
    candidate_schedule_counts = raw.groupby("state_key")["_candidate_h3_sha"].nunique()
    labels = {}
    for key in LABEL_COLUMNS:
        vals = pd.to_numeric(raw[key], errors="coerce").to_numpy(float)
        valid = vals[np.isfinite(vals)]
        safe_values = vals <= 0 if key != "tfv_delta" else vals < 0
        group_bal = raw.assign(_safe=safe_values).groupby("rainfall_sha256")["_safe"].mean() if len(raw) else pd.Series(dtype=float)
        labels[key] = {
            "finite": bool(np.isfinite(vals).all()),
            "stats": _stats(valid),
            "safe_or_improved_fraction_row": float(np.mean((vals <= 0) if key != "tfv_delta" else (vals < 0))) if len(vals) else None,
            "safe_or_improved_fraction_event_balanced": float(group_bal.mean()) if len(group_bal) else None,
            "collapse": bool(len(valid) > 0 and float(np.std(valid)) < 1e-12),
            "stored_vs_recomputed_max_abs_error": float(max(stored_error[key]) if stored_error[key] else math.inf),
            "stored_vs_recomputed_p99_abs_error": float(np.percentile(stored_error[key], 99) if stored_error[key] else math.inf),
        }
    k = pd.to_numeric(raw["actual_k"], errors="coerce")
    action_dist = Counter(int(x) for x in k.dropna().tolist())
    raw_audit = {
        "status": "pass" if all(gate_results.values()) and not parse_errors and bool(raw["case_uid"].astype(str).is_unique) and bool((k <= 8).all()) else "fail",
        "rows": len(raw),
        "case_uid_unique": bool(raw["case_uid"].astype(str).is_unique),
        "gate_results": gate_results,
        "parse_error_count": len(parse_errors),
        "parse_error_examples": parse_errors[:50],
        "rainfall_groups": int(raw["rainfall_sha256"].nunique()),
        "states": int(raw["state_key"].nunique()),
        "cases_per_state": _stats(state_counts.tolist()),
        "distinct_candidate_schedules_per_state": _stats(candidate_schedule_counts.tolist()),
        "states_with_at_least_3_distinct_candidates": int((candidate_schedule_counts >= 3).sum()),
        "states_with_at_least_3_fraction": float((candidate_schedule_counts >= 3).mean()) if len(candidate_schedule_counts) else 0.0,
        "actual_k": {"min": int(k.min()), "max": int(k.max()), "histogram": dict(sorted(action_dist.items()))} if len(k) else {},
        "k_gt_8": int((k > 8).sum()),
        "labels": labels,
        "action_collisions_rows": dict(collisions),
        "duplicate_rainfall_state_action_rows": int(raw.duplicated(["rainfall_sha256", "state_key", "_candidate_h3_sha"], keep=False).sum()),
    }
    return raw_audit, raw


def _history_compatible(root: Path, raw: pd.DataFrame, step1: pd.DataFrame, graph: dict[str, Any], cache_items: int = 4) -> dict[str, Any]:
    required = _build_usecols([str(x) for x in graph["node_ids"]], [str(x) for x in graph["facility_ids"]])
    by_rain: defaultdict[str, set[str]] = defaultdict(set)
    by_rain_event: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    allowed = step1[~step1["formal_split"].astype(str).isin(["formal_blind", "challenge", "locked_validation"])]
    for _, row in allowed[["rainfall_sha256", "event_id", "detail_path"]].drop_duplicates().iterrows():
        path = str(row["detail_path"])
        by_rain[str(row["rainfall_sha256"])].add(path)
        event = _nonempty(row.get("event_id", ""))
        if event:
            by_rain_event[(str(row["rainfall_sha256"]), event)].add(path)
    cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
    bounds: dict[str, tuple[float, float]] = {}
    def detail(path: Path) -> pd.DataFrame:
        key = str(path.resolve())
        if key in cache:
            value = cache.pop(key); cache[key] = value; return value
        value = _read_detail(path, required)
        cache[key] = value
        while len(cache) > cache_items:
            cache.popitem(last=False)
        return value
    def window(d: pd.DataFrame, anchor: float) -> bool:
        return _detail_extract_window(d, anchor, [str(x) for x in graph["node_ids"]], [str(x) for x in graph["facility_ids"]]) is not None
    state_rows = []
    for state, group in raw.groupby("state_key", sort=True):
        first = group.iloc[0]
        rain, event, cp = str(first["rainfall_sha256"]), _nonempty(first.get("event_id", "")), float(first["checkpoint_min"])
        candidate = Path(str(first.get("source_detail_path_candidate", "")))
        compatible = False
        failure = "no history candidate"
        try:
            candidate_detail = detail(candidate)
            if not window(candidate_detail, cp):
                raise ValueError("candidate lacks checkpoint Step1 window")
            anchor_signature = _detail_extract_window(candidate_detail, cp, [str(x) for x in graph["node_ids"]], [str(x) for x in graph["facility_ids"]])
            sig = (anchor_signature["depth_history"][-1], anchor_signature["actions"][-1], anchor_signature["rainfall"][-1:])
            candidates = sorted(by_rain_event.get((rain, event), set())) or sorted(by_rain.get(rain, set()))
            for raw_path in candidates:
                path = Path(raw_path)
                if not path.exists():
                    continue
                key = str(path.resolve())
                if key not in bounds:
                    elapsed = pd.to_numeric(pd.read_csv(path, usecols=["elapsed_min"])["elapsed_min"], errors="coerce").dropna()
                    bounds[key] = (float(elapsed.min()), float(elapsed.max())) if len(elapsed) else (math.inf, -math.inf)
                lo, hi = bounds[key]
                if lo > cp - 120.0 + 1e-6 or hi < cp - 1e-6:
                    continue
                d = detail(path)
                s = _detail_extract_window(d, cp, [str(x) for x in graph["node_ids"]], [str(x) for x in graph["facility_ids"]])
                if s is None or not all(np.allclose(a, b, atol=1e-6, rtol=0.0) for a, b in zip(sig, (s["depth_history"][-1], s["actions"][-1], s["rainfall"][-1:]))):
                    continue
                if all(window(d, anchor) for anchor in [cp - 60.0 + 5.0 * i for i in range(13)]):
                    compatible = True
                    failure = ""
                    break
            if not compatible and not failure:
                failure = "no same-state 120-min history"
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        state_rows.append({"state_key": str(state), "rainfall_group": rain, "compatible": compatible, "failure": failure})
    states = pd.DataFrame(state_rows)
    inp = set(raw["rainfall_sha256"].astype(str))
    got = set(states.loc[states["compatible"], "rainfall_group"].astype(str)) if not states.empty else set()
    return {
        "input_rainfall_groups": len(inp),
        "compatible_rainfall_groups": len(got),
        "input_states": len(states),
        "compatible_states": int(states["compatible"].sum()) if not states.empty else 0,
        "lost_rainfall_groups": sorted(inp - got),
        "alternate_state_recovered_rainfall_groups": sorted(set(states.loc[states["rainfall_group"].isin(got) & ~states["compatible"], "rainfall_group"].astype(str))) if not states.empty else [],
        "failed_state_examples": states.loc[~states["compatible"]].head(100).to_dict("records") if not states.empty else [],
        "minimum_required_groups": 69,
    }


def audit_targets(paths: dict[str, Path], raw: pd.DataFrame, root: Path, cache_items: int = 2) -> dict[str, Any]:
    graph = _load_graph_topology(root)
    nodes_df, _ = _parse_inp_topology(root / "data" / "wuhan_v8_storage_retrofit.inp")
    storage_ids = nodes_df.loc[nodes_df["node_type"].astype(str).str.casefold().eq("storage"), "node_id"].astype(str).tolist()
    outfall_ids = nodes_df.loc[nodes_df["node_type"].astype(str).str.casefold().isin(["outfall", "outfall_node"]), "node_id"].astype(str).tolist()
    columns = _target_columns([str(x) for x in graph["node_ids"]], storage_ids, [str(x) for x in graph["facility_ids"]], outfall_ids)
    all_cols = sorted(set(sum(columns.values(), ["elapsed_min"])))
    cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
    per_file = {}
    def one(path: Path) -> dict[str, Any]:
        key = str(path.resolve())
        if key in cache:
            value = cache.pop(key); cache[key] = value; return value
        header = _detail_header(path)
        present = [c for c in all_cols if c in set(header)]
        missing = {name: [c for c in cols if c not in set(header)] for name, cols in columns.items()}
        totals = {name: 0 for name in TARGET_NAMES}; finite = {name: 0 for name in TARGET_NAMES}; rows = 0
        if present:
            chunks = pd.read_csv(path, usecols=present, chunksize=2048, low_memory=False)
            for chunk in chunks:
                rows += len(chunk)
                for name, cols in columns.items():
                    present_cols = [c for c in cols if c in chunk.columns]
                    totals[name] += len(chunk) * len(cols)
                    if len(present_cols) == len(cols):
                        finite[name] += int(np.isfinite(chunk[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)).sum())
        result = {"rows": rows, "missing_columns": {k: v[:20] for k, v in missing.items()}, "finite_fraction": {k: float(finite[k] / totals[k]) if totals[k] else 0.0 for k in TARGET_NAMES}, "complete": {k: not missing[k] and finite[k] == totals[k] and totals[k] > 0 for k in TARGET_NAMES}}
        cache[key] = result
        while len(cache) > cache_items:
            cache.popitem(last=False)
        return result
    branch_cols = [f"source_detail_path_{r}" for r in ROLES]
    unique = sorted({str(x) for c in branch_cols for x in raw[c].dropna().tolist() if str(x)})
    for i, raw_path in enumerate(unique, start=1):
        path = Path(raw_path)
        try:
            per_file[raw_path] = one(path) if path.exists() else {"rows": 0, "missing_columns": {k: ["file_missing"] for k in TARGET_NAMES}, "finite_fraction": {k: 0.0 for k in TARGET_NAMES}, "complete": {k: False for k in TARGET_NAMES}}
        except Exception as exc:
            per_file[raw_path] = {"rows": 0, "error": f"{type(exc).__name__}: {exc}", "missing_columns": {k: ["read_error"] for k in TARGET_NAMES}, "finite_fraction": {k: 0.0 for k in TARGET_NAMES}, "complete": {k: False for k in TARGET_NAMES}}
        if i % 100 == 0:
            print(json.dumps({"phase": "TARGET_COVERAGE", "files": i, "total": len(unique)}, ensure_ascii=False), flush=True)
    complete_rows = {k: 0 for k in TARGET_NAMES}; complete_files = {k: 0 for k in TARGET_NAMES}; finite_values = {k: 0 for k in TARGET_NAMES}; total_values = {k: 0 for k in TARGET_NAMES}
    for result in per_file.values():
        for k in TARGET_NAMES:
            complete_files[k] += int(result["complete"][k])
            finite_values[k] += int(result["finite_fraction"][k] * max(result.get("rows", 0), 1))
            total_values[k] += max(result.get("rows", 0), 1)
    row_complete = {k: [] for k in TARGET_NAMES}
    for _, row in raw.iterrows():
        for k in TARGET_NAMES:
            row_complete[k].append(all(per_file.get(str(row[f"source_detail_path_{role}"]), {}).get("complete", {}).get(k, False) for role in ROLES))
    group_complete = {k: sorted(set(raw.loc[row_complete[k], "rainfall_sha256"].astype(str))) for k in TARGET_NAMES}
    all_formal = np.asarray([all(row_complete[k][i] for k in TARGET_NAMES) for i in range(len(raw))], dtype=bool)
    return {
        "network_node_count": len(graph["node_ids"]), "storage_node_count": len(storage_ids), "facility_count": len(graph["facility_ids"]), "outfall_node_count": len(outfall_ids),
        "unique_detail_files": len(unique), "per_target": {k: {"complete_detail_count": complete_files[k], "finite_fraction_file_weighted": float(finite_values[k] / total_values[k]) if total_values[k] else 0.0, "complete_rows": int(sum(row_complete[k])), "complete_states": int(raw.loc[row_complete[k], "state_key"].nunique()), "complete_rainfall_groups": len(group_complete[k])} for k in TARGET_NAMES},
        "formal_complete_detail_count": int(sum(all(per_file[p]["complete"][k] for k in TARGET_NAMES) for p in per_file)),
        "formal_complete_rows": int(all_formal.sum()), "formal_complete_states": int(raw.loc[all_formal, "state_key"].nunique()), "formal_complete_rainfall_groups": int(raw.loc[all_formal, "rainfall_sha256"].nunique()),
        "missing_columns_top20": {k: Counter(c for p in per_file.values() for c in p.get("missing_columns", {}).get(k, [])).most_common(20) for k in TARGET_NAMES},
        "storage_supervised": bool(all_formal.any() and all(per_file[p]["complete"]["storage_volume"] for p in per_file)),
        "facility_flow_supervised": bool(all_formal.any() and all(per_file[p]["complete"]["managed_facility_flow"] for p in per_file)),
        "outfall_supervised": bool(all_formal.any() and all(per_file[p]["complete"]["outfall_flow"] for p in per_file)),
        "outfall_schema_present": bool(outfall_ids) and any(bool(x.get("complete", {}).get("outfall_flow")) for x in per_file.values()),
    }


def _load_plan_files(eval_dir: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(eval_dir.glob("*_plan.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            result.extend(obj.get("events", []))
        except (OSError, json.JSONDecodeError):
            continue
    return result


def audit_evaluation(paths: dict[str, Path], identity: dict[str, Any], history: dict[str, Any]) -> dict[str, Any]:
    events = _load_plan_files(paths["evaluation_dir"])
    by_role = defaultdict(list)
    for event in events:
        by_role[str(event.get("formal_f2_role", ""))].append(event)
    checkpoint_counts = {role: sum(len(e.get("checkpoints", [])) for e in rows) for role, rows in by_role.items()}
    calibration_ready = []
    for event in by_role.get("calibration", []):
        cps = [float(c.get("checkpoint_min")) for c in event.get("checkpoints", []) if pd.notna(c.get("checkpoint_min"))]
        calibration_ready.append({"event_id": _nonempty(event.get("event_id")), "rainfall_sha256": _nonempty(event.get("rainfall_sha256")), "checkpoint_count": len(cps), "checkpoint_ge_120_count": sum(x >= 120 for x in cps), "history_group_available": _nonempty(event.get("rainfall_sha256")) not in set(history.get("lost_rainfall_groups", [])), "future_coverage_provable": False})
    blind_empty = all(not e.get("checkpoints") for e in by_role.get("formal_blind", []))
    expected = {"calibration": 12, "locked_validation": 16, "challenge": 12, "formal_blind": 24}
    counts = {role: len(by_role.get(role, [])) for role in expected}
    return {
        "counts": counts, "checkpoint_counts": checkpoint_counts, "blind_checkpoints_empty": blind_empty,
        "selection_uses_control_outcome": any(bool(e.get("selection_uses_control_outcome", True)) for e in events),
        "post_reveal_exclusion_allowed": any(bool(e.get("post_reveal_exclusion_allowed", True)) for e in events),
        "calibration_checkpoint_readiness": calibration_ready,
        "calibration_events_with_no_checkpoint_ge_120": sum(x["checkpoint_ge_120_count"] == 0 for x in calibration_ready),
        "calibration_future_coverage_provable": False,
        "rainfall_descriptors_missing": sorted({k for e in events for k in ("duration_min", "rainfall_family") if e.get(k) in (None, "")}),
        "ready": counts["calibration"] == 12 and counts["locked_validation"] == 16 and counts["challenge"] == 12 and counts["formal_blind"] >= 24 and blind_empty and not any(bool(e.get("selection_uses_control_outcome", True)) for e in events) and not any(bool(e.get("post_reveal_exclusion_allowed", True)) for e in events) and all(x["checkpoint_ge_120_count"] > 0 for x in calibration_ready),
    }


def audit_r0(paths: dict[str, Path], raw: pd.DataFrame) -> dict[str, Any]:
    r0 = json.loads(paths["r0_audit"].read_text(encoding="utf-8"))
    alignment = pd.read_csv(paths["alignment"], low_memory=False)
    split = _read_parquet(paths["split"], ["split_group_key"])
    raw_manifest_hash = sha256_file(paths["raw"])
    ledger_hash = sha256_file(paths["ledger"])
    step1_hash = sha256_file(paths["step1"])
    raw_groups = set(raw["rainfall_sha256"].astype(str))
    split_groups = set(split["split_group_key"].astype(str))
    return {
        "r0_status": r0.get("status"), "r0_rainfall_groups": r0.get("rainfall_groups"), "r0_case_counts": r0.get("counterfactual_cases"), "r0_sources": r0.get("sources", []),
        "case_alignment_rows": len(alignment), "raw_accepted_rows": len(raw), "case_alignment_rows_match_raw": len(alignment) == len(raw), "split_group_count": len(split_groups), "raw_group_count": len(raw_groups), "split_groups_match_raw": split_groups == raw_groups,
        "raw_manifest_sha256": raw_manifest_hash, "event_ledger_sha256": ledger_hash, "step1_manifest_sha256": step1_hash,
        "r0_lineage_hashes_present": all(bool(r0.get(k)) for k in ("raw_manifest_sha256", "event_ledger_sha256", "step1_manifest_sha256")),
        "ready": r0.get("status") == "pass" and len(alignment) == len(raw) and split_groups == raw_groups and r0.get("rainfall_groups") == len(raw_groups),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--max-semantic-files", type=int, default=128)
    ap.add_argument("--history-cache-items", type=int, default=4)
    ap.add_argument("--target-cache-items", type=int, default=2)
    args = ap.parse_args()
    root = args.project_root.resolve()
    formal = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    reuse = formal.parent / "data_reuse"
    out = formal / "precompute_readiness"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "prepare_audit": formal / "prepare/FORMAL_F2_PREPARE_AUDIT.json", "step1": formal / "prepare/FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet", "step1_audit": formal / "prepare/FORMAL_F2_STEP1_POOL_AUDIT.json", "raw": formal / "step2/FORMAL_F2_STEP2_RAW_MANIFEST.parquet", "raw_audit": formal / "step2/FORMAL_F2_STEP2_RAW_ADMISSION_AUDIT.json", "ledger": formal / "prepare/FORMAL_F2_EVENT_LEDGER.csv", "evaluation_dir": formal / "evaluation_plan", "r0_audit": reuse / "FORMAL_F2_R0_ADAPTER_AUDIT.json", "alignment": reuse / "case_alignment_audit.csv", "split": reuse / "split_group_manifest.parquet",
    }
    started = time.time()
    print("[IDENTITY] start", flush=True)
    identity = audit_identity(root, paths)
    print("[STEP1] start", flush=True)
    step1, step1_data = audit_step1(root, paths, args.max_semantic_files)
    print("[RAW] start", flush=True)
    raw, raw_frame = audit_raw(paths, step1_data["graph"])
    print("[GAT_DRY_RUN] start", flush=True)
    gat = _history_compatible(root, raw_frame, step1_data["frame"], step1_data["graph"], args.history_cache_items)
    print("[TARGETS] start", flush=True)
    targets = audit_targets(paths, raw_frame, root, args.target_cache_items)
    print("[EVALUATION] start", flush=True)
    evaluation = audit_evaluation(paths, identity, gat)
    print("[R0] start", flush=True)
    r0 = audit_r0(paths, raw_frame)
    prepare = json.loads(paths["prepare_audit"].read_text(encoding="utf-8"))
    step1_pool = json.loads(paths["step1_audit"].read_text(encoding="utf-8"))
    result = {
        "formal_generation_id": prepare.get("formal_generation_id", "PROJECT6_V42_FORMAL_F2"), "audit": "Formal F2 Precompute Readiness", "generated_at_epoch": time.time(), "elapsed_sec": time.time() - started,
        "01_ledger_ready": bool(identity["identity_conflict_count"] == 0 and identity["reserved_event_ids_without_rainfall_group"] == 0 and all(v == 0 for v in identity["split_overlap"].values())),
        "02_step1_data_ready": bool(step1["status"] == "pass" and step1["failed_window_count"] == 0 and step1["role_summary"].get("train", {}).get("rainfall_groups", 0) >= 65),
        "03_raw_step2_ready": bool(raw["status"] == "pass" and raw["states_with_at_least_3_distinct_candidates"] > 0),
        "04_evaluation_plan_ready": bool(evaluation["ready"]),
        "05_r0_adapter_ready": bool(r0["ready"]),
        "identity": identity, "step1": step1, "step1_pool_source_audit": step1_pool, "raw_step2": raw, "causal_gat": gat, "hydraulic_targets": targets, "evaluation": evaluation, "r0": r0,
        "causal_gat_compatible_groups": gat["compatible_rainfall_groups"], "formal_complete_hydraulic_target_groups": targets["formal_complete_rainfall_groups"], "step1_group_effective_sample_size": step1["windows_per_rainfall_group"].get("effective_group_count"), "step2_ge3_candidate_state_fraction": raw["states_with_at_least_3_fraction"],
        "scientific_blockers": [],
    }
    if gat["compatible_rainfall_groups"] < 69:
        result["scientific_blockers"].append("causal GAT-compatible rainfall groups below 69")
    if targets["formal_complete_rainfall_groups"] < 69:
        result["scientific_blockers"].append("formal complete hydraulic target rainfall groups below 69")
    if not targets["outfall_supervised"]:
        result["scientific_blockers"].append("outfall flow supervision is not complete")
    if not evaluation["ready"]:
        result["scientific_blockers"].append("evaluation plan is not fully executable from existing precompute artifacts")
    result["READY_FOR_STEP1"] = bool(result["01_ledger_ready"] and result["02_step1_data_ready"] and result["03_raw_step2_ready"] and result["05_r0_adapter_ready"] and gat["compatible_rainfall_groups"] >= 69)
    result["READY_FOR_STEP2"] = bool(result["READY_FOR_STEP1"] and targets["formal_complete_rainfall_groups"] >= 69 and targets["storage_supervised"] and targets["facility_flow_supervised"] and targets["outfall_supervised"])
    result["READY_FOR_CALIBRATION"] = bool(result["READY_FOR_STEP2"] and result["04_evaluation_plan_ready"])
    result["status"] = "pass" if result["READY_FOR_STEP1"] else "fail"
    path = out / "FORMAL_F2_PRECOMPUTE_READINESS.json"
    path.write_text(json.dumps(_jsonable(result), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (out / "FORMAL_F2_TARGET_COVERAGE.json").write_text(json.dumps(_jsonable(targets), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "READY_FOR_STEP1": result["READY_FOR_STEP1"], "READY_FOR_STEP2": result["READY_FOR_STEP2"], "READY_FOR_CALIBRATION": result["READY_FOR_CALIBRATION"], "causal_gat_compatible_groups": gat["compatible_rainfall_groups"], "formal_complete_hydraulic_target_groups": targets["formal_complete_rainfall_groups"], "output": str(path)}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
