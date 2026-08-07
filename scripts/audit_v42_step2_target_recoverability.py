"""Audit and deterministically recover FAST8 Step-2 targets without SWMM."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.simulation.v42_hydraulic_recorder import storage_volume_from_depth_v42
from sewerrtc.v4.v42_formal_runtime import load_actuators
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology, _parse_inp_topology


ABS_TOL = 1.0e-6
REL_TOL = 1.0e-9


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [str(value).strip() for value in next(csv.reader(handle))]


def _parse_inp_storage_geometry(path: Path) -> dict[str, dict[str, Any]]:
    storage: dict[str, dict[str, Any]] = {}
    curves: dict[str, dict[str, Any]] = {}
    section = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            text = raw.strip()
            if not text or text.startswith(";"):
                continue
            if text.startswith("[") and "]" in text:
                section = text[1 : text.index("]")].strip().upper()
                continue
            text = text.split(";", 1)[0].strip()
            parts = text.split()
            if section == "STORAGE" and len(parts) >= 6:
                kind = parts[4].upper()
                entry: dict[str, Any] = {
                    "shape": kind,
                    "max_depth": float(parts[2]),
                }
                if kind == "FUNCTIONAL":
                    entry["functional_params"] = [float(value) for value in parts[5:8]]
                elif kind == "TABULAR":
                    entry["curve_name"] = parts[5]
                storage[parts[0]] = entry
            elif section == "CURVES" and len(parts) >= 3:
                name = parts[0]
                try:
                    x, y = float(parts[-2]), float(parts[-1])
                except ValueError:
                    continue
                if len(parts) >= 4 and parts[1].upper() in {"STORAGE", "PUMP1", "PUMP2", "PUMP3", "PUMP4", "SHAPE"}:
                    curves[name] = {"type": parts[1].upper(), "depth": [x], "area": [y]}
                elif name in curves:
                    curves[name]["depth"].append(x)
                    curves[name]["area"].append(y)
    for node in storage.values():
        if node["shape"] == "TABULAR":
            curve = curves.get(str(node["curve_name"]))
            if not curve or curve.get("type") != "STORAGE":
                node["curve_error"] = "missing_STORAGE_curve"
            else:
                node["curve_depth"] = curve["depth"]
                node["curve_area"] = curve["area"]
    return storage


def _horizon_indices(elapsed: np.ndarray, checkpoint: float) -> np.ndarray | None:
    indices: list[int] = []
    for target in checkpoint + np.arange(10.0, 121.0, 10.0):
        found = np.flatnonzero(np.isclose(elapsed, target, atol=1.0e-6, rtol=0.0))
        if len(found) != 1:
            return None
        indices.append(int(found[0]))
    return np.asarray(indices, dtype=np.int64)


def _finite_group(frame: pd.DataFrame, columns: list[str], indices: np.ndarray) -> bool:
    return bool(columns and all(column in frame for column in columns) and np.isfinite(frame.loc[indices, columns].to_numpy(float)).all())


def _canonical_or_alias_map(headers: list[str], facility_ids: list[str]) -> dict[str, str | None]:
    lower = {str(column).casefold(): str(column) for column in headers}
    out: dict[str, str | None] = {}
    for facility in facility_ids:
        exact = f"flow:{facility}"
        if exact in headers:
            out[facility] = exact
            continue
        candidates = []
        target = facility.casefold()
        for column in headers:
            key = str(column).casefold()
            for prefix in ("flow:", "q:", "link_flow:", "flow_", "q_"):
                if key.startswith(prefix) and key[len(prefix) :] == target:
                    candidates.append(str(column))
        out[facility] = candidates[0] if len(set(candidates)) == 1 else None
    return out


def _storage_validation(
    paths: list[Path],
    storage_ids: list[str],
    geometry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    stats = {
        node: {"samples": 0, "mae_sum": 0.0, "sq_sum": 0.0, "max_abs": 0.0, "max_rel": 0.0}
        for node in storage_ids
    }
    files = 0
    for path in paths:
        headers = set(_read_header(path))
        columns = ["elapsed_min"] + sum(([f"h:{node}", f"storage_volume:{node}"] for node in storage_ids), [])
        if not all(column in headers for column in columns):
            continue
        files += 1
        for chunk in pd.read_csv(path, usecols=columns, chunksize=4096, low_memory=False):
            for node in storage_ids:
                depth = chunk[f"h:{node}"].to_numpy(float)
                stored = chunk[f"storage_volume:{node}"].to_numpy(float)
                mask = np.isfinite(depth) & np.isfinite(stored)
                if not mask.any():
                    continue
                entry = geometry[node]
                predicted = storage_volume_from_depth_v42(
                    depth[mask],
                    shape=entry["shape"],
                    functional_params=entry.get("functional_params"),
                    curve_depth=entry.get("curve_depth"),
                    curve_area=entry.get("curve_area"),
                )
                error = predicted - stored[mask]
                absolute = np.abs(error)
                relative = absolute / np.maximum(np.abs(stored[mask]), 1.0)
                result = stats[node]
                result["samples"] += int(mask.sum())
                result["mae_sum"] += float(absolute.sum())
                result["sq_sum"] += float(np.square(error).sum())
                result["max_abs"] = max(result["max_abs"], float(absolute.max()))
                result["max_rel"] = max(result["max_rel"], float(relative.max()))
    per_node = {}
    for node, result in stats.items():
        samples = int(result.pop("samples"))
        n = max(samples, 1)
        mae_sum = result.pop("mae_sum")
        sq_sum = result.pop("sq_sum")
        result["samples"] = samples
        result["mae"] = mae_sum / n if samples else None
        result["rmse"] = float(np.sqrt(sq_sum / n)) if samples else None
        result["pass"] = bool(result["samples"] and result["max_abs"] <= ABS_TOL and result["max_rel"] <= REL_TOL)
        per_node[node] = result
    return {
        "audit_id": "STORAGE_VOLUME_RECOVERY_VALIDATION_V1",
        "files_checked": files,
        "nodes": per_node,
        "all_nodes_pass": bool(per_node) and all(item["pass"] for item in per_node.values()),
        "absolute_tolerance_m3": ABS_TOL,
        "relative_tolerance": REL_TOL,
        "formula": "SWMM functional area integration; tabular depth-area piecewise-linear integration",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    graph = _load_graph_topology(root)
    nodes, _ = _parse_inp_topology(root / "data" / "wuhan_v8_storage_retrofit.inp")
    node_ids = [str(value) for value in graph["node_ids"]]
    storage_ids = nodes.loc[nodes["node_type"].astype(str).str.casefold().eq("storage"), "node_id"].astype(str).tolist()
    facility_ids = load_actuators(root)["actuator_id"].astype(str).tolist()
    geometry = _parse_inp_storage_geometry(root / "data" / "wuhan_v8_storage_retrofit.inp")
    missing_geometry = [node for node in storage_ids if node not in geometry]

    manifest = pd.read_parquet(args.input)
    if "source_detail_path_candidate" not in manifest:
        raise RuntimeError("FAST8 input lacks source_detail_path_candidate")
    detail_paths = [Path(str(value)) for value in manifest["source_detail_path_candidate"].dropna().unique()]
    storage_validation = _storage_validation([path for path in detail_paths if path.exists()], storage_ids, geometry)
    (out / "STORAGE_VOLUME_RECOVERY_VALIDATION.json").write_text(_json(storage_validation), encoding="utf-8")
    storage_recovery_allowed = not missing_geometry and bool(storage_validation["all_nodes_pass"])

    per_file_maps: dict[str, dict[str, str | None]] = {}
    map_observations: dict[str, set[str]] = {facility: set() for facility in facility_ids}
    rows: list[dict[str, Any]] = []
    recovered_storage_arrays: dict[str, list[list[float]]] = {}
    recovered_flow_arrays: dict[str, list[list[float]]] = {}
    for row_index, row in manifest.iterrows():
        path = Path(str(row["source_detail_path_candidate"]))
        headers = _read_header(path) if path.exists() else []
        header_set = set(headers)
        facility_map = _canonical_or_alias_map(headers, facility_ids)
        per_file_maps[str(path)] = facility_map
        for facility, column in facility_map.items():
            if column is not None:
                map_observations[facility].add(column)
        storage_depth_columns = [f"h:{node}" for node in storage_ids]
        storage_volume_columns = [f"storage_volume:{node}" for node in storage_ids]
        facility_flow_columns = [facility_map[node] for node in facility_ids]
        read_columns = ["elapsed_min"] + [column for column in storage_depth_columns + storage_volume_columns + facility_flow_columns if column in header_set]
        frame = pd.read_csv(path, usecols=list(dict.fromkeys(read_columns))) if path.exists() and "elapsed_min" in header_set else pd.DataFrame()
        indices = _horizon_indices(frame["elapsed_min"].to_numpy(float), float(row["checkpoint_min"])) if not frame.empty else None
        depth_ok = indices is not None and _finite_group(frame, storage_depth_columns, indices)
        direct_storage = indices is not None and _finite_group(frame, storage_volume_columns, indices)
        direct_facility = indices is not None and all(column is not None for column in facility_flow_columns) and _finite_group(frame, [str(column) for column in facility_flow_columns if column is not None], indices)
        storage_recovered = False
        storage_values: np.ndarray | None = None
        if indices is not None and depth_ok and storage_recovery_allowed:
            storage_values = np.column_stack([
                storage_volume_from_depth_v42(
                    frame.loc[indices, f"h:{node}"].to_numpy(float),
                    shape=geometry[node]["shape"],
                    functional_params=geometry[node].get("functional_params"),
                    curve_depth=geometry[node].get("curve_depth"),
                    curve_area=geometry[node].get("curve_area"),
                )
                for node in storage_ids
            ])
            storage_recovered = bool(np.isfinite(storage_values).all()) and not direct_storage
        if direct_storage:
            storage_values = frame.loc[indices, storage_volume_columns].to_numpy(float)
        flow_values: np.ndarray | None = None
        if direct_facility:
            flow_values = frame.loc[indices, [str(column) for column in facility_flow_columns]].to_numpy(float)
        storage_ok = direct_storage or storage_recovered
        facility_ok = direct_facility
        node_depth_ok = bool(row.get("trajectory_depth_candidate", "")) and indices is not None
        node_flood_ok = bool(row.get("trajectory_flood_candidate", "")) and indices is not None
        full = bool(node_depth_ok and node_flood_ok and storage_ok and facility_ok)
        partial = bool(node_depth_ok or node_flood_ok or storage_ok or facility_ok)
        key = str(row.get("candidate_action_sha256", row_index))
        if storage_values is not None and storage_ok:
            recovered_storage_arrays[key] = storage_values.astype(float).tolist()
        if flow_values is not None and facility_ok:
            recovered_flow_arrays[key] = flow_values.astype(float).tolist()
        rows.append({
            "manifest_row": int(row_index),
            "state_key": str(row.get("state_key", "")),
            "candidate_action_sha256": key,
            "detail_path": str(path),
            "depth_available": bool(node_depth_ok),
            "flood_available": bool(node_flood_ok),
            "storage_direct": bool(direct_storage),
            "storage_recovered": bool(storage_recovered),
            "storage_available": bool(storage_ok),
            "facility_flow_direct": bool(direct_facility),
            "facility_flow_recovered": False,
            "facility_flow_available": bool(facility_ok),
            "PFV_available": bool(np.isfinite(float(row.get("pfv_delta", np.nan)))),
            "TFV_available": bool(np.isfinite(float(row.get("tfv_delta", np.nan)))),
            "full_hydraulic_supervision": full,
            "partial_hydraulic_supervision": partial,
            "action_effect_supervision": bool(np.isfinite(float(row.get("pfv_delta", np.nan))) and np.isfinite(float(row.get("tfv_delta", np.nan)))),
            "horizon_complete": indices is not None,
        })

    row_audit = pd.DataFrame(rows)
    row_audit.to_csv(out / "FAST8_TARGET_RECOVERABILITY_ROWS.csv", index=False)
    row_by_key = row_audit.set_index("candidate_action_sha256")
    for key, values in recovered_storage_arrays.items():
        if key in row_by_key.index:
            manifest.loc[manifest["candidate_action_sha256"].astype(str).eq(key), "trajectory_storage_volume_candidate"] = _json(values)
    for key, values in recovered_flow_arrays.items():
        if key in row_by_key.index:
            manifest.loc[manifest["candidate_action_sha256"].astype(str).eq(key), "trajectory_facility_flow_candidate"] = _json(values)
    for column, value in {
        "trajectory_storage_volume_candidate_available": row_audit.set_index("candidate_action_sha256")["storage_available"],
        "trajectory_facility_flow_candidate_available": row_audit.set_index("candidate_action_sha256")["facility_flow_available"],
        "control_core_target_coverage_complete": row_audit.set_index("candidate_action_sha256")["full_hydraulic_supervision"],
    }.items():
        manifest[column] = manifest["candidate_action_sha256"].astype(str).map(value).fillna(False).astype(bool)
    manifest["storage_finite_fraction_candidate"] = manifest["trajectory_storage_volume_candidate_available"].astype(float)
    manifest["facility_flow_finite_fraction_candidate"] = manifest["trajectory_facility_flow_candidate_available"].astype(float)
    recovered_manifest = out / "RECOVERED_FAST8_CONTROL_CORE_MANIFEST.parquet"
    manifest.to_parquet(recovered_manifest, index=False)

    facility_map_summary = {}
    for facility in facility_ids:
        observed = sorted(map_observations[facility])
        facility_map_summary[facility] = {
            "canonical_column": f"flow:{facility}",
            "observed_columns": observed,
            "status": "exact" if observed == [f"flow:{facility}"] else ("alias" if len(observed) == 1 else "unresolved"),
        }
    facility_audit = {
        "audit_id": "EXACT_FACILITY_FLOW_COLUMN_MAP_V1",
        "facility_count": len(facility_ids),
        "recoverable_count": sum(item["status"] in {"exact", "alias"} for item in facility_map_summary.values()),
        "unrecoverable_count": sum(item["status"] == "unresolved" for item in facility_map_summary.values()),
        "mapping": facility_map_summary,
        "rule": "only exact or unique deterministic column aliases; no setting/neighbor/pseudo-label inference",
    }
    (out / "EXACT_FACILITY_FLOW_COLUMN_MAP.json").write_text(_json(facility_audit), encoding="utf-8")

    summary = {
        "audit_id": "STEP2_TARGET_RECOVERABILITY_AUDIT_V1",
        "development_only": True,
        "formal_mainline_authorized": False,
        "input_manifest": str(args.input.resolve()),
        "input_manifest_sha256": _sha256(args.input),
        "input_rows": int(len(manifest)),
        "input_states": int(manifest["state_key"].astype(str).nunique()),
        "unique_detail_files": len(detail_paths),
        "storage_nodes": storage_ids,
        "facility_ids": facility_ids,
        "storage_geometry_nodes": len(geometry),
        "storage_geometry_missing": missing_geometry,
        "storage_recovery_allowed": storage_recovery_allowed,
        "rows_direct_complete": int((row_audit["full_hydraulic_supervision"] & ~row_audit["storage_recovered"]).sum()),
        "rows_storage_recovered": int(row_audit["storage_recovered"].sum()),
        "rows_facility_flow_recovered": int(row_audit["facility_flow_recovered"].sum()),
        "rows_both_recovered": int((row_audit["storage_recovered"] & row_audit["facility_flow_recovered"]).sum()),
        "rows_control_core_complete": int(row_audit["full_hydraulic_supervision"].sum()),
        "rows_still_incomplete": int((~row_audit["full_hydraulic_supervision"]).sum()),
        "facility_flow_recoverable_count": facility_audit["recoverable_count"],
        "facility_flow_unrecoverable_count": facility_audit["unrecoverable_count"],
        "storage_validation_pass": storage_recovery_allowed,
        "no_new_swmm_for_recovery": True,
        "output_manifest": str(recovered_manifest),
        "output_rows": str(out / "FAST8_TARGET_RECOVERABILITY_ROWS.csv"),
        "next": "HEAD_ONLY_ACTION_REPAIR_V1" if int((~row_audit["full_hydraulic_supervision"]).sum()) == 0 else "STEP2_CANDIDATE_RELATIVE_PARTIAL_SUPERVISION_V1",
    }
    (out / "STEP2_TARGET_RECOVERABILITY_AUDIT_V1.json").write_text(_json(summary), encoding="utf-8")
    print(_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
