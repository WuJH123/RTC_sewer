"""V4.2 trajectory dataset builder with fail-closed semantic alignment.

The canonical dataset contract is:

* 13 history frames at 5-minute spacing (60-minute look-back);
* 12 future steps at 10-minute spacing (H120);
* graph-node tensors explicitly reordered by node ID;
* action tensors explicitly reordered by the frozen Engineering36 IDs;
* Candidate, No-control, Dynamic Internal and Hold-Previous branches all
  required for full counterfactual supervision;
* no synthetic ones/zeros are substituted for a missing reference branch.

This module deliberately treats file/column order as untrusted.  Identity is
resolved from the physical node/facility IDs before arrays are assembled.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sewerrtc.v4.v42_priority_contract import (
    PFV_CORE_8_IDS,
    PriorityContractError,
    get_pfv_core_node_indices,
)

logger = logging.getLogger(__name__)

N_FACILITIES = 36
N_HISTORY_FRAMES = 13
N_HORIZON_STEPS = 12
HISTORY_INTERVAL_MIN = 5
HORIZON_INTERVAL_MIN = 10
TIME_ATOL_MIN = 1e-6
SURROGATE_ACTION_MAP_CONTRACT = "undirected_khop_inverse_distance_v1_radius10"
SURROGATE_ACTION_MAP_RADIUS = 10
NODE_STATIC_COLS = [
    "invert",
    "max_depth",
    "ponded_area",
    "degree_in",
    "degree_out",
    "is_storage",
    "is_outfall",
]
BRANCH_ROLES = (
    "candidate",
    "no_control",
    "dynamic_internal_rules",
    "hold_previous",
)
REFERENCE_BRANCHES = BRANCH_ROLES[1:]


def build_surrogate_action_node_map(
    graph: dict[str, Any], *, radius: int = SURROGATE_ACTION_MAP_RADIUS
) -> np.ndarray:
    """Map each actuator to its bounded physical network influence domain.

    The raw graph map contains only facility endpoints.  That is sufficient
    for local action context, but it leaves downstream/upstream PFV nodes
    action-blind.  The surrogate uses an undirected hydraulic influence
    neighbourhood with inverse-distance weights; training and inference share
    this deterministic map.
    """
    def field(name: str):
        try:
            return graph[name]
        except (KeyError, TypeError):
            return getattr(graph, name)

    n_nodes = int(field("n_nodes"))
    n_facilities = int(field("n_facilities"))
    node_ids = [str(x) for x in field("node_ids")]
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}
    edge_index = np.asarray(field("edge_index"), dtype=np.int64)
    if edge_index.shape[0] != 2:
        raise ValueError("graph edge_index must have shape [2, E]")
    adjacency = [set() for _ in range(n_nodes)]
    for source, target in zip(edge_index[0], edge_index[1]):
        u, v = int(source), int(target)
        if not (0 <= u < n_nodes and 0 <= v < n_nodes):
            raise ValueError("graph edge_index contains an out-of-range node")
        # Backwater and storage effects are not strictly downstream-only.
        adjacency[u].add(v)
        adjacency[v].add(u)

    try:
        endpoints = graph.get("facility_endpoints", [])
    except AttributeError:
        endpoints = []
    influence = np.zeros((n_facilities, n_nodes), dtype=np.float32)
    for facility_index, endpoint in enumerate(endpoints):
        starts = [
            node_index[str(endpoint.get(name, ""))]
            for name in ("from_node", "to_node")
            if str(endpoint.get(name, "")) in node_index
        ]
        if not starts:
            raise ValueError(
                f"facility {facility_index} has no graph-resolvable endpoint"
            )
        distances = {node: 0 for node in starts}
        queue: deque[int] = deque(starts)
        while queue:
            node = queue.popleft()
            distance = distances[node]
            if distance >= int(radius):
                continue
            for neighbour in adjacency[node]:
                if neighbour not in distances:
                    distances[neighbour] = distance + 1
                    queue.append(neighbour)
        for node, distance in distances.items():
            influence[facility_index, node] = 1.0 / float(1 + distance)

    if len(endpoints) != n_facilities:
        # Step1GraphAssets retains the canonical endpoint-only action map but
        # not the INP endpoint names. Recover the same start nodes directly
        # from that map so runtime and training share one influence contract.
        raw_map = np.asarray(field("action_node_map"), dtype=np.float32)
        if raw_map.shape != (n_facilities, n_nodes):
            raise ValueError("graph action_node_map shape differs from graph dimensions")
        influence.fill(0.0)
        for facility_index in range(n_facilities):
            starts = np.flatnonzero(np.isfinite(raw_map[facility_index]) & (raw_map[facility_index] != 0.0)).tolist()
            if not starts:
                raise ValueError(f"facility {facility_index} has no action-map endpoint")
            distances = {int(node): 0 for node in starts}
            queue = deque(starts)
            while queue:
                node = queue.popleft()
                distance = distances[node]
                if distance >= int(radius):
                    continue
                for neighbour in adjacency[node]:
                    if neighbour not in distances:
                        distances[neighbour] = distance + 1
                        queue.append(neighbour)
            for node, distance in distances.items():
                influence[facility_index, node] = 1.0 / float(1 + distance)

    if not np.isfinite(influence).all() or not np.any(influence):
        raise ValueError("surrogate action influence map is empty or nonfinite")
    return influence


@dataclass
class TrajectoryDatasetResult:
    manifest: pd.DataFrame
    graph_schema: dict
    node_feature_schema: dict
    edge_feature_schema: dict
    action_schema: dict
    sample_count: int = 0
    reference_dedup_count: int = 0
    warnings: list[str] = field(default_factory=list)


def _load_engineering36_ids(project_root: Path) -> list[str]:
    """Load the frozen Engineering36 order; fail if it is not exactly 36 IDs."""
    path = Path(project_root) / "data" / "project6_v8_storage_retrofit_control_enabled_ids.txt"
    if not path.exists():
        raise FileNotFoundError(f"Engineering36 order file not found: {path}")
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(ids) != N_FACILITIES:
        raise ValueError(
            f"Engineering36 contract requires {N_FACILITIES} IDs, found {len(ids)}"
        )
    folded = [x.casefold() for x in ids]
    if len(set(folded)) != len(ids):
        raise ValueError("Engineering36 order contains duplicate IDs")
    return ids


def _id_lookup(values: Iterable[str], *, what: str) -> dict[str, str]:
    """Build a case-insensitive unique lookup mapping canonicalised ID -> raw ID."""
    lookup: dict[str, str] = {}
    for raw in values:
        text = str(raw)
        key = text.casefold()
        if key in lookup and lookup[key] != text:
            raise ValueError(f"Ambiguous {what} IDs differing only by case: {lookup[key]!r}, {text!r}")
        lookup[key] = text
    return lookup


def _resolve_columns_by_ids(
    df: pd.DataFrame,
    prefix: str,
    expected_ids: list[str],
    *,
    allow_nan: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Extract prefixed columns in *expected ID order*, never source-file order."""
    prefixed = [str(c) for c in df.columns if str(c).startswith(prefix)]
    suffix_lookup = _id_lookup((c[len(prefix):] for c in prefixed), what=f"{prefix} column")
    raw_column_by_suffix = {str(c)[len(prefix):].casefold(): str(c) for c in prefixed}

    resolved: list[str] = []
    missing: list[str] = []
    for expected in expected_ids:
        key = str(expected).casefold()
        if key not in suffix_lookup:
            missing.append(str(expected))
            continue
        resolved.append(raw_column_by_suffix[key])
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required {prefix} columns: {missing[:10]}"
        )
    if len(resolved) != len(expected_ids):
        raise ValueError(
            f"Resolved {len(resolved)} {prefix} columns for {len(expected_ids)} expected IDs"
        )

    numeric = df[resolved].apply(pd.to_numeric, errors="coerce")
    if not allow_nan and numeric.isna().any().any():
        where = np.argwhere(numeric.isna().to_numpy())
        r, c = where[0]
        raise ValueError(
            f"NaN/non-numeric value in required column {resolved[int(c)]!r} at row {int(r)}"
        )
    arr = numeric.fillna(0.0).to_numpy(dtype=np.float32)
    return resolved, arr


def _load_graph_topology(project_root: Path) -> dict:
    """Load graph topology and build action-node mapping in Engineering36 order."""
    from sewerrtc.graph.graph_builder import build_node_link_graph, node_feature_matrix

    project_root = Path(project_root)
    inp_path = project_root / "data" / "wuhan_v8_storage_retrofit.inp"
    nodes_df, links_df = _parse_inp_topology(inp_path)
    nodes_enriched, links_enriched, edge_index = build_node_link_graph(nodes_df, links_df)
    node_static, node_static_cols = node_feature_matrix(nodes_enriched)

    node_ids = [str(x) for x in nodes_enriched["node_id"].tolist()]
    if len({x.casefold() for x in node_ids}) != len(node_ids):
        raise ValueError("Graph node IDs are not unique")
    node_index = {x.casefold(): i for i, x in enumerate(node_ids)}

    facility_ids = _load_engineering36_ids(project_root)
    if "link_id" not in links_enriched.columns:
        raise KeyError("graph_builder output has no link_id column")
    link_rows: dict[str, pd.Series] = {}
    for _, row in links_enriched.iterrows():
        link_id = str(row["link_id"])
        key = link_id.casefold()
        if key in link_rows:
            raise ValueError(f"Duplicate graph link ID {link_id!r}")
        link_rows[key] = row

    missing_facilities = [x for x in facility_ids if x.casefold() not in link_rows]
    if missing_facilities:
        raise ValueError(
            "Engineering36 facilities missing from parsed INP graph: "
            f"{missing_facilities}"
        )

    n_nodes = len(node_ids)
    action_node_map = np.zeros((N_FACILITIES, n_nodes), dtype=np.float32)
    facility_endpoints: list[dict[str, str]] = []
    for idx, facility_id in enumerate(facility_ids):
        row = link_rows[facility_id.casefold()]
        from_n = str(row.get("from_node", ""))
        to_n = str(row.get("to_node", ""))
        for endpoint in (from_n, to_n):
            key = endpoint.casefold()
            if key not in node_index:
                raise ValueError(
                    f"Facility {facility_id} endpoint {endpoint!r} missing from graph nodes"
                )
            action_node_map[idx, node_index[key]] = 1.0
        facility_endpoints.append(
            {"facility_id": facility_id, "from_node": from_n, "to_node": to_n}
        )

    return {
        "n_nodes": n_nodes,
        "n_edges": int(edge_index.shape[1]),
        "edge_index": edge_index,
        "node_static": node_static,
        "node_static_cols": node_static_cols,
        "action_node_map": action_node_map,
        "node_ids": node_ids,
        "facility_ids": facility_ids,
        "facility_endpoints": facility_endpoints,
        "n_facilities": N_FACILITIES,
    }


def _parse_inp_topology(inp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal INP parser for node/link topology used by the V4.2 graph."""
    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    current_section = ""
    with open(inp_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if stripped.startswith("["):
                current_section = stripped.split("]")[0].lstrip("[").strip().upper()
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            if current_section == "JUNCTIONS":
                nodes.append(
                    {
                        "node_id": parts[0],
                        "invert": float(parts[1]) if len(parts) > 1 else 0.0,
                        "max_depth": float(parts[2]) if len(parts) > 2 else 0.0,
                        "ponded_area": 0.0,
                        "node_type": "junction",
                    }
                )
            elif current_section == "STORAGE":
                nodes.append(
                    {
                        "node_id": parts[0],
                        "invert": float(parts[1]) if len(parts) > 1 else 0.0,
                        "max_depth": float(parts[2]) if len(parts) > 2 else 0.0,
                        "ponded_area": float(parts[3]) if len(parts) > 3 else 0.0,
                        "node_type": "storage",
                    }
                )
            elif current_section == "OUTFALLS":
                nodes.append(
                    {
                        "node_id": parts[0],
                        "invert": float(parts[1]) if len(parts) > 1 else 0.0,
                        "max_depth": 0.0,
                        "ponded_area": 0.0,
                        "node_type": "outfall",
                    }
                )
            elif current_section == "CONDUITS":
                links.append(
                    {
                        "link_id": parts[0],
                        "from_node": parts[1],
                        "to_node": parts[2],
                        "link_type": "conduit",
                    }
                )
            elif current_section in ("PUMPS", "ORIFICES", "WEIRS", "OUTLETS"):
                links.append(
                    {
                        "link_id": parts[0],
                        "from_node": parts[1],
                        "to_node": parts[2],
                        "link_type": current_section.lower().rstrip("s"),
                    }
                )
    nodes_df = pd.DataFrame(nodes)
    links_df = pd.DataFrame(links)
    if nodes_df.empty:
        raise ValueError(f"No nodes parsed from {inp_path}")
    if links_df.empty:
        raise ValueError(f"No links parsed from {inp_path}")
    return nodes_df, links_df


def _read_detail(detail_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(detail_path)
        if df.empty:
            return None
        return df
    except Exception as exc:
        logger.warning("Failed to read detail %s: %s", detail_path, exc)
        return None


def _extract_columns(df: pd.DataFrame, prefix: str) -> tuple[list[str], np.ndarray]:
    """Legacy generic extractor retained for diagnostics/tests.

    Formal trajectory construction must use :func:`_resolve_columns_by_ids`.
    """
    cols = [str(c) for c in df.columns if str(c).startswith(prefix)]
    if not cols:
        return [], np.zeros((len(df), 0), dtype=np.float32)
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    return cols, numeric.fillna(0.0).to_numpy(dtype=np.float32)


def _indices_for_times(elapsed: np.ndarray, times: list[float]) -> list[int]:
    if np.isnan(elapsed).any():
        raise ValueError("elapsed_min contains NaN")
    out: list[int] = []
    for t in times:
        matches = np.flatnonzero(np.isclose(elapsed, t, atol=TIME_ATOL_MIN, rtol=0.0))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one detail row at elapsed_min={t}, found {len(matches)}"
            )
        out.append(int(matches[0]))
    if len(set(out)) != len(out):
        raise ValueError("Temporal extraction produced duplicate row indices")
    return out


def _extract_trajectory_windows(
    detail: pd.DataFrame,
    checkpoint_min: float,
    *,
    n_history: int = N_HISTORY_FRAMES,
    n_horizon: int = N_HORIZON_STEPS,
    history_interval: int = HISTORY_INTERVAL_MIN,
    horizon_interval: int = HORIZON_INTERVAL_MIN,
    expected_node_ids: list[str] | None = None,
    expected_facility_ids: list[str] | None = None,
    require_rainfall: bool = False,
) -> dict[str, np.ndarray] | None:
    """Extract ID-aligned history/future windows at exact contract timestamps."""
    if "elapsed_min" not in detail.columns:
        raise KeyError("detail.csv is missing elapsed_min")
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(dtype=np.float64)
    history_times = [
        checkpoint_min - (n_history - 1 - i) * history_interval
        for i in range(n_history)
    ]
    future_times = [
        checkpoint_min + (i + 1) * horizon_interval
        for i in range(n_horizon)
    ]
    try:
        history_indices = _indices_for_times(elapsed, history_times)
        future_indices = _indices_for_times(elapsed, future_times)
    except ValueError as exc:
        logger.warning("Temporal alignment failed at checkpoint %.3f: %s", checkpoint_min, exc)
        return None

    if expected_node_ids is not None:
        h_cols, h_arr = _resolve_columns_by_ids(detail, "h:", expected_node_ids)
        f_cols, f_arr = _resolve_columns_by_ids(detail, "flood:", expected_node_ids)
    else:
        h_cols, h_arr = _extract_columns(detail, "h:")
        f_cols, f_arr = _extract_columns(detail, "flood:")

    if expected_facility_ids is not None:
        a_cols, a_arr = _resolve_columns_by_ids(detail, "a:", expected_facility_ids)
    else:
        a_cols, a_arr = _extract_columns(detail, "a:")

    rain_col = "rainfall_mm_h"
    if rain_col not in detail.columns:
        if require_rainfall:
            raise KeyError("detail.csv is missing rainfall_mm_h")
        rain_arr = np.zeros((len(detail), 1), dtype=np.float32)
    else:
        rain_numeric = pd.to_numeric(detail[rain_col], errors="coerce")
        if rain_numeric.isna().any():
            raise ValueError("rainfall_mm_h contains NaN/non-numeric values")
        rain_arr = rain_numeric.to_numpy(dtype=np.float32)[:, None]

    sv_cols, sv_arr = _extract_columns(detail, "storage_volume:")
    return {
        "history_depth": h_arr[history_indices],
        "history_actions": a_arr[history_indices],
        "history_flood": f_arr[history_indices],
        "history_rainfall": rain_arr[history_indices].ravel(),
        "history_storage_volume": sv_arr[history_indices],
        "trajectory_depth": h_arr[future_indices],
        "trajectory_actions": a_arr[future_indices],
        "trajectory_flood": f_arr[future_indices],
        "trajectory_rainfall": rain_arr[future_indices].ravel(),
        "trajectory_storage_volume": sv_arr[future_indices],
        "history_elapsed_min": np.asarray(history_times, dtype=np.float32),
        "future_elapsed_min": np.asarray(future_times, dtype=np.float32),
        "checkpoint_min": float(checkpoint_min),
        "node_cols": h_cols,
        "action_cols": a_cols,
        "flood_cols": f_cols,
        "storage_cols": sv_cols,
    }


def _integration_dt_seconds(trajectory: dict[str, np.ndarray], fallback_dt_sec: float) -> np.ndarray:
    times = np.asarray(trajectory.get("future_elapsed_min", []), dtype=np.float64)
    if times.size == 0:
        return np.full(N_HORIZON_STEPS, float(fallback_dt_sec), dtype=np.float64)
    checkpoint = float(trajectory.get("checkpoint_min", times[0] - HORIZON_INTERVAL_MIN))
    boundaries = np.concatenate([[checkpoint], times])
    dt = np.diff(boundaries) * 60.0
    if len(dt) != len(times) or np.any(dt <= 0):
        raise ValueError("Invalid/non-monotonic future timestamps for KPI integration")
    return dt


def _compute_kpis_from_trajectory(
    trajectory: dict[str, np.ndarray],
    priority_node_indices: list[int],
    dt_sec: int = 600,
) -> dict[str, float]:
    """Compute PFV/TFV volume and Peak rate from ID-aligned flood-rate arrays."""
    flood_traj = np.asarray(trajectory["trajectory_flood"], dtype=np.float64)
    if flood_traj.ndim != 2 or flood_traj.shape[0] == 0:
        raise ValueError("trajectory_flood must be a non-empty [H, N] array")
    if not np.isfinite(flood_traj).all():
        raise ValueError("trajectory_flood contains NaN/Inf")
    dt = _integration_dt_seconds(trajectory, dt_sec)
    if dt.shape[0] != flood_traj.shape[0]:
        raise ValueError("Future time-step count does not match flood trajectory")

    total_rate = flood_traj.sum(axis=1)
    tfv = float(np.sum(total_rate * dt))
    peak = float(np.max(total_rate))

    if not priority_node_indices:
        raise PriorityContractError("PFV requires the frozen PFV_CORE8 node set")
    if any(i < 0 or i >= flood_traj.shape[1] for i in priority_node_indices):
        raise PriorityContractError("Priority node index is outside the flood trajectory")
    priority_rate = flood_traj[:, priority_node_indices].sum(axis=1)
    pfv = float(np.sum(priority_rate * dt))
    return {"PFV": pfv, "TFV": tfv, "peak_TFV_rate": peak}


def _scan_run_dirs(train_root: Path) -> dict[str, Path]:
    case_map: dict[str, Path] = {}
    for round_dir in sorted(train_root.glob("round*/runs")):
        if not round_dir.is_dir():
            continue
        for run_dir in sorted(round_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            completion = run_dir / "completion.json"
            if not completion.exists():
                continue
            try:
                comp = json.loads(completion.read_text(encoding="utf-8"))
                case_id = str(comp.get("case_id", ""))
                if case_id:
                    case_map[case_id] = run_dir
            except Exception:
                continue
    return case_map


def _scan_all_run_dirs(output_root: Path) -> dict[str, Path]:
    case_map: dict[str, Path] = {}
    for name in ("train1600_v3", "train1600"):
        root = Path(output_root) / name
        if root.exists():
            case_map.update(_scan_run_dirs(root))
    return case_map


def _build_case_id(row: pd.Series) -> str:
    return str(row.get("case_id", ""))


def _require_reference_branches(ref_data: dict[str, dict], *, case_id: str) -> None:
    missing = [name for name in REFERENCE_BRANCHES if name not in ref_data]
    if missing:
        raise ValueError(
            f"{case_id}: missing required reference branches {missing}; "
            "V4.2 full supervision is fail-closed"
        )


def _require_action_matrix(arr: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != (n_rows, n_cols):
        raise ValueError(
            f"Expected action matrix {(n_rows, n_cols)}, got {tuple(arr.shape)}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("Action matrix contains NaN/Inf")
    return arr


def _node_order_sha(node_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(node_ids).encode("utf-8")).hexdigest()


def build_trajectory_dataset(
    *,
    project_root: Path,
    output_root: Path,
    config: dict,
    train_only: bool = True,
) -> TrajectoryDatasetResult:
    """Build fail-closed V4.2 trajectories from accepted Train1600 samples."""
    project_root = Path(project_root)
    output_root = Path(output_root)
    warnings: list[str] = []

    manifest_path = output_root / "train1600_v3" / "dataset" / "train1600_v3_sample_manifest.csv"
    if not manifest_path.exists():
        frozen = sorted(
            (output_root / "audits" / "frozen_evidence" / "train1600_v3").glob(
                "*/dataset/train1600_v3_sample_manifest.csv"
            )
        )
        if frozen:
            manifest_path = frozen[0]
        else:
            raise FileNotFoundError(f"Train1600 manifest not found at {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    from sewerrtc.v4.train_v4_loader import compute_acceptance

    manifest = manifest.loc[compute_acceptance(manifest)].copy()
    if train_only:
        manifest = manifest.loc[manifest["split"].astype(str) == "train"].copy()
    if manifest.empty:
        raise ValueError("No accepted Train split samples found")

    ledger_path = output_root / "inventory" / "event_usage_ledger.csv"
    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        train_event_ids = set(
            ledger.loc[ledger["assigned_split"].astype(str) == "train", "event_id"].astype(str)
        )
        non_train = set(manifest["event_id"].astype(str)) - train_event_ids
        if non_train:
            warnings.append(
                f"manifest has {len(non_train)} events not assigned to train: {sorted(non_train)[:5]}"
            )
    else:
        warnings.append(f"event_usage_ledger not found at {ledger_path}")

    graph = _load_graph_topology(project_root)
    node_ids = graph["node_ids"]
    facility_ids = graph["facility_ids"]
    priority_node_indices = get_pfv_core_node_indices(list(node_ids))

    case_map = _scan_all_run_dirs(output_root)
    if not case_map:
        raise FileNotFoundError("No run directories found. Cannot extract trajectories.")

    records: list[dict[str, Any]] = []
    reference_cache: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
    skipped = 0

    for _, row in manifest.iterrows():
        event_id = str(row["event_id"])
        checkpoint_id = str(row["checkpoint_id"])
        checkpoint_min = float(row.get("checkpoint_min", 100.0))
        case_id = _build_case_id(row)
        run_dir = case_map.get(case_id)
        if run_dir is None:
            skipped += 1
            warnings.append(f"run dir not found for case_id={case_id}")
            continue
        completion_path = run_dir / "completion.json"
        if not completion_path.exists():
            skipped += 1
            warnings.append(f"completion.json missing for {case_id}")
            continue
        try:
            comp = json.loads(completion_path.read_text(encoding="utf-8"))
            branches = comp.get("branches", {})

            cand_branch = branches.get("candidate", {})
            cand_path = Path(str(cand_branch.get("detail_path", "")))
            if not cand_path.exists():
                raise FileNotFoundError(f"candidate detail missing: {cand_path}")
            cand_detail = _read_detail(cand_path)
            if cand_detail is None:
                raise ValueError("candidate detail is empty/unreadable")
            cand_traj = _extract_trajectory_windows(
                cand_detail,
                checkpoint_min,
                expected_node_ids=node_ids,
                expected_facility_ids=facility_ids,
                require_rainfall=True,
            )
            if cand_traj is None:
                raise ValueError("candidate trajectory temporal alignment failed")

            ref_key = (event_id, checkpoint_id)
            if ref_key not in reference_cache:
                ref_data: dict[str, dict[str, np.ndarray]] = {}
                for branch_name in REFERENCE_BRANCHES:
                    br_path_text = str(branches.get(branch_name, {}).get("detail_path", ""))
                    br_path = Path(br_path_text)
                    if not br_path_text or not br_path.exists():
                        continue
                    br_detail = _read_detail(br_path)
                    if br_detail is None:
                        continue
                    br_traj = _extract_trajectory_windows(
                        br_detail,
                        checkpoint_min,
                        expected_node_ids=node_ids,
                        expected_facility_ids=facility_ids,
                        require_rainfall=True,
                    )
                    if br_traj is not None:
                        ref_data[branch_name] = br_traj
                _require_reference_branches(ref_data, case_id=case_id)
                reference_cache[ref_key] = ref_data
            ref_data = reference_cache[ref_key]
            _require_reference_branches(ref_data, case_id=case_id)

            cand_action = _require_action_matrix(
                cand_traj["trajectory_actions"], N_HORIZON_STEPS, N_FACILITIES
            )
            history_action = _require_action_matrix(
                cand_traj["history_actions"], N_HISTORY_FRAMES, N_FACILITIES
            )
            ref_actions = {
                name: _require_action_matrix(
                    ref_data[name]["trajectory_actions"], N_HORIZON_STEPS, N_FACILITIES
                )
                for name in REFERENCE_BRANCHES
            }

            cand_kpis = _compute_kpis_from_trajectory(cand_traj, priority_node_indices)
            ref_kpis = {
                name: _compute_kpis_from_trajectory(ref_data[name], priority_node_indices)
                for name in REFERENCE_BRANCHES
            }
            pfv_delta = cand_kpis["PFV"] - ref_kpis["no_control"]["PFV"]
            tfv_delta = cand_kpis["TFV"] - ref_kpis["dynamic_internal_rules"]["TFV"]
            peak_delta = (
                cand_kpis["peak_TFV_rate"]
                - ref_kpis["dynamic_internal_rules"]["peak_TFV_rate"]
            )

            record = {
                "event_id": event_id,
                "checkpoint_id": checkpoint_id,
                "state_key": f"{event_id}::{checkpoint_id}",
                "split": "train",
                "case_id": case_id,
                "checkpoint_min": checkpoint_min,
                "node_order_sha256": _node_order_sha(node_ids),
                "facility_order_sha256": hashlib.sha256(
                    "\n".join(facility_ids).encode("utf-8")
                ).hexdigest(),
                "candidate_action_seq": json.dumps(cand_action.tolist(), allow_nan=False),
                "ref_no_control_action_seq": json.dumps(
                    ref_actions["no_control"].tolist(), allow_nan=False
                ),
                "ref_dynamic_internal_action_seq": json.dumps(
                    ref_actions["dynamic_internal_rules"].tolist(), allow_nan=False
                ),
                "ref_hold_previous_action_seq": json.dumps(
                    ref_actions["hold_previous"].tolist(), allow_nan=False
                ),
                "history_depth": json.dumps(cand_traj["history_depth"].tolist(), allow_nan=False),
                "history_actions": json.dumps(history_action.tolist(), allow_nan=False),
                "history_elapsed_min": json.dumps(
                    cand_traj["history_elapsed_min"].tolist(), allow_nan=False
                ),
                "future_elapsed_min": json.dumps(
                    cand_traj["future_elapsed_min"].tolist(), allow_nan=False
                ),
                "trajectory_depth_candidate": json.dumps(
                    cand_traj["trajectory_depth"].tolist(), allow_nan=False
                ),
                "trajectory_depth_no_control": json.dumps(
                    ref_data["no_control"]["trajectory_depth"].tolist(), allow_nan=False
                ),
                "trajectory_depth_dynamic_internal": json.dumps(
                    ref_data["dynamic_internal_rules"]["trajectory_depth"].tolist(),
                    allow_nan=False,
                ),
                "trajectory_depth_hold_previous": json.dumps(
                    ref_data["hold_previous"]["trajectory_depth"].tolist(),
                    allow_nan=False,
                ),
                "trajectory_flood_candidate": json.dumps(
                    cand_traj["trajectory_flood"].tolist(), allow_nan=False
                ),
                "trajectory_flood_no_control": json.dumps(
                    ref_data["no_control"]["trajectory_flood"].tolist(), allow_nan=False
                ),
                "trajectory_flood_dynamic_internal": json.dumps(
                    ref_data["dynamic_internal_rules"]["trajectory_flood"].tolist(),
                    allow_nan=False,
                ),
                "trajectory_flood_hold_previous": json.dumps(
                    ref_data["hold_previous"]["trajectory_flood"].tolist(),
                    allow_nan=False,
                ),
                "rainfall_forecast": json.dumps(
                    cand_traj["trajectory_rainfall"].tolist(), allow_nan=False
                ),
                "pfv_candidate_m3": cand_kpis["PFV"],
                "pfv_no_control_m3": ref_kpis["no_control"]["PFV"],
                "tfv_candidate_m3": cand_kpis["TFV"],
                "tfv_dynamic_internal_m3": ref_kpis["dynamic_internal_rules"]["TFV"],
                "peak_candidate_m3s": cand_kpis["peak_TFV_rate"],
                "peak_dynamic_internal_m3s": ref_kpis["dynamic_internal_rules"]["peak_TFV_rate"],
                "pfv_delta": pfv_delta,
                "tfv_delta": tfv_delta,
                "peak_delta": peak_delta,
                "pfv_safe_label": int(pfv_delta <= 0.0),
                "tfv_improved_label": int(tfv_delta <= 0.0),
                "peak_noninferior_label": int(peak_delta <= 0.0),
            }
            # Preserve grouping evidence when present in the source manifest.
            for key in ("rainfall_sha256", "actual_schedule_sha256", "candidate_family"):
                if key in row.index:
                    record[key] = row[key]
            records.append(record)
        except Exception as exc:
            skipped += 1
            warnings.append(f"{case_id}: rejected: {type(exc).__name__}: {exc}")
            continue

    if not records:
        raise ValueError("No trajectory records passed the fail-closed V4.2 contract")
    result_df = pd.DataFrame(records)

    graph_schema = {
        "n_nodes": graph["n_nodes"],
        "n_edges": graph["n_edges"],
        "n_facilities": N_FACILITIES,
        "facility_ids": facility_ids,
        "n_history_frames": N_HISTORY_FRAMES,
        "n_horizon_steps": N_HORIZON_STEPS,
        "history_interval_min": HISTORY_INTERVAL_MIN,
        "horizon_interval_min": HORIZON_INTERVAL_MIN,
        "node_static_dim": len(NODE_STATIC_COLS),
        "node_static_columns": NODE_STATIC_COLS,
        "edge_index_shape": [2, graph["n_edges"]],
        "action_node_map_shape": [N_FACILITIES, graph["n_nodes"]],
    }
    node_feature_schema = {
        "n_nodes": graph["n_nodes"],
        "node_ids": node_ids,
        "depth_columns": node_ids,
        "flood_columns": node_ids,
        "priority_node_ids": list(PFV_CORE_8_IDS),
    }
    edge_feature_schema = {
        "n_edges": graph["n_edges"],
        "edge_index_shape": [2, graph["n_edges"]],
    }
    action_schema = {
        "n_facilities": N_FACILITIES,
        "facility_ids": facility_ids,
        "action_node_map_shape": [N_FACILITIES, graph["n_nodes"]],
        "horizon_steps": N_HORIZON_STEPS,
        "history_frames": N_HISTORY_FRAMES,
        "alignment": "by_facility_id_fail_closed",
    }
    return TrajectoryDatasetResult(
        manifest=result_df,
        graph_schema=graph_schema,
        node_feature_schema=node_feature_schema,
        edge_feature_schema=edge_feature_schema,
        action_schema=action_schema,
        sample_count=len(records),
        reference_dedup_count=len(reference_cache),
        warnings=warnings,
    )


def _pad_action_matrix(arr: np.ndarray, n_cols: int) -> np.ndarray:
    """Legacy helper retained for compatibility; formal builder does not use it."""
    out = np.zeros((N_HORIZON_STEPS, n_cols), dtype=float)
    if arr.size == 0:
        return out
    rows = min(arr.shape[0], N_HORIZON_STEPS)
    cols = min(arr.shape[1], n_cols) if arr.ndim > 1 else min(arr.shape[0], n_cols)
    if arr.ndim == 1:
        out[0, :cols] = arr[:cols]
    else:
        out[:rows, :cols] = arr[:rows, :cols]
    return out


def write_trajectory_dataset(
    result: TrajectoryDatasetResult,
    output_dir: Path,
) -> dict[str, str]:
    """Write trajectory dataset and semantic schemas."""
    from sewerrtc.v4.runtime import atomic_write_json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    parquet_path = output_dir / "trajectory_manifest_v42.parquet"
    result.manifest.to_parquet(parquet_path, index=False)
    written["manifest_parquet"] = str(parquet_path)
    csv_path = output_dir / "trajectory_manifest_v42.csv"
    result.manifest.to_csv(csv_path, index=False)
    written["manifest_csv"] = str(csv_path)

    for name, schema in [
        ("graph_schema_v42.json", result.graph_schema),
        ("node_feature_schema_v42.json", result.node_feature_schema),
        ("edge_feature_schema_v42.json", result.edge_feature_schema),
        ("action_schema_v42.json", result.action_schema),
    ]:
        path = output_dir / name
        atomic_write_json(path, schema)
        written[name] = str(path)

    summary = {
        "sample_count": result.sample_count,
        "reference_dedup_count": result.reference_dedup_count,
        "n_warnings": len(result.warnings),
        "warnings": result.warnings[:100],
        "schema": result.graph_schema,
        "fail_closed_references": True,
        "id_aligned_nodes": True,
        "id_aligned_engineering36": True,
    }
    summary_path = output_dir / "trajectory_dataset_v42_summary.json"
    atomic_write_json(summary_path, summary)
    written["summary"] = str(summary_path)
    return written
