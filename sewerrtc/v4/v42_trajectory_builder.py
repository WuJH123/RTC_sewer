"""V4.2 trajectory dataset builder.

Builds a spatial-temporal trajectory dataset from the frozen V4.1 Train1600
split.  Each sample contains 13-frame history (5-min intervals, 60 min look-back)
and 12-step future (10-min intervals, 120 min horizon) for four branches:
Candidate, No-control, Dynamic Internal Rules, Hold Previous.

Reference branches are deduplicated per (event_id, checkpoint_id) to save disk.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.v4.v42_priority_contract import PFV_CORE_8_IDS, get_pfv_core_node_indices, PriorityContractError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FACILITIES = 36
N_HISTORY_FRAMES = 13  # 13 frames × 5 min = 60 min look-back
N_HORIZON_STEPS = 12
HISTORY_INTERVAL_MIN = 5
HORIZON_INTERVAL_MIN = 10
NODE_STATIC_COLS = [
    "invert", "max_depth", "ponded_area",
    "degree_in", "degree_out", "is_storage", "is_outfall",
]

BRANCH_ROLES = ("candidate", "no_control", "dynamic_internal_rules", "hold_previous")

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Graph topology helpers
# ---------------------------------------------------------------------------

def _load_graph_topology(project_root: Path) -> dict:
    """Load graph topology using graph_builder and return schema dict."""
    from sewerrtc.graph.graph_builder import (
        build_node_link_graph,
        make_actuator_table,
        node_feature_matrix,
    )

    inp_path = project_root / "data" / "wuhan_v8_storage_retrofit.inp"
    nodes_csv = project_root / "data" / "project6_v3_facility_semantics_36.csv"

    # Parse INP to get node/link DataFrames (use existing parser)
    nodes_df, links_df = _parse_inp_topology(inp_path)

    nodes_enriched, links_enriched, edge_index = build_node_link_graph(
        nodes_df, links_df
    )
    node_static, node_static_cols = node_feature_matrix(nodes_enriched)
    actuators = make_actuator_table(nodes_enriched, links_enriched, max_actuators=36)

    node_ids = list(nodes_enriched["node_id"])
    node_index = {n: i for i, n in enumerate(node_ids)}

    # Build action_node_map [36, N]
    n_nodes = len(node_ids)
    action_node_map = np.zeros((N_FACILITIES, n_nodes), dtype=np.float32)
    for _, row in actuators.iterrows():
        idx = int(row.get("actuator_index", 0))
        if idx >= N_FACILITIES:
            continue
        from_n = str(row.get("from_node", ""))
        to_n = str(row.get("to_node", ""))
        if from_n in node_index:
            action_node_map[idx, node_index[from_n]] = 1.0
        if to_n in node_index:
            action_node_map[idx, node_index[to_n]] = 1.0

    return {
        "n_nodes": n_nodes,
        "n_edges": int(edge_index.shape[1]),
        "edge_index": edge_index,
        "node_static": node_static,
        "node_static_cols": node_static_cols,
        "action_node_map": action_node_map,
        "node_ids": node_ids,
        "n_facilities": min(len(actuators), N_FACILITIES),
    }


def _parse_inp_topology(inp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal INP parser for node/link topology."""
    nodes: list[dict] = []
    links: list[dict] = []
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
                nodes.append({
                    "node_id": parts[0],
                    "invert": float(parts[1]) if len(parts) > 1 else 0.0,
                    "max_depth": float(parts[2]) if len(parts) > 2 else 0.0,
                    "ponded_area": 0.0,
                    "node_type": "junction",
                })
            elif current_section == "STORAGE":
                nodes.append({
                    "node_id": parts[0],
                    "invert": float(parts[1]) if len(parts) > 1 else 0.0,
                    "max_depth": float(parts[2]) if len(parts) > 2 else 0.0,
                    "ponded_area": float(parts[3]) if len(parts) > 3 else 0.0,
                    "node_type": "storage",
                })
            elif current_section == "OUTFALLS":
                nodes.append({
                    "node_id": parts[0],
                    "invert": float(parts[1]) if len(parts) > 1 else 0.0,
                    "max_depth": 0.0,
                    "ponded_area": 0.0,
                    "node_type": "outfall",
                })
            elif current_section == "CONDUITS":
                links.append({
                    "link_id": parts[0],
                    "from_node": parts[1],
                    "to_node": parts[2],
                    "link_type": "conduit",
                })
            elif current_section in ("PUMPS", "ORIFICES", "WEIRS", "OUTLETS"):
                links.append({
                    "link_id": parts[0],
                    "from_node": parts[1],
                    "to_node": parts[2],
                    "link_type": current_section.lower().rstrip("s"),
                })

    nodes_df = pd.DataFrame(nodes)
    links_df = pd.DataFrame(links)
    if nodes_df.empty:
        raise ValueError(f"No nodes parsed from {inp_path}")
    if links_df.empty:
        raise ValueError(f"No links parsed from {inp_path}")
    return nodes_df, links_df


# ---------------------------------------------------------------------------
# Detail CSV reader
# ---------------------------------------------------------------------------

def _read_detail(detail_path: Path) -> pd.DataFrame | None:
    """Read a SWMM detail CSV, returning None on failure."""
    try:
        df = pd.read_csv(detail_path)
        if df.empty:
            return None
        return df
    except Exception as exc:
        logger.warning("Failed to read detail %s: %s", detail_path, exc)
        return None


def _extract_columns(
    df: pd.DataFrame, prefix: str
) -> tuple[list[str], np.ndarray]:
    """Extract columns matching a prefix (e.g. 'h:', 'a:', 'flood:')."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return [], np.zeros((len(df), 0), dtype=np.float32)
    arr = df[cols].fillna(0.0).to_numpy(dtype=np.float32)
    return cols, arr


def _extract_trajectory_windows(
    detail: pd.DataFrame,
    checkpoint_min: float,
    *,
    n_history: int = N_HISTORY_FRAMES,
    n_horizon: int = N_HORIZON_STEPS,
    history_interval: int = HISTORY_INTERVAL_MIN,
    horizon_interval: int = HORIZON_INTERVAL_MIN,
) -> dict[str, np.ndarray] | None:
    """Extract history and future trajectory windows from a detail CSV.

    History: n_history frames at history_interval-min spacing before checkpoint.
    Future: n_horizon steps at horizon_interval-min spacing after checkpoint.
    """
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(
        dtype=np.float64
    )

    # History indices: checkpoint_min, checkpoint_min - interval, ...
    history_times = [
        checkpoint_min - (n_history - 1 - i) * history_interval
        for i in range(n_history)
    ]
    history_indices = []
    for t in history_times:
        idx = int(np.argmin(np.abs(elapsed - t)))
        history_indices.append(idx)

    # Future indices: checkpoint_min + interval, + 2*interval, ...
    future_times = [
        checkpoint_min + (i + 1) * horizon_interval for i in range(n_horizon)
    ]
    future_indices = []
    for t in future_times:
        idx = int(np.argmin(np.abs(elapsed - t)))
        future_indices.append(idx)

    # Validate indices are within bounds
    all_indices = history_indices + future_indices
    if max(all_indices) >= len(detail) or min(all_indices) < 0:
        return None

    # Extract node depth columns
    h_cols, h_arr = _extract_columns(detail, "h:")
    # Extract action columns
    a_cols, a_arr = _extract_columns(detail, "a:")
    # Extract flood columns
    f_cols, f_arr = _extract_columns(detail, "flood:")
    # Extract rainfall
    rain_col = "rainfall_mm_h"
    rain_arr = (
        detail[[rain_col]].fillna(0.0).to_numpy(dtype=np.float32)
        if rain_col in detail.columns
        else np.zeros((len(detail), 1), dtype=np.float32)
    )
    # Extract storage volumes
    sv_cols, sv_arr = _extract_columns(detail, "storage_volume:")

    result = {
        "history_depth": h_arr[history_indices],  # [7, N]
        "history_actions": a_arr[history_indices],  # [7, n_links]
        "history_flood": f_arr[history_indices],  # [7, N]
        "history_rainfall": rain_arr[history_indices].ravel(),  # [7]
        "history_storage_volume": sv_arr[history_indices],  # [7, n_storage]
        "trajectory_depth": h_arr[future_indices],  # [12, N]
        "trajectory_actions": a_arr[future_indices],  # [12, n_links]
        "trajectory_flood": f_arr[future_indices],  # [12, N]
        "trajectory_rainfall": rain_arr[future_indices].ravel(),  # [12]
        "trajectory_storage_volume": sv_arr[future_indices],  # [12, n_storage]
        "node_cols": h_cols,
        "action_cols": a_cols,
        "flood_cols": f_cols,
        "storage_cols": sv_cols,
    }
    return result


# ---------------------------------------------------------------------------
# KPI computation from trajectories
# ---------------------------------------------------------------------------

def _compute_kpis_from_trajectory(
    trajectory: dict[str, np.ndarray],
    priority_node_indices: list[int],
    dt_sec: int = 600,
) -> dict[str, float]:
    """Compute PFV, TFV, Peak from trajectory arrays."""
    flood_traj = trajectory["trajectory_flood"]  # [12, N]
    if flood_traj.size == 0:
        return {"PFV": 0.0, "TFV": 0.0, "peak_TFV_rate": 0.0}

    total_rate = flood_traj.sum(axis=1)  # [12]
    tfv = float(total_rate.sum() * dt_sec)
    peak = float(total_rate.max())

    if priority_node_indices:
        valid_pr = [i for i in priority_node_indices if i < flood_traj.shape[1]]
        if valid_pr:
            pfv = float(flood_traj[:, valid_pr].sum() * dt_sec)
        else:
            pfv = 0.0
    else:
        pfv = 0.0

    return {"PFV": pfv, "TFV": tfv, "peak_TFV_rate": peak}


# ---------------------------------------------------------------------------
# Run directory scanner
# ---------------------------------------------------------------------------

def _scan_run_dirs(train_root: Path) -> dict[str, Path]:
    """Build a mapping from case_id -> run directory for all train rounds."""
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
                # case_id is at top level of completion.json
                case_id = str(comp.get("case_id", ""))
                if case_id:
                    case_map[case_id] = run_dir
            except Exception:
                continue
    return case_map


def _scan_all_run_dirs(output_root: Path) -> dict[str, Path]:
    """Scan all possible locations for run directories."""
    case_map: dict[str, Path] = {}
    # Search under train1600_v3
    train_root = output_root / "train1600_v3"
    if train_root.exists():
        case_map.update(_scan_run_dirs(train_root))
    # Also search under train1600
    train_root2 = output_root / "train1600"
    if train_root2.exists():
        case_map.update(_scan_run_dirs(train_root2))
    return case_map


# ---------------------------------------------------------------------------
# Manifest case_id builder
# ---------------------------------------------------------------------------

def _build_case_id(row: pd.Series) -> str:
    """Reconstruct the case_id from manifest row fields."""
    return str(row.get("case_id", ""))


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_trajectory_dataset(
    *,
    project_root: Path,
    output_root: Path,
    config: dict,
    train_only: bool = True,
) -> TrajectoryDatasetResult:
    """Build the V4.2 trajectory dataset from Train split data.

    Parameters
    ----------
    project_root : Path
        Project root (contains sewerrtc/, data/, configs/).
    output_root : Path
        Output root (contains project6_dual_reference_v4/final_v4/...).
    config : dict
        Pipeline configuration dict.
    train_only : bool
        If True, only process Train split samples.

    Returns
    -------
    TrajectoryDatasetResult
        The assembled trajectory dataset with schemas and statistics.
    """
    project_root = Path(project_root)
    output_root = Path(output_root)
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Load the Train1600 manifest
    # ------------------------------------------------------------------
    manifest_path = (
        output_root
        / "train1600_v3"
        / "dataset"
        / "train1600_v3_sample_manifest.csv"
    )
    if not manifest_path.exists():
        # Try frozen evidence path
        frozen = sorted(
            (output_root / "audits" / "frozen_evidence" / "train1600_v3").glob(
                "*/dataset/train1600_v3_sample_manifest.csv"
            )
        )
        if frozen:
            manifest_path = frozen[0]
        else:
            raise FileNotFoundError(
                f"Train1600 manifest not found at {manifest_path}"
            )

    logger.info("Loading manifest from %s", manifest_path)
    manifest = pd.read_csv(manifest_path)

    # ------------------------------------------------------------------
    # 2. Filter to accepted Train split samples
    # ------------------------------------------------------------------
    from sewerrtc.v4.train_v4_loader import compute_acceptance

    accepted_mask = compute_acceptance(manifest)
    manifest = manifest.loc[accepted_mask].copy()

    if train_only:
        manifest = manifest.loc[manifest["split"].astype(str) == "train"].copy()

    logger.info("Processing %d accepted Train split samples", len(manifest))

    if manifest.empty:
        raise ValueError("No accepted Train split samples found")

    # ------------------------------------------------------------------
    # 3. Load event usage ledger (verify train events)
    # ------------------------------------------------------------------
    ledger_path = output_root / "inventory" / "event_usage_ledger.csv"
    train_event_ids: set[str] = set()
    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        train_events = ledger.loc[
            ledger["assigned_split"].astype(str) == "train"
        ]
        train_event_ids = set(train_events["event_id"].astype(str))
        logger.info("Train split has %d events from ledger", len(train_event_ids))

        # Verify manifest events are in train split
        manifest_events = set(manifest["event_id"].astype(str))
        non_train = manifest_events - train_event_ids
        if non_train:
            warnings.append(
                f"manifest has {len(non_train)} events not in train split: "
                f"{sorted(non_train)[:5]}"
            )
    else:
        warnings.append(f"event_usage_ledger not found at {ledger_path}")
        train_event_ids = set(manifest["event_id"].astype(str))

    # ------------------------------------------------------------------
    # 4. Load graph topology
    # ------------------------------------------------------------------
    logger.info("Loading graph topology")
    graph = _load_graph_topology(project_root)
    n_nodes = graph["n_nodes"]
    node_ids = graph["node_ids"]
    node_index = {n: i for i, n in enumerate(node_ids)}

    # Load priority nodes via contract — fail-closed
    try:
        priority_node_indices = get_pfv_core_node_indices(list(node_ids))
    except Exception as exc:
        raise PriorityContractError(
            f"Failed to resolve PFV core 8 indices: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # 5. Scan run directories
    # ------------------------------------------------------------------
    logger.info("Scanning run directories")
    case_map = _scan_all_run_dirs(output_root)
    logger.info("Found %d run directories", len(case_map))

    if not case_map:
        raise FileNotFoundError(
            "No run directories found. Cannot extract trajectories."
        )

    # ------------------------------------------------------------------
    # 6. Process each sample
    # ------------------------------------------------------------------
    records: list[dict[str, Any]] = []
    reference_cache: dict[tuple[str, str], dict] = {}  # (event, checkpoint) -> ref data
    skipped = 0
    processed = 0

    for row_idx, (_, row) in enumerate(manifest.iterrows()):
        event_id = str(row["event_id"])
        checkpoint_id = str(row["checkpoint_id"])
        checkpoint_min = float(row.get("checkpoint_min", 100.0))
        case_id = _build_case_id(row)

        # Find run directory
        run_dir = case_map.get(case_id)
        if run_dir is None:
            skipped += 1
            if skipped <= 5:
                warnings.append(f"run dir not found for case_id={case_id}")
            continue

        completion_path = run_dir / "completion.json"
        if not completion_path.exists():
            skipped += 1
            continue

        try:
            comp = json.loads(completion_path.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue

        branches = comp.get("branches", {})

        # --- Load candidate branch ---
        cand_branch = branches.get("candidate", {})
        cand_detail_path = cand_branch.get("detail_path", "")
        if not cand_detail_path or not Path(cand_detail_path).exists():
            skipped += 1
            continue

        cand_detail = _read_detail(Path(cand_detail_path))
        if cand_detail is None:
            skipped += 1
            continue

        cand_traj = _extract_trajectory_windows(cand_detail, checkpoint_min)
        if cand_traj is None:
            skipped += 1
            warnings.append(
                f"trajectory window extraction failed for {case_id}"
            )
            continue

        # --- Load reference branches (with dedup) ---
        ref_key = (event_id, checkpoint_id)
        if ref_key in reference_cache:
            ref_data = reference_cache[ref_key]
        else:
            ref_data = {}
            for branch_name in ("no_control", "dynamic_internal_rules", "hold_previous"):
                br = branches.get(branch_name, {})
                br_path = br.get("detail_path", "")
                if br_path and Path(br_path).exists():
                    br_detail = _read_detail(Path(br_path))
                    if br_detail is not None:
                        br_traj = _extract_trajectory_windows(
                            br_detail, checkpoint_min
                        )
                        if br_traj is not None:
                            ref_data[branch_name] = br_traj
                        else:
                            warnings.append(
                                f"ref trajectory failed: {case_id}/{branch_name}"
                            )
            reference_cache[ref_key] = ref_data

        # --- Extract action sequences ---
        cand_action_seq = cand_traj["trajectory_actions"]  # [12, n_links]
        # Pad/trim to [12, 36]
        cand_action_seq = _pad_action_matrix(cand_action_seq, N_FACILITIES)

        ref_action_seqs = {}
        ref_depth_seqs = {}
        for branch_name in ("no_control", "dynamic_internal_rules", "hold_previous"):
            if branch_name in ref_data:
                ref_action_seqs[branch_name] = _pad_action_matrix(
                    ref_data[branch_name]["trajectory_actions"], N_FACILITIES
                )
                ref_depth_seqs[branch_name] = ref_data[branch_name][
                    "trajectory_depth"
                ]

        # --- Compute KPIs and labels ---
        cand_kpis = _compute_kpis_from_trajectory(
            cand_traj, priority_node_indices
        )
        ref_kpis = {}
        for branch_name in ("no_control", "dynamic_internal_rules", "hold_previous"):
            if branch_name in ref_data:
                ref_kpis[branch_name] = _compute_kpis_from_trajectory(
                    ref_data[branch_name], priority_node_indices
                )

        # PFV delta vs no_control
        pfv_no_control = ref_kpis.get("no_control", {}).get("PFV", cand_kpis["PFV"])
        pfv_delta = cand_kpis["PFV"] - pfv_no_control

        # TFV delta vs dynamic_internal_rules
        tfv_di = ref_kpis.get("dynamic_internal_rules", {}).get(
            "TFV", cand_kpis["TFV"]
        )
        tfv_delta = cand_kpis["TFV"] - tfv_di

        # Peak delta vs dynamic_internal_rules
        peak_di = ref_kpis.get("dynamic_internal_rules", {}).get(
            "peak_TFV_rate", cand_kpis["peak_TFV_rate"]
        )
        peak_delta = cand_kpis["peak_TFV_rate"] - peak_di

        # Classification labels (consistent with V4.1 semantics)
        pfv_safe = int(pfv_delta <= 0.0)  # candidate PFV <= no_control PFV
        tfv_improved = int(tfv_delta <= 0.0)  # candidate TFV <= DI TFV
        peak_noninferior = int(peak_delta <= 0.0)  # candidate Peak <= DI Peak

        # --- Build record ---
        record = {
            "event_id": event_id,
            "checkpoint_id": checkpoint_id,
            "state_key": f"{event_id}::{checkpoint_id}",
            "split": "train",
            # Action sequences
            "candidate_action_seq": json.dumps(
                cand_action_seq.tolist(), allow_nan=False
            ),
            "ref_no_control_action_seq": json.dumps(
                ref_action_seqs.get(
                    "no_control",
                    np.ones((N_HORIZON_STEPS, N_FACILITIES), dtype=float),
                ).tolist(),
                allow_nan=False,
            ),
            "ref_dynamic_internal_action_seq": json.dumps(
                ref_action_seqs.get(
                    "dynamic_internal_rules",
                    np.ones((N_HORIZON_STEPS, N_FACILITIES), dtype=float),
                ).tolist(),
                allow_nan=False,
            ),
            "ref_hold_previous_action_seq": json.dumps(
                ref_action_seqs.get(
                    "hold_previous",
                    np.ones((N_HORIZON_STEPS, N_FACILITIES), dtype=float),
                ).tolist(),
                allow_nan=False,
            ),
            # History
            "history_depth": json.dumps(
                cand_traj["history_depth"].tolist(), allow_nan=False
            ),
            "history_actions": json.dumps(
                _pad_action_matrix(
                    cand_traj["history_actions"], N_FACILITIES
                ).tolist(),
                allow_nan=False,
            ),
            # Future depth trajectories
            "trajectory_depth_candidate": json.dumps(
                cand_traj["trajectory_depth"].tolist(), allow_nan=False
            ),
            "trajectory_depth_no_control": json.dumps(
                ref_depth_seqs.get(
                    "no_control",
                    np.zeros((N_HORIZON_STEPS, n_nodes), dtype=float),
                ).tolist(),
                allow_nan=False,
            ),
            "trajectory_depth_dynamic_internal": json.dumps(
                ref_depth_seqs.get(
                    "dynamic_internal_rules",
                    np.zeros((N_HORIZON_STEPS, n_nodes), dtype=float),
                ).tolist(),
                allow_nan=False,
            ),
            "trajectory_depth_hold_previous": json.dumps(
                ref_depth_seqs.get(
                    "hold_previous",
                    np.zeros((N_HORIZON_STEPS, n_nodes), dtype=float),
                ).tolist(),
                allow_nan=False,
            ),
            # Rainfall forecast
            "rainfall_forecast": json.dumps(
                cand_traj["trajectory_rainfall"].tolist(), allow_nan=False
            ),
            # KPI deltas and labels
            "pfv_delta": pfv_delta,
            "tfv_delta": tfv_delta,
            "peak_delta": peak_delta,
            "pfv_safe_label": pfv_safe,
            "tfv_improved_label": tfv_improved,
            "peak_noninferior_label": peak_noninferior,
        }
        records.append(record)
        processed += 1

        if (row_idx + 1) % 100 == 0:
            logger.info(
                "Processed %d/%d samples (%d skipped)",
                processed + skipped,
                len(manifest),
                skipped,
            )

    logger.info(
        "Trajectory extraction complete: %d processed, %d skipped, "
        "%d reference dedup pairs",
        processed,
        skipped,
        len(reference_cache),
    )

    if not records:
        raise ValueError(
            "No trajectory records were built. Check data paths and "
            "detail.csv availability."
        )

    # ------------------------------------------------------------------
    # 7. Build result
    # ------------------------------------------------------------------
    result_df = pd.DataFrame(records)

    graph_schema = {
        "n_nodes": graph["n_nodes"],
        "n_edges": graph["n_edges"],
        "n_facilities": graph["n_facilities"],
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
        "node_ids": graph["node_ids"],
        "depth_columns": [
            c.removeprefix("h:") for c in cand_traj["node_cols"]
        ] if "cand_traj" in dir() else [],
        "flood_columns": [
            c.removeprefix("flood:") for c in cand_traj["flood_cols"]
        ] if "cand_traj" in dir() else [],
    }

    edge_feature_schema = {
        "n_edges": graph["n_edges"],
        "edge_index_shape": [2, graph["n_edges"]],
    }

    action_schema = {
        "n_facilities": N_FACILITIES,
        "action_node_map_shape": [N_FACILITIES, graph["n_nodes"]],
        "horizon_steps": N_HORIZON_STEPS,
        "history_frames": N_HISTORY_FRAMES,
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
    """Pad or trim action matrix to [12, n_cols]."""
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


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_trajectory_dataset(
    result: TrajectoryDatasetResult,
    output_dir: Path,
) -> dict[str, str]:
    """Write trajectory dataset to disk as Parquet + JSON schemas.

    Returns a dict of written file paths.
    """
    from sewerrtc.v4.runtime import atomic_write_json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    # Parquet manifest
    parquet_path = output_dir / "trajectory_manifest_v42.parquet"
    result.manifest.to_parquet(parquet_path, index=False)
    written["manifest_parquet"] = str(parquet_path)

    # Also write CSV for human inspection
    csv_path = output_dir / "trajectory_manifest_v42.csv"
    result.manifest.to_csv(csv_path, index=False)
    written["manifest_csv"] = str(csv_path)

    # JSON schemas
    for name, schema in [
        ("graph_schema_v42.json", result.graph_schema),
        ("node_feature_schema_v42.json", result.node_feature_schema),
        ("edge_feature_schema_v42.json", result.edge_feature_schema),
        ("action_schema_v42.json", result.action_schema),
    ]:
        path = output_dir / name
        atomic_write_json(path, schema)
        written[name] = str(path)

    # Dataset summary
    summary = {
        "sample_count": result.sample_count,
        "reference_dedup_count": result.reference_dedup_count,
        "n_warnings": len(result.warnings),
        "warnings": result.warnings[:20],
        "schema": result.graph_schema,
    }
    summary_path = output_dir / "trajectory_dataset_v42_summary.json"
    atomic_write_json(summary_path, summary)
    written["summary"] = str(summary_path)

    logger.info(
        "Wrote trajectory dataset: %d samples, %d ref dedup pairs, "
        "%d files to %s",
        result.sample_count,
        result.reference_dedup_count,
        len(written),
        output_dir,
    )
    return written
