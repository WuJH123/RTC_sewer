from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .gat_audit import PROJECT4_CACHE, sha256_file
from .state_contract import (
    FACILITY_STATE_FIELDS,
    NODE_STATE_FIELDS,
    STORAGE_STATE_FIELDS,
    TEMPORAL_FRAME_OFFSETS_MIN,
)


STATE_INPUT_COLUMNS = [
    "sample_id",
    "trajectory_id",
    "event_id",
    "policy_id",
    "trajectory_key",
    "decision_time",
    "source_mode",
    "state_history_path",
    "facility_history_path",
    "storage_history_path",
    "frame_count",
    "history_window_min",
    "contains_future_data",
    "missing_flow_encoded_as_zero",
    "gat_node_state_validation_eligible",
    "full_project6_augmented_state_eligible",
    "eligibility_status",
    "exclusion_reason",
]


V42_FRAME_COUNT = len(TEMPORAL_FRAME_OFFSETS_MIN)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _managed_facility_ids(root: Path) -> list[str]:
    ids_path = root / "data" / "project6_v8_storage_retrofit_control_enabled_ids.txt"
    ids: list[str] = []
    for line in ids_path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            ids.append(item)
    return ids


def _priority_ids(root: Path) -> set[str]:
    path = root / "data" / "project5_design" / "priority_pfv_core_nodes.txt"
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _float_or_nan(value: str | float | int | None, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _project6_topology(root: Path):
    # Reuse the exact parser used by the V4.2 trajectory builder so the state
    # truth package and surrogate graph share node/facility identity.
    from sewerrtc.v4.v42_trajectory_builder import _parse_inp_topology

    inp = root / "data" / "wuhan_v8_storage_retrofit.inp"
    nodes, links = _parse_inp_topology(inp)
    node_meta = {
        str(row.node_id): {
            "invert": float(row.invert),
            "max_depth": float(row.max_depth),
            "node_type": str(row.node_type),
        }
        for row in nodes.itertuples(index=False)
    }
    link_meta = {
        str(row.link_id): {
            "from_node": str(row.from_node),
            "to_node": str(row.to_node),
            "link_type": str(row.link_type),
        }
        for row in links.itertuples(index=False)
    }
    return node_meta, link_meta


def _exact_history_indices(
    elapsed: list[float],
    decision_elapsed_min: list[float] | None,
) -> list[int]:
    by_elapsed = {
        round(value, 6): idx
        for idx, value in enumerate(elapsed)
        if np.isfinite(value)
    }
    if decision_elapsed_min is None:
        candidate_values = [value for value in elapsed if np.isfinite(value) and value >= 60.0]
    else:
        candidate_values = [float(value) for value in decision_elapsed_min if np.isfinite(value)]
    eligible: list[int] = []
    for value in candidate_values:
        idx = by_elapsed.get(round(value, 6))
        if idx is None:
            continue
        if all(round(value + offset, 6) in by_elapsed for offset in TEMPORAL_FRAME_OFFSETS_MIN):
            eligible.append(idx)
    # Preserve input/trajectory order and remove duplicate decision rows.
    return list(dict.fromkeys(eligible))


def _build_project6_history_npz(
    *,
    detail_path: Path,
    trajectory_id: str,
    event_id: str,
    facility_ids: list[str],
    out_dir: Path,
    max_samples: int,
    decision_elapsed_min: list[float] | None = None,
) -> tuple[Path | None, Path | None, Path | None, dict[str, Any]]:
    """Materialise *true-state* Project6 history without pretending it is GAT output."""
    rows = _read_csv(detail_path)
    if not rows:
        return None, None, None, {"status": "blocked", "failure_reason": "detail_file_empty"}
    elapsed = [_float_or_nan(row.get("elapsed_min")) for row in rows]
    eligible_indices = _exact_history_indices(elapsed, decision_elapsed_min)
    if max_samples and max_samples > 0:
        eligible_indices = eligible_indices[:max_samples]
    if not eligible_indices:
        return None, None, None, {"status": "blocked", "failure_reason": "no_60min_5min_history_samples"}

    root = Path(__file__).resolve().parents[2]
    node_meta, link_meta = _project6_topology(root)
    priority_ids = _priority_ids(root)
    sentinels = {"MH0200770", "HS1355904"}

    columns = list(rows[0].keys())
    detail_node_ids = [col[2:] for col in columns if col.startswith("h:")]
    node_ids = [nid for nid in node_meta if nid in set(detail_node_ids)]
    if not node_ids:
        return None, None, None, {"status": "blocked", "failure_reason": "no_graph_aligned_node_depth_columns"}
    storage_ids = [nid for nid in node_ids if node_meta[nid]["node_type"] == "storage"]

    node_features = list(NODE_STATE_FIELDS)
    facility_features = list(FACILITY_STATE_FIELDS)
    storage_features = list(STORAGE_STATE_FIELDS)
    facility_type_codes = {"orifice": 1.0, "weir": 2.0, "pump": 3.0, "outlet": 4.0}

    sample_count = len(eligible_indices)
    state_history = np.full(
        (sample_count, V42_FRAME_COUNT, len(node_ids), len(node_features)),
        np.nan,
        dtype=np.float32,
    )
    facility_history = np.full(
        (sample_count, V42_FRAME_COUNT, len(facility_ids), len(facility_features)),
        np.nan,
        dtype=np.float32,
    )
    storage_history = np.full(
        (sample_count, V42_FRAME_COUNT, len(storage_ids), len(storage_features)),
        np.nan,
        dtype=np.float32,
    )

    by_elapsed = {
        round(value, 6): idx
        for idx, value in enumerate(elapsed)
        if np.isfinite(value)
    }
    causality_failures: list[str] = []
    missing_flow_count = 0
    for sample_i, row_index in enumerate(eligible_indices):
        decision_elapsed = float(elapsed[row_index])
        for frame_i, offset in enumerate(TEMPORAL_FRAME_OFFSETS_MIN):
            source_elapsed = round(decision_elapsed + offset, 6)
            source_index = by_elapsed.get(source_elapsed)
            if source_index is None:
                # Formal V4.2 state histories use exact 5-min frames.  Do not
                # fill a missing state from a future observation or fabricate it.
                causality_failures.append(
                    f"{trajectory_id}:{decision_elapsed}:missing_exact_frame:{offset}"
                )
                continue
            source = rows[source_index]

            for node_i, node_id in enumerate(node_ids):
                meta = node_meta[node_id]
                depth = _float_or_nan(source.get(f"h:{node_id}"))
                head = _float_or_nan(source.get(f"head:{node_id}"))
                flood = _float_or_nan(source.get(f"flood:{node_id}"), 0.0)
                observed = float(np.isfinite(depth))
                max_depth = float(meta["max_depth"])
                invert = float(meta["invert"])
                if not np.isfinite(head) and np.isfinite(depth):
                    head = invert + depth
                filling = (
                    depth / max_depth
                    if np.isfinite(depth) and max_depth > 1.0e-8
                    else np.nan
                )
                headroom = max_depth - depth if np.isfinite(depth) else np.nan
                values = {
                    # SWMM detail is authoritative truth.  It must never be
                    # relabelled as a GAT reconstruction.
                    "reconstructed_depth": np.nan,
                    "observed_depth": depth,
                    "depth_source": 2.0,
                    "depth_quality": observed,
                    "filling_degree": filling,
                    "hydraulic_head": head,
                    "depth_headroom": headroom,
                    "rim_margin": headroom,
                    "surcharge_margin": headroom,
                    "flooding_rate": flood,
                    "node_type": {"junction": 1.0, "storage": 2.0, "outfall": 3.0}.get(meta["node_type"], 0.0),
                    "is_priority": float(node_id in priority_ids),
                    "is_sentinel_candidate": float(node_id in sentinels),
                    "is_storage": float(meta["node_type"] == "storage"),
                    "observation_mask": observed,
                    "uncertainty": np.nan,
                    "ood_score": np.nan,
                }
                for feature_i, feature_name in enumerate(node_features):
                    state_history[sample_i, frame_i, node_i, feature_i] = _float_or_nan(
                        values.get(feature_name), np.nan
                    )

            prior_index = by_elapsed.get(round(source_elapsed - 5.0, 6))
            previous_source = rows[prior_index] if prior_index is not None else None
            for fac_i, facility_id in enumerate(facility_ids):
                meta = link_meta.get(facility_id, {})
                setting = _float_or_nan(
                    source.get(f"setting:{facility_id}")
                    if source.get(f"setting:{facility_id}") not in (None, "")
                    else source.get(f"a:{facility_id}")
                )
                flow = _float_or_nan(source.get(f"flow:{facility_id}"))
                if not np.isfinite(flow):
                    missing_flow_count += 1
                previous_setting = setting
                if previous_source is not None:
                    previous_setting = _float_or_nan(
                        previous_source.get(f"setting:{facility_id}")
                        if previous_source.get(f"setting:{facility_id}") not in (None, "")
                        else previous_source.get(f"a:{facility_id}")
                    )
                up_id = str(meta.get("from_node", ""))
                down_id = str(meta.get("to_node", ""))
                up_head = _float_or_nan(source.get(f"head:{up_id}")) if up_id else np.nan
                down_head = _float_or_nan(source.get(f"head:{down_id}")) if down_id else np.nan
                values = {
                    "facility_id": float(fac_i),
                    "facility_type": facility_type_codes.get(str(meta.get("link_type", "")), 0.0),
                    "anchor_setting": setting,
                    "native_target_setting": setting,
                    "requested_setting": setting,
                    "projected_setting": setting,
                    "target_setting": setting,
                    "actual_current_setting": setting,
                    "previous_actual_setting": previous_setting,
                    "setting_rate": (
                        (setting - previous_setting) / 5.0
                        if np.isfinite(setting) and np.isfinite(previous_setting)
                        else np.nan
                    ),
                    "upstream_head": up_head,
                    "downstream_head": down_head,
                    "head_difference": (
                        up_head - down_head
                        if np.isfinite(up_head) and np.isfinite(down_head)
                        else np.nan
                    ),
                    "local_flow": flow,
                    "capacity_ratio": np.nan,
                    "flow_direction": np.sign(flow) if np.isfinite(flow) else np.nan,
                    "flow_trend": np.nan,
                    "residual_override_active": 0.0,
                    "override_ttl": 0.0,
                    "released_to_native": 1.0,
                    "data_quality": float(np.isfinite(setting)),
                    "ood": np.nan,
                }
                for feature_i, feature_name in enumerate(facility_features):
                    facility_history[sample_i, frame_i, fac_i, feature_i] = _float_or_nan(
                        values.get(feature_name), np.nan
                    )

            for storage_i, storage_id in enumerate(storage_ids):
                depth = _float_or_nan(source.get(f"h:{storage_id}"))
                max_depth = float(node_meta[storage_id]["max_depth"])
                current_volume = _float_or_nan(source.get(f"storage_volume:{storage_id}"))
                values = {
                    "current_volume": current_volume,
                    # Full volume cannot be reconstructed from max depth alone
                    # for arbitrary SWMM storage curves.
                    "full_volume": np.nan,
                    "filling_ratio": (
                        depth / max_depth
                        if np.isfinite(depth) and max_depth > 1.0e-8
                        else np.nan
                    ),
                    "remaining_capacity": np.nan,
                    "inlet_flow": np.nan,
                    "outlet_flow": np.nan,
                    "net_flow": np.nan,
                    "depth": depth,
                    "headroom": max_depth - depth if np.isfinite(depth) else np.nan,
                    "terminal_risk_proxy": np.nan,
                    "data_source": 2.0,
                    "uncertainty": np.nan,
                }
                for feature_i, feature_name in enumerate(storage_features):
                    storage_history[sample_i, frame_i, storage_i, feature_i] = _float_or_nan(
                        values.get(feature_name), np.nan
                    )

    history_dir = out_dir / "project6_retrofit_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    state_path = history_dir / f"{trajectory_id}__state_history.npz"
    facility_path = history_dir / f"{trajectory_id}__facility_history.npz"
    storage_path = history_dir / f"{trajectory_id}__storage_history.npz"
    np.savez_compressed(
        state_path,
        state_history=state_history,
        node_ids=np.asarray(node_ids, dtype=str),
        feature_names=np.asarray(node_features, dtype=str),
        frame_offsets_min=np.asarray(TEMPORAL_FRAME_OFFSETS_MIN, dtype=np.float32),
        source_role="swmm_true_state_not_gat_reconstruction",
        event_id=event_id,
        trajectory_id=trajectory_id,
    )
    np.savez_compressed(
        facility_path,
        facility_history=facility_history,
        facility_ids=np.asarray(facility_ids, dtype=str),
        feature_names=np.asarray(facility_features, dtype=str),
        frame_offsets_min=np.asarray(TEMPORAL_FRAME_OFFSETS_MIN, dtype=np.float32),
        event_id=event_id,
        trajectory_id=trajectory_id,
    )
    np.savez_compressed(
        storage_path,
        storage_history=storage_history,
        storage_ids=np.asarray(storage_ids, dtype=str),
        feature_names=np.asarray(storage_features, dtype=str),
        frame_offsets_min=np.asarray(TEMPORAL_FRAME_OFFSETS_MIN, dtype=np.float32),
        event_id=event_id,
        trajectory_id=trajectory_id,
    )
    report = {
        "status": "ready",
        "sample_count": sample_count,
        "node_count": len(node_ids),
        "facility_count": len(facility_ids),
        "storage_count": len(storage_ids),
        "frame_count": V42_FRAME_COUNT,
        "state_shape": list(state_history.shape),
        "facility_shape": list(facility_history.shape),
        "storage_shape": list(storage_history.shape),
        "state_source": "swmm_true_state_not_gat_reconstruction",
        "causality_failure_count": len(causality_failures),
        "causality_failures": causality_failures[:20],
        "missing_flow_count": int(missing_flow_count),
        "gat_uncertainty_materialized": False,
        "ood_materialized": False,
        "full_gat_integrated_state_eligible": False,
    }
    return state_path, facility_path, storage_path, report


def build_state_input_manifest(
    *,
    source_mode: str,
    out_dir: Path,
    trajectory_root: Path | None = None,
    validation_manifest: Path | None = None,
    control_checkpoint_catalog: Path | None = None,
    max_samples: int = 100,
) -> tuple[int, dict[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "state_input_manifest_v1.csv"
    gap_path = out_dir / "state_trajectory_gap_report.json"
    paths = {"manifest": manifest_path, "gap_report": gap_path}

    # Historical Project4 assets remain admissible only for node-level GAT
    # validation.  Their 7-frame layout must never unlock the V4.2 formal state
    # pipeline.
    if source_mode in {"project4_gat_validation", "project4_diagnostic_contaminated"}:
        if not PROJECT4_CACHE.exists():
            _write_json(
                gap_path,
                {
                    "status": "blocked",
                    "failure_reason": "project4_cache_missing",
                    "cache_path": str(PROJECT4_CACHE),
                    "completion_marker": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return 3, paths
        rows = [
            {
                "sample_id": f"project4_gat_validation_{i:05d}",
                "trajectory_id": f"project4_gat_validation_{i:05d}",
                "event_id": "unknown_project4_cache_event",
                "policy_id": "project4_gat_validation",
                "trajectory_key": f"project4_gat_validation_{i:05d}|unknown_project4_cache_event|project4_gat_validation",
                "decision_time": "",
                "source_mode": source_mode,
                "state_history_path": str(PROJECT4_CACHE),
                "facility_history_path": "",
                "storage_history_path": "",
                "frame_count": 7,
                "history_window_min": 60,
                "contains_future_data": "false",
                "missing_flow_encoded_as_zero": "false",
                "gat_node_state_validation_eligible": "true",
                "full_project6_augmented_state_eligible": "false",
                "eligibility_status": "legacy_node_state_validation_only",
                "exclusion_reason": "legacy_7frame_project4_cache_not_formal_v42_state_input",
            }
            for i in range(max(1, int(max_samples)))
        ]
        _write_csv(manifest_path, rows, STATE_INPUT_COLUMNS)
        _write_json(
            gap_path,
            {
                "status": "node_state_validation_only",
                "source_mode": source_mode,
                "cache_path": str(PROJECT4_CACHE),
                "cache_sha256": sha256_file(PROJECT4_CACHE),
                "gat_node_state_validation_eligible": True,
                "full_project6_augmented_state_eligible": False,
                "formal_v42_state_eligible": False,
                "completion_marker": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 0, paths

    if source_mode == "gat_independent_holdout":
        if validation_manifest is None or not validation_manifest.exists():
            _write_json(
                gap_path,
                {
                    "status": "blocked",
                    "failure_reason": "gat_independent_holdout_manifest_required",
                    "validation_manifest": str(validation_manifest) if validation_manifest else "",
                    "completion_marker": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return 3, paths
        with validation_manifest.open("r", encoding="utf-8-sig", newline="") as f:
            holdout_rows = list(csv.DictReader(f))
        rows: list[dict[str, Any]] = []
        for i, row in enumerate(holdout_rows[: max_samples or len(holdout_rows)]):
            rows.append(
                {
                    "sample_id": row.get("holdout_id") or f"gat_independent_holdout_{i:05d}",
                    "trajectory_id": row.get("holdout_id") or f"gat_independent_holdout_{i:05d}",
                    "event_id": row.get("event_id", ""),
                    "policy_id": "gat_independent_holdout",
                    "trajectory_key": "|".join(
                        [
                            row.get("holdout_id") or f"gat_independent_holdout_{i:05d}",
                            row.get("event_id", ""),
                            "gat_independent_holdout",
                        ]
                    ),
                    "decision_time": "",
                    "source_mode": source_mode,
                    "state_history_path": row.get("node_truth_path")
                    or row.get("cache_path")
                    or row.get("trajectory_path", ""),
                    "facility_history_path": "",
                    "storage_history_path": "",
                    "frame_count": int(row.get("frame_count") or 7),
                    "history_window_min": 60,
                    "contains_future_data": "false",
                    "missing_flow_encoded_as_zero": "false",
                    "gat_node_state_validation_eligible": "true",
                    "full_project6_augmented_state_eligible": "false",
                    "eligibility_status": "gat_independent_node_state_validation",
                    "exclusion_reason": "node_truth_only_not_full_project6_state",
                }
            )
        if not rows:
            _write_json(
                gap_path,
                {
                    "status": "blocked",
                    "failure_reason": "gat_independent_holdout_manifest_empty",
                    "validation_manifest": str(validation_manifest),
                    "completion_marker": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return 3, paths
        _write_csv(manifest_path, rows, STATE_INPUT_COLUMNS)
        _write_json(
            gap_path,
            {
                "status": "node_state_validation_only",
                "source_mode": source_mode,
                "validation_manifest": str(validation_manifest),
                "validation_manifest_sha256": sha256_file(validation_manifest),
                "gat_node_state_validation_eligible": True,
                "full_project6_augmented_state_eligible": False,
                "formal_v42_state_eligible": False,
                "completion_marker": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 0, paths

    if source_mode == "project6_retrofit_baseline":
        if trajectory_root is None or not trajectory_root.exists():
            _write_json(
                gap_path,
                {
                    "status": "blocked",
                    "failure_reason": "project6_retrofit_trajectory_root_missing",
                    "trajectory_root": str(trajectory_root) if trajectory_root else "",
                    "completion_marker": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return 3, paths
        root = Path(__file__).resolve().parents[2]
        manifest = trajectory_root / "baseline_trajectory_manifest.csv"
        if not manifest.exists():
            _write_json(
                gap_path,
                {
                    "status": "blocked",
                    "failure_reason": "baseline_trajectory_manifest_missing",
                    "trajectory_root": str(trajectory_root),
                    "completion_marker": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return 3, paths
        facility_ids = _managed_facility_ids(root)
        selected_by_trajectory: dict[str, list[float]] | None = None
        if control_checkpoint_catalog is not None and control_checkpoint_catalog.exists():
            selected_by_trajectory = {}
            for checkpoint in _read_csv(control_checkpoint_catalog):
                if str(checkpoint.get("round0_candidate_eligible", "true")).lower() == "false":
                    continue
                trajectory_id = checkpoint.get("trajectory_id", "")
                if not trajectory_id:
                    continue
                elapsed_min = _float_or_nan(checkpoint.get("elapsed_min"))
                if np.isnan(elapsed_min):
                    continue
                selected_by_trajectory.setdefault(trajectory_id, []).append(elapsed_min)

        manifest_rows = [
            row
            for row in _read_csv(manifest)
            if row.get("status") == "completed"
            or (row.get("status") == "skipped_existing" and row.get("detail_file", "").strip())
        ]
        if selected_by_trajectory is not None:
            manifest_rows = [
                row
                for row in manifest_rows
                if row.get("trajectory_id", "") in selected_by_trajectory
            ]
        rows: list[dict[str, Any]] = []
        gap_rows: list[dict[str, Any]] = []
        remaining = int(max_samples or 0)
        for row in manifest_rows:
            if remaining == 0 and max_samples:
                break
            detail_path = Path(row.get("detail_file", ""))
            if not detail_path.exists():
                gap_rows.append(
                    {
                        "trajectory_id": row.get("trajectory_id", ""),
                        "status": "blocked",
                        "failure_reason": "detail_file_missing",
                    }
                )
                continue
            per_trajectory_limit = remaining if remaining > 0 else 0
            state_path, facility_path, storage_path, report = _build_project6_history_npz(
                detail_path=detail_path,
                trajectory_id=row.get("trajectory_id", ""),
                event_id=row.get("event_id", ""),
                facility_ids=facility_ids,
                out_dir=out_dir,
                max_samples=per_trajectory_limit,
                decision_elapsed_min=(
                    selected_by_trajectory.get(row.get("trajectory_id", ""))
                    if selected_by_trajectory is not None
                    else None
                ),
            )
            gap_rows.append({"trajectory_id": row.get("trajectory_id", ""), **report})
            if (
                report.get("status") != "ready"
                or state_path is None
                or facility_path is None
                or storage_path is None
            ):
                continue
            sample_count = int(report["sample_count"])
            remaining = max(0, remaining - sample_count) if remaining > 0 else 0
            rows.append(
                {
                    "sample_id": row.get("trajectory_id", ""),
                    "trajectory_id": row.get("trajectory_id", ""),
                    "event_id": row.get("event_id", ""),
                    "policy_id": row.get("policy_id", ""),
                    "trajectory_key": "|".join(
                        [
                            row.get("trajectory_id", ""),
                            row.get("event_id", ""),
                            row.get("policy_id", ""),
                        ]
                    ),
                    "decision_time": "",
                    "source_mode": source_mode,
                    "state_history_path": str(state_path),
                    "facility_history_path": str(facility_path),
                    "storage_history_path": str(storage_path),
                    "frame_count": V42_FRAME_COUNT,
                    "history_window_min": 60,
                    "contains_future_data": "false",
                    "missing_flow_encoded_as_zero": "false",
                    # The SWMM package is useful as true-state/offline truth but
                    # has no GAT uncertainty/OOD, so it cannot claim a complete
                    # online GAT-integrated augmented state.
                    "gat_node_state_validation_eligible": "true",
                    "full_project6_augmented_state_eligible": "false",
                    "eligibility_status": "project6_true_state_ready_not_gat_integrated",
                    "exclusion_reason": "requires_frozen_gat_inference_uncertainty_ood_for_formal_online_state",
                }
            )
        if not rows:
            _write_csv(manifest_path, [], STATE_INPUT_COLUMNS)
            _write_json(
                gap_path,
                {
                    "status": "blocked",
                    "failure_reason": "no_project6_retrofit_baseline_state_samples",
                    "trajectory_root": str(trajectory_root),
                    "gap_rows": gap_rows,
                    "completion_marker": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return 3, paths
        _write_csv(manifest_path, rows, STATE_INPUT_COLUMNS)
        _write_json(
            gap_path,
            {
                "status": "completed_true_state_only",
                "source_mode": source_mode,
                "trajectory_root": str(trajectory_root),
                "baseline_trajectory_manifest": str(manifest),
                "baseline_trajectory_manifest_sha256": sha256_file(manifest),
                "manifest_rows": len(rows),
                "processable_baseline_rows": len(manifest_rows),
                "history_frame_count": V42_FRAME_COUNT,
                "history_offsets_min": TEMPORAL_FRAME_OFFSETS_MIN,
                "state_source": "swmm_true_state_not_gat_reconstruction",
                "gat_node_state_validation_eligible": True,
                "full_project6_augmented_state_eligible": False,
                "formal_v42_gat_integrated_state_eligible": False,
                "gap_rows": gap_rows,
                "completion_marker": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 0, paths

    _write_json(
        gap_path,
        {
            "status": "failed",
            "failure_reason": "unsupported_source_mode",
            "source_mode": source_mode,
            "allowed_source_modes": [
                "project4_diagnostic_contaminated",
                "project4_gat_validation",
                "gat_independent_holdout",
                "project6_retrofit_baseline",
            ],
            "completion_marker": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return 7, paths
