from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .gat_audit import PROJECT4_CACHE, sha256_file
from .state_contract import FACILITY_STATE_FIELDS, NODE_STATE_FIELDS, STORAGE_STATE_FIELDS


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


def _float_or_nan(value: str | float | int | None, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


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
    rows = _read_csv(detail_path)
    if not rows:
        return None, None, None, {"status": "blocked", "failure_reason": "detail_file_empty"}
    elapsed = [_float_or_nan(row.get("elapsed_min")) for row in rows]
    by_elapsed = {round(value, 6): idx for idx, value in enumerate(elapsed) if not np.isnan(value)}
    if decision_elapsed_min is None:
        eligible_indices = [
            i
            for i, value in enumerate(elapsed)
            if not np.isnan(value)
            and value >= 60.0
            and all(round(value + offset, 6) in by_elapsed for offset in [0, -10, -20, -30, -40, -50, -60])
        ]
    else:
        eligible_indices = []
        for value in decision_elapsed_min:
            key = round(value, 6)
            idx = by_elapsed.get(key)
            if idx is None:
                continue
            if value >= 60.0 and all(round(value + offset, 6) in by_elapsed for offset in [0, -10, -20, -30, -40, -50, -60]):
                eligible_indices.append(idx)
    if max_samples and max_samples > 0:
        eligible_indices = eligible_indices[:max_samples]
    if not eligible_indices:
        return None, None, None, {"status": "blocked", "failure_reason": "no_60min_history_samples"}

    columns = list(rows[0].keys())
    node_ids = [col[2:] for col in columns if col.startswith("h:")]
    if not node_ids:
        return None, None, None, {"status": "blocked", "failure_reason": "no_node_depth_columns"}
    frame_offsets = [0, -10, -20, -30, -40, -50, -60]
    node_features = list(NODE_STATE_FIELDS)
    facility_features = list(FACILITY_STATE_FIELDS)
    storage_features = list(STORAGE_STATE_FIELDS)
    facility_type_codes = {"orifice": 1.0, "weir": 2.0, "pump": 3.0}
    state_history = np.full((len(eligible_indices), 7, len(node_ids), len(node_features)), np.nan, dtype=np.float32)
    facility_history = np.full((len(eligible_indices), 7, len(facility_ids), len(facility_features)), np.nan, dtype=np.float32)
    storage_history = np.full((len(eligible_indices), 7, 0, len(storage_features)), np.nan, dtype=np.float32)
    causality_failures: list[str] = []
    for sample_i, row_index in enumerate(eligible_indices):
        decision_elapsed = elapsed[row_index]
        for frame_i, offset in enumerate(frame_offsets):
            frame_elapsed = round(decision_elapsed + offset, 6)
            source_index = by_elapsed.get(frame_elapsed)
            if source_index is None:
                prior = [value for value in by_elapsed if value <= frame_elapsed]
                if not prior:
                    causality_failures.append(f"{trajectory_id}:{decision_elapsed}:missing_frame:{offset}")
                    continue
                source_elapsed = max(prior)
                source_index = by_elapsed[source_elapsed]
            else:
                source_elapsed = frame_elapsed
            source = rows[source_index]
            data_age = max(0.0, decision_elapsed - source_elapsed)
            for node_i, node_id in enumerate(node_ids):
                depth = _float_or_nan(source.get(f"h:{node_id}"))
                flood = _float_or_nan(source.get(f"flood:{node_id}"))
                observed = 0.0 if np.isnan(depth) else 1.0
                values = {
                    "reconstructed_depth": depth,
                    "observed_depth": depth,
                    "depth_source": 1.0,
                    "depth_quality": observed,
                    "filling_degree": np.nan,
                    "hydraulic_head": depth,
                    "depth_headroom": np.nan,
                    "rim_margin": np.nan,
                    "surcharge_margin": np.nan,
                    "flooding_rate": flood,
                    "node_type": 0.0,
                    "is_priority": 0.0,
                    "is_sentinel_candidate": 1.0 if node_id in {"MH0200770", "HS1355904"} else 0.0,
                    "is_storage": 0.0,
                    "observation_mask": observed,
                    "uncertainty": np.nan,
                    "ood_score": np.nan,
                }
                for feature_i, feature_name in enumerate(node_features):
                    state_history[sample_i, frame_i, node_i, feature_i] = _float_or_nan(values.get(feature_name), np.nan)
            for fac_i, facility_id in enumerate(facility_ids):
                setting = _float_or_nan(source.get(f"setting:{facility_id}") or source.get(f"a:{facility_id}"))
                flow = _float_or_nan(source.get(f"flow:{facility_id}"))
                previous_setting = setting
                prior_index = by_elapsed.get(round(source_elapsed - 10.0, 6))
                if prior_index is not None:
                    previous_setting = _float_or_nan(rows[prior_index].get(f"setting:{facility_id}") or rows[prior_index].get(f"a:{facility_id}"))
                facility_type = "pump" if facility_id in {"add350.1", "ADD301.2", "ADD301.3"} else ("orifice" if "RTC_" in facility_id or "." in facility_id else "weir")
                values = {
                    "facility_id": float(fac_i),
                    "facility_type": facility_type_codes.get(facility_type, 0.0),
                    "anchor_setting": setting,
                    "native_target_setting": setting,
                    "requested_setting": setting,
                    "projected_setting": setting,
                    "target_setting": setting,
                    "actual_current_setting": setting,
                    "previous_actual_setting": previous_setting,
                    "setting_rate": setting - previous_setting if np.isfinite(setting) and np.isfinite(previous_setting) else np.nan,
                    "upstream_head": np.nan,
                    "downstream_head": np.nan,
                    "head_difference": np.nan,
                    "local_flow": flow,
                    "capacity_ratio": np.nan,
                    "flow_direction": np.sign(flow) if np.isfinite(flow) else np.nan,
                    "flow_trend": np.nan,
                    "residual_override_active": 0.0,
                    "override_ttl": 0.0,
                    "released_to_native": 1.0,
                    "data_quality": 0.0 if np.isnan(setting) else 1.0,
                    "ood": np.nan,
                }
                for feature_i, feature_name in enumerate(facility_features):
                    facility_history[sample_i, frame_i, fac_i, feature_i] = _float_or_nan(values.get(feature_name), np.nan)

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
        event_id=event_id,
        trajectory_id=trajectory_id,
    )
    np.savez_compressed(
        facility_path,
        facility_history=facility_history,
        facility_ids=np.asarray(facility_ids, dtype=str),
        feature_names=np.asarray(facility_features, dtype=str),
        event_id=event_id,
        trajectory_id=trajectory_id,
    )
    np.savez_compressed(
        storage_path,
        storage_history=storage_history,
        storage_ids=np.asarray([], dtype=str),
        feature_names=np.asarray(storage_features, dtype=str),
        event_id=event_id,
        trajectory_id=trajectory_id,
    )
    report = {
        "status": "ready",
        "sample_count": len(eligible_indices),
        "node_count": len(node_ids),
        "facility_count": len(facility_ids),
        "state_shape": list(state_history.shape),
        "facility_shape": list(facility_history.shape),
        "storage_shape": list(storage_history.shape),
        "causality_failure_count": len(causality_failures),
        "causality_failures": causality_failures[:20],
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
        # This manifest intentionally does not point to fabricated Project6
        # facility/storage/controller-memory tensors. It is only valid for
        # node-level GAT state validation and must not unlock full augmented
        # Project6 state generation.
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
                "eligibility_status": "node_state_validation_only",
                "exclusion_reason": "diagnostic_contaminated_project4_cache_lacks_project6_facility_storage_pump_ttl_fallback_fields",
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
        rows = []
        for i, row in enumerate(holdout_rows[: max_samples or len(holdout_rows)]):
            rows.append(
                {
                    "sample_id": row.get("holdout_id") or f"gat_independent_holdout_{i:05d}",
                    "trajectory_id": row.get("holdout_id") or f"gat_independent_holdout_{i:05d}",
                    "event_id": row.get("event_id", ""),
                    "policy_id": "gat_independent_holdout",
                    "trajectory_key": "|".join([row.get("holdout_id") or f"gat_independent_holdout_{i:05d}", row.get("event_id", ""), "gat_independent_holdout"]),
                    "decision_time": "",
                    "source_mode": source_mode,
                    "state_history_path": row.get("node_truth_path") or row.get("cache_path") or row.get("trajectory_path", ""),
                    "facility_history_path": "",
                    "storage_history_path": "",
                    "frame_count": 7,
                    "history_window_min": 60,
                    "contains_future_data": "false",
                    "missing_flow_encoded_as_zero": "false",
                    "gat_node_state_validation_eligible": "true",
                    "full_project6_augmented_state_eligible": "false",
                    "eligibility_status": "gat_independent_node_state_validation",
                    "exclusion_reason": "",
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
            manifest_rows = [row for row in manifest_rows if row.get("trajectory_id", "") in selected_by_trajectory]
        rows: list[dict[str, Any]] = []
        gap_rows: list[dict[str, Any]] = []
        remaining = int(max_samples or 0)
        for row in manifest_rows:
            if remaining == 0 and max_samples:
                break
            detail_path = Path(row.get("detail_file", ""))
            if not detail_path.exists():
                gap_rows.append({"trajectory_id": row.get("trajectory_id", ""), "status": "blocked", "failure_reason": "detail_file_missing"})
                continue
            per_trajectory_limit = remaining if remaining > 0 else 0
            state_path, facility_path, storage_path, report = _build_project6_history_npz(
                detail_path=detail_path,
                trajectory_id=row.get("trajectory_id", ""),
                event_id=row.get("event_id", ""),
                facility_ids=facility_ids,
                out_dir=out_dir,
                max_samples=per_trajectory_limit,
                decision_elapsed_min=selected_by_trajectory.get(row.get("trajectory_id", "")) if selected_by_trajectory is not None else None,
            )
            gap_rows.append({"trajectory_id": row.get("trajectory_id", ""), **report})
            if report.get("status") != "ready" or state_path is None or facility_path is None or storage_path is None:
                continue
            sample_count = int(report["sample_count"])
            remaining = max(0, remaining - sample_count) if remaining > 0 else 0
            rows.append(
                {
                    "sample_id": row.get("trajectory_id", ""),
                    "trajectory_id": row.get("trajectory_id", ""),
                    "event_id": row.get("event_id", ""),
                    "policy_id": row.get("policy_id", ""),
                    "trajectory_key": "|".join([row.get("trajectory_id", ""), row.get("event_id", ""), row.get("policy_id", "")]),
                    "decision_time": "",
                    "source_mode": source_mode,
                    "state_history_path": str(state_path),
                    "facility_history_path": str(facility_path),
                    "storage_history_path": str(storage_path),
                    "frame_count": 7,
                    "history_window_min": 60,
                    "contains_future_data": "false",
                    "missing_flow_encoded_as_zero": "false",
                    "gat_node_state_validation_eligible": "true",
                    "full_project6_augmented_state_eligible": "true",
                    "eligibility_status": "project6_full_baseline_state_ready",
                    "exclusion_reason": "",
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
                    "status": "completed",
                    "source_mode": source_mode,
                    "trajectory_root": str(trajectory_root),
                    "baseline_trajectory_manifest": str(manifest),
                    "baseline_trajectory_manifest_sha256": sha256_file(manifest),
                    "manifest_rows": len(rows),
                    "processable_baseline_rows": len(manifest_rows),
                    "accepted_baseline_statuses": ["completed", "skipped_existing_with_detail_file"],
                    "gap_rows": gap_rows,
                    "full_project6_augmented_state_eligible": True,
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
            "allowed_source_modes": ["project4_diagnostic_contaminated", "gat_independent_holdout", "project6_retrofit_baseline"],
            "completion_marker": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return 7, paths
