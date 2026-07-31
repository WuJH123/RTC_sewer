from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .state_contract import (
    FACILITY_STATE_FIELDS,
    NODE_STATE_FIELDS,
    PUMP_STATE_FIELDS,
    STORAGE_STATE_FIELDS,
    TEMPORAL_FRAME_OFFSETS_MIN,
)


FORMAL_FRAME_COUNT = len(TEMPORAL_FRAME_OFFSETS_MIN)
LEGACY_NODE_ONLY_FRAME_COUNTS = {7, FORMAL_FRAME_COUNT}

REQUIRED_MANIFEST_COLUMNS = [
    "sample_id",
    "trajectory_id",
    "event_id",
    "policy_id",
    "trajectory_key",
    "decision_time",
    "state_history_path",
    "facility_history_path",
    "storage_history_path",
    "frame_count",
    "history_window_min",
    "contains_future_data",
    "missing_flow_encoded_as_zero",
    "gat_node_state_validation_eligible",
    "full_project6_augmented_state_eligible",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def _feature_index_rows(
    names: list[str], source: str, controller_visible: bool, unit: str = "mixed"
) -> list[dict[str, Any]]:
    return [
        {
            "feature_name": name,
            "index": i,
            "source": source,
            "controller_visible": str(bool(controller_visible)).lower(),
            "truth_only": str(not bool(controller_visible)).lower(),
            "unit": unit,
            "missing_encoding": "nan_with_quality_or_availability_feature",
            "causality": "current_or_past_only",
            "materialized_status": "materialized",
        }
        for i, name in enumerate(names)
    ]


def _write_feature_index(
    path: Path, names: list[str], source: str, controller_visible: bool
) -> Path:
    _write_json(
        path,
        {
            "feature_count": len(names),
            "features": _feature_index_rows(names, source, controller_visible),
        },
    )
    return path


def validate_state_input_manifest(
    path: Path, out_dir: Path
) -> tuple[bool, list[str], list[dict[str, str]]]:
    del out_dir
    failures: list[str] = []
    if not path.exists():
        failures.append(f"state_input_manifest_not_found:{path}")
        return False, failures, []
    rows = _read_csv(path)
    if not rows:
        failures.append("state_input_manifest_empty")
        return False, failures, []
    columns = set(rows[0].keys())
    for column in REQUIRED_MANIFEST_COLUMNS:
        if column not in columns:
            failures.append(f"missing_manifest_column:{column}")
    return not failures, failures, rows


def build_runtime_state_features(
    *,
    config_path: Path,
    lock_path: Path,
    state_input_manifest: Path,
    out_dir: Path,
    max_samples: int | None = None,
    state_validation_mode: str = "full_project6_augmented_state",
) -> tuple[int, dict[str, Path]]:
    """Validate materialised state tensors without upgrading truth to GAT state.

    Formal Project6 augmented state requires 13 five-minute frames and an input
    manifest that explicitly declares ``full_project6_augmented_state_eligible``.
    Historical Project4/holdout assets may still be audited in node-only modes,
    but a 7-frame node-only audit can never unlock the formal V4.2 controller.
    """
    del config_path
    node_only_modes = {"project4_node_only", "gat_independent_node_only"}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sample_manifest": out_dir / "augmented_state_sample_manifest.csv",
        "shape_audit": out_dir / "augmented_state_shape_audit.json",
        "causality_audit": out_dir / "augmented_state_causality_audit.csv",
        "missingness_audit": out_dir / "augmented_state_missingness_audit.csv",
        "facility_audit": out_dir / "augmented_state_facility_audit.csv",
        "node_feature_index": out_dir / "node_feature_index.json",
        "facility_feature_index": out_dir / "facility_feature_index.json",
        "storage_feature_index": out_dir / "storage_feature_index.json",
        "feature_materialization_audit": out_dir / "feature_materialization_audit.csv",
        "gap_report": out_dir / "state_input_gap_report.json",
    }
    failures: list[str] = []
    if not lock_path.exists():
        failures.append("primary_gat_lock_missing")
    _, manifest_failures, rows = validate_state_input_manifest(
        state_input_manifest, out_dir
    )
    failures.extend(manifest_failures)
    if failures:
        _write_json(
            paths["gap_report"],
            {
                "status": "blocked",
                "failures": failures,
                "required_manifest_columns": REQUIRED_MANIFEST_COLUMNS,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 3, paths

    base = state_input_manifest.parent
    rows = rows[:max_samples] if max_samples else rows
    sample_rows: list[dict[str, Any]] = []
    causality_rows: list[dict[str, Any]] = []
    missingness_rows: list[dict[str, Any]] = []
    facility_rows: list[dict[str, Any]] = []
    materialization_rows: list[dict[str, Any]] = []
    state_shapes: list[list[Any]] = []
    facility_shapes: list[list[Any]] = []
    storage_shapes: list[list[Any]] = []
    actual_node_feature_names: list[str] = []
    actual_facility_feature_names: list[str] = []
    actual_storage_feature_names: list[str] = []

    for row in rows:
        sample_failures: list[str] = []
        full_project6_eligible = str(
            row.get("full_project6_augmented_state_eligible", "")
        ).lower() in {"true", "1", "yes"}
        node_eligible = str(row.get("gat_node_state_validation_eligible", "")).lower() in {
            "true",
            "1",
            "yes",
        }
        node_validation_only = node_eligible and not full_project6_eligible
        node_only = state_validation_mode in node_only_modes

        if node_only:
            if not node_eligible:
                sample_failures.append("gat_node_state_validation_not_eligible")
        else:
            if node_validation_only:
                sample_failures.append(
                    "node_state_validation_only_not_full_project6_augmented_state"
                )
            if not full_project6_eligible:
                sample_failures.append("full_project6_augmented_state_not_eligible")

        try:
            frame_count = int(row.get("frame_count") or -1)
        except ValueError:
            frame_count = -1
        try:
            history_window = int(row.get("history_window_min") or -1)
        except ValueError:
            history_window = -1

        if node_only:
            if frame_count not in LEGACY_NODE_ONLY_FRAME_COUNTS:
                sample_failures.append("node_only_frame_count_unsupported")
        elif frame_count != FORMAL_FRAME_COUNT:
            sample_failures.append(
                f"formal_frame_count_not_{FORMAL_FRAME_COUNT}"
            )
        if history_window < 60:
            sample_failures.append("history_window_less_than_60min")
        if str(row.get("contains_future_data", "")).lower() not in {"false", "0", "no"}:
            sample_failures.append("future_data_present")
        if str(row.get("missing_flow_encoded_as_zero", "")).lower() not in {
            "false",
            "0",
            "no",
        }:
            sample_failures.append("missing_flow_encoded_as_zero")

        state_path = _resolve(base, row["state_history_path"])
        facility_path = _resolve(base, row["facility_history_path"])
        storage_path = _resolve(base, row.get("storage_history_path", ""))
        if not state_path.exists():
            sample_failures.append("state_history_path_missing")
        if not node_only and not facility_path.exists():
            sample_failures.append("facility_history_path_missing")
        if not node_only and not storage_path.exists():
            sample_failures.append("storage_history_path_missing")

        if node_only and not sample_failures:
            state_shapes.append([1, frame_count, "node_only", "depth_validation"])
            facility_shapes.append([0, 0, 0, 0])
            storage_shapes.append([0, 0, 0, 0])
        elif not sample_failures:
            try:
                state_npz = np.load(state_path, allow_pickle=False)
                facility_npz = np.load(facility_path, allow_pickle=False)
                storage_npz = np.load(storage_path, allow_pickle=False)
                state_history = state_npz["state_history"]
                facility_history = facility_npz["facility_history"]
                storage_history = storage_npz["storage_history"]
                state_feature_names = (
                    [str(v) for v in state_npz["feature_names"].tolist()]
                    if "feature_names" in state_npz.files
                    else []
                )
                facility_feature_names = (
                    [str(v) for v in facility_npz["feature_names"].tolist()]
                    if "feature_names" in facility_npz.files
                    else []
                )
                storage_feature_names = (
                    [str(v) for v in storage_npz["feature_names"].tolist()]
                    if "feature_names" in storage_npz.files
                    else []
                )
                if not actual_node_feature_names:
                    actual_node_feature_names = state_feature_names
                if not actual_facility_feature_names:
                    actual_facility_feature_names = facility_feature_names
                if not actual_storage_feature_names:
                    actual_storage_feature_names = storage_feature_names
                state_shapes.append(list(state_history.shape))
                facility_shapes.append(list(facility_history.shape))
                storage_shapes.append(list(storage_history.shape))
                if state_history.ndim != 4 or state_history.shape[1] != FORMAL_FRAME_COUNT:
                    sample_failures.append(
                        f"state_history_shape_not_samples_{FORMAL_FRAME_COUNT}_N_F"
                    )
                if (
                    facility_history.ndim != 4
                    or facility_history.shape[1] != FORMAL_FRAME_COUNT
                    or facility_history.shape[2] != 36
                ):
                    sample_failures.append(
                        f"facility_history_shape_not_samples_{FORMAL_FRAME_COUNT}_36_F"
                    )
                if storage_history.ndim != 4 or storage_history.shape[1] != FORMAL_FRAME_COUNT:
                    sample_failures.append(
                        f"storage_history_shape_not_samples_{FORMAL_FRAME_COUNT}_S_F"
                    )
                if state_history.ndim == 4 and state_history.shape[-1] != len(state_feature_names):
                    sample_failures.append("node_feature_name_count_mismatch")
                if facility_history.ndim == 4 and facility_history.shape[-1] != len(facility_feature_names):
                    sample_failures.append("facility_feature_name_count_mismatch")
                if storage_history.ndim == 4 and storage_history.shape[-1] != len(storage_feature_names):
                    sample_failures.append("storage_feature_name_count_mismatch")
                if state_feature_names != NODE_STATE_FIELDS:
                    sample_failures.append("node_feature_schema_mismatch")
                if facility_feature_names != FACILITY_STATE_FIELDS:
                    sample_failures.append("facility_feature_schema_mismatch")
                if storage_feature_names != STORAGE_STATE_FIELDS:
                    sample_failures.append("storage_feature_schema_mismatch")
            except Exception as exc:
                sample_failures.append(f"state_npz_load_failed:{exc}")

        sample_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "trajectory_id": row.get("trajectory_id"),
                "event_id": row.get("event_id"),
                "policy_id": row.get("policy_id"),
                "trajectory_key": row.get("trajectory_key"),
                "decision_time": row.get("decision_time"),
                "state_history_path": str(state_path),
                "facility_history_path": str(facility_path),
                "storage_history_path": str(storage_path),
                "status": "blocked" if sample_failures else "ready",
                "failure_reason": ";".join(sample_failures),
                "full_project6_augmented_state_eligible": str(
                    full_project6_eligible
                ).lower(),
                "gat_node_state_validation_eligible": str(node_eligible).lower(),
            }
        )
        for i, offset in enumerate(
            TEMPORAL_FRAME_OFFSETS_MIN if not node_only else range(frame_count)
        ):
            causality_rows.append(
                {
                    "sample_id": row.get("sample_id"),
                    "frame_index": i,
                    "offset_min": offset if not node_only else "legacy_declared",
                    "decision_time": row.get("decision_time"),
                    "valid_before_decision": str(
                        "future_data_present" not in sample_failures
                    ).lower(),
                    "aggregation_method": "declared_by_state_input_manifest",
                    "quality": "blocked" if sample_failures else "ready",
                }
            )
        missingness_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "missing_flow_encoded_as_zero": row.get("missing_flow_encoded_as_zero"),
                "missingness_status": (
                    "blocked"
                    if "missing_flow_encoded_as_zero" in sample_failures
                    else "declared_not_zero_filled"
                ),
            }
        )
        facility_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "facility_count_required": 36,
                "add350_speed_fields_required": True,
                "ADD301_binary_fields_required": True,
                "status": "blocked"
                if sample_failures
                else "ready_for_field_level_validation",
            }
        )

    _write_csv(
        paths["sample_manifest"],
        sample_rows,
        [
            "sample_id",
            "trajectory_id",
            "event_id",
            "policy_id",
            "trajectory_key",
            "decision_time",
            "state_history_path",
            "facility_history_path",
            "storage_history_path",
            "status",
            "failure_reason",
            "full_project6_augmented_state_eligible",
            "gat_node_state_validation_eligible",
        ],
    )
    _write_csv(
        paths["causality_audit"],
        causality_rows,
        [
            "sample_id",
            "frame_index",
            "offset_min",
            "decision_time",
            "valid_before_decision",
            "aggregation_method",
            "quality",
        ],
    )
    _write_csv(
        paths["missingness_audit"],
        missingness_rows,
        ["sample_id", "missing_flow_encoded_as_zero", "missingness_status"],
    )
    _write_csv(
        paths["facility_audit"],
        facility_rows,
        [
            "sample_id",
            "facility_count_required",
            "add350_speed_fields_required",
            "ADD301_binary_fields_required",
            "status",
        ],
    )

    _write_feature_index(
        paths["node_feature_index"],
        actual_node_feature_names or NODE_STATE_FIELDS,
        "project6_runtime_state_npz" if actual_node_feature_names else "contract_only_not_materialized",
        True,
    )
    _write_feature_index(
        paths["facility_feature_index"],
        actual_facility_feature_names or FACILITY_STATE_FIELDS,
        "project6_runtime_facility_npz" if actual_facility_feature_names else "contract_only_not_materialized",
        True,
    )
    _write_feature_index(
        paths["storage_feature_index"],
        actual_storage_feature_names or STORAGE_STATE_FIELDS,
        "project6_runtime_storage_npz" if actual_storage_feature_names else "contract_only_not_materialized",
        True,
    )

    for group, actual, expected in [
        ("node", actual_node_feature_names, NODE_STATE_FIELDS),
        ("facility", actual_facility_feature_names, FACILITY_STATE_FIELDS),
        ("storage", actual_storage_feature_names, STORAGE_STATE_FIELDS),
    ]:
        for i, name in enumerate(expected):
            materialization_rows.append(
                {
                    "feature_group": group,
                    "feature_name": name,
                    "index": i,
                    "expected": "true",
                    "materialized": str(name in actual).lower(),
                    "materialized_index": actual.index(name) if name in actual else "",
                    "status": "materialized" if name in actual else "missing",
                }
            )
    _write_csv(
        paths["feature_materialization_audit"],
        materialization_rows,
        [
            "feature_group",
            "feature_name",
            "index",
            "expected",
            "materialized",
            "materialized_index",
            "status",
        ],
    )

    any_blocked = any(row["status"] == "blocked" for row in sample_rows)
    node_only_complete = state_validation_mode in node_only_modes and not any_blocked
    full_complete = state_validation_mode not in node_only_modes and not any_blocked
    _write_json(
        paths["shape_audit"],
        {
            "status": "blocked" if any_blocked else "completed",
            "state_validation_mode": state_validation_mode,
            "formal_history_frame_count": FORMAL_FRAME_COUNT,
            "formal_history_offsets_min": TEMPORAL_FRAME_OFFSETS_MIN,
            "legacy_node_only_validation_complete": node_only_complete,
            "formal_full_project6_state_complete": full_complete,
            "state_history_expected": f"[samples,{FORMAL_FRAME_COUNT},N,F_node]",
            "facility_history_expected": f"[samples,{FORMAL_FRAME_COUNT},36,F_facility]",
            "storage_history_expected": f"[samples,{FORMAL_FRAME_COUNT},S,F_storage]",
            "state_shapes_seen": state_shapes,
            "facility_shapes_seen": facility_shapes,
            "storage_shapes_seen": storage_shapes,
            "node_state_fields": NODE_STATE_FIELDS,
            "facility_state_fields": FACILITY_STATE_FIELDS,
            "storage_state_fields": STORAGE_STATE_FIELDS,
            "actual_node_feature_names": actual_node_feature_names,
            "actual_facility_feature_names": actual_facility_feature_names,
            "actual_storage_feature_names": actual_storage_feature_names,
            "pump_state_fields": PUMP_STATE_FIELDS,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_json(
        paths["gap_report"],
        {
            "status": "blocked" if any_blocked else "completed",
            "blocked_sample_count": sum(
                1 for row in sample_rows if row["status"] == "blocked"
            ),
            "runtime_state_features_generated": not any_blocked,
            "legacy_node_only_validation_complete": node_only_complete,
            "full_project6_augmented_state_complete": full_complete,
            "formal_v42_history_13x5min": full_complete,
            "unlocks_round0": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return (3 if any_blocked else 0), paths
