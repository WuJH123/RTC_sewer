from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .state_contract import FACILITY_STATE_FIELDS, NODE_STATE_FIELDS, PUMP_STATE_FIELDS, build_state_feature_contract


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _network_path_from_config(config_path: Path) -> Path | None:
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    for raw in text.splitlines():
        line = raw.strip()
        if "wuhan_v8_storage_retrofit.inp" in line:
            if ":" in line:
                value = line.split(":", 1)[1].strip().strip("'\"")
                path = Path(value)
                if not path.is_absolute():
                    path = config_path.parent.parent / path
                return path
    default = config_path.parent.parent / "data" / "wuhan_v8_storage_retrofit.inp"
    return default if default.exists() else None


def _gat_lock_status(project_root: Path) -> tuple[str, str]:
    gat_dir = project_root / "outputs" / "project6_pfvfirst_dualfallback_10min_v3" / "gat"
    lock_path = gat_dir / "gat_primary_selection_lock.json"
    gate_path = gat_dir / "gat_sr0p15_robustness_gate.json"
    if not lock_path.exists():
        return "pending_manual_execution", "pending"
    robustness = "pending"
    if gate_path.exists():
        try:
            robustness = str(json.loads(gate_path.read_text(encoding="utf-8-sig")).get("status") or "pending")
        except Exception:
            robustness = "unreadable"
    return "locked", robustness


def build_augmented_state_contract_outputs(config_path: Path, gat_compatibility_path: Path, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    network_path = _network_path_from_config(config_path)
    config_hash = sha256_file(config_path)
    network_hash = sha256_file(network_path) if network_path else None
    contract = build_state_feature_contract(config_sha256=config_hash, network_sha256=network_hash)
    contract["gat_compatibility_report"] = str(gat_compatibility_path)
    contract["gat_compatibility_hash"] = sha256_file(gat_compatibility_path)
    contract["build_status"] = "schema_contract_only"
    contract["formal_state_pipeline_frozen"] = False
    contract["action_data_generated"] = False
    selection_lock_status, gat_robustness_status = _gat_lock_status(config_path.parent.parent)
    contract["selected_primary_gat"] = "sr0p15"
    contract["selection_decision_status"] = "user_confirmed"
    contract["selection_lock_status"] = selection_lock_status
    contract["gat_robustness_status"] = gat_robustness_status
    paths = {
        "state_feature_contract": out_dir / "state_feature_contract.json",
        "state_feature_schema": out_dir / "state_feature_schema.json",
        "facility_state_schema": out_dir / "facility_state_schema.csv",
        "temporal_state_alignment_audit": out_dir / "temporal_state_alignment_audit.csv",
        "state_quality_contract": out_dir / "state_quality_contract.json",
        "local_flow_feature_contract": out_dir / "local_flow_feature_contract.json",
    }
    paths["state_feature_contract"].write_text(json.dumps(contract, indent=2), encoding="utf-8")
    schema = {
        "node_state_fields": NODE_STATE_FIELDS,
        "facility_state_fields": FACILITY_STATE_FIELDS,
        "pump_state_fields": PUMP_STATE_FIELDS,
        "state_tensor_shape": {
            "temporal_frames": 7,
            "node_axis": "retrofit_inp_node_order",
            "facility_axis": 36,
            "selected_primary_gat": "sr0p15",
            "selection_decision_status": "user_confirmed",
            "selection_lock_status": selection_lock_status,
            "gat_robustness_status": gat_robustness_status,
        },
    }
    paths["state_feature_schema"].write_text(json.dumps(schema, indent=2), encoding="utf-8")
    with paths["facility_state_schema"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["field", "required", "online_available", "notes"])
        writer.writeheader()
        for field in FACILITY_STATE_FIELDS + PUMP_STATE_FIELDS:
            writer.writerow({"field": field, "required": True, "online_available": "depends_on_source", "notes": ""})
    with paths["temporal_state_alignment_audit"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "offset_min",
                "measurement_time",
                "source_time",
                "state_estimation_time",
                "decision_time",
                "data_age_min",
                "valid_before_decision",
                "interpolation_method",
                "missingness_mask",
                "quality_flag",
            ],
        )
        writer.writeheader()
        for i, offset in enumerate([0, -10, -20, -30, -40, -50, -60]):
            writer.writerow(
                {
                    "frame_index": i,
                    "offset_min": offset,
                    "measurement_time": "",
                    "source_time": "",
                    "state_estimation_time": "",
                    "decision_time": "",
                    "data_age_min": "",
                    "valid_before_decision": "",
                    "interpolation_method": "causal_last_observation_carried_forward",
                    "missingness_mask": "required",
                    "quality_flag": "schema_only",
                }
            )
    paths["state_quality_contract"].write_text(
        json.dumps(
            {
                "future_data_violation_blocks_online_control": True,
                "stale_data_flag_required": True,
                "ood_score_required": True,
                "sentinel_threshold_unresolved": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["local_flow_feature_contract"].write_text(
        json.dumps(
            {
                "missing_flow_is_not_zero": True,
                "availability_mask_required": True,
                "future_truth_flow_forbidden": True,
                "covered_links": [
                    "36 managed facilities",
                    "direct upstream/downstream links",
                    "priority influence paths",
                    "sentinel candidate paths",
                    "storage inlet/outlet groups",
                    "pump upstream/downstream paths",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths
