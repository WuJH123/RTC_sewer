from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REQUIRED_CLONE_FIELDS = [
    "node_depths",
    "node_inflows",
    "link_flows",
    "link_depths",
    "storage_state",
    "actual_facility_settings",
    "native_target_settings",
    "add350_1_actual_speed",
    "ADD301_2_binary_state",
    "ADD301_3_binary_state",
    "pump_on_off_duration",
    "controller_memory",
    "override_ttl",
    "fallback_mode",
    "rainfall_state",
    "continuation_policy",
]


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_state_clone_contract_outputs(config_path: Path, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "state_clone_contract": out_dir / "state_clone_contract.json",
        "controller_state_manifest_schema": out_dir / "controller_state_manifest.schema.json",
        "state_clone_equivalence_schema": out_dir / "state_clone_equivalence.schema.csv",
    }
    contract = {
        "contract_name": "project6_v3_state_clone_equivalence",
        "config_sha256": sha256_file(config_path),
        "status": "implemented_not_run",
        "completion_marker_allowed": False,
        "required_clone_fields": REQUIRED_CLONE_FIELDS,
        "same_state_eligibility_false_if_missing": ["controller_memory", "override_ttl", "fallback_mode"],
        "test_protocol": [
            "continue original simulation",
            "save complete checkpoint and controller memory",
            "restore from checkpoint",
            "do not change action",
            "continue with identical inputs",
            "compare future state and KPI",
        ],
    }
    paths["state_clone_contract"].write_text(json.dumps(contract, indent=2), encoding="utf-8")
    manifest_schema = {
        "type": "object",
        "required": REQUIRED_CLONE_FIELDS,
        "properties": {field: {"description": "required for same-state restoration"} for field in REQUIRED_CLONE_FIELDS},
    }
    paths["controller_state_manifest_schema"].write_text(json.dumps(manifest_schema, indent=2), encoding="utf-8")
    with paths["state_clone_equivalence_schema"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "tolerance", "unit", "comparison", "blocks_same_state_if_failed"],
        )
        writer.writeheader()
        for metric in [
            "node_depths",
            "node_inflows",
            "link_flows",
            "link_depths",
            "storage_state",
            "actual_facility_settings",
            "native_target_settings",
            "add350_1_actual_speed",
            "ADD301_2_binary_state",
            "ADD301_3_binary_state",
            "PFV",
            "TFV",
            "peak_TFV_rate",
        ]:
            writer.writerow(
                {
                    "metric": metric,
                    "tolerance": "to_be_set_above_SWMM_numerical_noise_floor",
                    "unit": "native",
                    "comparison": "continued_vs_restored",
                    "blocks_same_state_if_failed": True,
                }
            )
    return paths
