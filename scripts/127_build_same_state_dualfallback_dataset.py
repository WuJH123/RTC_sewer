"""Build same-state dual-fallback effect dataset from completed case details.

This script intentionally fails fast if required branch outputs are missing. It
must be run only after the user has generated same-state cases manually.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_COLUMNS = {
    "case_id",
    "event_id",
    "checkpoint_id",
    "branch_id",
    "selected_fallback_id",
    "actual_executed_action_sequence",
    "continuation_policy_id",
    "kpi_contract_version",
}


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def validate_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    header = manifest_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    missing = sorted(REQUIRED_COLUMNS - set(header))
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    return {"manifest": str(manifest_path), "required_columns_present": True}


def build_dataset(config_path: Path, case_manifest: Path, out_dir: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    validation = validate_manifest(case_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": str(config_path),
        "case_manifest": str(case_manifest),
        "validation": validation,
        "dataset_semantics": {
            "pfv_reference": "internal_rules_and_selected_fallback",
            "tfv_peak_reference": "selected_fallback",
            "action_input": "actual_executed_action_sequence",
            "forbid_requested_only_actions": True
        },
        "status": "schema_validated_dataset_builder_skeleton",
        "next_required_input": "completed same-state branch detail files"
    }
    (out_dir / "dataset_build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--case-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    print(json.dumps(build_dataset(Path(args.config), Path(args.case_manifest), Path(args.out_dir)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
