"""Train PFV-first dual-fallback effect model.

The implementation is a guarded CLI skeleton. It validates inputs and records
the intended training contract. Model training logic should be connected only
after a same-state dataset with sufficient coverage exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def validate_dataset(dataset_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if dataset_path.suffix.lower() not in {".npz", ".parquet"}:
        raise ValueError(f"unsupported dataset format: {dataset_path}")


def create_training_manifest(config_path: Path, dataset_path: Path, out_dir: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    validate_dataset(dataset_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": str(config_path),
        "dataset": str(dataset_path),
        "model_goal": "PFV-first learned candidate scoring against Internal and selected fallback",
        "required_inputs": [
            "current_gat_state",
            "operational_forecast_120min",
            "actual_executed_action_sequence",
            "selected_fallback_action_sequence",
            "internal_reference_prediction",
            "ood_features"
        ],
        "required_outputs": [
            "pfv_improvement_lcb_vs_internal",
            "pfv_improvement_lcb_vs_selected_fallback",
            "tfv_delta_ucb_vs_selected_fallback",
            "peak_delta_ucb_vs_selected_fallback",
            "uncertainty",
            "ood_score"
        ],
        "gate_thresholds": cfg.get("model_gate", {}),
        "status": "input_validated_training_not_implemented_until_dataset_contract_passes"
    }
    (out_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    print(json.dumps(create_training_manifest(Path(args.config), Path(args.dataset), Path(args.out_dir)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
