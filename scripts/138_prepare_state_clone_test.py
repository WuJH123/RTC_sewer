#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.state_clone_contract import write_state_clone_contract_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare state-clone equivalence schemas without running SWMM.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        print(json.dumps({"status": "failed", "reason": "config_not_found", "config": str(config_path)}, indent=2))
        return 6
    outputs = write_state_clone_contract_outputs(config_path=config_path, out_dir=out_dir)
    augmented_manifest = out_dir / "augmented_state_sample_manifest.csv"
    shape_audit = out_dir / "augmented_state_shape_audit.json"
    gap_report = out_dir / "state_input_gap_report.json"
    if not augmented_manifest.exists() or not shape_audit.exists() or not gap_report.exists():
        report = {
            "status": "blocked",
            "reason": "runtime_augmented_state_outputs_missing",
            "required": [str(augmented_manifest), str(shape_audit), str(gap_report)],
            "implemented_artifacts": {k: str(v) for k, v in outputs.items()},
            "completion_marker_allowed": False,
        }
        print(json.dumps(report, indent=2))
        return 3
    gap = json.loads(gap_report.read_text(encoding="utf-8"))
    if gap.get("status") != "completed":
        report = {
            "status": "blocked",
            "reason": "runtime_augmented_state_not_completed",
            "state_gap_report": str(gap_report),
            "state_gap_status": gap.get("status"),
            "implemented_artifacts": {k: str(v) for k, v in outputs.items()},
            "completion_marker_allowed": False,
        }
        print(json.dumps(report, indent=2))
        return 3
    report = {
        "status": "blocked",
        "hotstart_equivalence_status": "not_run",
        "formal_same_state_unlock_allowed": False,
        "reason": "state_clone_contract_and_runtime_state_inputs_are_available, but formal hotstart equivalence still requires a dedicated SWMM checkpoint run",
        "implemented_artifacts": {k: str(v) for k, v in outputs.items()},
        "runtime_state_manifest": str(augmented_manifest),
        "state_gap_report": str(gap_report),
        "completion_marker_allowed": False,
    }
    print(json.dumps(report, indent=2))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
