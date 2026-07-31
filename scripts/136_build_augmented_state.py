#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.augmented_state import build_augmented_state_contract_outputs


def _lock_status(root: Path) -> tuple[str, str]:
    lock_path = root / "outputs" / "project6_pfvfirst_dualfallback_10min_v3" / "gat" / "gat_primary_selection_lock.json"
    gate_path = root / "outputs" / "project6_pfvfirst_dualfallback_10min_v3" / "gat" / "gat_sr0p15_robustness_gate.json"
    if not lock_path.exists():
        return "pending_manual_execution", "pending"
    robustness = "pending"
    if gate_path.exists():
        try:
            gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
            robustness = str(gate.get("status") or "pending")
        except Exception:
            robustness = "unreadable"
    return "locked", robustness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Project6 V3 augmented-state contracts and schemas.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gat-compatibility", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    compat_path = Path(args.gat_compatibility)
    out_dir = Path(args.out_dir)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not compat_path.is_absolute():
        compat_path = ROOT / compat_path
    out_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        print(json.dumps({"status": "failed", "reason": "config_not_found", "config": str(config_path)}, indent=2))
        return 6
    if not compat_path.exists():
        print(json.dumps({"status": "blocked", "reason": "gat_compatibility_report_not_found", "path": str(compat_path)}, indent=2))
        return 3
    outputs = build_augmented_state_contract_outputs(config_path=config_path, gat_compatibility_path=compat_path, out_dir=out_dir)
    selection_lock_status, robustness_status = _lock_status(ROOT)
    report = {
        "status": "completed_state_contract_schema_build",
        "selected_primary_gat": "sr0p15",
        "selection_decision_status": "user_confirmed",
        "selection_lock_status": selection_lock_status,
        "gat_robustness_status": robustness_status,
        "runtime_state_features_generated": False,
        "unlocks_round0": False,
        "sentinel_contract_status": "human_resolution_required",
        "does_not_generate_action_data": True,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
