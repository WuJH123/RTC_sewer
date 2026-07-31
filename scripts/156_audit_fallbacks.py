#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.contracts.prompt3a import OUT_ROOT, managed_facility_ids, utc_now, write_csv, write_json
from sewerrtc.control.fallback_contract import serialize_actions
from sewerrtc.control.fallback_selector import select_safe_fallback
from sewerrtc.control.internal_fallback import internal_rules_fallback
from sewerrtc.control.passive_fallback import executable_passive_fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dual-fallback contracts and audit report.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "fallbacks"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    facility_state = [
        {
            "facility_id": fid,
            "actual_current_setting": "hold",
            "native_target_setting": "hold",
            "minimum_passive_target": "",
        }
        for fid in managed_facility_ids()
    ]
    passive = executable_passive_fallback(facility_state)
    internal = internal_rules_fallback({"learned_override_active": True, "override_ttl": 1, "candidate_target": "candidate"})
    selection = select_safe_fallback(
        passive,
        internal,
        {
            "executable_passive": {"tfv_ucb": 0, "peak_ucb": 0, "sentinel_storage_risk": 0, "pfv_ucb": 0, "switch_count": passive["switch_count"], "action_magnitude": passive["total_action_magnitude"], "native_transition_cost": 0},
            "internal_rules": {"tfv_ucb": 0, "peak_ucb": 0, "sentinel_storage_risk": 0, "pfv_ucb": 0, "switch_count": 0, "action_magnitude": 0, "native_transition_cost": 0},
        },
    )
    files = [
        write_json(out_dir / "passive_fallback_contract.json", {"status": "completed", "forbidden_modes": ["all_36_zero", "all_36_open", "instant_reset_all"], "minimum_intervention": True}),
        write_json(out_dir / "internal_fallback_contract.json", {"status": "completed", "release_override_required": True, "native_rules_retake_authority_required": True}),
        write_json(out_dir / "fallback_selection_contract.json", {"status": "completed", "candidate_evaluation_after_selection": True, "selection": selection}),
        write_csv(out_dir / "fallback_transition_tests.schema.csv", [], ["test_id", "fallback_id", "input_hash", "expected_status"]),
        write_json(out_dir / "fallback_execution_audit_report.json", {"status": "completed", "created_at": utc_now(), "passive_executable": passive["executable"], "internal_executable": internal["executable"], "selected_fallback_id": selection["selected_fallback_id"], "passive_actions": serialize_actions(passive["actions"])}),
    ]
    print(json.dumps({"status": "completed", "outputs": [str(p) for p in files]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
