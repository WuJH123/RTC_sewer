#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.contracts.prompt3a import OUT_ROOT, read_csv, write_csv, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run no more than 20 Round 0 cases without executing full Round 0.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--round0-manifest", default=str(OUT_ROOT / "round0" / "paired_manifest_round0.csv"))
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "round0"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    rows = read_csv(Path(args.round0_manifest))[:20]
    branch_rows = []
    action_rows = []
    kpi_rows = []
    fallback_rows = []
    planned_count = 0
    excluded_count = 0
    blocking_reasons: list[str] = []
    for row in rows:
        case_id = row.get("case_id", "")
        planned = row.get("feasibility") == "planned"
        if planned:
            planned_count += 1
        else:
            excluded_count += 1
        same_state_status = "ready_for_shadow_execution" if row.get("checkpoint_id") else "blocked_missing_checkpoint"
        controller_memory_status = "pending_runtime_capture" if not row.get("controller_memory_hash") else "available"
        hotstart_status = "pending_runtime_capture" if not row.get("state_clone_hash") else "available"
        if planned and same_state_status.startswith("blocked"):
            blocking_reasons.append(f"{case_id}:{same_state_status}")
        override_count = int(row.get("override_count") or 0)
        binary_legality = row.get("binary_legality") or "pending"
        interlock_status = "structural_precheck_pending"
        if override_count > 8:
            binary_legality = "blocked_override_count_exceeds_K"
            blocking_reasons.append(f"{case_id}:override_count_exceeds_K")
        branch_rows.append(
            {
                "case_id": case_id,
                "same_state_status": same_state_status,
                "hotstart_status": hotstart_status,
                "controller_memory_status": controller_memory_status,
            }
        )
        action_rows.append(
            {
                "case_id": case_id,
                "binary_legality": binary_legality,
                "add350_override_status": "blocked_until_bounds_verified" if row.get("add350_residual_override") == "True" else "not_requested",
                "interlock_status": interlock_status,
            }
        )
        kpi_rows.append({"case_id": case_id, "kpi_status": "not_evaluated_in_structural_dryrun"})
        fallback_rows.append(
            {
                "case_id": case_id,
                "fallback_selected_before_candidate": str(row.get("selected_fallback", "") != "").lower(),
                "status": "ready" if row.get("selected_fallback") else "blocked_missing_selected_fallback",
            }
        )
    report = {
        "status": "structural_only",
        "candidate_preview_count": len(rows),
        "planned_preview_count": planned_count,
        "excluded_preview_count": excluded_count,
        "blocking_reasons": blocking_reasons,
        "same_state_hotstart_execution_status": "not_run",
        "hydraulic_branch_execution_status": "not_run",
        "completion_marker_allowed": False,
        "full_round0_unlock_allowed": False,
    }
    write_csv(out_dir / "round0_dryrun_manifest.csv", rows)
    write_csv(out_dir / "round0_dryrun_branch_audit.csv", branch_rows, ["case_id", "same_state_status", "hotstart_status", "controller_memory_status"])
    write_csv(out_dir / "round0_dryrun_action_audit.csv", action_rows, ["case_id", "binary_legality", "add350_override_status", "interlock_status"])
    write_csv(out_dir / "round0_dryrun_kpi_audit.csv", kpi_rows, ["case_id", "kpi_status"])
    write_csv(out_dir / "round0_dryrun_fallback_audit.csv", fallback_rows, ["case_id", "fallback_selected_before_candidate", "status"])
    write_json(out_dir / "round0_dryrun_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
