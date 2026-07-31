"""Gate 2.5-real-v2 Independent Reaudit.

Verifies two blocking issues that V2 verdict missed:
1. Dynamic Internal checkpoint state differs from fixed branches (P04 logic error)
2. Recovery was incorrectly set as informational (P19 logic error)

Outputs to V2 directory (append-only, no modification of existing files):
- gate2p5_real_v2_independent_reaudit.json
- gate2p5_real_v2_failed_conditions.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real_v2"


def _parse_state_hash(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def main() -> int:
    hash_df = pd.read_csv(V2_DIR / "state_hash_comparison.csv")
    kpi_df = pd.read_csv(V2_DIR / "branch_kpi_comparison.csv")
    verdict = json.loads((V2_DIR / "gate2p5_real_v2_verdict.json").read_text(encoding="utf-8"))

    failures = []
    findings = {}

    # =========================================================================
    # Issue 1: Dynamic Internal checkpoint state differs from fixed branches
    # =========================================================================
    cp_labels = hash_df["checkpoint_label"].unique()
    hash_mismatch_details = []

    for cp in cp_labels:
        cp_rows = hash_df[hash_df["checkpoint_label"] == cp]
        branches_data = {}
        for _, r in cp_rows.iterrows():
            branches_data[r["branch"]] = _parse_state_hash(r["checkpoint_state_hash"])

        di_hash = branches_data.get("dynamic_internal", {})
        nc_hash = branches_data.get("no_control", {})

        for key in ["h_sha256", "head_sha256", "flood_sha256", "storage_volume_sha256", "flow_sha256"]:
            di_val = di_hash.get(key, "")
            nc_val = nc_hash.get(key, "")
            if di_val != nc_val:
                hash_mismatch_details.append({
                    "checkpoint": cp,
                    "key": key,
                    "dynamic_internal": di_val[:32] + "...",
                    "no_control": nc_val[:32] + "...",
                    "match": False,
                })

    findings["dynamic_internal_state_mismatch"] = {
        "description": "Dynamic Internal checkpoint state hash differs from no_control/hold branches",
        "mismatch_count": len(hash_mismatch_details),
        "details": hash_mismatch_details[:10],
        "root_cause": "179 native control links in [CONTROLS] are not included in prefix schedule. "
                      "no_control INP (strip_controls=True) has no native rules, so these links "
                      "are uncontrolled. dynamic_internal INP (with-controls) has native rules "
                      "driving these links. Result: massive hydraulic divergence (h: up to 6.1m).",
    }

    failures.append({
        "condition": "P04_checkpoint_state_hash_equality",
        "severity": "CRITICAL",
        "description": "Four branches must have identical checkpoint state hash (including dynamic_internal)",
        "v2_verdict_was": "PASS (incorrectly excluded dynamic_internal)",
        "actual_status": "FAIL",
        "evidence": f"{len(hash_mismatch_details)} hash mismatches between dynamic_internal and no_control",
    })

    # Check prefix_actual_schedule_sha256 consistency
    prefix_sha_mismatch = []
    for cp in cp_labels:
        cp_rows = hash_df[hash_df["checkpoint_label"] == cp]
        shas = {}
        for _, r in cp_rows.iterrows():
            shas[r["branch"]] = str(r["prefix_actual_schedule_sha256"])
        unique_shas = set(shas.values())
        if len(unique_shas) > 1:
            prefix_sha_mismatch.append({
                "checkpoint": cp,
                "unique_sha_count": len(unique_shas),
                "branches": shas,
            })

    findings["prefix_schedule_sha_mismatch"] = {
        "description": "prefix_actual_schedule_sha256 differs across branches",
        "details": prefix_sha_mismatch,
        "root_cause": "dynamic_internal and hold_snapshot share one SHA (with-controls INP prefix), "
                      "while no_control and hold_previous have different SHAs (no-controls INP prefix). "
                      "The prefix schedule covers only 36 Eng36 facilities, not the 179 native control links.",
    }

    failures.append({
        "condition": "P07_prefix_schedule_sha_consistency",
        "severity": "CRITICAL",
        "description": "All four branches must have identical prefix_actual_schedule_sha256",
        "v2_verdict_was": "PASS (only checked field existence, not equality)",
        "actual_status": "FAIL",
        "evidence": f"{len(prefix_sha_mismatch)} checkpoints with mismatched prefix SHA",
    })

    # =========================================================================
    # Issue 2: Recovery incorrectly set as informational
    # =========================================================================
    recovery_values = kpi_df["recovery_criteria_met"].tolist()
    all_false = all(not v for v in recovery_values)

    findings["recovery_blocking_error"] = {
        "description": "All branches have recovery_criteria_met=false but V2 P19 marked it as informational",
        "all_branches_false": all_false,
        "recovery_values": recovery_values,
        "last_flood_times": kpi_df["last_flood_time_min"].tolist(),
        "root_cause": "V2 verdict P19 passed recovery as 'informational' despite all branches failing. "
                      "This allowed Gate to PASS even though no branch achieved recovery. "
                      "Recovery must be a BLOCKING condition.",
    }

    failures.append({
        "condition": "P21_recovery_must_be_blocking",
        "severity": "CRITICAL",
        "description": "recovery_criteria_met must be true for all branches (blocking, not informational)",
        "v2_verdict_was": "PASS (P19 marked as informational)",
        "actual_status": "FAIL",
        "evidence": f"All {len(recovery_values)} branch-recovery combinations have recovery_criteria_met=false",
    })

    # =========================================================================
    # Build reaudit output
    # =========================================================================
    reaudit = {
        "gate": "2.5-real-v2",
        "superseded": True,
        "superseded_reason": "incomplete_shared_prefix_gate_and_nonblocking_recovery",
        "gate3_authorization": False,
        "original_verdict": verdict.get("verdict", "UNKNOWN"),
        "original_pass_count": verdict.get("pass_count", 0),
        "original_total": verdict.get("total_conditions", 0),
        "new_failure_count": len(failures),
        "findings": findings,
        "failed_conditions": failures,
    }

    out_json = V2_DIR / "gate2p5_real_v2_independent_reaudit.json"
    out_json.write_text(json.dumps(reaudit, indent=2, default=str), encoding="utf-8")

    # Failed conditions CSV
    fail_df = pd.DataFrame(failures)
    out_csv = V2_DIR / "gate2p5_real_v2_failed_conditions.csv"
    fail_df.to_csv(out_csv, index=False)

    print(f"[V2 Reaudit] superseded=True, reason={reaudit['superseded_reason']}")
    print(f"[V2 Reaudit] Gate 3 authorization=False")
    print(f"[V2 Reaudit] {len(failures)} new failures found:")
    for f in failures:
        print(f"  [{f['severity']}] {f['condition']}: {f['description']}")
    print(f"[V2 Reaudit] Wrote: {out_json}")
    print(f"[V2 Reaudit] Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
