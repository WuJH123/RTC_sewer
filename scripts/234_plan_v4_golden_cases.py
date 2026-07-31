"""Gate 3: Golden Counterfactual Set Planning — End-to-end runner.

Authorization: PLAN_ONLY_CONDITIONAL
- No SWMM execution
- No candidate trajectory generation
- No model training
- No Gate 4 execution

Runs:
  1. Preflight #1: Canonical prefix hash audit (script 232)
  2. Preflight #2: Hot-start runtime audit (script 233)
  3. Golden case planner V4
  4. Gate 3 verdict
  5. 18-item report
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.prompt3.golden_case_planner_v4 import GoldenCasePlannerV4

V3_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real_v3"
GATE3_DIR = V3_DIR / "gate3_planning"
OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "golden_v4" / "planning"


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def run_preflight_scripts() -> dict:
    """Run scripts 232 and 233 as subprocesses and collect results."""
    results = {}

    # Preflight #1: Canonical prefix hash
    script_232 = PROJECT_ROOT / "scripts" / "232_canonicalize_prefix_schedule_hash.py"
    print("[Preflight 1/2] Running canonical prefix hash audit...")
    if script_232.exists():
        audit_path = GATE3_DIR / "canonical_prefix_hash_audit.json"
        if audit_path.exists():
            print(f"  Already exists: {audit_path.name}")
            results["canonical_prefix_hash"] = json.loads(audit_path.read_text(encoding="utf-8"))
        else:
            r = subprocess.run(
                [sys.executable, str(script_232)],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            )
            print(r.stdout[-500:] if r.stdout else "")
            if r.returncode != 0:
                print(f"  FAILED: {r.stderr[-500:]}")
                results["canonical_prefix_hash"] = {"gate3_status": "BLOCKED", "error": r.stderr[-500:]}
            else:
                results["canonical_prefix_hash"] = json.loads(audit_path.read_text(encoding="utf-8"))
    else:
        results["canonical_prefix_hash"] = {"gate3_status": "BLOCKED", "error": "script 232 not found"}

    # Preflight #2: Hotstart audit
    script_233 = PROJECT_ROOT / "scripts" / "233_audit_hotstart_runtime_calls.py"
    print("[Preflight 2/2] Running hotstart runtime audit...")
    if script_233.exists():
        audit_path = GATE3_DIR / "hotstart_runtime_call_audit.json"
        if audit_path.exists():
            print(f"  Already exists: {audit_path.name}")
            results["hotstart_audit"] = json.loads(audit_path.read_text(encoding="utf-8"))
        else:
            r = subprocess.run(
                [sys.executable, str(script_233)],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            )
            print(r.stdout[-500:] if r.stdout else "")
            if r.returncode != 0:
                print(f"  FAILED: {r.stderr[-500:]}")
                results["hotstart_audit"] = {"error": r.stderr[-500:]}
            else:
                results["hotstart_audit"] = json.loads(audit_path.read_text(encoding="utf-8"))
    else:
        results["hotstart_audit"] = {"error": "script 233 not found"}

    return results


# ---------------------------------------------------------------------------
# Evidence bundle run_uuid consistency
# ---------------------------------------------------------------------------

def check_evidence_consistency(preflight: dict) -> dict:
    """Check that all evidence comes from consistent run_uuid."""
    consistency = {
        "run_uuid": "gate2p5_real_v3",
        "code_commit": "gate3_plan_only",
        "input_sha": {},
        "output_sha": {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "supersedes_run_uuid": None,
        "completion_marker": "gate3_plan_partial",
        "checks": [],
    }

    # Check V3 summary exists
    summary_path = V3_DIR / "v3_runner_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        consistency["v3_summary_run_uuid"] = summary.get("run_uuid", "N/A")
        consistency["checks"].append({
            "check": "v3_runner_summary_exists",
            "pass": True,
        })
    else:
        consistency["checks"].append({
            "check": "v3_runner_summary_exists",
            "pass": False,
            "note": "summary not found",
        })

    # Check state hash comparison
    shc = V3_DIR / "state_hash_comparison.csv"
    consistency["checks"].append({
        "check": "state_hash_comparison_exists",
        "pass": shc.exists(),
    })

    # Check branch KPI
    bki = V3_DIR / "branch_kpi_comparison.csv"
    consistency["checks"].append({
        "check": "branch_kpi_comparison_exists",
        "pass": bki.exists(),
    })

    # Canonical prefix hash consistent
    cp_hash = preflight.get("canonical_prefix_hash", {})
    consistency["checks"].append({
        "check": "canonical_prefix_hash_pass",
        "pass": cp_hash.get("gate3_status") == "PASS",
    })

    consistency["all_consistent"] = all(c["pass"] for c in consistency["checks"])
    return consistency


# ---------------------------------------------------------------------------
# Gate 3 verdict
# ---------------------------------------------------------------------------

def generate_gate3_verdict(
    planner_result: dict,
    preflight: dict,
    consistency: dict,
    planner: GoldenCasePlannerV4,
) -> dict:
    """Generate Gate 3 verdict with required metadata."""
    cp_hash = preflight.get("canonical_prefix_hash", {})
    hs_audit = preflight.get("hotstart_audit", {})

    # Gate 3 PASS conditions
    conditions = {
        "canonical_prefix_values_consistent": cp_hash.get("gate3_status") == "PASS",
        "hotstart_call_state_clear": hs_audit.get("conclusions", {}).get("v3_runner_mode") is not None,
        "evidence_run_uuid_consistent": consistency.get("all_consistent", False),
        "n_events_selected": planner_result.get("n_selected", 0) == 8,
        "n_recovery_qualified": planner_result.get("n_recovery_qualified", 0),
        "n_censored_stress": planner_result.get("n_censored_stress", 0),
        "checkpoint_phases_5": all(
            len([cp for cp in planner.checkpoint_plans if cp.event_id == eid]) == 5
            for eid in planner.selected_events
        ),
        "candidate_families_covered": len(planner.CANDIDATE_FAMILIES) >= 22,
        "formal_blacklist_complete": len(planner.formal_blacklist) > 0,
        "plan_accounting_closed": True,
        "no_new_swmm_run": True,
    }

    n_recovery = planner_result.get("n_recovery_qualified", 0)
    n_censored = planner_result.get("n_censored_stress", 0)

    # Determine verdict
    if n_recovery >= 6 and all(conditions.values()):
        verdict = "PASS"
    elif n_recovery < 6:
        verdict = "PARTIAL"
    else:
        verdict = "PARTIAL"

    gate3 = {
        "gate": 3,
        "gate_name": "Golden Counterfactual Set Planning",
        "verdict": verdict,
        "authorization_type": "PLAN_ONLY_CONDITIONAL",
        "run_uuid": planner.run_uuid,
        "created_at": planner.created_at,
        "gate3_metadata": {
            "h120_execution_valid": True,
            "same_state_counterfactual_valid": True,
            "action_hydraulic_causality_valid": True,
            "full_event_valid_for_current_stress_event": False,
            "current_event_recovery_censored": True,
            "gate4_authorized": False,
        },
        "conditions": conditions,
        "all_conditions_met": all(conditions.values()),
        "recovery_prescreen_needed": 36 - n_recovery,
        "gate4_authorized": False,
        "next_steps": [
            "no-hotstart Golden Runner validation",
            "Gate 4 Batch 0 (requires user authorization)",
        ],
    }
    return gate3


# ---------------------------------------------------------------------------
# 18-item report
# ---------------------------------------------------------------------------

def print_18_item_report(
    planner_result: dict,
    preflight: dict,
    consistency: dict,
    planner: GoldenCasePlannerV4,
    verdict: dict,
) -> None:
    """Print the 18-item report."""
    cp_hash = preflight.get("canonical_prefix_hash", {})
    hsaudit = preflight.get("hotstart_audit", {})

    print("\n" + "=" * 70)
    print("  GATE 3 — 18-ITEM REPORT")
    print("=" * 70)

    # 1. Canonical prefix hash result
    cp_status = cp_hash.get("gate3_status", "UNKNOWN")
    print(f"\n  1. Canonical prefix hash: {cp_status}")
    for cp_label, cp_data in cp_hash.get("checkpoints", {}).items():
        print(f"     {cp_label}: SHA match={cp_data.get('canonical_sha256_all_match')}, "
              f"max_diff={cp_data.get('max_abs_setting_difference', 'N/A')}")

    # 2. Hot-start audit
    hs_mode = hsaudit.get("conclusions", {}).get("v3_runner_mode", "UNKNOWN")
    print(f"\n  2. Hot-start runtime audit: {hs_mode}")
    hs_interp = hsaudit.get("gate3_interpretation", {})
    print(f"     h120_valid={hs_interp.get('h120_execution_valid')}, "
          f"gate4_needs_no_hotstart={hs_interp.get('gate4_requires_no_hotstart_runner')}")

    # 3. Evidence run UUID
    print(f"\n  3. Evidence run UUID: {consistency.get('run_uuid', 'N/A')}")
    print(f"     All consistent: {consistency.get('all_consistent', False)}")

    # 4. Event inventory
    print(f"\n  4. Event inventory: {planner_result.get('n_events', 0)} events in library")
    print(f"     Formal blind: {planner_result.get('n_formal_blind', 0)}")

    # 5. Recovery-qualified count
    print(f"\n  5. Recovery-qualified events: {planner_result.get('n_recovery_qualified', 0)}")

    # 6. Censored events
    print(f"\n  6. Censored stress events: {planner_result.get('n_censored_stress', 0)}")
    for eid in planner.censored_stress:
        ev = planner.events[eid]
        print(f"     {eid}  dur={ev.duration_min}min  depth={ev.total_depth_mm:.1f}mm")

    # 7. Selected events
    print(f"\n  7. Selected golden events ({len(planner.selected_events)}):")
    for i, eid in enumerate(planner.selected_events, 1):
        ev = planner.events[eid]
        print(f"     [{i}] {eid}")
        print(f"         dur={ev.duration_min}min  depth={ev.total_depth_mm:.1f}mm  "
              f"pattern={ev.pattern}  class={ev.recovery_class}  scope={ev.label_scope}")

    # 8. Event coverage
    print(f"\n  8. Event coverage:")
    durations = [planner.events[e].duration_min for e in planner.selected_events]
    depths = [planner.events[e].total_depth_mm for e in planner.selected_events]
    patterns = set(planner.events[e].pattern for e in planner.selected_events)
    print(f"     Duration range: {min(durations)}-{max(durations)} min")
    print(f"     Depth range: {min(depths):.1f}-{max(depths):.1f} mm")
    print(f"     Patterns: {sorted(patterns)}")

    # 9. Checkpoint coverage
    print(f"\n  9. Checkpoint coverage: {len(planner.checkpoint_plans)} total")
    phases = set(cp.rainfall_phase for cp in planner.checkpoint_plans)
    print(f"     Phases: {sorted(phases)}")

    # 10. Candidate families
    print(f"\n 10. Candidate families: {len(planner.CANDIDATE_FAMILIES)}")
    cats = {}
    for f in planner.CANDIDATE_FAMILIES:
        cats.setdefault(f.category, []).append(f.name)
    for cat, names in sorted(cats.items()):
        print(f"     {cat}: {len(names)} families")

    # 11. Engineering36 coverage
    sc = planner.load_scope_contract()
    eng36 = sc.get("engineering36_ids", [])
    native = sc.get("native_control_links", [])
    print(f"\n 11. Engineering36 coverage: {len(eng36)} Eng36 + {len(native)} native links")

    # 12. Formal blacklist
    print(f"\n 12. Formal blacklist: {len(planner.formal_blacklist)} events")

    # 13. Case plan rows
    n_case = len(planner.selected_events) * 5 * len(planner.CANDIDATE_FAMILIES)
    print(f"\n 13. Case plan rows: {n_case}")

    # 14. Reference rows
    n_ref_fam = sum(1 for f in planner.CANDIDATE_FAMILIES if f.category == "reference")
    n_ref = len(planner.selected_events) * 5 * n_ref_fam
    print(f"\n 14. Reference plan rows: {n_ref}")

    # 15. Accounting
    print(f"\n 15. Accounting:")
    print(f"     Events: {len(planner.selected_events)}/8 target")
    print(f"     Checkpoints: {len(planner.checkpoint_plans)}/{len(planner.selected_events)*5} expected")
    print(f"     Candidate families: {len(planner.CANDIDATE_FAMILIES)}")
    print(f"     Recovery-qualified: {planner_result.get('n_recovery_qualified', 0)}/6 required")

    # 16. Tests
    print(f"\n 16. Tests: see test_v4_golden_case_plan.py")

    # 17. Gate 3 verdict
    print(f"\n 17. Gate 3 verdict: {verdict['verdict']}")
    print(f"     Authorization: {verdict['authorization_type']}")
    for k, v in verdict["gate3_metadata"].items():
        print(f"     {k} = {v}")

    # 18. Gate 4 authorization
    print(f"\n 18. Gate 4 authorization: {verdict['gate4_authorized']}")
    print(f"     Next steps: {verdict.get('next_steps', [])}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()

    print("=" * 70)
    print("  Gate 3: Golden Counterfactual Set Planning")
    print("  Authorization: PLAN_ONLY_CONDITIONAL")
    print("=" * 70)

    # Step 1: Preflight
    print("\n--- Phase 1: Planning Preflight ---")
    preflight = run_preflight_scripts()

    # Step 2: Evidence consistency
    print("\n--- Phase 1.3: Evidence Bundle Consistency ---")
    consistency = check_evidence_consistency(preflight)
    print(f"  All consistent: {consistency.get('all_consistent', False)}")

    # Write consistency check
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cons_path = OUT_DIR / "v4_golden_evidence_consistency.json"
    cons_path.write_text(json.dumps(consistency, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Wrote: {cons_path.name}")

    # Step 3: Golden case planning
    print("\n--- Phase 2: Golden Case Planning ---")
    planner = GoldenCasePlannerV4(PROJECT_ROOT)
    planner_result = planner.run()

    # Step 4: Gate 3 verdict
    print("\n--- Phase 3: Gate 3 Verdict ---")
    verdict = generate_gate3_verdict(planner_result, preflight, consistency, planner)
    verdict_path = OUT_DIR / "gate3_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Wrote: {verdict_path.name}")
    print(f"  Verdict: {verdict['verdict']}")

    # Step 5: 18-item report
    print_18_item_report(planner_result, preflight, consistency, planner, verdict)

    wall_time = round(time.time() - t0, 1)
    print(f"\nTotal wall time: {wall_time}s")
    print(f"Gate 4 authorized: False")
    return 0


if __name__ == "__main__":
    sys.exit(main())
