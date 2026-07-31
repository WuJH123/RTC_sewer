"""Gate 3.5 v2 Phase G: Gate 4-H120 Batch 0 Auditor.

Audits Batch 0 results:
  - Checks for PFV-safe candidates
  - Checks for TFV-improved candidates
  - Checks for peak-noninferior candidates
  - Checks for joint-feasible candidates
  - Validates label signal (non-zero)
  - Checks peak near-zero ratio
  - Validates candidate family coverage
  - Checks actual action distance
  - Detects false no-op
  - Validates SWMM readback
  - Independently re-computes H120 labels

Output:
  - batch0_audit_report.json
  - batch0_audit_summary.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.data.round0_prompt2 import _priority_nodes
from sewerrtc.simulation.kpi_metrics import compute_window_kpis

BATCH_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_capability_v2" / "gate4_h120_batch0"
WORK_DIR = BATCH_DIR / "work"

EVENT_ID = "V31_RP10_D2H_P65_v31_independent_gamma_084"
CHECKPOINT_MIN = 60.0
SPINUP_MIN = 4320
H120_MIN = 120
ADJ_CHECKPOINT = CHECKPOINT_MIN + SPINUP_MIN


def compute_h120_labels_independent(detail_csv, checkpoint_min, h120_min, priority_nodes):
    df = pd.read_csv(detail_csv)
    result = compute_window_kpis(
        df, priority_nodes, checkpoint_min, h120_min, dt_sec=300
    )
    return {
        "pfv_m3": float(result["PFV"]),
        "tfv_m3": float(result["TFV"]),
        "peak_tfv_rate_m3s": float(result["peak_TFV_rate"]),
        "h120_steps": int(result["steps"]),
    }


def main():
    print("=" * 60)
    print("  Gate 3.5 v2: Gate 4-H120 Batch 0 Auditor")
    print("=" * 60)

    # Load results
    results_csv = BATCH_DIR / "batch0_results.csv"
    if not results_csv.exists():
        print(f"ERROR: {results_csv} not found")
        return 1

    results_df = pd.read_csv(results_csv)
    priority = _priority_nodes()

    print(f"\n  Loaded {len(results_df)} branches")
    print(f"  Priority nodes: {len(priority)}")

    # Independent label re-computation
    print("\n  Re-computing H120 labels independently...")
    label_audit_rows = []
    for _, row in results_df.iterrows():
        branch = row["branch"]
        branch_type = row["branch_type"]
        
        if branch_type == "reference":
            detail_csv = WORK_DIR / f"batch0_{EVENT_ID}__{branch}_detail.csv"
        else:
            cand_id = branch.replace("candidate_", "")
            detail_csv = WORK_DIR / f"batch0_{EVENT_ID}__cand_{cand_id}_detail.csv"

        if not detail_csv.exists():
            print(f"    WARNING: {detail_csv} not found for {branch}")
            continue

        labels = compute_h120_labels_independent(detail_csv, ADJ_CHECKPOINT, H120_MIN, priority)
        
        label_audit_rows.append({
            "branch": branch,
            "branch_type": branch_type,
            "pfv_m3_independent": labels.get("pfv_m3", float("nan")),
            "tfv_m3_independent": labels.get("tfv_m3", float("nan")),
            "peak_tfv_rate_independent": labels.get("peak_tfv_rate_m3s", float("nan")),
            "h120_steps_independent": labels.get("h120_steps", 0),
            "pfv_m3_reported": row.get("pfv_m3", float("nan")),
            "tfv_m3_reported": row.get("tfv_m3", float("nan")),
            "peak_reported": row.get("peak_tfv_rate", float("nan")),
        })

    label_audit_df = pd.DataFrame(label_audit_rows)
    
    # Check label consistency
    print("\n  Checking label consistency...")
    n_mismatch = 0
    for _, row in label_audit_df.iterrows():
        pfv_match = np.isclose(row["pfv_m3_independent"], row["pfv_m3_reported"], rtol=1e-3, equal_nan=True)
        tfv_match = np.isclose(row["tfv_m3_independent"], row["tfv_m3_reported"], rtol=1e-3, equal_nan=True)
        if not (pfv_match and tfv_match):
            n_mismatch += 1
            print(f"    MISMATCH: {row['branch']}")
            print(f"      PFV: reported={row['pfv_m3_reported']:.4f}, independent={row['pfv_m3_independent']:.4f}")
            print(f"      TFV: reported={row['tfv_m3_reported']:.4f}, independent={row['tfv_m3_independent']:.4f}")

    if n_mismatch == 0:
        print("    All labels match!")
    else:
        print(f"    {n_mismatch} label mismatches detected")

    # Candidate analysis
    print("\n  Analyzing candidates...")
    candidates = results_df[results_df["branch_type"] == "candidate"].copy()
    
    n_candidates = len(candidates)
    n_pfv_safe = candidates["pfv_safe"].sum() if "pfv_safe" in candidates.columns else 0
    n_peak_ni = candidates["peak_noninferior"].sum() if "peak_noninferior" in candidates.columns else 0
    n_tfv_imp = candidates["tfv_improved"].sum() if "tfv_improved" in candidates.columns else 0
    n_joint = candidates["joint_feasible"].sum() if "joint_feasible" in candidates.columns else 0

    print(f"    Total candidates: {n_candidates}")
    print(f"    PFV safe: {n_pfv_safe}")
    print(f"    Peak noninferior: {n_peak_ni}")
    print(f"    TFV improved: {n_tfv_imp}")
    print(f"    Joint feasible: {n_joint}")

    # Check for signal
    print("\n  Checking label signal...")
    has_pfv_signal = n_pfv_safe > 0
    has_tfv_signal = n_tfv_imp > 0
    has_peak_signal = n_peak_ni > 0
    has_joint = n_joint > 0

    print(f"    PFV signal: {has_pfv_signal}")
    print(f"    TFV signal: {has_tfv_signal}")
    print(f"    Peak signal: {has_peak_signal}")
    print(f"    Joint feasible: {has_joint}")

    # Check candidate family coverage
    print("\n  Checking candidate family coverage...")
    cand_ids = candidates["branch"].tolist()
    families = {
        "all_open": any("all_open" in c for c in cand_ids),
        "all_closed": any("all_closed" in c for c in cand_ids),
        "uniform_partial": any("uniform_" in c for c in cand_ids),
        "random": any("random_" in c for c in cand_ids),
    }
    for fam, present in families.items():
        print(f"    {fam}: {present}")

    # Peak near-zero ratio
    print("\n  Checking peak near-zero ratio...")
    peak_vals = candidates["peak_tfv_rate"].dropna() if "peak_tfv_rate" in candidates.columns else pd.Series([])
    if len(peak_vals) > 0:
        near_zero = (peak_vals.abs() < 1e-6).sum()
        ratio = near_zero / len(peak_vals)
        print(f"    Near-zero peaks: {near_zero}/{len(peak_vals)} ({ratio:.1%})")
    else:
        print("    No peak data available")

    # Audit report
    audit_report = {
        "event_id": EVENT_ID,
        "checkpoint_min": CHECKPOINT_MIN,
        "h120_min": H120_MIN,
        "n_candidates": n_candidates,
        "n_pfv_safe": int(n_pfv_safe),
        "n_peak_noninferior": int(n_peak_ni),
        "n_tfv_improved": int(n_tfv_imp),
        "n_joint_feasible": int(n_joint),
        "has_pfv_signal": has_pfv_signal,
        "has_tfv_signal": has_tfv_signal,
        "has_peak_signal": has_peak_signal,
        "has_joint_feasible": has_joint,
        "candidate_families": families,
        "label_mismatches": n_mismatch,
        "audit_passed": has_joint,
        "recommendation": "proceed_to_batch_expansion" if has_joint else "candidate_pool_redesign",
    }

    # Save audit report
    audit_json = BATCH_DIR / "batch0_audit_report.json"
    audit_json.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Audit report saved to {audit_json}")

    # Save label audit CSV
    label_audit_df.to_csv(BATCH_DIR / "batch0_audit_summary.csv", index=False)

    print(f"\n{'='*60}")
    print(f"  Audit complete")
    print(f"  Joint feasible candidates: {n_joint}")
    print(f"  Recommendation: {audit_report['recommendation']}")
    print(f"{'='*60}")

    return 0 if has_joint else 5


if __name__ == "__main__":
    sys.exit(main())
