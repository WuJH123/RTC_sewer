"""Gate 5 Phase 6b: Audit Gate 5 Exact-SWMM Candidate Diagnosis.

Checks:
  - PFV-safe candidates exist
  - TFV-improved candidates exist
  - Peak-noninferior candidates exist
  - Joint-feasible candidates exist
  - All engineering constraints met
  - Label independent re-computation
  - Action uniqueness
  - Family coverage

Output:
  - gate5_audit_report.json
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

OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_capability_v2" / "gate4_h120_batch0"
GATE5_DIR = OUT_DIR / "gate5_exact_diagnosis"

EVENT_ID = "V31_RP10_D2H_P65_v31_independent_gamma_084"
CHECKPOINT_MIN = 60.0
SPINUP_MIN = 4320
ADJ_CHECKPOINT = CHECKPOINT_MIN + SPINUP_MIN
H120_MIN = 120

PFV_SAFETY_THRESHOLD = 1.0
TFV_IMPROVEMENT_THRESHOLD = 0.5
PEAK_NONINFERIOR_THRESHOLD = 0.001


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
    print("=" * 70)
    print("  Gate 5: Audit Report")
    print("=" * 70)

    results_csv = GATE5_DIR / "gate5_candidate_results.csv"
    if not results_csv.exists():
        print(f"ERROR: {results_csv} not found")
        return 1

    results_df = pd.read_csv(results_csv)
    priority = _priority_nodes()

    n_total = len(results_df)
    n_pfv_safe = results_df["pfv_safe"].sum() if "pfv_safe" in results_df.columns else 0
    n_tfv_imp = results_df["tfv_improved"].sum() if "tfv_improved" in results_df.columns else 0
    n_peak_ni = results_df["peak_noninferior"].sum() if "peak_noninferior" in results_df.columns else 0
    n_joint = results_df["joint_feasible"].sum() if "joint_feasible" in results_df.columns else 0

    # Family coverage
    families = results_df["family"].unique().tolist() if "family" in results_df.columns else []
    family_counts = results_df["family"].value_counts().to_dict() if "family" in results_df.columns else {}

    # K distribution
    k_vals = results_df["k_actual"].describe().to_dict() if "k_actual" in results_df.columns else {}

    # Action uniqueness
    action_hashes = results_df["action_hash"].unique() if "action_hash" in results_df.columns else []
    n_unique_actions = len(action_hashes)

    # Label diversity
    n_hard_neg = int((results_df["delta_pfv_vs_nc"] > PFV_SAFETY_THRESHOLD).sum()) if "delta_pfv_vs_nc" in results_df.columns else 0
    n_neutral = int(((results_df["delta_pfv_vs_nc"].abs() < 0.01) & (results_df["delta_tfv_vs_di"].abs() < 0.01)).sum()) if "delta_pfv_vs_nc" in results_df.columns else 0

    # Independent label re-computation (sample check)
    print("\n  Independent label re-computation (sample)...")
    n_mismatch = 0
    for _, row in results_df.head(5).iterrows():
        cid = row["candidate_id"]
        detail_csv = GATE5_DIR / f"gate5_{cid}_detail.csv"
        if detail_csv.exists():
            indep = compute_h120_labels_independent(detail_csv, ADJ_CHECKPOINT, H120_MIN, priority)
            reported_pfv = row.get("pfv_m3", float("nan"))
            reported_tfv = row.get("tfv_m3", float("nan"))
            indep_pfv = indep.get("pfv_m3", float("nan"))
            indep_tfv = indep.get("tfv_m3", float("nan"))
            if not (np.isclose(reported_pfv, indep_pfv, rtol=1e-3) and np.isclose(reported_tfv, indep_tfv, rtol=1e-3)):
                n_mismatch += 1
                print(f"    MISMATCH: {cid}")

    # PASS conditions
    pass_conditions = {
        "at_least_one_pfv_safe": int(n_pfv_safe) >= 1,
        "at_least_one_tfv_improved": int(n_tfv_imp) >= 1,
        "at_least_one_peak_noninferior": int(n_peak_ni) >= 1,
        "at_least_one_joint_feasible": int(n_joint) >= 1,
        "multiple_families": len(families) >= 3,
        "label_diversity": n_hard_neg > 0 or n_neutral > 0,
        "all_k_le_8": bool(results_df["k_actual"].max() <= 8) if "k_actual" in results_df.columns else False,
        "unique_actions": n_unique_actions == n_total,
    }

    gate5_pass = all(pass_conditions.values())

    # Recommendation
    if gate5_pass:
        recommendation = "proceed_to_pilot"
    elif n_joint == 0 and (n_pfv_safe > 0 or n_tfv_imp > 0):
        recommendation = "refine_candidate_search"
    else:
        recommendation = "candidate_coverage_failure"

    audit = {
        "event_id": EVENT_ID,
        "n_candidates_run": n_total,
        "n_pfv_safe": int(n_pfv_safe),
        "n_tfv_improved": int(n_tfv_imp),
        "n_peak_noninferior": int(n_peak_ni),
        "n_joint_feasible": int(n_joint),
        "families": families,
        "family_counts": family_counts,
        "k_distribution": {k: round(v, 4) for k, v in k_vals.items()},
        "n_unique_actions": n_unique_actions,
        "n_hard_negatives": n_hard_neg,
        "n_neutral": n_neutral,
        "label_mismatches": n_mismatch,
        "pass_conditions": {k: bool(v) for k, v in pass_conditions.items()},
        "gate5_pass": bool(gate5_pass),
        "recommendation": recommendation,
    }

    (GATE5_DIR / "gate5_audit_report.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n  Candidates: {n_total}")
    print(f"  PFV safe: {n_pfv_safe}")
    print(f"  TFV improved: {n_tfv_imp}")
    print(f"  Peak noninferior: {n_peak_ni}")
    print(f"  Joint feasible: {n_joint}")
    print(f"  Families: {families}")
    print(f"  Label mismatches: {n_mismatch}")
    print(f"  Gate 5 PASS: {gate5_pass}")
    print(f"  Recommendation: {recommendation}")
    print("=" * 70)

    return 0 if gate5_pass else 5


if __name__ == "__main__":
    sys.exit(main())
