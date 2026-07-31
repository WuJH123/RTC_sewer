"""Gate 2.5-real Stage 7: KPI and Recovery computation for all branches.

Computes for each branch at each checkpoint:
  - H120: PFV, TFV, peak_TFV_rate, priority peak depth, flood duration
  - Full event: PFV, TFV, peak_TFV_rate, recovery time, final ponding, water balance

Outputs (in outputs/project6_dual_reference_v4/recovery_validation/gate2p5_real/):
  - branch_kpi_comparison.csv
  - recovery_audit.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.simulation.kpi_metrics import compute_kpis
from sewerrtc.simulation.runtime_contracts import analyze_recovery
from sewerrtc.data.round0_prompt2 import _load_round0_actuators, _priority_nodes

OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real"
CATALOG_PATH = OUT_DIR / "checkpoint_catalog.csv"
SELECTION_PATH = OUT_DIR / "positive_control_selection.json"

BRANCH_NAMES = ["no_control", "dynamic_internal", "hold_internal_snapshot", "hold_previous"]


def _compute_h120_kpis(detail: pd.DataFrame, priority: list[str], duration_min: int,
                        dt_sec: int = 300) -> dict:
    """Compute KPIs over the H120 window (first 120 min of event)."""
    h120 = detail[detail["elapsed_min"] <= 120.0].copy()
    if h120.empty:
        return {"H120_PFV": 0.0, "H120_TFV": 0.0, "H120_peak_TFV_rate": 0.0,
                "H120_flood_duration_min": 0.0, "H120_priority_peak_depth": 0.0}
    kpis = compute_kpis(h120, priority, dt_sec=dt_sec)
    # Priority peak depth: max total flooding rate in priority nodes
    pr_cols = [c for c in h120.columns if c.startswith("flood:") and c.split(":", 1)[1] in priority]
    if pr_cols:
        pr_rate = h120[pr_cols].fillna(0.0).to_numpy(float).sum(axis=1)
        pr_peak_depth = float(pr_rate.max())
    else:
        pr_peak_depth = 0.0
    return {
        "H120_PFV": kpis.get("PFV", 0.0),
        "H120_TFV": kpis.get("TFV", 0.0),
        "H120_peak_TFV_rate": kpis.get("peak_TFV_rate", 0.0),
        "H120_flood_duration_min": kpis.get("flood_duration_min", 0.0),
        "H120_priority_peak_depth": pr_peak_depth,
    }


def _compute_full_event_kpis(detail: pd.DataFrame, priority: list[str],
                              dt_sec: int = 300) -> dict:
    """Compute KPIs over the full event."""
    kpis = compute_kpis(detail, priority, dt_sec=dt_sec)
    # Final ponding: flooding at last timestep
    flood_cols = [c for c in detail.columns if c.startswith("flood:")]
    if flood_cols:
        last_row = detail.iloc[-1]
        final_ponding = float(sum(max(0.0, float(last_row.get(c, 0.0))) for c in flood_cols))
    else:
        final_ponding = 0.0
    return {
        "full_PFV": kpis.get("PFV", 0.0),
        "full_TFV": kpis.get("TFV", 0.0),
        "full_peak_TFV_rate": kpis.get("peak_TFV_rate", 0.0),
        "full_flood_duration_min": kpis.get("flood_duration_min", 0.0),
        "final_ponding_m3": final_ponding,
        "total_rows": len(detail),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_csv(CATALOG_PATH)
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    primary_event = selection["primary_event"]
    priority = _priority_nodes()

    kpi_rows = []
    recovery_rows = []

    for _, cp_row in catalog.iterrows():
        cp_min = float(cp_row["checkpoint_elapsed_min"])
        cp_label = str(cp_row["checkpoint_label"])
        print(f"\n=== Checkpoint {cp_label} at t={cp_min:.0f}min ===")

        for branch in BRANCH_NAMES:
            # Map branch name to file
            if branch == "no_control":
                fname = f"branch_no_control_cp{cp_label}_detail.csv"
            elif branch == "dynamic_internal":
                fname = f"branch_dynamic_internal_cp{cp_label}_detail.csv"
            elif branch == "hold_internal_snapshot":
                fname = f"branch_hold_snapshot_cp{cp_label}_detail.csv"
            elif branch == "hold_previous":
                fname = f"branch_hold_previous_cp{cp_label}_detail.csv"
            else:
                continue

            detail_path = OUT_DIR / fname
            if not detail_path.exists():
                print(f"  WARNING: {fname} not found, skipping")
                continue

            detail = pd.read_csv(detail_path)
            dt_sec = 300  # 5-min steps

            # H120 KPIs
            h120 = _compute_h120_kpis(detail, priority, int(cp_row.get("duration_min", 300)), dt_sec)

            # Full event KPIs
            full = _compute_full_event_kpis(detail, priority, dt_sec)

            # Action changes post-checkpoint
            action_cols = sorted([c for c in detail.columns if c.startswith("a:")])
            post_cp = detail[detail["elapsed_min"] > cp_min]
            action_changes_post = 0
            for col in action_cols:
                vals = pd.to_numeric(post_cp[col], errors="coerce").fillna(1.0)
                action_changes_post += int((vals.diff().abs() > 1e-6).sum())

            row = {
                "checkpoint_label": cp_label,
                "checkpoint_elapsed_min": cp_min,
                "branch": branch,
                **h120,
                **full,
                "action_changes_post_checkpoint": action_changes_post,
            }
            kpi_rows.append(row)

            # Recovery analysis
            rec = analyze_recovery(
                detail, event_id=primary_event, policy_id=branch,
                trajectory_id=f"{branch}_cp{cp_label}",
                duration_min=int(cp_row.get("duration_min", 300)),
                priority_nodes=priority,
            )
            recovery_rows.append({
                "checkpoint_label": cp_label,
                "checkpoint_elapsed_min": cp_min,
                "branch": branch,
                "recovery_criteria_met": rec.get("recovery_criteria_met", False),
                "recovery_time_min": rec.get("recovery_time_min", None),
                "last_flood_time_min": rec.get("last_flood_time_min", None),
                "last_priority_flood_time_min": rec.get("last_priority_flood_time_min", None),
                "actual_tail_min": rec.get("actual_tail_min", 0.0),
                "restored_count": rec.get("restored_count", 0),
            })

            print(f"  {branch}: H120_PFV={h120['H120_PFV']:.1f}, full_TFV={full['full_TFV']:.1f}, "
                  f"peak={full['full_peak_TFV_rate']:.3f}, recovery={rec.get('recovery_criteria_met', False)}")

    # Write KPI comparison
    kpi_df = pd.DataFrame(kpi_rows)
    kpi_path = OUT_DIR / "branch_kpi_comparison.csv"
    kpi_df.to_csv(kpi_path, index=False)
    print(f"\n[Stage 7] Wrote {kpi_path}")

    # Write recovery audit
    rec_df = pd.DataFrame(recovery_rows)
    rec_path = OUT_DIR / "recovery_audit.csv"
    rec_df.to_csv(rec_path, index=False)
    print(f"[Stage 7] Wrote {rec_path}")

    # Summary
    print(f"\n=== Stage 7 Summary ===")
    for cp_label in catalog["checkpoint_label"].tolist():
        print(f"\n  Checkpoint {cp_label}:")
        cp_kpis = kpi_df[kpi_df["checkpoint_label"] == cp_label]
        for _, r in cp_kpis.iterrows():
            print(f"    {r['branch']:30s}: H120_PFV={r['H120_PFV']:10.1f}  full_TFV={r['full_TFV']:10.1f}  "
                  f"peak={r['full_peak_TFV_rate']:.3f}  changes_post={r['action_changes_post_checkpoint']}")
        cp_rec = rec_df[rec_df["checkpoint_label"] == cp_label]
        for _, r in cp_rec.iterrows():
            print(f"    {r['branch']:30s}: recovery={r['recovery_criteria_met']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
