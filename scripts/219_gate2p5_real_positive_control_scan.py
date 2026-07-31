"""Gate 2.5-real Stage 2: Scan rainfall events for positive native-rule control.

Runs SWMM with native [CONTROLS] preserved on representative events.
Selects primary and backup events where native rules trigger setting changes.

Outputs (in outputs/project6_dual_reference_v4/recovery_validation/gate2p5_real/):
  - positive_control_event_scan.csv
  - positive_control_selection.json
"""
from __future__ import annotations

import json
import sys
import time
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.simulation.pyswmm_runner import run_swmm_trajectory
from sewerrtc.data.round0_prompt2 import _load_round0_actuators, _priority_nodes

INP_PATH = PROJECT_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"
RAIN_TABLE = PROJECT_ROOT / "outputs" / "rainfall_library_v8_storage_variablepump" / "rainfall_event_table.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real"
WORK_DIR = OUT_DIR / "stage2_work"

# Select representative events: mix of durations and peak patterns
SCAN_EVENTS = [
    # 2h events - different patterns
    "V31_RP10_D2H_P50_v31_independent_gamma_087",
    "V31_RP10_D2H_P80_v31_s_curve_088",
    # 3h events - different patterns
    "V31_RP10_D3H_P35_v31_independent_gamma_093",
    "V31_RP10_D3H_P50_v31_s_curve_097",
    "V31_RP10_D3H_P65_v31_front_back_split_101",
    # 5h events - different patterns
    "V31_RP10_D5H_P35_v31_independent_gamma_108",
    "V31_RP10_D5H_P65_v31_s_curve_115",
    "V31_RP10_D5H_P80_v31_front_back_split_119",
]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _analyze_detail(detail_csv: Path, eng36_ids: list[str]) -> dict[str, Any]:
    """Analyze a detail CSV for native-rule activity."""
    df = pd.read_csv(detail_csv)
    if df.empty:
        return {"rows": 0, "native_change_detected": False}

    action_cols = [c for c in df.columns if c.startswith("a:")]
    setting_cols = [c for c in df.columns if c.startswith("setting:")]

    # Overall action analysis
    total_unique = 0
    total_changes = 0
    facility_changes: dict[str, int] = {}
    facility_unique: dict[str, int] = {}
    facility_std: dict[str, float] = {}

    for col in action_cols:
        aid = col.split(":", 1)[1]
        vals = pd.to_numeric(df[col], errors="coerce").fillna(1.0)
        unique_count = int(vals.nunique())
        changes = int((vals.diff().abs() > 1e-6).sum())
        std = float(vals.std()) if len(vals) > 1 else 0.0
        total_unique += unique_count
        total_changes += changes
        facility_changes[aid] = changes
        facility_unique[aid] = unique_count
        facility_std[aid] = std

    # Eng36-specific analysis
    eng36_changes = 0
    eng36_facilities_changed = 0
    eng36_binary_switches = 0
    eng36_variable_changes = 0
    for aid in eng36_ids:
        fc = facility_changes.get(aid, 0)
        if fc > 0:
            eng36_facilities_changed += 1
            eng36_changes += fc
            # Check if binary (only 0/1 values)
            col = f"a:{aid}"
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(1.0)
                unique_vals = set(vals.round(3).unique())
                if unique_vals <= {0.0, 1.0}:
                    eng36_binary_switches += int((vals.diff().abs() > 0.5).sum())
                else:
                    eng36_variable_changes += fc

    # Checkpoint-after analysis (after 40 min)
    post_40 = df[df["elapsed_min"] >= 40.0]
    post_changes = 0
    for col in action_cols:
        if col in post_40.columns:
            vals = pd.to_numeric(post_40[col], errors="coerce").fillna(1.0)
            post_changes += int((vals.diff().abs() > 1e-6).sum())

    # Flood analysis
    flood_cols = [c for c in df.columns if c.startswith("flood:")]
    pfv_cols = [f"flood:{n}" for n in _priority_nodes() if f"flood:{n}" in df.columns]
    flood_sum = df[flood_cols].fillna(0.0).sum(axis=1) if flood_cols else pd.Series([0.0])
    total_flood = float(flood_sum.sum() * 300)  # m3 (5-min steps)
    pfv_sum = df[pfv_cols].fillna(0.0).sum(axis=1) if pfv_cols else pd.Series([0.0])
    pfv = float(pfv_sum.sum() * 300)

    # Rainfall phase
    elapsed = df["elapsed_min"]
    max_elapsed = float(elapsed.max())
    rain_active = df[df["rainfall_mm_h"] > 0.01]
    rain_end = float(rain_active["elapsed_min"].max()) if not rain_active.empty else 0.0

    return {
        "rows": len(df),
        "total_action_changes": total_changes,
        "total_unique_settings": total_unique,
        "eng36_action_changes": eng36_changes,
        "eng36_facilities_changed": eng36_facilities_changed,
        "eng36_binary_switches": eng36_binary_switches,
        "eng36_variable_changes": eng36_variable_changes,
        "post_40min_action_changes": post_changes,
        "native_change_detected": total_changes > 0,
        "eng36_change_detected": eng36_changes > 0,
        "total_flood_m3": total_flood,
        "pfv_m3": pfv,
        "max_elapsed_min": max_elapsed,
        "rain_end_min": rain_end,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    actuators = _load_round0_actuators()
    priority = _priority_nodes()
    eng36_ids = actuators["actuator_id"].astype(str).tolist()

    # Load rainfall table
    rain_table = pd.read_csv(RAIN_TABLE)
    rain_table["event_id"] = rain_table["event_id"].astype(str)

    # Filter to scan events
    scan_rows = rain_table[rain_table["event_id"].isin(SCAN_EVENTS)]
    if scan_rows.empty:
        print("ERROR: No matching rainfall events found!")
        return 1

    print(f"[Stage 2] Scanning {len(scan_rows)} events for native-rule activity...")
    print(f"  Engineering36 facilities: {len(eng36_ids)}")
    print(f"  Priority nodes: {len(priority)}")

    scan_results = []
    for idx, (_, ev) in enumerate(scan_rows.iterrows(), 1):
        event_id = str(ev["event_id"])
        duration_min = int(ev["duration_min"])
        sim_duration_min = int(ev["simulation_duration_min"])
        rainfall_csv = str(ev["rainfall_csv"])

        print(f"\n[{idx}/{len(scan_rows)}] {event_id} (dur={duration_min}min, sim={sim_duration_min}min)")

        # Create event INP with controls preserved
        event_inp = WORK_DIR / f"{event_id}__with_controls.inp"
        if not event_inp.exists():
            mutate_inp_for_event(
                INP_PATH, rainfall_csv, event_inp,
                sim_duration_min, strip_controls=False,
            )

        # Run SWMM with internal_rules (native controls)
        detail_csv = WORK_DIR / f"{event_id}__internal_rules_detail.csv"
        if not detail_csv.exists():
            t0 = time.time()
            try:
                kpis = run_swmm_trajectory(
                    inp_path=event_inp,
                    policy_id="internal_rules",
                    actuators=actuators,
                    priority_nodes=priority,
                    out_detail_csv=detail_csv,
                    event_id=event_id,
                    duration_min=duration_min,
                    control_step_sec=300,
                    seed=2026,
                    simulation_duration_min=sim_duration_min,
                    recession_min=sim_duration_min - duration_min,
                )
                wall = time.time() - t0
                print(f"  SWMM completed in {wall:.1f}s, rows={kpis.get('rows', '?')}")
            except Exception as exc:
                print(f"  SWMM FAILED: {exc}")
                scan_results.append({
                    "event_id": event_id,
                    "duration_min": duration_min,
                    "simulation_duration_min": sim_duration_min,
                    "swmm_error": str(exc),
                    "native_change_detected": False,
                })
                continue
        else:
            print(f"  Reusing existing detail CSV")

        # Analyze
        analysis = _analyze_detail(detail_csv, eng36_ids)
        row = {
            "event_id": event_id,
            "duration_min": duration_min,
            "simulation_duration_min": sim_duration_min,
            "rainfall_csv": rainfall_csv,
            "rainfall_sha256": _file_sha256(Path(rainfall_csv)),
            "inp_sha256": _file_sha256(event_inp),
            **analysis,
        }
        scan_results.append(row)
        print(f"  Changes: total={analysis['total_action_changes']}, eng36={analysis['eng36_action_changes']}, "
              f"post40={analysis['post_40min_action_changes']}, PFV={analysis['pfv_m3']:.1f}m3")

    # Write scan results
    scan_df = pd.DataFrame(scan_results)
    scan_path = OUT_DIR / "positive_control_event_scan.csv"
    scan_df.to_csv(scan_path, index=False)
    print(f"\n[Stage 2] Wrote {scan_path}")

    # Select primary and backup events
    # Criteria: native_change_detected=True, eng36_change_detected=True,
    #           post_40min_action_changes > 0, prefer events with more changes
    candidates = scan_df[
        scan_df.get("native_change_detected", pd.Series([False] * len(scan_df))) &
        scan_df.get("eng36_change_detected", pd.Series([False] * len(scan_df)))
    ].copy()

    if candidates.empty:
        # Fallback: any event with native changes
        candidates = scan_df[scan_df.get("native_change_detected", pd.Series([False]))].copy()

    if candidates.empty:
        print("ERROR: No events with native-rule changes found!")
        selection = {
            "primary_event": None,
            "backup_event": None,
            "selection_rationale": "NO_EVENTS_WITH_NATIVE_CHANGES",
            "gate_blocked": True,
        }
    else:
        # Sort by total_action_changes descending
        candidates = candidates.sort_values("total_action_changes", ascending=False)
        primary = candidates.iloc[0]
        backup = candidates.iloc[1] if len(candidates) > 1 else candidates.iloc[0]

        selection = {
            "primary_event": str(primary["event_id"]),
            "primary_duration_min": int(primary["duration_min"]),
            "primary_sim_duration_min": int(primary["simulation_duration_min"]),
            "primary_total_changes": int(primary["total_action_changes"]),
            "primary_eng36_changes": int(primary["eng36_action_changes"]),
            "primary_post_40_changes": int(primary["post_40min_action_changes"]),
            "primary_pfv_m3": float(primary["pfv_m3"]),
            "backup_event": str(backup["event_id"]),
            "backup_duration_min": int(backup["duration_min"]),
            "backup_total_changes": int(backup["total_action_changes"]),
            "backup_eng36_changes": int(backup["eng36_action_changes"]),
            "selection_rationale": "max_native_rule_activity_with_eng36_changes",
            "selection_not_based_on_model_accuracy": True,
            "gate_blocked": False,
            "total_candidates": len(candidates),
            "total_scanned": len(scan_df),
        }

    sel_path = OUT_DIR / "positive_control_selection.json"
    sel_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Stage 2] Wrote {sel_path}")

    # Summary
    print(f"\n=== Stage 2 Summary ===")
    print(f"  Events scanned: {len(scan_df)}")
    print(f"  Events with native changes: {scan_df.get('native_change_detected', pd.Series([False])).sum()}")
    print(f"  Events with Eng36 changes: {scan_df.get('eng36_change_detected', pd.Series([False])).sum()}")
    if not selection.get("gate_blocked"):
        print(f"  Primary event: {selection['primary_event']}")
        print(f"  Backup event: {selection['backup_event']}")
    else:
        print(f"  GATE BLOCKED: No positive control events found")

    return 0


if __name__ == "__main__":
    sys.exit(main())
