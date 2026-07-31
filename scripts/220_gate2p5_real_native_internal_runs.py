"""Gate 2.5-real Stage 3: Run native_internal_from_start twice for determinism check.

Runs the primary positive-control event twice with identical parameters.
Compares action schedules, settings, KPIs, and recovery.

Outputs (in outputs/project6_dual_reference_v4/recovery_validation/gate2p5_real/):
  - native_internal_run1_detail.csv
  - native_internal_run2_detail.csv
  - native_internal_repeatability.csv
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

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
WORK_DIR = OUT_DIR / "stage3_work"
SELECTION_PATH = OUT_DIR / "positive_control_selection.json"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _df_sha256(df: pd.DataFrame, columns: list[str]) -> str:
    """Hash specific columns of a DataFrame."""
    h = hashlib.sha256()
    for col in sorted(columns):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(-999.0).to_numpy()
            h.update(col.encode())
            h.update(vals.tobytes())
    return h.hexdigest()


def _action_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("a:")])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Load selection
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    primary_event = selection["primary_event"]
    print(f"[Stage 3] Primary event: {primary_event}")

    # Load rainfall info
    rain_table = pd.read_csv(RAIN_TABLE)
    ev_row = rain_table[rain_table["event_id"].astype(str) == primary_event]
    if ev_row.empty:
        print(f"ERROR: Event {primary_event} not found in rainfall table!")
        return 1
    ev = ev_row.iloc[0]
    duration_min = int(ev["duration_min"])
    sim_duration_min = int(ev["simulation_duration_min"])
    rainfall_csv = str(ev["rainfall_csv"])

    # Load actuators and priority
    actuators = _load_round0_actuators()
    priority = _priority_nodes()
    eng36_ids = actuators["actuator_id"].astype(str).tolist()

    # Create event INP with controls preserved
    event_inp = WORK_DIR / f"{primary_event}__with_controls.inp"
    if not event_inp.exists():
        mutate_inp_for_event(
            INP_PATH, rainfall_csv, event_inp,
            sim_duration_min, strip_controls=False,
        )
    print(f"  INP: {event_inp} (sha256={_file_sha256(event_inp)[:16]}...)")

    repeatability_rows = []

    for run_tag in ["run1", "run2"]:
        detail_csv = WORK_DIR / f"{primary_event}__internal_{run_tag}_detail.csv"
        # Always re-run for determinism check (don't reuse Stage 2 output)
        t0 = time.time()
        print(f"\n  Running {run_tag}...")
        kpis = run_swmm_trajectory(
            inp_path=event_inp,
            policy_id="internal_rules",
            actuators=actuators,
            priority_nodes=priority,
            out_detail_csv=detail_csv,
            event_id=primary_event,
            duration_min=duration_min,
            control_step_sec=300,
            seed=2026,
            simulation_duration_min=sim_duration_min,
            recession_min=sim_duration_min - duration_min,
        )
        wall = time.time() - t0
        print(f"  {run_tag} completed in {wall:.1f}s, rows={kpis.get('rows', '?')}")

        # Copy to output dir
        out_detail = OUT_DIR / f"native_internal_{run_tag}_detail.csv"
        detail_df = pd.read_csv(detail_csv)
        detail_df.to_csv(out_detail, index=False)
        print(f"  Wrote {out_detail}")

        # Compute hashes
        action_cols = _action_columns(detail_df)
        action_hash = _df_sha256(detail_df, action_cols)
        setting_cols = [f"setting:{c.split(':')[1]}" for c in action_cols if f"setting:{c.split(':')[1]}" in detail_df.columns]
        setting_hash = _df_sha256(detail_df, setting_cols)
        flood_cols = [c for c in detail_df.columns if c.startswith("flood:")]
        flood_hash = _df_sha256(detail_df, flood_cols)

        repeatability_rows.append({
            "run": run_tag,
            "event_id": primary_event,
            "rows": len(detail_df),
            "wall_time_sec": wall,
            "action_columns_count": len(action_cols),
            "action_schedule_sha256": action_hash,
            "setting_sequence_sha256": setting_hash,
            "flood_sequence_sha256": flood_hash,
            "inp_sha256": _file_sha256(event_inp),
            "rainfall_sha256": _file_sha256(Path(rainfall_csv)),
            "PFV": kpis.get("PFV", 0.0),
            "TFV": kpis.get("TFV", 0.0),
            "peak_TFV_rate": kpis.get("peak_TFV_rate", 0.0),
            "flood_duration_min": kpis.get("flood_duration_min", 0.0),
            "action_changes": kpis.get("action_changes", 0.0),
            "recovery_criteria_met": kpis.get("recovery_criteria_met", False),
            "max_elapsed_min": float(detail_df["elapsed_min"].max()) if not detail_df.empty else 0.0,
        })

    # Compare runs
    if len(repeatability_rows) >= 2:
        r1, r2 = repeatability_rows
        action_match = r1["action_schedule_sha256"] == r2["action_schedule_sha256"]
        setting_match = r1["setting_sequence_sha256"] == r2["setting_sequence_sha256"]
        flood_match = r1["flood_sequence_sha256"] == r2["flood_sequence_sha256"]
        pfv_diff = abs(r1["PFV"] - r2["PFV"])
        tfv_diff = abs(r1["TFV"] - r2["TFV"])
        peak_diff = abs(r1["peak_TFV_rate"] - r2["peak_TFV_rate"])

        verdict = "DETERMINISTIC" if (action_match and setting_match and flood_match) else "NON_DETERMINISTIC"

        print(f"\n=== Determinism Check ===")
        print(f"  Action SHA match: {action_match}")
        print(f"  Setting SHA match: {setting_match}")
        print(f"  Flood SHA match: {flood_match}")
        print(f"  PFV diff: {pfv_diff:.6e}")
        print(f"  TFV diff: {tfv_diff:.6e}")
        print(f"  Peak diff: {peak_diff:.6e}")
        print(f"  VERDICT: {verdict}")

        # Add comparison to repeatability
        for row in repeatability_rows:
            row["determinism_verdict"] = verdict
            row["action_sha_match_run1_vs_run2"] = action_match
            row["pfv_abs_diff"] = pfv_diff
            row["tfv_abs_diff"] = tfv_diff
            row["peak_abs_diff"] = peak_diff

    # Write repeatability
    rep_df = pd.DataFrame(repeatability_rows)
    rep_path = OUT_DIR / "native_internal_repeatability.csv"
    rep_df.to_csv(rep_path, index=False)
    print(f"\n[Stage 3] Wrote {rep_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
