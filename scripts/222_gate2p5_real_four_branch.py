"""Gate 2.5-real Stage 5: Run four-branch comparison at each checkpoint.

For each checkpoint, runs:
  1. no_control (strip_controls=True)
  2. dynamic_internal_rules (prefix replay + native rules handoff)
  3. hold_internal_snapshot (freeze at checkpoint settings)
  4. hold_previous (freeze at pre-checkpoint settings)

Outputs (in outputs/project6_dual_reference_v4/recovery_validation/gate2p5_real/):
  - branch_{name}_detail.csv (per checkpoint per branch)
  - state_hash_comparison.csv
  - branch_action_comparison.csv
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
from sewerrtc.simulation.pyswmm_runner import run_swmm_trajectory, run_swmm_dynamic_internal
from sewerrtc.data.round0_prompt2 import _load_round0_actuators, _priority_nodes

INP_PATH = PROJECT_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"
RAIN_TABLE = PROJECT_ROOT / "outputs" / "rainfall_library_v8_storage_variablepump" / "rainfall_event_table.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real"
WORK_DIR = OUT_DIR / "stage5_work"
SELECTION_PATH = OUT_DIR / "positive_control_selection.json"
CATALOG_PATH = OUT_DIR / "checkpoint_catalog.csv"
RUN1_DETAIL = OUT_DIR / "native_internal_run1_detail.csv"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _df_action_hash(df: pd.DataFrame) -> str:
    action_cols = sorted([c for c in df.columns if c.startswith("a:")])
    h = hashlib.sha256()
    for col in action_cols:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(-999.0).to_numpy()
        h.update(col.encode())
        h.update(vals.tobytes())
    return h.hexdigest()


def _df_state_hash(df: pd.DataFrame, checkpoint_min: float) -> dict:
    """Compute state hashes at checkpoint."""
    row = df.iloc[(df["elapsed_min"] - checkpoint_min).abs().argsort()[:1]]
    hashes = {}
    # Node depths
    h_cols = sorted([c for c in df.columns if c.startswith("h:")])
    h = hashlib.sha256()
    for c in h_cols:
        h.update(c.encode())
        h.update(np.array([float(row[c].iloc[0]) if not pd.isna(row[c].iloc[0]) else -999.0]).tobytes())
    hashes["node_depth_sha256"] = h.hexdigest()
    # Flood
    f_cols = sorted([c for c in df.columns if c.startswith("flood:")])
    h = hashlib.sha256()
    for c in f_cols:
        h.update(c.encode())
        h.update(np.array([float(row[c].iloc[0]) if not pd.isna(row[c].iloc[0]) else -999.0]).tobytes())
    hashes["flood_sha256"] = h.hexdigest()
    # Actions
    a_cols = sorted([c for c in df.columns if c.startswith("a:")])
    h = hashlib.sha256()
    for c in a_cols:
        h.update(c.encode())
        h.update(np.array([float(row[c].iloc[0]) if not pd.isna(row[c].iloc[0]) else -999.0]).tobytes())
    hashes["action_sha256"] = h.hexdigest()
    return hashes


def _create_frozen_baseline(baseline_df: pd.DataFrame, freeze_min: float,
                            eng36_ids: list[str]) -> pd.DataFrame:
    """Create a baseline where actions after freeze_min are frozen at freeze_min values."""
    frozen = baseline_df.copy()
    # Find the row closest to freeze_min
    idx = (frozen["elapsed_min"] - freeze_min).abs().idxmin()
    freeze_row = frozen.loc[idx]
    # Freeze all action columns after freeze_min
    action_cols = [c for c in frozen.columns if c.startswith("a:")]
    mask = frozen["elapsed_min"] > freeze_min
    for col in action_cols:
        frozen.loc[mask, col] = freeze_row[col]
    return frozen


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Load inputs
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    primary_event = selection["primary_event"]
    catalog = pd.read_csv(CATALOG_PATH)
    baseline_detail = pd.read_csv(RUN1_DETAIL)

    rain_table = pd.read_csv(RAIN_TABLE)
    ev_row = rain_table[rain_table["event_id"].astype(str) == primary_event].iloc[0]
    duration_min = int(ev_row["duration_min"])
    sim_duration_min = int(ev_row["simulation_duration_min"])
    rainfall_csv = str(ev_row["rainfall_csv"])

    actuators = _load_round0_actuators()
    priority = _priority_nodes()
    eng36_ids = actuators["actuator_id"].astype(str).tolist()

    # Create event INPs
    inp_with_controls = WORK_DIR / f"{primary_event}__with_controls.inp"
    inp_no_controls = WORK_DIR / f"{primary_event}__no_controls.inp"
    if not inp_with_controls.exists():
        mutate_inp_for_event(INP_PATH, rainfall_csv, inp_with_controls,
                             sim_duration_min, strip_controls=False)
    if not inp_no_controls.exists():
        mutate_inp_for_event(INP_PATH, rainfall_csv, inp_no_controls,
                             sim_duration_min, strip_controls=True)

    print(f"[Stage 5] Event: {primary_event}")
    print(f"  Duration: {duration_min}min, Sim: {sim_duration_min}min")
    print(f"  Checkpoints: {catalog['checkpoint_elapsed_min'].tolist()}")

    hash_rows = []
    action_rows = []

    for _, cp_row in catalog.iterrows():
        cp_min = float(cp_row["checkpoint_elapsed_min"])
        cp_label = str(cp_row["checkpoint_label"])
        print(f"\n=== Checkpoint {cp_label} at t={cp_min:.0f}min ===")

        # Branch 1: no_control
        print(f"  Running no_control...")
        nc_detail = WORK_DIR / f"cp{cp_label}__no_control_detail.csv"
        t0 = time.time()
        nc_kpis = run_swmm_trajectory(
            inp_path=inp_no_controls, policy_id="internal_rules",
            actuators=actuators, priority_nodes=priority,
            out_detail_csv=nc_detail, event_id=primary_event,
            duration_min=duration_min, control_step_sec=300, seed=2026,
            simulation_duration_min=sim_duration_min,
            recession_min=sim_duration_min - duration_min,
        )
        print(f"    done in {time.time()-t0:.1f}s")
        nc_df = pd.read_csv(nc_detail)
        nc_df.to_csv(OUT_DIR / f"branch_no_control_cp{cp_label}_detail.csv", index=False)

        # Branch 2: dynamic_internal_rules
        print(f"  Running dynamic_internal_rules (override_start={cp_min})...")
        di_detail = WORK_DIR / f"cp{cp_label}__dynamic_internal_detail.csv"
        t0 = time.time()
        di_kpis = run_swmm_dynamic_internal(
            inp_path=inp_with_controls, actuators=actuators,
            priority_nodes=priority,
            internal_baseline_detail_csv=RUN1_DETAIL,
            out_detail_csv=di_detail, event_id=primary_event,
            duration_min=duration_min, override_start_min=cp_min,
            control_step_sec=300,
        )
        print(f"    done in {time.time()-t0:.1f}s")
        di_df = pd.read_csv(di_detail)
        di_df.to_csv(OUT_DIR / f"branch_dynamic_internal_cp{cp_label}_detail.csv", index=False)

        # Branch 3: hold_internal_snapshot (freeze entire baseline = prefix replay forever)
        print(f"  Running hold_internal_snapshot...")
        his_detail = WORK_DIR / f"cp{cp_label}__hold_snapshot_detail.csv"
        t0 = time.time()
        his_kpis = run_swmm_dynamic_internal(
            inp_path=inp_with_controls, actuators=actuators,
            priority_nodes=priority,
            internal_baseline_detail_csv=RUN1_DETAIL,
            out_detail_csv=his_detail, event_id=primary_event,
            duration_min=duration_min, override_start_min=float(sim_duration_min) + 1.0,
            control_step_sec=300,
            policy_id="hold_internal_snapshot",
        )
        print(f"    done in {time.time()-t0:.1f}s")
        his_df = pd.read_csv(his_detail)
        his_df.to_csv(OUT_DIR / f"branch_hold_snapshot_cp{cp_label}_detail.csv", index=False)

        # Branch 4: hold_previous (freeze at checkpoint-5min values)
        print(f"  Running hold_previous (freeze at {cp_min - 5:.0f}min)...")
        hp_baseline = _create_frozen_baseline(baseline_detail, cp_min - 5.0, eng36_ids)
        hp_baseline_path = WORK_DIR / f"cp{cp_label}__hold_previous_baseline.csv"
        hp_baseline.to_csv(hp_baseline_path, index=False)
        hp_detail = WORK_DIR / f"cp{cp_label}__hold_previous_detail.csv"
        t0 = time.time()
        hp_kpis = run_swmm_dynamic_internal(
            inp_path=inp_with_controls, actuators=actuators,
            priority_nodes=priority,
            internal_baseline_detail_csv=hp_baseline_path,
            out_detail_csv=hp_detail, event_id=primary_event,
            duration_min=duration_min, override_start_min=float(sim_duration_min) + 1.0,
            control_step_sec=300,
            policy_id="hold_previous",
        )
        print(f"    done in {time.time()-t0:.1f}s")
        hp_df = pd.read_csv(hp_detail)
        hp_df.to_csv(OUT_DIR / f"branch_hold_previous_cp{cp_label}_detail.csv", index=False)

        # Compute state hashes at checkpoint
        for branch_name, branch_df in [("no_control", nc_df), ("dynamic_internal", di_df),
                                        ("hold_internal_snapshot", his_df), ("hold_previous", hp_df)]:
            state_hashes = _df_state_hash(branch_df, cp_min)
            hash_rows.append({
                "checkpoint_label": cp_label,
                "checkpoint_elapsed_min": cp_min,
                "branch": branch_name,
                "network_sha256": _file_sha256(inp_with_controls if branch_name != "no_control" else inp_no_controls),
                "rainfall_sha256": _file_sha256(Path(rainfall_csv)),
                "action_schedule_sha256": _df_action_hash(branch_df),
                **state_hashes,
            })

            # Action stats
            action_cols = sorted([c for c in branch_df.columns if c.startswith("a:")])
            post_cp = branch_df[branch_df["elapsed_min"] > cp_min]
            pre_cp = branch_df[branch_df["elapsed_min"] <= cp_min]
            changes_post = 0
            unique_settings = {}
            for col in action_cols:
                aid = col.split(":", 1)[1]
                if col in post_cp.columns:
                    vals = pd.to_numeric(post_cp[col], errors="coerce").fillna(1.0)
                    changes_post += int((vals.diff().abs() > 1e-6).sum())
                    unique_settings[aid] = int(vals.nunique())

            action_rows.append({
                "checkpoint_label": cp_label,
                "checkpoint_elapsed_min": cp_min,
                "branch": branch_name,
                "total_rows": len(branch_df),
                "post_checkpoint_rows": len(post_cp),
                "post_checkpoint_action_changes": changes_post,
                "pfv": nc_kpis.get("PFV", 0) if branch_name == "no_control" else
                       di_kpis.get("PFV", 0) if branch_name == "dynamic_internal" else
                       his_kpis.get("PFV", 0) if branch_name == "hold_internal_snapshot" else
                       hp_kpis.get("PFV", 0),
            })

    # Write outputs
    hash_df = pd.DataFrame(hash_rows)
    hash_path = OUT_DIR / "state_hash_comparison.csv"
    hash_df.to_csv(hash_path, index=False)
    print(f"\n[Stage 5] Wrote {hash_path}")

    action_df = pd.DataFrame(action_rows)
    action_path = OUT_DIR / "branch_action_comparison.csv"
    action_df.to_csv(action_path, index=False)
    print(f"[Stage 5] Wrote {action_path}")

    # Summary
    print(f"\n=== Stage 5 Summary ===")
    for cp_label in catalog["checkpoint_label"].tolist():
        print(f"\n  Checkpoint {cp_label}:")
        cp_hashes = hash_df[hash_df["checkpoint_label"] == cp_label]
        cp_actions = action_df[action_df["checkpoint_label"] == cp_label]
        for _, r in cp_actions.iterrows():
            print(f"    {r['branch']}: post_changes={r['post_checkpoint_action_changes']}, "
                  f"PFV={r['pfv']:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
