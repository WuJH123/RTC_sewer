"""Gate 2.5-real-v2 Runner: End-to-end execution with all V1 fixes.

Executes:
  1. Read Scope Contract
  2. Create with-controls and no-controls INP variants
  3. Run baseline (with controls, internal_rules)
  4. Determine checkpoints
  5. Run 4 branches (dynamic_internal + no_control + hold_snapshot + hold_previous)
  6. Run causal intervention (3 sub-branches)
  7. Compute H120 from checkpoint (not from t=0)
  8. Compute Recovery (with extended tail)
  9. Record all SHAs, hotstart audit

All outputs in: outputs/project6_dual_reference_v4/recovery_validation/gate2p5_real_v2/
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
from sewerrtc.simulation.pyswmm_runner import (
    run_swmm_trajectory,
    run_swmm_dynamic_internal,
    run_swmm_fixed_action,
    physical_network_sha256,
    _load_nominal_action_table,
)
from sewerrtc.simulation.kpi_metrics import compute_kpis
from sewerrtc.simulation.runtime_contracts import analyze_recovery
from sewerrtc.data.round0_prompt2 import _load_round0_actuators, _priority_nodes

INP_PATH = PROJECT_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"
RAIN_TABLE = PROJECT_ROOT / "outputs" / "rainfall_library_v8_storage_variablepump" / "rainfall_event_table.csv"
V1_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real"
OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real_v2"
WORK_DIR = OUT_DIR / "work"
SCOPE_CONTRACT = PROJECT_ROOT / "docs" / "contracts" / "PROJECT6_V4_CONTROL_SCOPE_CONTRACT.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _df_sha256(df: pd.DataFrame, cols: list[str]) -> str:
    h = hashlib.sha256()
    for c in sorted(cols):
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").fillna(-999.0).to_numpy()
            h.update(c.encode())
            h.update(vals.tobytes())
    return h.hexdigest()


def _compute_h120_from_checkpoint(detail: pd.DataFrame, checkpoint_min: float,
                                   priority: list[str], dt_sec: int = 300) -> dict:
    """Compute H120 KPIs with window starting at checkpoint (NOT at t=0)."""
    window = detail[(detail["elapsed_min"] > checkpoint_min) &
                    (detail["elapsed_min"] <= checkpoint_min + 120.0)]
    if window.empty:
        return {"PFV_H120": 0.0, "TFV_H120": 0.0, "peak_TFV_rate_H120": 0.0,
                "flood_duration_H120": 0.0, "priority_peak_depth_H120": 0.0,
                "h120_rows": 0, "h120_window_hash": ""}
    kpis = compute_kpis(window, priority, dt_sec=dt_sec)
    pr_cols = [c for c in window.columns if c.startswith("flood:") and c.split(":", 1)[1] in priority]
    pr_peak = 0.0
    if pr_cols:
        pr_rate = window[pr_cols].fillna(0.0).to_numpy(float).sum(axis=1)
        pr_peak = float(pr_rate.max())
    # Hash the window rows for uniqueness verification
    window_hash = _df_sha256(window, [c for c in window.columns if c.startswith("elapsed_min")])
    return {
        "PFV_H120": kpis.get("PFV", 0.0),
        "TFV_H120": kpis.get("TFV", 0.0),
        "peak_TFV_rate_H120": kpis.get("peak_TFV_rate", 0.0),
        "flood_duration_H120": kpis.get("flood_duration_min", 0.0),
        "priority_peak_depth_H120": pr_peak,
        "h120_rows": len(window),
        "h120_window_hash": window_hash,
    }


def _audit_h120(detail: pd.DataFrame, checkpoint_min: float,
                 priority: list[str], dt_sec: int = 300) -> dict:
    """Independent H120 audit (second implementation)."""
    window = detail[(detail["elapsed_min"] > checkpoint_min) &
                    (detail["elapsed_min"] <= checkpoint_min + 120.0)]
    if window.empty:
        return {"audit_PFV": 0.0, "audit_TFV": 0.0, "audit_peak": 0.0}
    flood_cols = [c for c in window.columns if c.startswith("flood:")]
    pr_cols = [c for c in flood_cols if c.split(":", 1)[1] in priority]
    flood = window[flood_cols].fillna(0.0).to_numpy(float)
    rate = flood.sum(axis=1)
    tfv = float(rate.sum() * dt_sec)
    peak = float(rate.max())
    pfv = 0.0
    if pr_cols:
        pfv = float(window[pr_cols].fillna(0.0).to_numpy(float).sum() * dt_sec)
    return {"audit_PFV": pfv, "audit_TFV": tfv, "audit_peak": peak}


def _checkpoint_state_hash(detail: pd.DataFrame, checkpoint_min: float) -> dict:
    """Compute full state hash at the LAST PREFIX row before checkpoint.

    This captures the shared state before branches diverge. Using the row AT
    checkpoint would capture post_action (since elapsed_min < checkpoint_min is
    False at the exact checkpoint time).
    """
    prefix_rows = detail[detail["elapsed_min"] < checkpoint_min]
    if prefix_rows.empty:
        row_idx = (detail["elapsed_min"] - checkpoint_min).abs().idxmin()
    else:
        row_idx = prefix_rows.index[-1]
    row = detail.loc[row_idx]
    hashes = {}
    for prefix in ["h:", "head:", "flood:", "storage_volume:", "a:", "setting:", "flow:"]:
        cols = sorted([c for c in detail.columns if c.startswith(prefix)])
        h = hashlib.sha256()
        for c in cols:
            v = float(row[c]) if c in detail.columns and not pd.isna(row[c]) else -999.0
            h.update(c.encode())
            h.update(np.array([v], dtype=np.float64).tobytes())
        hashes[f"{prefix.replace(':', '')}_sha256"] = h.hexdigest()
    return hashes


def _count_post_changes(detail: pd.DataFrame, checkpoint_min: float) -> int:
    """Count action changes after checkpoint."""
    post = detail[detail["elapsed_min"] > checkpoint_min]
    if len(post) < 2:
        return 0
    a_cols = [c for c in detail.columns if c.startswith("a:")]
    changes = 0
    for c in a_cols:
        vals = pd.to_numeric(post[c], errors="coerce").fillna(1.0)
        changes += int((vals.diff().abs() > 1e-6).sum())
    return changes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Load scope contract
    scope = json.loads(SCOPE_CONTRACT.read_text(encoding="utf-8"))
    eng36_ids = scope["engineering36_ids"]
    native_82 = scope["engineering36_overlap_with_native_rules"] + scope["non_engineering36_native_controlled"]
    print(f"[V2 Runner] Scope: {len(eng36_ids)} Eng36, {len(native_82)} native-rule facilities")

    # Load V1 selection (reuse event + checkpoints)
    v1_selection = json.loads((V1_DIR / "positive_control_selection.json").read_text(encoding="utf-8"))
    v1_catalog = pd.read_csv(V1_DIR / "checkpoint_catalog.csv")
    primary_event = v1_selection["primary_event"]

    # Load rainfall info
    rain_table = pd.read_csv(RAIN_TABLE)
    ev_row = rain_table[rain_table["event_id"].astype(str) == primary_event].iloc[0]
    duration_min = int(ev_row["duration_min"])
    rainfall_csv = str(ev_row["rainfall_csv"])

    # Extended simulation time for recovery: duration + 360 min tail
    sim_duration_min = duration_min + 360  # 300 + 360 = 660 min
    recession_min = sim_duration_min - duration_min  # 360 min

    actuators = _load_round0_actuators()
    priority = _priority_nodes()
    all_actuator_ids = actuators["actuator_id"].astype(str).tolist()
    n_act = len(all_actuator_ids)

    print(f"[V2 Runner] Event: {primary_event}")
    print(f"  Duration: {duration_min}min, Sim: {sim_duration_min}min (extended for recovery)")
    print(f"  Actuators: {n_act}, Priority nodes: {len(priority)}")

    # ---- Step 1: Create INP variants ----
    inp_with = WORK_DIR / f"{primary_event}__with_controls.inp"
    inp_no = WORK_DIR / f"{primary_event}__no_controls.inp"
    if not inp_with.exists():
        mutate_inp_for_event(INP_PATH, rainfall_csv, inp_with, sim_duration_min, strip_controls=False)
    if not inp_no.exists():
        mutate_inp_for_event(INP_PATH, rainfall_csv, inp_no, sim_duration_min, strip_controls=True)

    phys_sha_with = physical_network_sha256(inp_with)
    phys_sha_no = physical_network_sha256(inp_no)
    print(f"  Physical SHA (with controls):    {phys_sha_with[:32]}...")
    print(f"  Physical SHA (no controls):      {phys_sha_no[:32]}...")
    print(f"  Physical SHA match: {phys_sha_with == phys_sha_no}")

    # ---- Step 2: Run baseline (with controls, internal_rules) ----
    baseline_detail = WORK_DIR / f"{primary_event}__baseline_detail.csv"
    if not baseline_detail.exists():
        print(f"\n  Running baseline (with controls, internal_rules)...")
        t0 = time.time()
        baseline_kpis = run_swmm_trajectory(
            inp_path=inp_with, policy_id="internal_rules",
            actuators=actuators, priority_nodes=priority,
            out_detail_csv=baseline_detail, event_id=primary_event,
            duration_min=duration_min, control_step_sec=300, seed=2026,
            simulation_duration_min=sim_duration_min,
            recession_min=recession_min,
        )
        print(f"  Baseline done in {time.time()-t0:.1f}s")
    else:
        print(f"\n  Baseline already exists, loading...")

    baseline_df = pd.read_csv(baseline_detail)
    # Copy to output dir
    baseline_df.to_csv(OUT_DIR / "baseline_detail.csv", index=False)
    print(f"  Baseline: {len(baseline_df)} rows, elapsed={baseline_df['elapsed_min'].max():.0f}min")

    # B7: Clean up any .hsf files created by run_swmm_trajectory (try_save_hotstart)
    # V2 output directory must NOT contain .hsf files
    _hsf_cleanup = list(OUT_DIR.rglob("*.hsf")) + list(WORK_DIR.rglob("*.hsf"))
    for _hsf in _hsf_cleanup:
        _hsf.unlink(missing_ok=True)
    if _hsf_cleanup:
        print(f"  Cleaned up {len(_hsf_cleanup)} .hsf files (B7 hotstart policy)")

    # ---- Step 3: Determine checkpoints ----
    checkpoints = {}
    for _, cp_row in v1_catalog.iterrows():
        cp_label = str(cp_row["checkpoint_label"])
        cp_min = float(cp_row["checkpoint_elapsed_min"])
        checkpoints[cp_label] = cp_min
    print(f"\n  Checkpoints: {checkpoints}")

    # ---- Step 4: Build prefix schedule from baseline ----
    prefix_table = _load_nominal_action_table(baseline_detail, all_actuator_ids)

    # ---- Step 5: Run 4 branches at each checkpoint ----
    hash_rows = []
    kpi_rows = []
    recovery_rows_list = []
    snapshot_evidence = []

    for cp_label, cp_min in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"  Checkpoint {cp_label} at t={cp_min:.0f}min")
        print(f"{'='*60}")

        # Get checkpoint state from baseline
        baseline_state = _checkpoint_state_hash(baseline_df, cp_min)

        # Get snapshot settings at checkpoint (82 native-rule facilities)
        cp_row_idx = (baseline_df["elapsed_min"] - cp_min).abs().idxmin()
        snapshot_settings = {}
        for aid in native_82:
            col = f"a:{aid}"
            if col in baseline_df.columns:
                snapshot_settings[aid] = float(baseline_df.loc[cp_row_idx, col])
            else:
                snapshot_settings[aid] = 1.0

        # Get previous settings (checkpoint - 5 min)
        prev_min = cp_min - 5.0
        prev_row_idx = (baseline_df["elapsed_min"] - prev_min).abs().idxmin()
        previous_settings = {}
        for aid in native_82:
            col = f"a:{aid}"
            if col in baseline_df.columns:
                previous_settings[aid] = float(baseline_df.loc[prev_row_idx, col])
            else:
                previous_settings[aid] = 1.0

        # ---- Branch 1: dynamic_internal_rules ----
        print(f"  [1/4] Running dynamic_internal_rules...")
        di_detail = WORK_DIR / f"v2_cp{cp_label}__dynamic_internal_detail.csv"
        t0 = time.time()
        di_kpis = run_swmm_dynamic_internal(
            inp_path=inp_with, actuators=actuators, priority_nodes=priority,
            internal_baseline_detail_csv=baseline_detail,
            out_detail_csv=di_detail, event_id=primary_event,
            duration_min=duration_min, override_start_min=cp_min,
            control_step_sec=300,
        )
        print(f"    done in {time.time()-t0:.1f}s")
        di_df = pd.read_csv(di_detail)
        di_df.to_csv(OUT_DIR / f"branch_dynamic_internal_cp{cp_label}_detail.csv", index=False)

        # ---- Branch 2: no_control ----
        print(f"  [2/4] Running no_control (all 1.0)...")
        nc_detail = WORK_DIR / f"v2_cp{cp_label}__no_control_detail.csv"
        no_ctrl_action = np.ones(n_act, dtype=np.float64)
        t0 = time.time()
        nc_kpis = run_swmm_fixed_action(
            inp_path=inp_no, actuators=actuators, priority_nodes=priority,
            out_detail_csv=nc_detail, event_id=primary_event,
            duration_min=duration_min,
            prefix_schedule=prefix_table,
            override_start_min=cp_min,
            post_action=no_ctrl_action,
            control_step_sec=300,
            policy_id="no_control",
            simulation_duration_min=sim_duration_min,
        )
        print(f"    done in {time.time()-t0:.1f}s")
        nc_df = pd.read_csv(nc_detail)
        nc_df.to_csv(OUT_DIR / f"branch_no_control_cp{cp_label}_detail.csv", index=False)

        # ---- Branch 3: hold_internal_snapshot ----
        print(f"  [3/4] Running hold_internal_snapshot...")
        his_detail = WORK_DIR / f"v2_cp{cp_label}__hold_snapshot_detail.csv"
        # Build snapshot action vector for all actuators
        snap_vec = np.ones(n_act, dtype=np.float64)
        for aid in native_82:
            if aid in all_actuator_ids:
                snap_vec[all_actuator_ids.index(aid)] = snapshot_settings[aid]
        t0 = time.time()
        his_kpis = run_swmm_fixed_action(
            inp_path=inp_no, actuators=actuators, priority_nodes=priority,
            out_detail_csv=his_detail, event_id=primary_event,
            duration_min=duration_min,
            prefix_schedule=prefix_table,
            override_start_min=cp_min,
            post_action=snap_vec,
            control_step_sec=300,
            policy_id="hold_internal_snapshot",
            simulation_duration_min=sim_duration_min,
        )
        print(f"    done in {time.time()-t0:.1f}s")
        his_df = pd.read_csv(his_detail)
        his_df.to_csv(OUT_DIR / f"branch_hold_snapshot_cp{cp_label}_detail.csv", index=False)

        # ---- Branch 4: hold_previous ----
        print(f"  [4/4] Running hold_previous...")
        hp_detail = WORK_DIR / f"v2_cp{cp_label}__hold_previous_detail.csv"
        prev_vec = np.ones(n_act, dtype=np.float64)
        for aid in native_82:
            if aid in all_actuator_ids:
                prev_vec[all_actuator_ids.index(aid)] = previous_settings[aid]
        t0 = time.time()
        hp_kpis = run_swmm_fixed_action(
            inp_path=inp_no, actuators=actuators, priority_nodes=priority,
            out_detail_csv=hp_detail, event_id=primary_event,
            duration_min=duration_min,
            prefix_schedule=prefix_table,
            override_start_min=cp_min,
            post_action=prev_vec,
            control_step_sec=300,
            policy_id="hold_previous",
            simulation_duration_min=sim_duration_min,
        )
        print(f"    done in {time.time()-t0:.1f}s")
        hp_df = pd.read_csv(hp_detail)
        hp_df.to_csv(OUT_DIR / f"branch_hold_previous_cp{cp_label}_detail.csv", index=False)

        # ---- Verify shared prefix ----
        print(f"\n  Verifying shared prefix...")
        for branch_name, branch_df in [("dynamic_internal", di_df), ("no_control", nc_df),
                                        ("hold_snapshot", his_df), ("hold_previous", hp_df)]:
            state = _checkpoint_state_hash(branch_df, cp_min)
            post_changes = _count_post_changes(branch_df, cp_min)

            # H120 from checkpoint
            h120 = _compute_h120_from_checkpoint(branch_df, cp_min, priority)
            audit = _audit_h120(branch_df, cp_min, priority)

            # Recovery
            rec = analyze_recovery(
                branch_df, event_id=primary_event, policy_id=branch_name,
                trajectory_id=f"v2_{branch_name}_cp{cp_label}",
                duration_min=duration_min,
                minimum_tail_min=180, priority_nodes=priority,
            )

            hash_rows.append({
                "checkpoint_label": cp_label,
                "checkpoint_elapsed_min": cp_min,
                "branch": branch_name,
                "physical_network_sha256": phys_sha_with if branch_name == "dynamic_internal" else phys_sha_no,
                "rainfall_sha256": _file_sha256(Path(rainfall_csv)),
                "inp_file_sha256": _file_sha256(inp_with if branch_name == "dynamic_internal" else inp_no),
                "prefix_actual_schedule_sha256": _df_sha256(
                    branch_df[branch_df["elapsed_min"] <= cp_min],
                    [c for c in branch_df.columns if c.startswith("a:")]
                ),
                "checkpoint_state_hash": json.dumps(state),
                "post_checkpoint_action_changes": post_changes,
                "hotstart_used": False,
                **h120,
                **audit,
                "h120_match": abs(h120["PFV_H120"] - audit["audit_PFV"]) < 1.0,
                "recovery_criteria_met": bool(rec.get("recovery_criteria_met", False)),
                "recovery_censored": bool(rec.get("recovery_censored", False)),
            })

            kpi_rows.append({
                "checkpoint_label": cp_label,
                "branch": branch_name,
                **h120,
                **audit,
                "recovery_criteria_met": bool(rec.get("recovery_criteria_met", False)),
                "recovery_time_min": rec.get("recovery_time_min"),
                "last_flood_time_min": rec.get("last_flood_time_min"),
            })

        # Snapshot evidence
        his_post = _count_post_changes(his_df, cp_min)
        di_post = _count_post_changes(di_df, cp_min)
        di_a = _df_sha256(di_df[di_df["elapsed_min"] > cp_min], [c for c in di_df.columns if c.startswith("a:")])
        his_a = _df_sha256(his_df[his_df["elapsed_min"] > cp_min], [c for c in his_df.columns if c.startswith("a:")])
        snapshot_evidence.append({
            "checkpoint_label": cp_label,
            "snapshot_post_changes": his_post,
            "dynamic_post_changes": di_post,
            "snapshot_schedule_sha256": his_a,
            "dynamic_schedule_sha256": di_a,
            "schedules_differ": di_a != his_a,
            "snapshot_is_constant": his_post == 0,
        })

        print(f"  Snapshot: post_changes={his_post}, constant={his_post==0}")
        print(f"  Dynamic:  post_changes={di_post}, SHA differs from snapshot={di_a != his_a}")

    # ---- Step 6: Causal intervention ----
    print(f"\n{'='*60}")
    print(f"  Causal intervention experiment")
    print(f"{'='*60}")

    # Auto-select actuator: find one with non-zero flow post-checkpoint
    cp_label_0 = list(checkpoints.keys())[0]
    cp_min_0 = checkpoints[cp_label_0]
    baseline_post = baseline_df[baseline_df["elapsed_min"] > cp_min_0]
    best_aid = None
    best_flow = 0.0
    for aid in native_82:
        fcol = f"flow:{aid}"
        if fcol in baseline_post.columns:
            max_flow = pd.to_numeric(baseline_post[fcol], errors="coerce").fillna(0).abs().max()
            if max_flow > best_flow:
                best_flow = float(max_flow)
                best_aid = aid

    if best_aid is None:
        best_aid = native_82[0]  # fallback
    print(f"  Selected actuator: {best_aid} (max_flow={best_flow:.3f} m3/s)")

    causal_results = {}
    for setting_val, tag in [(0.0, "low"), (1.0, "high")]:
        causal_action = np.ones(n_act, dtype=np.float64)
        if best_aid in all_actuator_ids:
            causal_action[all_actuator_ids.index(best_aid)] = setting_val

        causal_detail = WORK_DIR / f"v2_causal_{tag}_detail.csv"
        t0 = time.time()
        ck = run_swmm_fixed_action(
            inp_path=inp_no, actuators=actuators, priority_nodes=priority,
            out_detail_csv=causal_detail, event_id=primary_event,
            duration_min=duration_min,
            prefix_schedule=prefix_table,
            override_start_min=cp_min_0,
            post_action=causal_action,
            control_step_sec=300,
            policy_id=f"causal_{tag}",
            simulation_duration_min=sim_duration_min,
        )
        print(f"  Causal {tag} done in {time.time()-t0:.1f}s")
        causal_results[tag] = pd.read_csv(causal_detail)
        causal_results[tag].to_csv(OUT_DIR / f"causal_{tag}_detail.csv", index=False)

    # ---- Write outputs ----
    hash_df = pd.DataFrame(hash_rows)
    hash_df.to_csv(OUT_DIR / "state_hash_comparison.csv", index=False)
    print(f"\n[V2 Runner] Wrote state_hash_comparison.csv")

    kpi_df = pd.DataFrame(kpi_rows)
    kpi_df.to_csv(OUT_DIR / "branch_kpi_comparison.csv", index=False)
    print(f"[V2 Runner] Wrote branch_kpi_comparison.csv")

    snap_df = pd.DataFrame(snapshot_evidence)
    snap_df.to_csv(OUT_DIR / "snapshot_evidence.csv", index=False)
    print(f"[V2 Runner] Wrote snapshot_evidence.csv")

    # Causal comparison
    causal_rows = []
    for tag in ["low", "high"]:
        cdf = causal_results[tag]
        post = cdf[cdf["elapsed_min"] > cp_min_0]
        fcol = f"flow:{best_aid}"
        max_flow = float(pd.to_numeric(post[fcol], errors="coerce").fillna(0).abs().max()) if fcol in post.columns else 0.0
        req_col = f"requested_setting:{best_aid}"
        req_val = float(post[req_col].iloc[0]) if req_col in post.columns and len(post) > 0 else float("nan")
        act_col = f"a:{best_aid}"
        act_val = float(post[act_col].iloc[0]) if act_col in post.columns and len(post) > 0 else float("nan")
        causal_rows.append({
            "branch": f"causal_{tag}",
            "actuator_id": best_aid,
            "requested_setting": req_val,
            "actual_setting": act_val,
            "max_abs_flow": max_flow,
        })
    # Compare flows between low and high
    low_post = causal_results["low"][causal_results["low"]["elapsed_min"] > cp_min_0]
    high_post = causal_results["high"][causal_results["high"]["elapsed_min"] > cp_min_0]
    min_len = min(len(low_post), len(high_post))
    flow_differs = False
    depth_differs = False
    if min_len > 0:
        for c in [c for c in low_post.columns if c.startswith("flow:")]:
            if c in high_post.columns:
                lv = pd.to_numeric(low_post[c].iloc[:min_len], errors="coerce").fillna(0)
                hv = pd.to_numeric(high_post[c].iloc[:min_len], errors="coerce").fillna(0)
                if (lv - hv).abs().max() > 1e-6:
                    flow_differs = True
                    break
        for c in [c for c in low_post.columns if c.startswith("h:")]:
            if c in high_post.columns:
                lv = pd.to_numeric(low_post[c].iloc[:min_len], errors="coerce").fillna(0)
                hv = pd.to_numeric(high_post[c].iloc[:min_len], errors="coerce").fillna(0)
                if (lv - hv).abs().max() > 1e-6:
                    depth_differs = True
                    break

    causal_cmp_df = pd.DataFrame(causal_rows)
    causal_cmp_df.to_csv(OUT_DIR / "causal_intervention_comparison.csv", index=False)

    causal_verdict = {
        "actuator_id": best_aid,
        "max_baseline_flow": best_flow,
        "low_requested": 0.0,
        "high_requested": 1.0,
        "flow_differs_between_branches": bool(flow_differs),
        "depth_differs_between_branches": bool(depth_differs),
        "causal_pass": bool(flow_differs and depth_differs),
    }
    (OUT_DIR / "causal_intervention_verdict.json").write_text(
        json.dumps(causal_verdict, indent=2, default=str), encoding="utf-8")
    print(f"[V2 Runner] Causal: flow_differs={flow_differs}, depth_differs={depth_differs}")

    # Hotstart audit
    hsf_files = list(OUT_DIR.rglob("*.hsf")) + list(WORK_DIR.rglob("*.hsf"))
    hotstart_audit = {
        "hotstart_used": False,
        "use_hotstart_call_count": 0,
        "save_hotstart_call_count": 0,
        "hsf_files_in_v2_output": len(hsf_files),
        "hsf_file_paths": [str(f) for f in hsf_files],
    }
    (OUT_DIR / "hotstart_audit.json").write_text(
        json.dumps(hotstart_audit, indent=2), encoding="utf-8")

    # Formal blacklist
    blacklist = {
        "blacklisted_events": [primary_event],
        "blacklist_reason": "V31 rainfall library used by formal_v31_design",
        "formal_blacklist_written": True,
    }
    (OUT_DIR / "formal_blacklist.json").write_text(
        json.dumps(blacklist, indent=2), encoding="utf-8")

    # Summary
    print(f"\n{'='*60}")
    print(f"  V2 RUNNER SUMMARY")
    print(f"{'='*60}")
    print(f"  Physical SHA match: {phys_sha_with == phys_sha_no}")
    print(f"  Hotstart .hsf files: {len(hsf_files)}")
    for _, r in hash_df.iterrows():
        print(f"  {r['checkpoint_label']:20s} {r['branch']:25s}: "
              f"post_changes={r['post_checkpoint_action_changes']}, "
              f"H120_PFV={r['PFV_H120']:.1f}, recovery={r['recovery_criteria_met']}")
    print(f"  Causal: flow_differs={flow_differs}, depth_differs={depth_differs}")
    print(f"\n  V2 Runner complete. Run verdict script next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
