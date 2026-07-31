"""Gate 2.5-real-v3 Runner: End-to-end execution with shared prefix + blocking recovery.

Key V3 changes vs V2:
  1. Prefix schedule covers ALL 90 links (36 Eng36 + 82 native control links - 28 overlap)
  2. Baseline records setting: for 54 non-Eng36 native control links
  3. All 4 branches (including dynamic_internal) write all 90 links during prefix
  4. Recovery is BLOCKING (not informational)
  5. Extended simulation: duration + 720min tail = 1020min total

All outputs in: outputs/project6_dual_reference_v4/recovery_validation/gate2p5_real_v3/
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
    _load_full_prefix_table,
)
from sewerrtc.simulation.kpi_metrics import compute_kpis
from sewerrtc.simulation.runtime_contracts import analyze_recovery
from sewerrtc.data.round0_prompt2 import _load_round0_actuators, _priority_nodes

INP_PATH = PROJECT_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"
RAIN_TABLE = PROJECT_ROOT / "outputs" / "rainfall_library_v8_storage_variablepump" / "rainfall_event_table.csv"
V1_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real"
OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real_v3"
WORK_DIR = OUT_DIR / "work"
SCOPE_CONTRACT_V2 = PROJECT_ROOT / "docs" / "contracts" / "PROJECT6_V4_CONTROL_SCOPE_CONTRACT_V2.json"


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
    """Compute full state hash at the LAST PREFIX row before checkpoint."""
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

    # Load Scope Contract V2
    scope = json.loads(SCOPE_CONTRACT_V2.read_text(encoding="utf-8"))
    eng36_ids = scope["engineering36_ids"]
    native_control_links = scope["native_control_links"]
    overlap_28 = scope["engineering36_overlap_with_native_rules"]

    # Compute extra prefix links: native control links NOT in Eng36
    eng36_set = set(eng36_ids)
    extra_prefix_ids = [lid for lid in native_control_links if lid not in eng36_set]
    total_prefix = len(set(eng36_ids) | set(native_control_links))

    print(f"[V3 Runner] Scope Contract V2:")
    print(f"  Eng36 actuators: {len(eng36_ids)}")
    print(f"  Native control links: {len(native_control_links)}")
    print(f"  Overlap (Eng36 ∩ native): {len(overlap_28)}")
    print(f"  Extra prefix links (native - Eng36): {len(extra_prefix_ids)}")
    print(f"  Total prefix links: {total_prefix}")

    # Load V1 selection (reuse event + checkpoints)
    v1_selection = json.loads((V1_DIR / "positive_control_selection.json").read_text(encoding="utf-8"))
    v1_catalog = pd.read_csv(V1_DIR / "checkpoint_catalog.csv")
    primary_event = v1_selection["primary_event"]

    # Load rainfall info
    rain_table = pd.read_csv(RAIN_TABLE)
    ev_row = rain_table[rain_table["event_id"].astype(str) == primary_event].iloc[0]
    duration_min = int(ev_row["duration_min"])
    rainfall_csv = str(ev_row["rainfall_csv"])

    # V3: Extended simulation = duration + 720min tail (for recovery blocking)
    # recovery contract: minimum_tail=180, max_tail=720
    # Add 5min buffer to ensure full 1020min coverage (pyswmm reporting step alignment)
    sim_duration_min = duration_min + 720 + 5  # 300 + 720 + 5 = 1025 min
    recession_min = sim_duration_min - duration_min  # 725 min

    actuators = _load_round0_actuators()
    priority = _priority_nodes()
    all_actuator_ids = actuators["actuator_id"].astype(str).tolist()
    n_act = len(all_actuator_ids)

    print(f"\n[V3 Runner] Event: {primary_event}")
    print(f"  Duration: {duration_min}min, Sim: {sim_duration_min}min (duration + 725 tail, 5min buffer)")
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

    # ---- Step 2: Run baseline WITH extra recording links ----
    baseline_detail = WORK_DIR / f"{primary_event}__baseline_detail.csv"
    if not baseline_detail.exists():
        print(f"\n  Running baseline (with controls, internal_rules + {len(extra_prefix_ids)} extra recording links)...")
        t0 = time.time()
        baseline_kpis = run_swmm_trajectory(
            inp_path=inp_with, policy_id="internal_rules",
            actuators=actuators, priority_nodes=priority,
            out_detail_csv=baseline_detail, event_id=primary_event,
            duration_min=duration_min, control_step_sec=300, seed=2026,
            simulation_duration_min=sim_duration_min,
            recession_min=recession_min,
            extra_recording_link_ids=extra_prefix_ids,
        )
        print(f"  Baseline done in {time.time()-t0:.1f}s")
    else:
        print(f"\n  Baseline already exists, loading...")

    baseline_df = pd.read_csv(baseline_detail)
    baseline_df.to_csv(OUT_DIR / "baseline_detail.csv", index=False)
    print(f"  Baseline: {len(baseline_df)} rows, elapsed={baseline_df['elapsed_min'].max():.0f}min")

    # Verify extra recording columns exist
    extra_found = [lid for lid in extra_prefix_ids if f"setting:{lid}" in baseline_df.columns]
    print(f"  Extra recording links found in baseline: {len(extra_found)}/{len(extra_prefix_ids)}")

    # HSF cleanup
    _hsf_cleanup = list(OUT_DIR.rglob("*.hsf")) + list(WORK_DIR.rglob("*.hsf"))
    for _hsf in _hsf_cleanup:
        _hsf.unlink(missing_ok=True)
    if _hsf_cleanup:
        print(f"  Cleaned up {len(_hsf_cleanup)} .hsf files")

    # ---- Step 3: Determine checkpoints ----
    checkpoints = {}
    for _, cp_row in v1_catalog.iterrows():
        cp_label = str(cp_row["checkpoint_label"])
        cp_min = float(cp_row["checkpoint_elapsed_min"])
        checkpoints[cp_label] = cp_min
    print(f"\n  Checkpoints: {checkpoints}")

    # ---- Step 4: Build FULL prefix schedule (Eng36 + extra native links) ----
    prefix_table_eng36 = _load_nominal_action_table(baseline_detail, all_actuator_ids)
    _, extra_prefix_table, _, extra_found = _load_full_prefix_table(
        baseline_detail, all_actuator_ids, extra_prefix_ids)
    print(f"  Prefix table: Eng36={len(prefix_table_eng36)} timesteps, "
          f"Extra={len(extra_prefix_table)} timesteps, "
          f"Extra links found={len(extra_found)}")

    # ---- Step 5: Run 4 branches at each checkpoint ----
    hash_rows = []
    kpi_rows = []
    recovery_rows_list = []
    snapshot_evidence = []
    prefix_evidence = []

    # All native control links for snapshot (82 total)
    all_prefix_link_ids = sorted(set(eng36_ids) | set(native_control_links))

    for cp_label, cp_min in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"  Checkpoint {cp_label} at t={cp_min:.0f}min")
        print(f"{'='*60}")

        # Get snapshot settings at checkpoint for ALL prefix links
        cp_row_idx = (baseline_df["elapsed_min"] - cp_min).abs().idxmin()
        snapshot_settings = {}
        for lid in all_prefix_link_ids:
            # Try a: first, then setting:
            for col_prefix in ["a:", "setting:"]:
                col = f"{col_prefix}{lid}"
                if col in baseline_df.columns:
                    snapshot_settings[lid] = float(baseline_df.loc[cp_row_idx, col])
                    break
            else:
                snapshot_settings[lid] = 1.0

        # Previous settings (checkpoint - 5 min)
        prev_min = cp_min - 5.0
        prev_row_idx = (baseline_df["elapsed_min"] - prev_min).abs().idxmin()
        previous_settings = {}
        for lid in all_prefix_link_ids:
            for col_prefix in ["a:", "setting:"]:
                col = f"{col_prefix}{lid}"
                if col in baseline_df.columns:
                    previous_settings[lid] = float(baseline_df.loc[prev_row_idx, col])
                    break
            else:
                previous_settings[lid] = 1.0

        # ---- Branch 1: dynamic_internal_rules ----
        print(f"  [1/4] Running dynamic_internal_rules (with extra prefix)...")
        di_detail = WORK_DIR / f"v3_cp{cp_label}__dynamic_internal_detail.csv"
        t0 = time.time()
        di_kpis = run_swmm_dynamic_internal(
            inp_path=inp_with, actuators=actuators, priority_nodes=priority,
            internal_baseline_detail_csv=baseline_detail,
            out_detail_csv=di_detail, event_id=primary_event,
            duration_min=duration_min, override_start_min=cp_min,
            control_step_sec=300,
            extra_prefix_link_ids=extra_found,
            extra_prefix_table=extra_prefix_table,
            prefix_inp_path=inp_no,  # V3: use no-controls INP for prefix (shared prefix)
            hotstart_dir=WORK_DIR,
        )
        print(f"    done in {time.time()-t0:.1f}s")
        di_df = pd.read_csv(di_detail)
        di_df.to_csv(OUT_DIR / f"branch_dynamic_internal_cp{cp_label}_detail.csv", index=False)

        # ---- Branch 2: no_control ----
        print(f"  [2/4] Running no_control (with extra prefix)...")
        nc_detail = WORK_DIR / f"v3_cp{cp_label}__no_control_detail.csv"
        no_ctrl_action = np.ones(n_act, dtype=np.float64)
        t0 = time.time()
        nc_kpis = run_swmm_fixed_action(
            inp_path=inp_no, actuators=actuators, priority_nodes=priority,
            out_detail_csv=nc_detail, event_id=primary_event,
            duration_min=duration_min,
            prefix_schedule=prefix_table_eng36,
            override_start_min=cp_min,
            post_action=no_ctrl_action,
            control_step_sec=300,
            policy_id="no_control",
            simulation_duration_min=sim_duration_min,
            extra_prefix_link_ids=extra_found,
            extra_prefix_table=extra_prefix_table,
        )
        print(f"    done in {time.time()-t0:.1f}s")
        nc_df = pd.read_csv(nc_detail)
        nc_df.to_csv(OUT_DIR / f"branch_no_control_cp{cp_label}_detail.csv", index=False)

        # ---- Branch 3: hold_internal_snapshot ----
        print(f"  [3/4] Running hold_internal_snapshot (with extra prefix)...")
        his_detail = WORK_DIR / f"v3_cp{cp_label}__hold_snapshot_detail.csv"
        # Build snapshot action vector for Eng36 actuators only
        snap_vec = np.ones(n_act, dtype=np.float64)
        for aid in eng36_ids:
            if aid in all_actuator_ids and aid in snapshot_settings:
                snap_vec[all_actuator_ids.index(aid)] = snapshot_settings[aid]
        t0 = time.time()
        his_kpis = run_swmm_fixed_action(
            inp_path=inp_no, actuators=actuators, priority_nodes=priority,
            out_detail_csv=his_detail, event_id=primary_event,
            duration_min=duration_min,
            prefix_schedule=prefix_table_eng36,
            override_start_min=cp_min,
            post_action=snap_vec,
            control_step_sec=300,
            policy_id="hold_internal_snapshot",
            simulation_duration_min=sim_duration_min,
            extra_prefix_link_ids=extra_found,
            extra_prefix_table=extra_prefix_table,
        )
        print(f"    done in {time.time()-t0:.1f}s")
        his_df = pd.read_csv(his_detail)
        his_df.to_csv(OUT_DIR / f"branch_hold_snapshot_cp{cp_label}_detail.csv", index=False)

        # ---- Branch 4: hold_previous ----
        print(f"  [4/4] Running hold_previous (with extra prefix)...")
        hp_detail = WORK_DIR / f"v3_cp{cp_label}__hold_previous_detail.csv"
        prev_vec = np.ones(n_act, dtype=np.float64)
        for aid in eng36_ids:
            if aid in all_actuator_ids and aid in previous_settings:
                prev_vec[all_actuator_ids.index(aid)] = previous_settings[aid]
        t0 = time.time()
        hp_kpis = run_swmm_fixed_action(
            inp_path=inp_no, actuators=actuators, priority_nodes=priority,
            out_detail_csv=hp_detail, event_id=primary_event,
            duration_min=duration_min,
            prefix_schedule=prefix_table_eng36,
            override_start_min=cp_min,
            post_action=prev_vec,
            control_step_sec=300,
            policy_id="hold_previous",
            simulation_duration_min=sim_duration_min,
            extra_prefix_link_ids=extra_found,
            extra_prefix_table=extra_prefix_table,
        )
        print(f"    done in {time.time()-t0:.1f}s")
        hp_df = pd.read_csv(hp_detail)
        hp_df.to_csv(OUT_DIR / f"branch_hold_previous_cp{cp_label}_detail.csv", index=False)

        # ---- Verify shared prefix (ALL 4 branches including dynamic_internal) ----
        print(f"\n  Verifying shared prefix (all 4 branches)...")
        for branch_name, branch_df in [("dynamic_internal", di_df), ("no_control", nc_df),
                                        ("hold_snapshot", his_df), ("hold_previous", hp_df)]:
            state = _checkpoint_state_hash(branch_df, cp_min)
            post_changes = _count_post_changes(branch_df, cp_min)

            # H120 from checkpoint
            h120 = _compute_h120_from_checkpoint(branch_df, cp_min, priority)
            audit = _audit_h120(branch_df, cp_min, priority)

            # Recovery (V3: blocking, max_tail=720)
            rec = analyze_recovery(
                branch_df, event_id=primary_event, policy_id=branch_name,
                trajectory_id=f"v3_{branch_name}_cp{cp_label}",
                duration_min=duration_min,
                minimum_tail_min=180, priority_nodes=priority,
            )

            # Compute prefix SHA including extra link settings
            prefix_a_cols = [c for c in branch_df.columns if c.startswith("a:")]
            prefix_setting_extra_cols = [f"setting:{lid}" for lid in extra_found
                                         if f"setting:{lid}" in branch_df.columns]
            prefix_sha = _df_sha256(
                branch_df[branch_df["elapsed_min"] <= cp_min],
                prefix_a_cols + prefix_setting_extra_cols
            )

            hash_rows.append({
                "checkpoint_label": cp_label,
                "checkpoint_elapsed_min": cp_min,
                "branch": branch_name,
                "physical_network_sha256": phys_sha_with if branch_name == "dynamic_internal" else phys_sha_no,
                "rainfall_sha256": _file_sha256(Path(rainfall_csv)),
                "inp_file_sha256": _file_sha256(inp_with if branch_name == "dynamic_internal" else inp_no),
                "prefix_actual_schedule_sha256": prefix_sha,
                "checkpoint_state_hash": json.dumps(state),
                "post_checkpoint_action_changes": post_changes,
                "hotstart_used": False,
                "extra_prefix_links_written": len(extra_found),
                **h120,
                **audit,
                "h120_match": abs(h120["PFV_H120"] - audit["audit_PFV"]) < 1.0,
                "recovery_criteria_met": bool(rec.get("recovery_criteria_met", False)),
                "recovery_censored": bool(rec.get("recovery_censored", False)),
                "full_event_eligible": bool(rec.get("recovery_criteria_met", False) and
                                            not rec.get("recovery_censored", True)),
            })

            kpi_rows.append({
                "checkpoint_label": cp_label,
                "branch": branch_name,
                **h120,
                **audit,
                "recovery_criteria_met": bool(rec.get("recovery_criteria_met", False)),
                "recovery_time_min": rec.get("recovery_time_min"),
                "recovery_censored": bool(rec.get("recovery_censored", False)),
                "full_event_eligible": bool(rec.get("recovery_criteria_met", False) and
                                            not rec.get("recovery_censored", True)),
                "last_flood_time_min": rec.get("last_flood_time_min"),
                "actual_tail_min": rec.get("actual_tail_min"),
            })

        # Prefix equality evidence: compare last-prefix-row state across all 4 branches
        for branch_name, branch_df in [("dynamic_internal", di_df), ("no_control", nc_df),
                                        ("hold_snapshot", his_df), ("hold_previous", hp_df)]:
            prefix_rows_df = branch_df[branch_df["elapsed_min"] < cp_min]
            if not prefix_rows_df.empty:
                last_row = prefix_rows_df.iloc[-1]
                prefix_evidence.append({
                    "checkpoint_label": cp_label,
                    "branch": branch_name,
                    "last_prefix_elapsed": float(last_row["elapsed_min"]),
                    "h_sha256": _df_sha256(prefix_rows_df.tail(1),
                                           [c for c in branch_df.columns if c.startswith("h:")]),
                    "flow_sha256": _df_sha256(prefix_rows_df.tail(1),
                                              [c for c in branch_df.columns if c.startswith("flow:")]),
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

    # Auto-select actuator from Eng36 overlap with native rules
    cp_label_0 = list(checkpoints.keys())[0]
    cp_min_0 = checkpoints[cp_label_0]
    baseline_post = baseline_df[baseline_df["elapsed_min"] > cp_min_0]
    best_aid = None
    best_flow = 0.0
    for aid in overlap_28:
        fcol = f"flow:{aid}"
        if fcol in baseline_post.columns:
            max_flow = pd.to_numeric(baseline_post[fcol], errors="coerce").fillna(0).abs().max()
            if max_flow > best_flow:
                best_flow = float(max_flow)
                best_aid = aid

    if best_aid is None:
        best_aid = overlap_28[0] if overlap_28 else eng36_ids[0]
    print(f"  Selected actuator: {best_aid} (max_flow={best_flow:.3f} m3/s)")

    causal_results = {}
    for setting_val, tag in [(0.0, "low"), (1.0, "high")]:
        causal_action = np.ones(n_act, dtype=np.float64)
        if best_aid in all_actuator_ids:
            causal_action[all_actuator_ids.index(best_aid)] = setting_val

        causal_detail = WORK_DIR / f"v3_causal_{tag}_detail.csv"
        t0 = time.time()
        ck = run_swmm_fixed_action(
            inp_path=inp_no, actuators=actuators, priority_nodes=priority,
            out_detail_csv=causal_detail, event_id=primary_event,
            duration_min=duration_min,
            prefix_schedule=prefix_table_eng36,
            override_start_min=cp_min_0,
            post_action=causal_action,
            control_step_sec=300,
            policy_id=f"causal_{tag}",
            simulation_duration_min=sim_duration_min,
            extra_prefix_link_ids=extra_found,
            extra_prefix_table=extra_prefix_table,
        )
        print(f"  Causal {tag} done in {time.time()-t0:.1f}s")
        causal_results[tag] = pd.read_csv(causal_detail)
        causal_results[tag].to_csv(OUT_DIR / f"causal_{tag}_detail.csv", index=False)

    # ---- Write outputs ----
    hash_df = pd.DataFrame(hash_rows)
    hash_df.to_csv(OUT_DIR / "state_hash_comparison.csv", index=False)
    print(f"\n[V3 Runner] Wrote state_hash_comparison.csv")

    kpi_df = pd.DataFrame(kpi_rows)
    kpi_df.to_csv(OUT_DIR / "branch_kpi_comparison.csv", index=False)
    print(f"[V3 Runner] Wrote branch_kpi_comparison.csv")

    snap_df = pd.DataFrame(snapshot_evidence)
    snap_df.to_csv(OUT_DIR / "snapshot_evidence.csv", index=False)
    print(f"[V3 Runner] Wrote snapshot_evidence.csv")

    prefix_evidence_df = pd.DataFrame(prefix_evidence)
    prefix_evidence_df.to_csv(OUT_DIR / "prefix_trajectory_comparison.csv", index=False)

    # Prefix equality JSON
    prefix_eq = {"checkpoints": {}}
    for cp_label in checkpoints:
        cp_rows = prefix_evidence_df[prefix_evidence_df["checkpoint_label"] == cp_label]
        all_match = True
        if len(cp_rows) > 1:
            for col in ["h_sha256", "flow_sha256"]:
                vals = cp_rows[col].unique()
                if len(vals) > 1:
                    all_match = False
                    break
        prefix_eq["checkpoints"][cp_label] = {
            "all_branches_match": all_match,
            "branch_count": len(cp_rows),
        }
    (OUT_DIR / "prefix_equality_evidence.json").write_text(
        json.dumps(prefix_eq, indent=2), encoding="utf-8")

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
    print(f"[V3 Runner] Causal: flow_differs={flow_differs}, depth_differs={depth_differs}")

    # Hotstart audit
    hsf_files = list(OUT_DIR.rglob("*.hsf")) + list(WORK_DIR.rglob("*.hsf"))
    hotstart_audit = {
        "hotstart_used": False,
        "hsf_files_in_v3_output": len(hsf_files),
        "hsf_file_paths": [str(f) for f in hsf_files],
    }
    (OUT_DIR / "hotstart_audit.json").write_text(
        json.dumps(hotstart_audit, indent=2), encoding="utf-8")

    # Runner summary JSON
    summary = {
        "gate": "2.5-real-v3",
        "event_id": primary_event,
        "duration_min": duration_min,
        "sim_duration_min": sim_duration_min,
        "eng36_count": len(eng36_ids),
        "native_control_links_count": len(native_control_links),
        "extra_prefix_links_count": len(extra_found),
        "total_prefix_links": total_prefix,
        "physical_sha_match": phys_sha_with == phys_sha_no,
        "hotstart_files": len(hsf_files),
        "causal_pass": bool(flow_differs and depth_differs),
        "checkpoints": list(checkpoints.keys()),
    }
    (OUT_DIR / "v3_runner_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    # Summary
    print(f"\n{'='*60}")
    print(f"  V3 RUNNER SUMMARY")
    print(f"{'='*60}")
    print(f"  Physical SHA match: {phys_sha_with == phys_sha_no}")
    print(f"  Hotstart .hsf files: {len(hsf_files)}")
    print(f"  Total prefix links: {total_prefix} (Eng36={len(eng36_ids)}, native={len(native_control_links)}, extra={len(extra_found)})")
    for _, r in hash_df.iterrows():
        print(f"  {r['checkpoint_label']:20s} {r['branch']:25s}: "
              f"post_changes={r['post_checkpoint_action_changes']}, "
              f"H120_PFV={r['PFV_H120']:.1f}, recovery={r['recovery_criteria_met']}")
    print(f"  Causal: flow_differs={flow_differs}, depth_differs={depth_differs}")
    print(f"\n  V3 Runner complete. Run verdict script next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
