"""Gate 5 Phase 2: Exact-SWMM Ablation of uniform_90pct (16-way parallel).

Performs:
  1. leave-one-actuator-out (restore to mid-point)
  2. leave-one-facility-group-out
  3. single-facility perturbation (top-3 from LOO)

All SWMM runs are executed in parallel (up to 16 concurrent).

Output:
  - facility_marginal_effects.csv
  - facility_group_marginal_effects.csv
  - facility_perturbation_effects.csv
  - uniform90_ablation_audit.json
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.simulation.pyswmm_runner import run_swmm_fixed_action
from sewerrtc.simulation.kpi_metrics import compute_window_kpis
from sewerrtc.data.round0_prompt2 import _priority_nodes

INP_PATH = PROJECT_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"
RAIN_TABLE = PROJECT_ROOT / "outputs" / "rainfall_library_v8_storage_variablepump" / "rainfall_event_table.csv"
ACTUATOR_CSV = PROJECT_ROOT / "data" / "project6_v3_facility_semantics_36.csv"

EVENT_ID = "V31_RP10_D2H_P65_v31_independent_gamma_084"
CHECKPOINT_MIN = 60.0
SPINUP_MIN = 4320
ADJ_CHECKPOINT = CHECKPOINT_MIN + SPINUP_MIN
H120_MIN = 120
CONTROL_STEP_SEC = 300
UNIFORM_90_VAL = 0.90
MAX_WORKERS = 16

OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_capability_v2" / "gate4_h120_batch0"
ABLATION_DIR = OUT_DIR / "ablation_uniform90"
PARALLEL_DIR = ABLATION_DIR / "parallel_runs"

BINARY_PUMPS = ["ADD301.2", "ADD301.3"]


def load_actuators():
    df = pd.read_csv(ACTUATOR_CSV)
    if "actuator_id" not in df.columns and "facility_id" in df.columns:
        df["actuator_id"] = df["facility_id"]
    if "link_type" not in df.columns and "actuator_type" in df.columns:
        df["link_type"] = df["actuator_type"]
    return df


def classify_facilities(actuators):
    groups = {
        "orifices": [], "weirs": [], "pumps_binary": [], "pumps_continuous": [],
        "storage_inlet": [], "storage_outlet": [], "downstream_regulator": [],
    }
    for _, row in actuators.iterrows():
        fid = row["facility_id"]
        atype = row.get("actuator_type", "")
        role = row.get("storage_role", "none")
        if role == "storage_inlet":
            groups["storage_inlet"].append(fid)
        elif role == "storage_outlet":
            groups["storage_outlet"].append(fid)
        elif role == "downstream_regulator":
            groups["downstream_regulator"].append(fid)
        elif atype == "pump":
            groups["pumps_binary" if fid in BINARY_PUMPS else "pumps_continuous"].append(fid)
        elif atype == "orifice":
            groups["orifices"].append(fid)
        elif atype == "weir":
            groups["weirs"].append(fid)
    return groups


def compute_h120_labels(detail_csv, checkpoint_min, h120_min, priority_nodes):
    df = pd.read_csv(detail_csv)
    result = compute_window_kpis(
        df, priority_nodes, checkpoint_min, h120_min, dt_sec=CONTROL_STEP_SEC
    )
    return {
        "pfv_m3": float(result["PFV"]),
        "tfv_m3": float(result["TFV"]),
        "peak_tfv_rate_m3s": float(result["peak_TFV_rate"]),
        "h120_steps": int(result["steps"]),
    }


def create_inp_with_spinup(base_inp, rain_csv, out_inp, sim_dur_min, spinup_min=SPINUP_MIN, strip_controls=False):
    from datetime import datetime, timedelta
    total_dur = spinup_min + sim_dur_min
    mutate_inp_for_event(base_inp, rain_csv, out_inp, total_dur, strip_controls=strip_controls)
    lines = out_inp.read_text(encoding="utf-8", errors="ignore").splitlines()
    result = []
    spinup_td = timedelta(minutes=spinup_min)
    ts_name = "RTC_RAIN_TS"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(ts_name) and len(stripped) > len(ts_name):
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    ts_date = datetime.strptime(parts[1], "%m/%d/%Y")
                    ts_time = datetime.strptime(parts[2], "%H:%M")
                    orig_dt = ts_date.replace(hour=ts_time.hour, minute=ts_time.minute)
                    new_dt = orig_dt + spinup_td
                    new_line = f"{ts_name:<16} {new_dt:%m/%d/%Y} {new_dt:%H:%M} {parts[3]}"
                    result.append(new_line)
                    continue
                except (ValueError, IndexError):
                    pass
        result.append(line)
    out_inp.write_text("\n".join(result) + "\n", encoding="utf-8")
    return out_inp


def _run_single_ablation(args):
    """Worker function for parallel execution. Each runs in its own directory."""
    (ablation_id, source_inp, actuators_csv, priority, facility_ids,
     base_action_dict, override_dict, run_dir, adj_checkpoint, h120_min,
     control_step_sec, event_id) = args

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy INP to isolated directory
    local_inp = run_dir / Path(source_inp).name
    shutil.copy2(str(source_inp), str(local_inp))

    # Load actuators in worker process
    import pandas as pd
    import numpy as np
    actuators = pd.read_csv(actuators_csv)
    if "actuator_id" not in actuators.columns:
        actuators["actuator_id"] = actuators["facility_id"]
    if "link_type" not in actuators.columns and "actuator_type" in actuators.columns:
        actuators["link_type"] = actuators["actuator_type"]

    # Build action vector
    action = {fid: base_action_dict.get(fid, 0.5) for fid in facility_ids}
    for fid, val in override_dict.items():
        action[fid] = val
    action_vec = np.array([action[fid] for fid in facility_ids])

    detail_csv = run_dir / f"{ablation_id}_detail.csv"

    try:
        from sewerrtc.simulation.pyswmm_runner import run_swmm_fixed_action
        run_swmm_fixed_action(
            inp_path=str(local_inp),
            actuators=actuators,
            priority_nodes=priority,
            out_detail_csv=str(detail_csv),
            event_id=event_id,
            duration_min=0,
            prefix_schedule={},
            override_start_min=adj_checkpoint,
            post_action=action_vec,
            control_step_sec=control_step_sec,
            simulation_duration_min=None,
            policy_id=f"ablation_{ablation_id}",
        )
        labels = compute_h120_labels(detail_csv, adj_checkpoint, h120_min, priority)
        return {"ablation_id": ablation_id, "labels": labels, "detail": str(detail_csv), "error": None}
    except Exception as e:
        return {"ablation_id": ablation_id, "labels": {}, "detail": str(detail_csv), "error": str(e)}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  Gate 5 Phase 2: uniform_90pct Exact-SWMM Ablation (16-parallel)")
    print("=" * 70)

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    PARALLEL_DIR.mkdir(parents=True, exist_ok=True)

    actuators = load_actuators()
    priority = _priority_nodes()
    facility_ids = actuators["facility_id"].tolist()
    n_fac = len(facility_ids)
    groups = classify_facilities(actuators)

    rain_table = pd.read_csv(RAIN_TABLE)
    row = rain_table[rain_table["event_id"] == EVENT_ID]
    if row.empty:
        print(f"ERROR: {EVENT_ID} not found")
        return 1
    ev = row.iloc[0]
    rain_csv = str(ev["rainfall_csv"])
    sim_dur = int(ev["simulation_duration_min"])

    # Prepare master INP
    master_inp = ABLATION_DIR / f"ablation_{EVENT_ID}__with_ctrl.inp"
    if not master_inp.exists():
        create_inp_with_spinup(INP_PATH, rain_csv, master_inp, sim_dur, strip_controls=False)
        print("  Created master INP")

    # Base action: uniform 0.90
    base_action = {fid: UNIFORM_90_VAL for fid in facility_ids}
    for bp in BINARY_PUMPS:
        base_action[bp] = 1.0

    # Load reference labels from Batch 0
    batch_work = OUT_DIR / "work"
    nc_csv = batch_work / f"batch0_{EVENT_ID}__no_control_detail.csv"
    di_csv = batch_work / f"batch0_{EVENT_ID}__dynamic_internal_rules_detail.csv"
    nc_labels = compute_h120_labels(nc_csv, ADJ_CHECKPOINT, H120_MIN, priority) if nc_csv.exists() else {}
    di_labels = compute_h120_labels(di_csv, ADJ_CHECKPOINT, H120_MIN, priority) if di_csv.exists() else {}
    u90_csv = batch_work / f"batch0_{EVENT_ID}__cand_uniform_90pct_detail.csv"
    u90_labels = compute_h120_labels(u90_csv, ADJ_CHECKPOINT, H120_MIN, priority) if u90_csv.exists() else {}

    print(f"  Facilities: {n_fac}, Priority: {len(priority)}")
    print(f"  NC: PFV={nc_labels.get('pfv_m3',0):.2f}, TFV={nc_labels.get('tfv_m3',0):.2f}")
    print(f"  DI: PFV={di_labels.get('pfv_m3',0):.2f}, TFV={di_labels.get('tfv_m3',0):.2f}")
    print(f"  U90: PFV={u90_labels.get('pfv_m3',0):.2f}, TFV={u90_labels.get('tfv_m3',0):.2f}")

    # ── Build all tasks ──
    all_tasks = []
    task_meta = {}  # ablation_id -> {type, fid/group, ...}

    # 1. LOO tasks
    for fid in facility_ids:
        ablation_id = f"loo_{fid}"
        override = {fid: 0.0 if fid in BINARY_PUMPS else 0.5}
        task_args = (ablation_id, str(master_inp), str(ACTUATOR_CSV), priority,
                     facility_ids, base_action, override,
                     str(PARALLEL_DIR / ablation_id), ADJ_CHECKPOINT, H120_MIN,
                     CONTROL_STEP_SEC, EVENT_ID)
        all_tasks.append(task_args)
        task_meta[ablation_id] = {"type": "loo", "facility_id": fid}

    # 2. Group tasks
    for group_name, group_fids in groups.items():
        if not group_fids:
            continue
        ablation_id = f"restore_group_{group_name}"
        override = {fid: 0.5 for fid in group_fids}
        for bp in BINARY_PUMPS:
            if bp in override:
                override[bp] = 0.0
        task_args = (ablation_id, str(master_inp), str(ACTUATOR_CSV), priority,
                     facility_ids, base_action, override,
                     str(PARALLEL_DIR / ablation_id), ADJ_CHECKPOINT, H120_MIN,
                     CONTROL_STEP_SEC, EVENT_ID)
        all_tasks.append(task_args)
        task_meta[ablation_id] = {"type": "group", "group": group_name, "n_facilities": len(group_fids)}

    print(f"\n  Total tasks: {len(all_tasks)} ({len(facility_ids)} LOO + {sum(1 for g in groups.values() if g)} group)")
    print(f"  Max parallel: {MAX_WORKERS}")

    # ── Execute all tasks in parallel ──
    results_map = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {executor.submit(_run_single_ablation, task): task[0] for task in all_tasks}
        for future in as_completed(future_to_id):
            ablation_id = future_to_id[future]
            try:
                result = future.result()
            except Exception as e:
                result = {"ablation_id": ablation_id, "labels": {}, "error": str(e)}
            results_map[ablation_id] = result
            completed += 1
            labels = result.get("labels", {})
            pfv = labels.get("pfv_m3", float("nan"))
            tfv = labels.get("tfv_m3", float("nan"))
            err = result.get("error", "")
            status = "OK" if err is None else f"ERR"
            print(f"    [{completed}/{len(all_tasks)}] {ablation_id[:45]:45s} PFV={pfv:.2f} TFV={tfv:.2f} [{status}]")

    # ── Process LOO results ──
    print(f"\n  ── Processing LOO Results ──")
    loo_rows = []
    for fid in facility_ids:
        ablation_id = f"loo_{fid}"
        r = results_map.get(ablation_id, {"labels": {}, "error": "missing"})
        labels = r.get("labels", {})
        pfv = labels.get("pfv_m3", float("nan"))
        tfv = labels.get("tfv_m3", float("nan"))
        peak = labels.get("peak_tfv_rate_m3s", float("nan"))
        nc_pfv = nc_labels.get("pfv_m3", float("nan"))
        di_tfv = di_labels.get("tfv_m3", float("nan"))
        di_peak = di_labels.get("peak_tfv_rate_m3s", float("nan"))
        delta_pfv = pfv - nc_pfv if not (np.isnan(pfv) or np.isnan(nc_pfv)) else float("nan")
        delta_tfv = tfv - di_tfv if not (np.isnan(tfv) or np.isnan(di_tfv)) else float("nan")
        delta_peak = peak - di_peak if not (np.isnan(peak) or np.isnan(di_peak)) else float("nan")
        loo_rows.append({
            "facility_id": fid,
            "ablation": f"restore_{fid}_to_mid",
            "pfv_m3": pfv, "tfv_m3": tfv, "peak_tfv_rate": peak,
            "delta_pfv_vs_nc": round(delta_pfv, 4) if not np.isnan(delta_pfv) else float("nan"),
            "delta_tfv_vs_di": round(delta_tfv, 4) if not np.isnan(delta_tfv) else float("nan"),
            "delta_peak_vs_di": round(delta_peak, 6) if not np.isnan(delta_peak) else float("nan"),
            "error": r.get("error"),
        })
    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(ABLATION_DIR / "facility_marginal_effects.csv", index=False)

    # ── Process group results ──
    log_rows = []
    for group_name, group_fids in groups.items():
        if not group_fids:
            continue
        ablation_id = f"restore_group_{group_name}"
        r = results_map.get(ablation_id, {"labels": {}, "error": "missing"})
        labels = r.get("labels", {})
        pfv = labels.get("pfv_m3", float("nan"))
        tfv = labels.get("tfv_m3", float("nan"))
        peak = labels.get("peak_tfv_rate_m3s", float("nan"))
        nc_pfv = nc_labels.get("pfv_m3", float("nan"))
        di_tfv = di_labels.get("tfv_m3", float("nan"))
        di_peak = di_labels.get("peak_tfv_rate_m3s", float("nan"))
        delta_pfv = pfv - nc_pfv if not (np.isnan(pfv) or np.isnan(nc_pfv)) else float("nan")
        delta_tfv = tfv - di_tfv if not (np.isnan(tfv) or np.isnan(di_tfv)) else float("nan")
        delta_peak = peak - di_peak if not (np.isnan(peak) or np.isnan(di_peak)) else float("nan")
        log_rows.append({
            "group": group_name, "n_facilities": len(group_fids),
            "pfv_m3": pfv, "tfv_m3": tfv, "peak_tfv_rate": peak,
            "delta_pfv_vs_nc": round(delta_pfv, 4) if not np.isnan(delta_pfv) else float("nan"),
            "delta_tfv_vs_di": round(delta_tfv, 4) if not np.isnan(delta_tfv) else float("nan"),
            "delta_peak_vs_di": round(delta_peak, 6) if not np.isnan(delta_peak) else float("nan"),
            "error": r.get("error"),
        })
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(ABLATION_DIR / "facility_group_marginal_effects.csv", index=False)

    # ── Perturbation (needs LOO results to pick top-3) ──
    print(f"\n  ── Key Facility Perturbation ──")
    loo_valid = loo_df.dropna(subset=["delta_pfv_vs_nc"]).copy()
    if not loo_valid.empty:
        loo_valid["abs_delta_pfv"] = loo_valid["delta_pfv_vs_nc"].abs()
        top3 = loo_valid.nlargest(3, "abs_delta_pfv")["facility_id"].tolist()
    else:
        top3 = facility_ids[:3]

    pert_tasks = []
    pert_meta = {}
    for fid in top3:
        if fid in BINARY_PUMPS:
            continue
        for delta in [-0.10, -0.05, +0.05, +0.10]:
            new_val = max(0.0, min(1.0, UNIFORM_90_VAL + delta))
            ablation_id = f"perturb_{fid}_{delta:+.2f}"
            override = {fid: new_val}
            task_args = (ablation_id, str(master_inp), str(ACTUATOR_CSV), priority,
                         facility_ids, base_action, override,
                         str(PARALLEL_DIR / ablation_id), ADJ_CHECKPOINT, H120_MIN,
                         CONTROL_STEP_SEC, EVENT_ID)
            pert_tasks.append(task_args)
            pert_meta[ablation_id] = {"facility_id": fid, "perturbation": delta, "new_value": new_val}

    print(f"  Running {len(pert_tasks)} perturbation tasks (parallel={MAX_WORKERS})...")
    pert_results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {executor.submit(_run_single_ablation, task): task[0] for task in pert_tasks}
        for future in as_completed(future_to_id):
            aid = future_to_id[future]
            try:
                result = future.result()
            except Exception as e:
                result = {"ablation_id": aid, "labels": {}, "error": str(e)}
            pert_results[aid] = result

    perturbation_rows = []
    for aid, meta in pert_meta.items():
        r = pert_results.get(aid, {"labels": {}, "error": "missing"})
        labels = r.get("labels", {})
        perturbation_rows.append({
            "facility_id": meta["facility_id"],
            "perturbation": meta["perturbation"],
            "new_value": meta["new_value"],
            "pfv_m3": labels.get("pfv_m3", float("nan")),
            "tfv_m3": labels.get("tfv_m3", float("nan")),
            "peak_tfv_rate": labels.get("peak_tfv_rate_m3s", float("nan")),
            "error": r.get("error"),
        })
    pert_df = pd.DataFrame(perturbation_rows)
    pert_df.to_csv(ABLATION_DIR / "facility_perturbation_effects.csv", index=False)

    # ── Audit summary ──
    n_loo = len(loo_df)
    n_loo_ok = int(loo_df["error"].isna().sum()) if "error" in loo_df.columns else 0
    n_group = len(log_df)
    n_group_ok = int(log_df["error"].isna().sum()) if "error" in log_df.columns else 0
    n_pert = len(pert_df)
    n_pert_ok = int(pert_df["error"].isna().sum()) if "error" in pert_df.columns else 0

    u90_pfv = u90_labels.get("pfv_m3", 0)
    u90_tfv = u90_labels.get("tfv_m3", float("inf"))
    pfv_protective = [r["facility_id"] for _, r in loo_df.iterrows()
                      if r.get("error") is None and r.get("pfv_m3", float("inf")) < u90_pfv]
    tfv_improving = [r["facility_id"] for _, r in loo_df.iterrows()
                     if r.get("error") is None and r.get("tfv_m3", float("inf")) < u90_tfv]

    audit = {
        "n_leave_one_out": n_loo, "n_loo_ok": n_loo_ok,
        "n_leave_group_out": n_group, "n_group_ok": n_group_ok,
        "n_perturbations": n_pert, "n_pert_ok": n_pert_ok,
        "top3_pfv_impact_facilities": top3,
        "pfv_protective_facilities": pfv_protective[:10],
        "tfv_improving_facilities": tfv_improving[:10],
        "uniform90_pfv": u90_labels.get("pfv_m3", 0),
        "uniform90_tfv": u90_labels.get("tfv_m3", 0),
        "nc_pfv": nc_labels.get("pfv_m3", 0),
        "di_tfv": di_labels.get("tfv_m3", 0),
        "max_workers": MAX_WORKERS,
        "wall_time_sec": round(time.time() - t0, 1),
    }
    (ABLATION_DIR / "uniform90_ablation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'='*70}")
    print(f"  LOO: {n_loo_ok}/{n_loo} OK")
    print(f"  Group: {n_group_ok}/{n_group} OK")
    print(f"  Perturbation: {n_pert_ok}/{n_pert} OK")
    print(f"  PFV-protective: {len(pfv_protective)}, TFV-improving: {len(tfv_improving)}")
    print(f"  Wall time: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
