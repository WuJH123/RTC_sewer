"""Gate 5 Phase 6: Exact-SWMM Candidate-Space Diagnosis (16-way parallel).

Runs 40-100 precise candidates using V4CandidateGenerator.
Uses same event, checkpoint, prefix, no-hotstart, authoritative SWMM.
All SWMM runs executed in parallel (up to 16 concurrent).

Output:
  - gate5_candidate_results.csv
  - gate5_candidate_audit.json
"""
from __future__ import annotations

import hashlib
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
from sewerrtc.control.v4_candidate_generator import V4CandidateGenerator

INP_PATH = PROJECT_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"
RAIN_TABLE = PROJECT_ROOT / "outputs" / "rainfall_library_v8_storage_variablepump" / "rainfall_event_table.csv"
ACTUATOR_CSV = PROJECT_ROOT / "data" / "project6_v3_facility_semantics_36.csv"

EVENT_ID = "V31_RP10_D2H_P65_v31_independent_gamma_084"
CHECKPOINT_MIN = 60.0
SPINUP_MIN = 4320
ADJ_CHECKPOINT = CHECKPOINT_MIN + SPINUP_MIN
H120_MIN = 120
CONTROL_STEP_SEC = 300
MAX_CANDIDATES = 80
MAX_WORKERS = 16

OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_capability_v2" / "gate4_h120_batch0"
GATE5_DIR = OUT_DIR / "gate5_exact_diagnosis"
PARALLEL_DIR = GATE5_DIR / "parallel_runs"

# PFV/TFV/Peak thresholds
PFV_SAFETY_THRESHOLD = 1.0  # m3
TFV_IMPROVEMENT_THRESHOLD = 0.5  # m3
PEAK_NONINFERIOR_THRESHOLD = 0.001  # m3/s


def load_actuators():
    df = pd.read_csv(ACTUATOR_CSV)
    if "actuator_id" not in df.columns and "facility_id" in df.columns:
        df["actuator_id"] = df["facility_id"]
    if "link_type" not in df.columns and "actuator_type" in df.columns:
        df["link_type"] = df["actuator_type"]
    return df


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


def extract_di_action(detail_csv, checkpoint_min, spinup_min, facility_ids):
    """Extract DI actions at checkpoint."""
    df = pd.read_csv(detail_csv)
    if df.empty:
        return np.full(len(facility_ids), 0.5)
    adj_cp = checkpoint_min + spinup_min
    window = df[(df["elapsed_min"] >= adj_cp) & (df["elapsed_min"] < adj_cp + 5)]
    if window.empty:
        return np.full(len(facility_ids), 0.5)
    actions = []
    for fid in facility_ids:
        col = f"a:{fid}"
        if col in window.columns:
            actions.append(float(window[col].iloc[0]))
        else:
            actions.append(0.5)
    return np.array(actions)


def _run_single_candidate(args):
    """Worker: copy INP to isolated dir, run SWMM, compute labels."""
    (candidate_id, source_inp, actuators_csv, priority, facility_ids,
     action_vec, run_dir, adj_checkpoint, h120_min, control_step_sec,
     event_id, sim_dur_min) = args

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    local_inp = run_dir / Path(source_inp).name
    shutil.copy2(str(source_inp), str(local_inp))

    import pandas as pd
    actuators = pd.read_csv(actuators_csv)
    if "actuator_id" not in actuators.columns:
        actuators["actuator_id"] = actuators["facility_id"]
    if "link_type" not in actuators.columns and "actuator_type" in actuators.columns:
        actuators["link_type"] = actuators["actuator_type"]

    detail_csv = run_dir / f"gate5_{candidate_id}_detail.csv"

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
            policy_id=f"gate5_{candidate_id}",
        )
        labels = compute_h120_labels(str(detail_csv), adj_checkpoint, h120_min, priority)
        return {"candidate_id": candidate_id, "labels": labels, "error": None}
    except Exception as e:
        return {"candidate_id": candidate_id, "labels": {}, "error": str(e)}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  Gate 5: Exact-SWMM Candidate-Space Diagnosis (16-parallel)")
    print("=" * 70)

    GATE5_DIR.mkdir(parents=True, exist_ok=True)
    PARALLEL_DIR.mkdir(parents=True, exist_ok=True)

    actuators = load_actuators()
    priority = _priority_nodes()
    facility_ids = actuators["facility_id"].tolist()
    n_fac = len(facility_ids)
    semantics = pd.read_csv(ACTUATOR_CSV)

    rain_table = pd.read_csv(RAIN_TABLE)
    row = rain_table[rain_table["event_id"] == EVENT_ID]
    if row.empty:
        print(f"ERROR: {EVENT_ID} not found")
        return 1
    ev = row.iloc[0]
    rain_csv = str(ev["rainfall_csv"])
    sim_dur = int(ev["simulation_duration_min"])

    # Prepare master INP
    wc_inp = GATE5_DIR / f"gate5_{EVENT_ID}__with_ctrl.inp"
    if not wc_inp.exists():
        create_inp_with_spinup(INP_PATH, rain_csv, wc_inp, sim_dur, strip_controls=False)
        print("  Created with-control INP")

    # Load references from Batch 0
    batch_work = OUT_DIR / "work"
    nc_csv = batch_work / f"batch0_{EVENT_ID}__no_control_detail.csv"
    di_csv = batch_work / f"batch0_{EVENT_ID}__dynamic_internal_rules_detail.csv"

    nc_labels = compute_h120_labels(str(nc_csv), ADJ_CHECKPOINT, H120_MIN, priority) if nc_csv.exists() else {}
    di_labels = compute_h120_labels(str(di_csv), ADJ_CHECKPOINT, H120_MIN, priority) if di_csv.exists() else {}

    print(f"\n  NC: PFV={nc_labels.get('pfv_m3', 0):.2f}, TFV={nc_labels.get('tfv_m3', 0):.2f}")
    print(f"  DI: PFV={di_labels.get('pfv_m3', 0):.2f}, TFV={di_labels.get('tfv_m3', 0):.2f}")

    # Extract DI action at checkpoint
    di_action = extract_di_action(str(di_csv), CHECKPOINT_MIN, SPINUP_MIN, facility_ids) if di_csv.exists() else np.full(n_fac, 0.5)

    # Generate candidates
    generator = V4CandidateGenerator(
        facility_ids=facility_ids,
        facility_semantics=semantics,
        priority_nodes=priority,
        max_k=8,
    )

    # Load sensitivity ranking from ablation
    sensitivity_map = {}
    ablation_dir = OUT_DIR / "ablation_uniform90"
    if (ablation_dir / "facility_marginal_effects.csv").exists():
        eff = pd.read_csv(ablation_dir / "facility_marginal_effects.csv")
        if not eff.empty and "delta_pfv_vs_nc" in eff.columns:
            eff_valid = eff.dropna(subset=["delta_pfv_vs_nc"])
            if not eff_valid.empty:
                eff_valid["abs_impact"] = eff_valid["delta_pfv_vs_nc"].abs()
                ranked = eff_valid.sort_values("abs_impact", ascending=False)["facility_id"].tolist()
                sensitivity_map["ranking"] = ranked

    candidates = generator.generate_all(di_action, max_total=MAX_CANDIDATES)
    print(f"\n  Generated {len(candidates)} candidates")
    print(f"  Families: {set(c.family for c in candidates)}")

    # Build parallel tasks
    tasks = []
    for cand in candidates:
        task_args = (
            cand.candidate_id, str(wc_inp), str(ACTUATOR_CSV), priority,
            facility_ids, cand.action,
            str(PARALLEL_DIR / cand.candidate_id),
            ADJ_CHECKPOINT, H120_MIN, CONTROL_STEP_SEC, EVENT_ID, sim_dur,
        )
        tasks.append(task_args)

    print(f"\n  Running {len(tasks)} candidates with {MAX_WORKERS} parallel workers...")

    # Execute in parallel
    results_map = {}
    n_done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_single_candidate, t): t[0] for t in tasks}
        for future in as_completed(futures):
            cid = futures[future]
            try:
                result = future.result()
                results_map[cid] = result
            except Exception as e:
                results_map[cid] = {"candidate_id": cid, "labels": {}, "error": str(e)}
            n_done += 1
            if n_done % 10 == 0 or n_done == len(tasks):
                print(f"    Progress: {n_done}/{len(tasks)}")

    # Build result rows
    all_results = []
    cand_map = {c.candidate_id: c for c in candidates}

    for cid, res in results_map.items():
        cand = cand_map[cid]
        labels = res["labels"]
        error = res.get("error")

        pfv = labels.get("pfv_m3", float("nan"))
        tfv = labels.get("tfv_m3", float("nan"))
        peak = labels.get("peak_tfv_rate_m3s", float("nan"))

        nc_pfv = nc_labels.get("pfv_m3", float("nan"))
        di_tfv = di_labels.get("tfv_m3", float("nan"))
        di_peak = di_labels.get("peak_tfv_rate_m3s", float("nan"))

        delta_pfv = pfv - nc_pfv if not (np.isnan(pfv) or np.isnan(nc_pfv)) else float("nan")
        delta_tfv = tfv - di_tfv if not (np.isnan(tfv) or np.isnan(di_tfv)) else float("nan")
        delta_peak = peak - di_peak if not (np.isnan(peak) or np.isnan(di_peak)) else float("nan")

        pfv_safe = bool(delta_pfv <= PFV_SAFETY_THRESHOLD) if not np.isnan(delta_pfv) else False
        tfv_imp = bool(delta_tfv < -TFV_IMPROVEMENT_THRESHOLD) if not np.isnan(delta_tfv) else False
        peak_ni = bool(delta_peak <= PEAK_NONINFERIOR_THRESHOLD) if not np.isnan(delta_peak) else False
        joint = pfv_safe and tfv_imp and peak_ni

        result = {
            "candidate_id": cid,
            "family": cand.family,
            "k_actual": cand.k_actual,
            "pfv_m3": pfv,
            "tfv_m3": tfv,
            "peak_tfv_rate": peak,
            "delta_pfv_vs_nc": round(delta_pfv, 4) if not np.isnan(delta_pfv) else float("nan"),
            "delta_tfv_vs_di": round(delta_tfv, 4) if not np.isnan(delta_tfv) else float("nan"),
            "delta_peak_vs_di": round(delta_peak, 6) if not np.isnan(delta_peak) else float("nan"),
            "pfv_safe": pfv_safe,
            "tfv_improved": tfv_imp,
            "peak_noninferior": peak_ni,
            "joint_feasible": joint,
            "action_hash": hashlib.sha256(cand.action.tobytes()).hexdigest()[:16],
            "description": cand.description,
            "error": error,
        }
        all_results.append(result)

        status = "JOINT" if joint else ("PFV" if pfv_safe else "") + ("+TFV" if tfv_imp else "") + ("+Peak" if peak_ni else "")
        print(f"    {cid[:40]:40s} K={cand.k_actual} dPFV={delta_pfv:+.2f} dTFV={delta_tfv:+.2f} dPk={delta_peak:+.4f} [{status or '-'}]")

    # ── Save results ──
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(GATE5_DIR / "gate5_candidate_results.csv", index=False)

    n_total = len(all_results)
    n_pfv_safe = sum(1 for r in all_results if r["pfv_safe"])
    n_tfv_imp = sum(1 for r in all_results if r["tfv_improved"])
    n_peak_ni = sum(1 for r in all_results if r["peak_noninferior"])
    n_joint = sum(1 for r in all_results if r["joint_feasible"])
    n_errors = sum(1 for r in all_results if r.get("error"))

    # Family coverage
    families = list(set(r["family"] for r in all_results))
    family_counts = {}
    for r in all_results:
        f = r["family"]
        family_counts[f] = family_counts.get(f, 0) + 1

    # K distribution
    k_vals = [r["k_actual"] for r in all_results]

    # Label diversity
    n_hard_neg = sum(1 for r in all_results if not np.isnan(r["delta_pfv_vs_nc"]) and r["delta_pfv_vs_nc"] > PFV_SAFETY_THRESHOLD)
    n_neutral = sum(1 for r in all_results
                    if not np.isnan(r.get("delta_pfv_vs_nc", 0)) and not np.isnan(r.get("delta_tfv_vs_di", 0))
                    and abs(r["delta_pfv_vs_nc"]) < 0.01 and abs(r["delta_tfv_vs_di"]) < 0.01)

    # Action uniqueness
    action_hashes = set(r["action_hash"] for r in all_results)

    # PASS conditions
    pass_conditions = {
        "at_least_one_pfv_safe": n_pfv_safe >= 1,
        "at_least_one_tfv_improved": n_tfv_imp >= 1,
        "at_least_one_peak_noninferior": n_peak_ni >= 1,
        "at_least_one_joint_feasible": n_joint >= 1,
        "multiple_families": len(families) >= 3,
        "label_diversity": n_hard_neg > 0 or n_neutral > 0,
        "all_k_le_8": max(k_vals) <= 8 if k_vals else False,
        "unique_actions": len(action_hashes) == n_total,
    }
    gate5_pass = all(pass_conditions.values())

    audit = {
        "event_id": EVENT_ID,
        "checkpoint_min": CHECKPOINT_MIN,
        "n_candidates_run": n_total,
        "n_errors": n_errors,
        "n_pfv_safe": n_pfv_safe,
        "n_tfv_improved": n_tfv_imp,
        "n_peak_noninferior": n_peak_ni,
        "n_joint_feasible": n_joint,
        "families": families,
        "family_counts": family_counts,
        "k_distribution": {"min": int(min(k_vals)), "max": int(max(k_vals)), "mean": round(float(np.mean(k_vals)), 2)} if k_vals else {},
        "n_unique_actions": len(action_hashes),
        "n_hard_negatives": n_hard_neg,
        "n_neutral": n_neutral,
        "pass_conditions": {k: bool(v) for k, v in pass_conditions.items()},
        "gate5_pass": bool(gate5_pass),
        "max_workers": MAX_WORKERS,
        "wall_time_sec": round(time.time() - t0, 1),
    }

    (GATE5_DIR / "gate5_candidate_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'='*70}")
    print(f"  Gate 5 Summary")
    print(f"{'='*70}")
    print(f"  Candidates run: {n_total} (errors: {n_errors})")
    print(f"  PFV safe: {n_pfv_safe}")
    print(f"  TFV improved: {n_tfv_imp}")
    print(f"  Peak noninferior: {n_peak_ni}")
    print(f"  Joint feasible: {n_joint}")
    print(f"  Families: {families}")
    print(f"  K range: [{min(k_vals) if k_vals else '-'}, {max(k_vals) if k_vals else '-'}]")
    print(f"  Gate 5 PASS: {gate5_pass}")
    print(f"  Wall time: {time.time() - t0:.1f}s")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
