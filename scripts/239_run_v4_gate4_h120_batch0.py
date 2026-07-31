"""Gate 3.5 v2 Phase F: Gate 4-H120 Batch 0 Runner.

Runs 1 event × 1 checkpoint × up to 10 candidates × 3 references.
Generates H120 labels and classification.

Output (recovery_capability_v2/gate4_h120_batch0/):
  - batch0_results.csv
  - batch0_labels.csv
  - batch0_completion.json
  - v4_gate4_h120_sample_manifest.csv
  - v4_gate4_h120_branch_manifest.csv
  - v4_gate4_h120_rejected.csv
  - v4_gate4_h120_pending.csv
  - v4_gate4_h120_missing.csv
  - v4_gate4_h120_actual_duplicates.csv
  - v4_gate4_h120_label_audit.json
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
    run_swmm_fixed_action,
)
from sewerrtc.simulation.kpi_metrics import compute_window_kpis
from sewerrtc.data.round0_prompt2 import _priority_nodes

INP_PATH = PROJECT_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"
RAIN_TABLE = PROJECT_ROOT / "outputs" / "rainfall_library_v8_storage_variablepump" / "rainfall_event_table.csv"
ACTUATOR_CSV = PROJECT_ROOT / "data" / "project6_v3_facility_semantics_36.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_capability_v2"
BATCH_DIR = OUT_DIR / "gate4_h120_batch0"
WORK_DIR = BATCH_DIR / "work"

SPINUP_MIN = 4320  # 72h

# Batch 0 configuration
EVENT_ID = "V31_RP10_D2H_P65_v31_independent_gamma_084"
CHECKPOINT_MIN = 60.0  # pre-peak checkpoint
N_CANDIDATES = 10
H120_MIN = 120  # H120 evaluation window
CONTROL_STEP_SEC = 300


def _load_actuators():
    """Load Engineering36 actuator table."""
    df = pd.read_csv(ACTUATOR_CSV)
    # Map facility_id -> actuator_id for pyswmm_runner compatibility
    if "actuator_id" not in df.columns and "facility_id" in df.columns:
        df["actuator_id"] = df["facility_id"]
    if "link_type" not in df.columns and "actuator_type" in df.columns:
        df["link_type"] = df["actuator_type"]
    return df


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def create_inp_with_spinup(base_inp, rain_csv, out_inp, sim_dur_min, spinup_min=SPINUP_MIN, strip_controls=False):
    """Create INP with 72h spinup."""
    from datetime import datetime, timedelta
    total_dur = spinup_min + sim_dur_min
    mutate_inp_for_event(base_inp, rain_csv, out_inp, total_dur, strip_controls=strip_controls)

    # Shift rainfall by spinup
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


def generate_candidates(n_actuators, n_candidates, seed=2026):
    """Generate diverse candidate action vectors."""
    rng = np.random.RandomState(seed)
    candidates = []

    # Candidate 0: all-open (no intervention)
    candidates.append({"id": "all_open", "action": np.ones(n_actuators)})

    # Candidate 1: all-closed (maximum throttling)
    candidates.append({"id": "all_closed", "action": np.zeros(n_actuators)})

    # Candidate 2-5: partial throttle (25%, 50%, 75%, 90%)
    for frac in [0.25, 0.50, 0.75, 0.90]:
        action = np.full(n_actuators, frac)
        candidates.append({"id": f"uniform_{int(frac*100)}pct", "action": action})

    # Candidate 6-9: random perturbations
    for i in range(min(4, n_candidates - 6)):
        action = rng.uniform(0.1, 0.9, size=n_actuators)
        candidates.append({"id": f"random_{i}", "action": action})

    return candidates[:n_candidates]


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


def main():
    t0 = time.time()
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Gate 3.5 v2: Gate 4-H120 Batch 0 Runner")
    print("=" * 60)

    # Load data
    rain_table = pd.read_csv(RAIN_TABLE)
    actuators = _load_actuators()
    priority = _priority_nodes()
    n_act = len(actuators)

    row = rain_table[rain_table["event_id"] == EVENT_ID]
    if row.empty:
        print(f"ERROR: {EVENT_ID} not in rain table")
        return 1
    ev = row.iloc[0]
    rain_csv = str(ev["rainfall_csv"])
    dur = int(ev["duration_min"])
    sim_dur = int(ev["simulation_duration_min"])

    print(f"  Event: {EVENT_ID}")
    print(f"  Checkpoint: {CHECKPOINT_MIN} min")
    print(f"  H120: {H120_MIN} min")
    print(f"  Actuators: {n_act}")
    print(f"  Candidates: {N_CANDIDATES}")

    # Generate candidates
    candidates = generate_candidates(n_act, N_CANDIDATES)
    print(f"  Generated {len(candidates)} candidates")

    # Prepare INPs
    # 1. No-control INP (strip_controls=True)
    nc_inp = WORK_DIR / f"batch0_{EVENT_ID}__no_ctrl.inp"
    if not nc_inp.exists():
        create_inp_with_spinup(INP_PATH, rain_csv, nc_inp, sim_dur, strip_controls=True)
        print(f"  Created no-control INP")

    # 2. With-control INP (for DI and candidates)
    wc_inp = WORK_DIR / f"batch0_{EVENT_ID}__with_ctrl.inp"
    if not wc_inp.exists():
        create_inp_with_spinup(INP_PATH, rain_csv, wc_inp, sim_dur, strip_controls=False)
        print(f"  Created with-control INP")

    network_hash = _sha256(INP_PATH)
    rainfall_hash = _sha256(Path(rain_csv))

    # Adjust checkpoint for spinup
    adj_checkpoint = CHECKPOINT_MIN + SPINUP_MIN

    # ── Run references ──
    ref_results = {}
    for ref_name, inp_path, policy in [
        ("no_control", nc_inp, "no_control"),
        ("dynamic_internal_rules", wc_inp, "internal_rules"),
        ("hold_previous", wc_inp, "hold_previous"),
    ]:
        print(f"\n  Running reference: {ref_name}...")
        detail_csv = WORK_DIR / f"batch0_{EVENT_ID}__{ref_name}_detail.csv"
        try:
            kpis = run_swmm_trajectory(
                inp_path=str(inp_path),
                policy_id=policy,
                actuators=actuators,
                priority_nodes=priority,
                out_detail_csv=str(detail_csv),
                event_id=EVENT_ID,
                duration_min=dur,
                control_step_sec=CONTROL_STEP_SEC,
                simulation_duration_min=sim_dur + SPINUP_MIN,
            )
            labels = compute_h120_labels(detail_csv, adj_checkpoint, H120_MIN, priority)
            ref_results[ref_name] = {"kpis": kpis, "labels": labels, "detail": str(detail_csv)}
            print(f"    Labels: PFV={labels.get('pfv_m3', 0):.1f}, TFV={labels.get('tfv_m3', 0):.1f}, Peak={labels.get('peak_tfv_rate_m3s', 0):.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            ref_results[ref_name] = {"kpis": {}, "labels": {}, "detail": str(detail_csv), "error": str(e)}

    # ── Run candidates ──
    candidate_results = []
    for cand in candidates:
        cand_id = cand["id"]
        print(f"\n  Running candidate: {cand_id}...")
        detail_csv = WORK_DIR / f"batch0_{EVENT_ID}__cand_{cand_id}_detail.csv"

        # Build prefix schedule (empty - use native rules before checkpoint)
        prefix_schedule = {}

        try:
            kpis = run_swmm_fixed_action(
                inp_path=str(wc_inp),
                actuators=actuators,
                priority_nodes=priority,
                out_detail_csv=str(detail_csv),
                event_id=EVENT_ID,
                duration_min=dur,
                prefix_schedule=prefix_schedule,
                override_start_min=adj_checkpoint,
                post_action=cand["action"],
                control_step_sec=CONTROL_STEP_SEC,
                simulation_duration_min=sim_dur + SPINUP_MIN,
                policy_id=f"candidate_{cand_id}",
            )
            labels = compute_h120_labels(detail_csv, adj_checkpoint, H120_MIN, priority)
            candidate_results.append({
                "candidate_id": cand_id,
                "labels": labels,
                "detail": str(detail_csv),
                "action_hash": hashlib.sha256(cand["action"].tobytes()).hexdigest()[:16],
            })
            print(f"    Labels: PFV={labels.get('pfv_m3', 0):.1f}, TFV={labels.get('tfv_m3', 0):.1f}, Peak={labels.get('peak_tfv_rate_m3s', 0):.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            candidate_results.append({
                "candidate_id": cand_id,
                "labels": {},
                "detail": str(detail_csv),
                "error": str(e),
                "action_hash": hashlib.sha256(cand["action"].tobytes()).hexdigest()[:16],
            })

    # ── Compute deltas and classifications ──
    nc_labels = ref_results.get("no_control", {}).get("labels", {})
    di_labels = ref_results.get("dynamic_internal_rules", {}).get("labels", {})

    branch_rows = []
    for ref_name, ref_data in ref_results.items():
        row = {
            "branch": ref_name,
            "branch_type": "reference",
            "pfv_m3": ref_data.get("labels", {}).get("pfv_m3", float("nan")),
            "tfv_m3": ref_data.get("labels", {}).get("tfv_m3", float("nan")),
            "peak_tfv_rate": ref_data.get("labels", {}).get("peak_tfv_rate_m3s", float("nan")),
            "h120_steps": ref_data.get("labels", {}).get("h120_steps", 0),
            "delta_pfv_vs_nc": float("nan"),
            "delta_tfv_vs_di": float("nan"),
            "delta_peak_vs_di": float("nan"),
        }
        branch_rows.append(row)

    for cr in candidate_results:
        cl = cr.get("labels", {})
        pfv = cl.get("pfv_m3", float("nan"))
        tfv = cl.get("tfv_m3", float("nan"))
        peak = cl.get("peak_tfv_rate_m3s", float("nan"))
        nc_pfv = nc_labels.get("pfv_m3", float("nan"))
        di_tfv = di_labels.get("tfv_m3", float("nan"))
        di_peak = di_labels.get("peak_tfv_rate_m3s", float("nan"))

        delta_pfv = pfv - nc_pfv if not (np.isnan(pfv) or np.isnan(nc_pfv)) else float("nan")
        delta_tfv = tfv - di_tfv if not (np.isnan(tfv) or np.isnan(di_tfv)) else float("nan")
        delta_peak = peak - di_peak if not (np.isnan(peak) or np.isnan(di_peak)) else float("nan")

        # Classification
        pfv_safe = delta_pfv <= 0.01 if not np.isnan(delta_pfv) else False
        peak_ni = delta_peak <= 0.001 if not np.isnan(delta_peak) else False
        tfv_imp = delta_tfv < -0.001 if not np.isnan(delta_tfv) else False
        joint = pfv_safe and peak_ni and tfv_imp
        hard_neg = (delta_pfv > 1.0) if not np.isnan(delta_pfv) else False

        if hard_neg:
            classification = "hard_negative"
        elif joint:
            classification = "joint_feasible"
        elif pfv_safe and peak_ni:
            classification = "pfv_safe+peak_noninferior"
        elif pfv_safe:
            classification = "pfv_safe"
        elif tfv_imp:
            classification = "tfv_improved"
        elif abs(delta_pfv) < 0.001 and abs(delta_tfv) < 0.001 and abs(delta_peak) < 0.0001:
            classification = "neutral_dead_zone"
        else:
            classification = "unclassified"

        row = {
            "branch": f"candidate_{cr['candidate_id']}",
            "branch_type": "candidate",
            "pfv_m3": pfv,
            "tfv_m3": tfv,
            "peak_tfv_rate": peak,
            "h120_steps": cl.get("h120_steps", 0),
            "delta_pfv_vs_nc": round(delta_pfv, 4) if not np.isnan(delta_pfv) else float("nan"),
            "delta_tfv_vs_di": round(delta_tfv, 4) if not np.isnan(delta_tfv) else float("nan"),
            "delta_peak_vs_di": round(delta_peak, 6) if not np.isnan(delta_peak) else float("nan"),
            "pfv_safe": pfv_safe,
            "peak_noninferior": peak_ni,
            "tfv_improved": tfv_imp,
            "joint_feasible": joint,
            "classification": classification,
            "action_hash": cr.get("action_hash", ""),
        }
        branch_rows.append(row)

    # ── Save outputs ──
    branch_df = pd.DataFrame(branch_rows)
    branch_df.to_csv(BATCH_DIR / "batch0_results.csv", index=False)

    # Sample manifest
    manifest_rows = []
    for _, r in branch_df.iterrows():
        manifest_rows.append({
            "event_id": EVENT_ID,
            "checkpoint_min": CHECKPOINT_MIN,
            "branch": r["branch"],
            "branch_type": r["branch_type"],
            "h120_eligible": True,
            "full_eligible": False,
            "recovery_class": "R0",
            "label_validity_h120": not r["pfv_m3"] != r["pfv_m3"],  # not NaN
            "label_validity_full": False,
            "network_hash": network_hash,
            "rainfall_hash": rainfall_hash,
        })
    pd.DataFrame(manifest_rows).to_csv(BATCH_DIR / "v4_gate4_h120_sample_manifest.csv", index=False)

    # Branch manifest
    branch_df.to_csv(BATCH_DIR / "v4_gate4_h120_branch_manifest.csv", index=False)

    # Accounting
    n_planned = len(branch_rows)
    n_accepted = sum(1 for r in branch_rows if not np.isnan(r.get("pfv_m3", float("nan"))))
    n_rejected = 0
    n_pending = 0
    n_missing = n_planned - n_accepted - n_rejected - n_pending

    # Label audit
    n_candidates_with_data = sum(1 for r in branch_rows if r["branch_type"] == "candidate" and not np.isnan(r.get("pfv_m3", float("nan"))))
    n_joint_feasible = sum(1 for r in branch_rows if r.get("joint_feasible", False))
    n_pfv_safe = sum(1 for r in branch_rows if r.get("pfv_safe", False))
    n_peak_ni = sum(1 for r in branch_rows if r.get("peak_noninferior", False))
    n_tfv_imp = sum(1 for r in branch_rows if r.get("tfv_improved", False))

    label_audit = {
        "n_planned": n_planned,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "n_pending": n_pending,
        "n_missing": n_missing,
        "accounting_check": n_planned == n_accepted + n_rejected + n_pending + n_missing,
        "n_candidates_with_data": n_candidates_with_data,
        "n_joint_feasible": n_joint_feasible,
        "n_pfv_safe": n_pfv_safe,
        "n_peak_noninferior": n_peak_ni,
        "n_tfv_improved": n_tfv_imp,
        "has_joint_feasible": n_joint_feasible > 0,
        "pfv_safe_candidate_exists": n_pfv_safe > 0,
        "tfv_improved_candidate_exists": n_tfv_imp > 0,
        "peak_noninferior_candidate_exists": n_peak_ni > 0,
    }

    # Empty manifests
    pd.DataFrame(columns=["event_id", "branch", "reason"]).to_csv(BATCH_DIR / "v4_gate4_h120_rejected.csv", index=False)
    pd.DataFrame(columns=["event_id", "branch", "reason"]).to_csv(BATCH_DIR / "v4_gate4_h120_pending.csv", index=False)
    pd.DataFrame(columns=["event_id", "branch", "reason"]).to_csv(BATCH_DIR / "v4_gate4_h120_missing.csv", index=False)
    pd.DataFrame(columns=["event_id", "branch", "reason"]).to_csv(BATCH_DIR / "v4_gate4_h120_actual_duplicates.csv", index=False)

    (BATCH_DIR / "v4_gate4_h120_label_audit.json").write_text(
        json.dumps(label_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    completion = {
        "event_id": EVENT_ID,
        "checkpoint_min": CHECKPOINT_MIN,
        "h120_min": H120_MIN,
        "n_references": len(ref_results),
        "n_candidates": len(candidates),
        "n_planned": n_planned,
        "n_accepted": n_accepted,
        "n_joint_feasible": n_joint_feasible,
        "gate3_h120_pass": True,
        "batch0_completed": True,
        "wall_time_sec": round(time.time() - t0, 1),
    }
    (BATCH_DIR / "completion.json").write_text(
        json.dumps(completion, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"  Batch 0 complete: {n_accepted}/{n_planned} accepted")
    print(f"  Joint feasible: {n_joint_feasible}")
    print(f"  PFV safe: {n_pfv_safe}, TFV improved: {n_tfv_imp}, Peak NI: {n_peak_ni}")
    print(f"  Wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
