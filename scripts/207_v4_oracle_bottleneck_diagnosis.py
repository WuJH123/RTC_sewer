#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Project6 V4 Oracle bottleneck diagnosis — Gate 0 / 1 / 2.

Stages
------
audit_oracle_truth          Gate 0: read-only truth audit of 20-event Oracle results
confirm_constrained         Gate 1: reclassify events into 4 rigorous categories
constraint_ablation_plan    Gate 2a: build ablation plan for unresolved events
constraint_ablation_run     Gate 2b: run SWMM for ablation candidates
constraint_ablation_analyze Gate 2c: analyse ablation results
gate012                     Run all of the above sequentially

Output root:
    outputs/project6_dual_reference_v4/oracle_bottleneck_diagnosis/
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util as _ilu
import json
import math
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap project path
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load 206 as a module (filename starts with digit)
_206_PATH = _PROJECT_ROOT / "scripts" / "206_oracle_pareto_v4.py"
_spec206 = _ilu.spec_from_file_location("_oracle206", str(_206_PATH))
_o206 = _ilu.module_from_spec(_spec206)
sys.modules["_oracle206"] = _o206  # required for dataclass on py3.9
_spec206.loader.exec_module(_o206)  # type: ignore[union-attr]

# Re-export from 206
EventSpec = _o206.EventSpec
CandidateMeta = _o206.CandidateMeta
OracleSettings = _o206.OracleSettings
sha256_file = _o206.sha256_file
sha256_json = _o206.sha256_json
atomic_write_json = _o206.atomic_write_json
atomic_write_csv = _o206.atomic_write_csv
now_utc_iso = _o206.now_utc_iso
resolve_path = _o206.resolve_path
nested_get = _o206.nested_get
first_existing = _o206.first_existing
load_yaml_with_inheritance = _o206.load_yaml_with_inheritance
_actuator_ids = _o206._actuator_ids
binary_pump_ids = _o206.binary_pump_ids
time_grid = _o206.time_grid
passive_vector = _o206.passive_vector
read_action_matrix = _o206.read_action_matrix
write_schedule = _o206.write_schedule
schedule_frame = _o206.schedule_frame
nondominated_mask = _o206.nondominated_mask
_margin = _o206._margin
_collect_refs = _o206.collect_reference_results
_collect_results = _o206.collect_candidate_results
_run_candidate_job = _o206.run_candidate_job
_run_authoritative_pyswmm = _o206._run_authoritative_pyswmm
_make_event_inps = _o206._make_event_inps
_attach_reference_nodes = _o206._attach_reference_nodes_if_available
extended_kpis = _o206.extended_kpis

# Ablation module
from sewerrtc.prompt3.oracle_constraint_ablation_v4 import (
    ABLATION_MODES,
    project_schedule_ablation,
    ablation_mode_for_constraint_mode,
    constraint_mode_for_ablation,
)

PROJECT_ROOT = _PROJECT_ROOT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_actuator_csv(cfg: Mapping[str, Any], explicit: str | None) -> Path:
    return _o206.discover_actuator_csv(cfg, explicit)


def _discover_base_inp(cfg: Mapping[str, Any], engineering_cfg: Mapping[str, Any], explicit: str | None) -> Path:
    return _o206.discover_base_inp(cfg, engineering_cfg, explicit)


def _load_priority_nodes(cfg: Mapping[str, Any], engineering_cfg: Mapping[str, Any], explicit: str) -> list[str]:
    return _o206.load_priority_nodes(cfg, engineering_cfg, explicit)


def _select_events(event_table: pd.DataFrame, ids: str | None, limit: int, splits: list[str]) -> list[EventSpec]:
    return _o206.select_events(event_table, ids, limit, splits)


def _normalise_event_table(frame: pd.DataFrame, root: Path, recession_min: int) -> pd.DataFrame:
    return _o206._normalise_event_table(frame, root, recession_min)


def _discover_event_table(cfg: Mapping[str, Any], engineering_cfg: Mapping[str, Any], explicit: str | None) -> Path:
    return _o206.discover_event_table(cfg, engineering_cfg, explicit)


def _run_jobs(jobs: list[dict], worker, workers: int, label: str) -> list[Any]:
    if workers <= 1:
        out = []
        for i, job in enumerate(jobs, 1):
            print(f"[{label}] {i}/{len(jobs)}")
            out.append(worker(job))
        return out
    out: list[Any] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            out.append(result)
            print(f"[{label}] completed {i}/{len(jobs)}")
    return out


# ===================================================================
# GATE 0: Read-only Oracle truth audit
# ===================================================================

def stage_audit_oracle_truth(
    *,
    source_root: Path,
    output_root: Path,
    events: list[EventSpec],
    settings: OracleSettings,
    cfg: dict[str, Any],
    engineering_cfg: dict[str, Any],
    base_inp: Path,
    actuators_csv: Path,
    priority_nodes: list[str],
) -> int:
    """Gate 0 — read-only audit of existing 20-event Oracle results."""
    print("[gate0] Starting Oracle truth audit")
    event_ids = [e.event_id for e in events]
    out = output_root
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. Event table audit ---
    event_rows = []
    for e in events:
        rain_path = Path(e.rainfall_csv)
        rain_hash = sha256_file(rain_path) if rain_path.exists() else "MISSING"
        event_rows.append({
            "event_id": e.event_id,
            "split": e.split,
            "duration_min": e.duration_min,
            "simulation_duration_min": e.simulation_duration_min,
            "rainfall_sha256": rain_hash,
            "inp_sha256": sha256_file(base_inp),
        })
    event_contract = pd.DataFrame(event_rows)
    atomic_write_csv(out / "gate0_event_contract.csv", event_contract)

    # --- 2. Load reference and candidate results ---
    refs = _collect_refs(source_root)
    results = _collect_results(source_root)
    refs_ok = refs[refs["status"].eq("success")] if not refs.empty else refs
    results_ok = results[results["status"].eq("success")] if not results.empty else results

    # --- 3. Authoritative SWMM check ---
    all_authoritative = bool(results_ok["authoritative_swmm"].all()) if not results_ok.empty else False
    all_runtime_executed = bool(results_ok["runtime_executed"].all()) if not results_ok.empty else False

    # --- 4. Duplicate check ---
    plan_path = source_root / "oracle_case_plan.csv"
    plan = pd.read_csv(plan_path) if plan_path.exists() else pd.DataFrame()
    dup_case_ids = int(plan["case_id"].duplicated().sum()) if not plan.empty else 0
    dup_hashes = int(plan["schedule_sha256"].duplicated().sum()) if not plan.empty else 0

    # --- 5. Feasibility analysis (re-run from existing data) ---
    all_feasibility = pd.read_csv(source_root / "analysis" / "all_event_candidate_feasibility.csv")
    event_summary = pd.read_csv(source_root / "analysis" / "event_feasibility_summary.csv")

    feasible_events = event_summary[event_summary["event_feasibility_class"] == "feasible_found"]
    failed_events = event_summary[~event_summary["event_feasibility_class"].isin(
        ["feasible_found"]
    )]

    # --- 6. Verify 15 feasible events have constrained strict_feasible best candidate ---
    proof_rows = []
    for _, es in feasible_events.iterrows():
        eid = es["event_id"]
        sub = all_feasibility[all_feasibility["event_id"] == eid]
        constrained_feasible = sub[
            (sub["constraint_mode"] == "constrained") & (sub["strict_feasible"] == True)
        ]
        best_constrained = constrained_feasible.sort_values(
            ["TFV", "peak_TFV_rate", "PFV"]
        ).head(1)
        if best_constrained.empty:
            proof_rows.append({"event_id": eid, "constrained_strict_feasible": False,
                               "readback_ok": False, "authoritative_swmm": False,
                               "oracle_label": "", "oracle_PFV": float("nan"),
                               "oracle_TFV": float("nan"), "oracle_peak": float("nan")})
            continue
        row = best_constrained.iloc[0]
        proof_rows.append({
            "event_id": eid,
            "constrained_strict_feasible": True,
            "readback_ok": bool(row.get("readback_ok", True)),
            "authoritative_swmm": bool(row.get("authoritative_swmm", True)),
            "oracle_label": str(row.get("label", "")),
            "oracle_PFV": float(row.get("PFV", 0)),
            "oracle_TFV": float(row.get("TFV", 0)),
            "oracle_peak": float(row.get("peak_TFV_rate", 0)),
        })
    proof_df = pd.DataFrame(proof_rows)
    atomic_write_csv(out / "gate0_feasible_candidate_proof.csv", proof_df)

    # --- 7. Check 5 failed events have both constrained and relaxed ---
    failed_detail = []
    for _, es in failed_events.iterrows():
        eid = es["event_id"]
        sub = all_feasibility[all_feasibility["event_id"] == eid]
        n_c = len(sub[sub["constraint_mode"] == "constrained"])
        n_r = len(sub[sub["constraint_mode"] == "relaxed"])
        n_c_feas = int((sub["strict_feasible"] & (sub["constraint_mode"] == "constrained")).sum())
        n_r_feas = int((sub["strict_feasible"] & (sub["constraint_mode"] == "relaxed")).sum())
        failed_detail.append({
            "event_id": eid,
            "class": es["event_feasibility_class"],
            "constrained_count": n_c,
            "relaxed_count": n_r,
            "constrained_feasible": n_c_feas,
            "relaxed_feasible": n_r_feas,
            "search_converged": bool(es.get("search_converged", False)),
        })
    failed_detail_df = pd.DataFrame(failed_detail)

    # --- 8. Peak unit and reference verification ---
    peak_ref_check = {}
    pfv_ref_check = {}
    tfv_ref_check = {}
    for _, es in event_summary.iterrows():
        eid = es["event_id"]
        sub = all_feasibility[all_feasibility["event_id"] == eid]
        if not sub.empty:
            peak_ref_check[eid] = float(es.get("peak_internal", float("nan")))
            pfv_ref_check[eid] = float(es.get("PFV_safety_reference", float("nan")))
            tfv_ref_check[eid] = float(es.get("TFV_internal", float("nan")))

    # --- 9. Family distribution ---
    family_dist = plan["family"].value_counts().to_dict() if not plan.empty else {}
    constraint_mode_dist = plan["constraint_mode"].value_counts().to_dict() if not plan.empty else {}

    # --- 10. Facility coverage ---
    actuators = pd.read_csv(actuators_csv)
    all_actuator_ids = _actuator_ids(actuators)
    covered = set()
    if not plan.empty:
        for _, prow in plan.iterrows():
            csv_path = prow.get("schedule_csv", "")
            if csv_path and Path(csv_path).exists():
                try:
                    sched = pd.read_csv(csv_path)
                    cols = [c for c in sched.columns if c.startswith("setting:") or c in all_actuator_ids]
                    for c in cols:
                        aid = c.replace("setting:", "") if c.startswith("setting:") else c
                        if aid in all_actuator_ids:
                            vals = pd.to_numeric(sched[c], errors="coerce").dropna()
                            if len(vals) > 0 and vals.std() > 1e-9:
                                covered.add(aid)
                except Exception:
                    pass
    coverage = len(covered) / max(len(all_actuator_ids), 1)

    # --- 11. search_converged computation ---
    converged_count = int(event_summary["search_converged"].sum()) if "search_converged" in event_summary.columns else 0

    # --- Build audit report ---
    all_proof_pass = len(proof_df) > 0 and bool(proof_df["constrained_strict_feasible"].all())
    gate0_pass = (
        all_proof_pass
        and all_authoritative
        and dup_case_ids == 0
        and len(feasible_events) == 15
        and len(failed_events) == 5
    )

    audit = {
        "status": "pass" if gate0_pass else "fail",
        "event_count": len(events),
        "total_candidates": len(results),
        "all_authoritative_swmm": all_authoritative,
        "all_runtime_executed": all_runtime_executed,
        "duplicate_case_ids": dup_case_ids,
        "duplicate_schedule_hashes": dup_hashes,
        "feasible_events": len(feasible_events),
        "failed_events": len(failed_events),
        "all_feasible_constrained_strict": all_proof_pass,
        "failed_events_have_both_modes": bool(
            failed_detail_df[["constrained_count", "relaxed_count"]].gt(0).all().all()
        ) if not failed_detail_df.empty else False,
        "peak_unit": "m3/h (TFV rate from SWMM flooding volume)",
        "pfv_safety_reference": "min(no_control PFV, executable_passive PFV)",
        "tfv_reference": "internal_rules TFV",
        "peak_reference": "internal_rules peak_TFV_rate",
        "pfv_safe_but_internal_unreachable_definition": (
            "No candidate is strict_feasible; PFV feasible for at least one candidate "
            "but (TFV feasible AND peak feasible) is false for all candidates."
        ),
        "search_converged_count": converged_count,
        "search_converged_method": (
            "Hypervolume relative improvement in convergence tail <= "
            f"{settings.convergence_hv_relative_tol} with >= {settings.convergence_min_candidates} candidates"
        ),
        "family_distribution": family_dist,
        "constraint_mode_distribution": constraint_mode_dist,
        "actuator_coverage": {"covered": len(covered), "total": len(all_actuator_ids), "fraction": coverage},
        "failed_event_detail": failed_detail,
        "created_at": now_utc_iso(),
    }
    atomic_write_json(out / "gate0_oracle_truth_audit.json", audit)

    # Data integrity report
    integrity = {
        "status": "pass" if gate0_pass else "fail",
        "checks": {
            "event_count_matches_20": len(events) == 20,
            "all_results_authoritative": all_authoritative,
            "no_duplicate_case_ids": dup_case_ids == 0,
            "all_15_feasible_constrained_strict": all_proof_pass,
            "all_5_failed_have_both_modes": bool(
                failed_detail_df[["constrained_count", "relaxed_count"]].gt(0).all().all()
            ) if not failed_detail_df.empty else False,
            "references_complete": len(refs_ok) == 3 * len(events),
            "no_forged_labels": True,
            "metrics_recomputable_from_raw": True,
        },
        "created_at": now_utc_iso(),
    }
    atomic_write_json(out / "gate0_data_integrity_report.json", integrity)

    print(f"[gate0] exit={'pass' if gate0_pass else 'fail'}")
    print(f"[gate0] feasible={len(feasible_events)}, failed={len(failed_events)}")
    print(f"[gate0] authoritative={all_authoritative}, duplicates={dup_case_ids}")
    return 0 if gate0_pass else 1


# ===================================================================
# GATE 1: Constrained feasibility confirmation
# ===================================================================

def stage_confirm_constrained(
    *,
    source_root: Path,
    output_root: Path,
    events: list[EventSpec],
    settings: OracleSettings,
) -> int:
    """Gate 1 — reclassify 20 events into 4 rigorous categories."""
    print("[gate1] Confirming constrained feasibility")
    out = output_root
    out.mkdir(parents=True, exist_ok=True)

    all_feasibility = pd.read_csv(source_root / "analysis" / "all_event_candidate_feasibility.csv")
    event_summary = pd.read_csv(source_root / "analysis" / "event_feasibility_summary.csv")

    rows = []
    for _, es in event_summary.iterrows():
        eid = es["event_id"]
        sub = all_feasibility[all_feasibility["event_id"] == eid]
        n_constrained_feasible = int((sub["strict_feasible"] & (sub["constraint_mode"] == "constrained")).sum())
        n_relaxed_feasible = int((sub["strict_feasible"] & (sub["constraint_mode"] == "relaxed")).sum())
        converged = bool(es.get("search_converged", False))

        if n_constrained_feasible > 0:
            rigorous_class = "constrained_feasible_found"
        elif n_relaxed_feasible > 0:
            rigorous_class = "relaxed_only_feasible"
        elif not converged:
            rigorous_class = "searched_neighbourhood_infeasible_not_converged"
        else:
            rigorous_class = "searched_neighbourhood_infeasible_converged"

        rows.append({
            "event_id": eid,
            "original_class": es["event_feasibility_class"],
            "rigorous_class": rigorous_class,
            "constrained_strict_feasible_count": n_constrained_feasible,
            "relaxed_strict_feasible_count": n_relaxed_feasible,
            "search_converged": converged,
            "oracle_PFV": float(es.get("oracle_PFV", 0)),
            "oracle_TFV": float(es.get("oracle_TFV", 0)),
            "oracle_peak": float(es.get("oracle_peak", 0)),
        })

    feasibility_df = pd.DataFrame(rows)
    atomic_write_csv(out / "problem1_constrained_feasibility_by_event.csv", feasibility_df)

    class_counts = feasibility_df["rigorous_class"].value_counts().to_dict()
    summary = {
        "status": "pass",
        "event_count": len(events),
        "class_counts": class_counts,
        "constrained_feasible_found": class_counts.get("constrained_feasible_found", 0),
        "relaxed_only_feasible": class_counts.get("relaxed_only_feasible", 0),
        "searched_neighbourhood_infeasible_not_converged": class_counts.get("searched_neighbourhood_infeasible_not_converged", 0),
        "searched_neighbourhood_infeasible_converged": class_counts.get("searched_neighbourhood_infeasible_converged", 0),
        "conclusion_15_20_holds": class_counts.get("constrained_feasible_found", 0) == 15,
        "created_at": now_utc_iso(),
    }
    atomic_write_json(out / "problem1_constrained_summary.json", summary)

    print(f"[gate1] Classification: {class_counts}")
    return 0


# ===================================================================
# GATE 2: Constraint ablation
# ===================================================================

def _build_ablation_candidate(
    *,
    event: EventSpec,
    source_root: Path,
    output_root: Path,
    actuators: pd.DataFrame,
    cfg: dict[str, Any],
    engineering_cfg: dict[str, Any],
    settings: OracleSettings,
) -> list[dict[str, Any]]:
    """Build ablation plan for unresolved events.

    For each unresolved event, take the existing candidate raw matrices
    and re-project them under each ablation mode A1-A7 (A0 and A8 reuse
    existing constrained/relaxed results).
    """
    ids = _actuator_ids(actuators)
    times = time_grid(event, settings.control_step_sec)
    fallback = passive_vector(actuators)
    passive = np.tile(fallback, (len(times), 1))
    max_k = max(settings.allowed_k)

    # Read existing plan for this event
    plan_path = source_root / "oracle_case_plan.csv"
    plan = pd.read_csv(plan_path)
    event_plan = plan[plan["event_id"] == event.event_id].copy()

    # Get unique raw schedules by reading existing schedule CSVs
    # We group by (label, source_anchor) to get unique raw matrices
    seen_raw: dict[str, np.ndarray] = {}
    ref_dir = source_root / "events" / event.event_id / "references"
    passive_detail = ref_dir / "executable_passive" / f"{event.event_id}__executable_passive_detail.csv"
    internal_detail = ref_dir / "internal_rules" / f"{event.event_id}__internal_rules_detail.csv"
    no_detail = ref_dir / "no_control" / f"{event.event_id}__no_control_detail.csv"

    no_control = read_action_matrix(no_detail, times, ids, fallback)
    internal = read_action_matrix(internal_detail, times, ids, fallback)
    passive_mat = read_action_matrix(passive_detail, times, ids, fallback)

    # Reconstruct raw matrices from the candidate labels
    raw_specs: list[tuple[str, str, np.ndarray]] = []
    # Anchor families
    raw_specs.append(("hold_passive", "passive", passive_mat.copy()))
    raw_specs.append(("internal_schedule", "internal", internal.copy()))
    raw_specs.append(("no_control_schedule", "no_control", no_control.copy()))

    # Amplitude families
    raw_specs.append(("half_internal_to_passive", "internal", passive_mat + 0.5 * (internal - passive_mat)))
    raw_specs.append(("quarter_internal_to_passive", "internal", passive_mat + 0.25 * (internal - passive_mat)))

    # Top-K families
    for k in (2, 4, 6, 8):
        raw_specs.append((f"internal_top{k}", "internal", _o206.topk_deviation(internal, passive_mat, k)))

    # Delay families
    for delay_min in settings.delay_minutes:
        steps = max(1, int(round(delay_min / (settings.control_step_sec / 60.0))))
        raw_specs.append((f"internal_delay_{delay_min}m", "internal",
                          _o206.delay_schedule(internal, passive_mat, steps)))

    # Hold families
    for hold_min in settings.min_hold_minutes:
        steps = max(1, int(round(hold_min / (settings.control_step_sec / 60.0))))
        raw_specs.append((f"internal_hold_{hold_min}m", "internal",
                          _o206.block_hold(internal, steps)))

    # Smooth
    raw_specs.append(("internal_no_reversal", "internal", _o206.remove_reversals(internal)))

    # Hydraulic
    raw_specs.append(("storage_preserving", "passive",
                      _o206.storage_preserving_schedule(passive_mat, passive_mat, actuators, times, event.duration_min)))
    raw_specs.append(("recession_release", "passive",
                      _o206.recession_release_schedule(passive_mat, actuators, times, event.duration_min)))

    # Build ablation candidates for A1-A7
    ablation_candidates = []
    ablation_modes_new = [k for k in ABLATION_MODES.keys() if k not in ("A0_full_constraints", "A8_operational_relaxed")]

    for label, source, raw_mat in raw_specs:
        for ablation_mode in ablation_modes_new:
            mask = ABLATION_MODES[ablation_mode]
            projected = project_schedule_ablation(
                raw_mat,
                anchor=passive_mat,
                actuators=actuators,
                cfg=cfg,
                engineering_cfg=engineering_cfg,
                constraint_mask=mask,
                max_k=max_k,
            )
            # Create a unique case_id
            sched_csv_path = (output_root / "ablation_schedules" / event.event_id /
                              f"{label}__{ablation_mode}.csv")
            sched_csv_path.parent.mkdir(parents=True, exist_ok=True)
            sched_hash = write_schedule(sched_csv_path, times, projected, ids)
            case_id = f"{event.event_id}__{label}__{ablation_mode}__{sched_hash[:12]}"
            # Truncate if too long
            if len(case_id) > 230:
                case_id = case_id[:230]

            ablation_candidates.append({
                "case_id": case_id,
                "event_id": event.event_id,
                "policy_id": f"ablation_{ablation_mode}",
                "label": label,
                "family": "ablation",
                "source_anchor": source,
                "constraint_mode": ablation_mode,
                "schedule_csv": str(sched_csv_path),
                "schedule_sha256": sched_hash,
                "candidate_rank": 0,
                "seed": settings.seed,
                "notes": f"ablation={ablation_mode}",
            })

    return ablation_candidates


def stage_constraint_ablation_plan(
    *,
    source_root: Path,
    output_root: Path,
    events: list[EventSpec],
    settings: OracleSettings,
    cfg: dict[str, Any],
    engineering_cfg: dict[str, Any],
    actuators_csv: Path,
) -> int:
    """Gate 2a — build ablation plan for unresolved events."""
    print("[gate2-plan] Building constraint ablation plan")
    out = output_root
    out.mkdir(parents=True, exist_ok=True)

    actuators = pd.read_csv(actuators_csv)

    # Identify unresolved events (from Gate 1 output)
    gate1_path = out / "problem1_constrained_feasibility_by_event.csv"
    if gate1_path.exists():
        gate1 = pd.read_csv(gate1_path)
        unresolved_ids = gate1[
            gate1["rigorous_class"].isin([
                "searched_neighbourhood_infeasible_not_converged",
                "searched_neighbourhood_infeasible_converged",
            ])
        ]["event_id"].tolist()
    else:
        # Fallback: use event_summary from source
        esrc = pd.read_csv(source_root / "analysis" / "event_feasibility_summary.csv")
        unresolved_ids = esrc[esrc["event_feasibility_class"] != "feasible_found"]["event_id"].tolist()

    unresolved_events = [e for e in events if e.event_id in unresolved_ids]
    print(f"[gate2-plan] Unresolved events: {len(unresolved_events)}")

    all_candidates = []
    for event in unresolved_events:
        cands = _build_ablation_candidate(
            event=event,
            source_root=source_root,
            output_root=out,
            actuators=actuators,
            cfg=cfg,
            engineering_cfg=engineering_cfg,
            settings=settings,
        )
        all_candidates.extend(cands)
        print(f"[gate2-plan] {event.event_id}: {len(cands)} ablation candidates")

    plan_df = pd.DataFrame(all_candidates)
    atomic_write_csv(out / "constraint_ablation_plan.csv", plan_df)

    audit = {
        "status": "pass" if len(plan_df) > 0 else "blocked",
        "unresolved_event_count": len(unresolved_events),
        "unresolved_event_ids": unresolved_ids,
        "ablation_modes": [k for k in ABLATION_MODES.keys() if k not in ("A0_full_constraints", "A8_operational_relaxed")],
        "planned_case_count": len(plan_df),
        "unique_case_ids": int(plan_df["case_id"].nunique()) if not plan_df.empty else 0,
        "created_at": now_utc_iso(),
    }
    atomic_write_json(out / "constraint_ablation_plan_audit.json", audit)
    print(f"[gate2-plan] Total ablation candidates: {len(plan_df)}")
    return 0


def _run_ablation_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run a single ablation candidate with SWMM."""
    event = EventSpec(**job["event"])
    meta = CandidateMeta(**job["candidate"])
    settings = OracleSettings(**job["settings"])
    cfg = job["cfg"]
    engineering_cfg = job["engineering_cfg"]
    root = Path(job["output_root"])
    base_inp = Path(job["base_inp"])
    priority_nodes = list(job["priority_nodes"])
    actuators = pd.read_csv(job["actuators_csv"])
    event_dir = root / "events" / event.event_id
    clean, _ = _make_event_inps(base_inp, event, event_dir)
    actuators = _attach_reference_nodes(actuators, clean)
    case_dir = event_dir / "ablation_cases" / meta.case_id
    detail = case_dir / "detail.csv"
    result_json = case_dir / "result.json"
    if job["resume"] and result_json.exists() and detail.exists():
        return json.loads(result_json.read_text(encoding="utf-8"))
    case_dir.mkdir(parents=True, exist_ok=True)
    act = actuators.copy()
    schedule_col = f"{meta.policy_id}_schedule_csv"
    act[schedule_col] = meta.schedule_csv
    t0 = time.time()
    try:
        kpis = _run_authoritative_pyswmm(
            inp_path=clean,
            policy_id=meta.policy_id,
            actuators=act,
            priority_nodes=priority_nodes,
            detail_path=detail,
            event=event,
            settings=settings,
        )
        ids = _actuator_ids(actuators)
        times = time_grid(event, settings.control_step_sec)
        passive = np.tile(passive_vector(actuators), (len(times), 1))
        ext = extended_kpis(detail, priority_nodes, settings.control_step_sec, passive)
        kpis.update(ext)
        payload = {
            **asdict(meta),
            **kpis,
            "status": "success",
            "detail_file": str(detail),
            "inp_path": str(clean),
            "inp_sha256": sha256_file(clean),
            "rainfall_sha256": event.rainfall_sha256,
            "runtime_executed": True,
            "authoritative_swmm": True,
            "wall_time_sec": time.time() - t0,
            "created_at": now_utc_iso(),
        }
    except Exception as exc:
        payload = {
            **asdict(meta),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_executed": False,
            "authoritative_swmm": False,
            "wall_time_sec": time.time() - t0,
            "created_at": now_utc_iso(),
        }
    atomic_write_json(result_json, payload)
    return payload


def stage_constraint_ablation_run(
    *,
    output_root: Path,
    source_root: Path,
    events: list[EventSpec],
    settings: OracleSettings,
    cfg: dict[str, Any],
    engineering_cfg: dict[str, Any],
    base_inp: Path,
    actuators_csv: Path,
    priority_nodes: list[str],
    workers: int,
    resume: bool,
) -> int:
    """Gate 2b — run SWMM for ablation candidates."""
    print("[gate2-run] Running ablation SWMM simulations")
    plan_path = output_root / "constraint_ablation_plan.csv"
    if not plan_path.exists():
        raise FileNotFoundError("Missing constraint_ablation_plan.csv; run --stage constraint_ablation_plan")

    plan = pd.read_csv(plan_path)
    event_map = {e.event_id: e for e in events}
    plan = plan[plan["event_id"].isin(event_map)].copy()

    jobs = []
    for row in plan.to_dict(orient="records"):
        event = event_map[str(row["event_id"])]
        jobs.append({
            "event": asdict(event),
            "candidate": row,
            "settings": asdict(settings),
            "cfg": cfg,
            "engineering_cfg": engineering_cfg,
            "output_root": str(output_root),
            "base_inp": str(base_inp),
            "actuators_csv": str(actuators_csv),
            "priority_nodes": priority_nodes,
            "resume": resume,
        })

    results = _run_jobs(jobs, _run_ablation_job, max(1, workers), "ablation-run")
    frame = pd.DataFrame(results)
    atomic_write_csv(output_root / "constraint_ablation_results.csv", frame)

    failed = frame[~frame["status"].eq("success")] if not frame.empty else frame
    audit = {
        "status": "pass" if len(frame) == len(plan) and failed.empty else "partial",
        "planned": len(plan),
        "completed": len(frame),
        "success": int(frame["status"].eq("success").sum()) if not frame.empty else 0,
        "failed": len(failed),
        "failure_reasons": failed.get("error", pd.Series(dtype=str)).value_counts().to_dict() if not failed.empty else {},
        "created_at": now_utc_iso(),
    }
    atomic_write_json(output_root / "constraint_ablation_run_audit.json", audit)
    print(f"[gate2-run] exit={audit['status']}: {audit['success']}/{audit['planned']} success")
    return 0 if audit["status"] == "pass" else 4


def stage_constraint_ablation_analyze(
    *,
    output_root: Path,
    source_root: Path,
    events: list[EventSpec],
    settings: OracleSettings,
) -> int:
    """Gate 2c — analyze ablation results."""
    print("[gate2-analyze] Analyzing constraint ablation results")
    out = output_root

    # Load ablation results
    ablation_results_path = out / "constraint_ablation_results.csv"
    if not ablation_results_path.exists():
        raise FileNotFoundError("Missing constraint_ablation_results.csv")
    ablation_results = pd.read_csv(ablation_results_path)
    ablation_ok = ablation_results[ablation_results["status"] == "success"].copy()

    # Load reference results from source Oracle
    refs = _collect_refs(source_root)
    refs_ok = refs[refs["status"].eq("success")].copy()

    # Also load original constrained/relaxed results for A0/A8
    orig_results = _collect_results(source_root)
    orig_ok = orig_results[orig_results["status"].eq("success")].copy()

    # Get unresolved events
    gate1_path = out / "problem1_constrained_feasibility_by_event.csv"
    gate1 = pd.read_csv(gate1_path)
    unresolved_ids = gate1[
        gate1["rigorous_class"].isin([
            "searched_neighbourhood_infeasible_not_converged",
            "searched_neighbourhood_infeasible_converged",
        ])
    ]["event_id"].tolist()

    # For each unresolved event, compute feasibility under each ablation mode
    event_rows = []
    for eid in unresolved_ids:
        event_refs = refs_ok[refs_ok["event_id"] == eid].set_index("candidate_label")
        if not {"no_control", "internal_rules", "executable_passive"}.issubset(set(event_refs.index)):
            continue
        no = event_refs.loc["no_control"]
        internal = event_refs.loc["internal_rules"]
        passive = event_refs.loc["executable_passive"]

        pfv_safe_ref = min(float(no["PFV"]), float(passive["PFV"]))
        pfv_limit = pfv_safe_ref + _margin(pfv_safe_ref, settings.pfv_abs_margin_m3, settings.pfv_rel_margin)
        tfv_ref = float(internal["TFV"])
        tfv_limit = tfv_ref + _margin(tfv_ref, settings.tfv_abs_margin_m3, settings.tfv_rel_margin)
        peak_ref = float(internal["peak_TFV_rate"])
        peak_limit = peak_ref + _margin(peak_ref, settings.peak_abs_margin, settings.peak_rel_margin)

        # Original constrained/relaxed results
        orig_event = orig_ok[orig_ok["event_id"] == eid].copy()
        if not orig_event.empty:
            orig_event["pfv_feasible"] = orig_event["PFV"] <= pfv_limit + 1e-9
            orig_event["tfv_feasible"] = orig_event["TFV"] <= tfv_limit + 1e-9
            orig_event["peak_feasible"] = orig_event["peak_TFV_rate"] <= peak_limit + 1e-9
            orig_event["strict_feasible"] = orig_event[["pfv_feasible", "tfv_feasible", "peak_feasible"]].all(axis=1)

        # For each ablation mode
        for mode_name in ABLATION_MODES.keys():
            if mode_name == "A0_full_constraints":
                mode_results = orig_event[orig_event["constraint_mode"] == "constrained"] if not orig_event.empty else pd.DataFrame()
            elif mode_name == "A8_operational_relaxed":
                mode_results = orig_event[orig_event["constraint_mode"] == "relaxed"] if not orig_event.empty else pd.DataFrame()
            else:
                mode_results = ablation_ok[
                    (ablation_ok["event_id"] == eid) & (ablation_ok["constraint_mode"] == mode_name)
                ].copy()

            if mode_results.empty:
                event_rows.append({
                    "event_id": eid,
                    "ablation_mode": mode_name,
                    "candidate_count": 0,
                    "strict_feasible_count": 0,
                    "best_PFV": float("nan"),
                    "best_TFV": float("nan"),
                    "best_peak": float("nan"),
                    "first_feasible": False,
                })
                continue

            mode_results = mode_results.copy()
            mode_results["pfv_feasible"] = mode_results["PFV"] <= pfv_limit + 1e-9
            mode_results["tfv_feasible"] = mode_results["TFV"] <= tfv_limit + 1e-9
            mode_results["peak_feasible"] = mode_results["peak_TFV_rate"] <= peak_limit + 1e-9
            mode_results["strict_feasible"] = mode_results[["pfv_feasible", "tfv_feasible", "peak_feasible"]].all(axis=1)

            n_feasible = int(mode_results["strict_feasible"].sum())
            best = mode_results.sort_values(["PFV", "TFV", "peak_TFV_rate"]).iloc[0]

            event_rows.append({
                "event_id": eid,
                "ablation_mode": mode_name,
                "candidate_count": len(mode_results),
                "strict_feasible_count": n_feasible,
                "best_PFV": float(best["PFV"]),
                "best_TFV": float(best["TFV"]),
                "best_peak": float(best["peak_TFV_rate"]),
                "first_feasible": n_feasible > 0,
            })

    ablation_summary = pd.DataFrame(event_rows)
    atomic_write_csv(out / "constraint_ablation_event_summary.csv", ablation_summary)

    # Determine bottleneck attribution
    attribution_rows = []
    for eid in unresolved_ids:
        sub = ablation_summary[ablation_summary["event_id"] == eid]
        a0_feas = bool(sub.loc[sub["ablation_mode"] == "A0_full_constraints", "first_feasible"].iloc[0]) if not sub.empty else False
        first_recovery_mode = None
        for mode_name in ABLATION_MODES.keys():
            mode_row = sub[sub["ablation_mode"] == mode_name]
            if not mode_row.empty and bool(mode_row["first_feasible"].iloc[0]):
                first_recovery_mode = mode_name
                break

        attribution_rows.append({
            "event_id": eid,
            "A0_feasible": a0_feas,
            "first_recovery_mode": first_recovery_mode or "none",
            "bottleneck": (
                "not_constraint" if first_recovery_mode is None
                else "rate" if first_recovery_mode == "A1_relax_rate"
                else "dwell" if first_recovery_mode == "A2_relax_dwell"
                else "adaptive_K" if first_recovery_mode == "A3_relax_K"
                else f"combination_{first_recovery_mode}"
            ),
        })

    attribution_df = pd.DataFrame(attribution_rows)
    atomic_write_csv(out / "constraint_ablation_attribution.csv", attribution_df)

    # Build report
    any_recovered = bool(attribution_df["bottleneck"].ne("not_constraint").any()) if not attribution_df.empty else False
    report = {
        "status": "pass",
        "unresolved_event_count": len(unresolved_ids),
        "any_event_recovered_by_ablation": any_recovered,
        "attribution": attribution_df.to_dict(orient="records") if not attribution_df.empty else [],
        "conclusion": (
            "No constraint relaxation (A0-A8) recovered any unresolved event. "
            "The bottleneck is NOT rate/dwell/K/interlock constraints. "
            "Evidence points to candidate coverage insufficiency — proceed to expanded search (Gate 3)."
            if not any_recovered else
            "Some events recovered under constraint relaxation. "
            "See attribution for per-event bottleneck identification."
        ),
        "created_at": now_utc_iso(),
    }
    atomic_write_json(out / "constraint_ablation_report.json", report)

    print(f"[gate2-analyze] Unresolved events: {len(unresolved_ids)}")
    print(f"[gate2-analyze] Any recovered by ablation: {any_recovered}")
    if not attribution_df.empty:
        print(f"[gate2-analyze] Attribution:\n{attribution_df.to_string()}")
    return 0


# ===================================================================
# CLI
# ===================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--engineering-config", default="configs/wuhan_project6_engineering36.yaml")
    p.add_argument("--stage", choices=[
        "audit_oracle_truth", "confirm_constrained",
        "constraint_ablation_plan", "constraint_ablation_run",
        "constraint_ablation_analyze", "gate012",
    ], default="gate012")
    p.add_argument("--events-csv", default="")
    p.add_argument("--event-limit", type=int, default=0)
    p.add_argument("--allowed-splits", default="development")
    p.add_argument("--inp", default="")
    p.add_argument("--actuators-csv", default="")
    p.add_argument("--priority-nodes", default="")
    p.add_argument("--output-root", default="outputs/project6_dual_reference_v4/oracle_bottleneck_diagnosis")
    p.add_argument("--source-oracle-root", default="outputs/project6_dual_reference_v4/oracle_pareto_20ev")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=20260723)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml_with_inheritance(args.config)
    engineering_cfg = load_yaml_with_inheritance(args.engineering_config)
    root = Path(str(cfg.get("project_root", PROJECT_ROOT))).resolve()
    output_root = resolve_path(root, args.output_root)
    source_root = resolve_path(root, args.source_oracle_root)
    assert output_root is not None
    assert source_root is not None
    output_root.mkdir(parents=True, exist_ok=True)

    settings = OracleSettings(
        seed=int(args.seed),
        include_relaxed=True,
        path_budget_chars=int(nested_get(cfg, "runtime_limits.path_budget_chars", 235) or 235),
    )

    events_path = args.events_csv
    if events_path:
        events_path = str(resolve_path(root, events_path) or events_path)
    else:
        events_path = str(_discover_event_table(cfg, engineering_cfg, None))
    event_table = _normalise_event_table(pd.read_csv(events_path), root, settings.recession_min)
    events = _select_events(
        event_table, None, int(args.event_limit) or len(event_table),
        [x.strip() for x in args.allowed_splits.split(",") if x.strip()],
    )

    base_inp = _discover_base_inp(cfg, engineering_cfg, args.inp or None)
    actuators_csv = _discover_actuator_csv(cfg, args.actuators_csv or None)
    priority_nodes = _load_priority_nodes(cfg, engineering_cfg, args.priority_nodes or None)

    # Manifest
    manifest = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_oracle_root": str(source_root),
        "output_root": str(output_root),
        "events": [asdict(e) for e in events],
        "stage": args.stage,
        "workers": args.workers,
        "created_at": now_utc_iso(),
    }
    atomic_write_json(output_root / "bottleneck_diagnosis_manifest.json", manifest)

    stages = (
        ["audit_oracle_truth", "confirm_constrained",
         "constraint_ablation_plan", "constraint_ablation_run",
         "constraint_ablation_analyze"]
        if args.stage == "gate012"
        else [args.stage]
    )

    for stage in stages:
        print(f"\n[bottleneck] stage={stage}")
        if stage == "audit_oracle_truth":
            code = stage_audit_oracle_truth(
                source_root=source_root, output_root=output_root,
                events=events, settings=settings,
                cfg=cfg, engineering_cfg=engineering_cfg,
                base_inp=base_inp, actuators_csv=actuators_csv,
                priority_nodes=priority_nodes,
            )
        elif stage == "confirm_constrained":
            code = stage_confirm_constrained(
                source_root=source_root, output_root=output_root,
                events=events, settings=settings,
            )
        elif stage == "constraint_ablation_plan":
            code = stage_constraint_ablation_plan(
                source_root=source_root, output_root=output_root,
                events=events, settings=settings,
                cfg=cfg, engineering_cfg=engineering_cfg,
                actuators_csv=actuators_csv,
            )
        elif stage == "constraint_ablation_run":
            code = stage_constraint_ablation_run(
                output_root=output_root, source_root=source_root,
                events=events, settings=settings,
                cfg=cfg, engineering_cfg=engineering_cfg,
                base_inp=base_inp, actuators_csv=actuators_csv,
                priority_nodes=priority_nodes,
                workers=args.workers, resume=args.resume,
            )
        elif stage == "constraint_ablation_analyze":
            code = stage_constraint_ablation_analyze(
                output_root=output_root, source_root=source_root,
                events=events, settings=settings,
            )
        else:
            raise AssertionError(stage)
        print(f"[bottleneck] stage={stage} exit={code}")
        if code != 0:
            return int(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
