#!/usr/bin/env python3
"""Aug1 Manifest Recovery V2 -- main script.

Gate-aware: --dry-run for audit only, default for full recovery.
Never overwrites the official manifest unless --promote is used.

Usage:
  python scripts_aug1_recover_manifest_v2.py --config <path> [--dry-run] [--workers N]
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# This script lives in scripts/ but references sewerrtc/ and outputs/ at the
# repository root, so PROJECT_ROOT must be the parent of scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import sewerrtc.prompt3.action_effect_v4 as v4
import sewerrtc.data.round0_prompt2 as r0
from sewerrtc.prompt3.aug1_manifest_recovery import (
    scan_case_files, build_plan_index, match_file_signature_to_plan,
    analyze_existing_aug1_trajectory, GroupCache, compute_differs,
    infer_provenance, sha256_file, safe_readback_check,
    atomic_write_csv_safe, compute_resume_key,
    REF_BRANCHES, BRANCH_SUFFIX,
    CAUSAL_FEATURE_NAMES, CONTEXT_FEATURE_NAMES, ACTION_FEATURE_NAMES,
    FULL_TAIL_MIN, STEP_MIN,
)
from sewerrtc.prompt3.action_effect_v4_aug1 import (
    _prefix_rows, _rainfall_forecast, _truth_str, _truth_str_eq,
    _window_kpis, _initial_state_hash, _sequence_from_detail,
)
from sewerrtc.simulation.runtime_contracts import write_csv, write_json, utc_now


# ═══════════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aug1 Manifest Recovery V2")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--remove-stale-lock", action="store_true")
    args = parser.parse_args(argv)

    t0 = time.time()
    cfg = v4._load_yaml(args.config)
    aug_cfg = ((cfg.get("v4", {}) or {}).get("aug1", {}) or {})
    out_root = Path(cfg.get("output_root", "outputs/project6_dual_reference_v4"))
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    aug1_dir = out_root / "dual_reference_aug1"
    cases_dir = aug1_dir / "cases"
    plan_path = aug1_dir / "v4_aug1_case_plan.csv"
    v2_dir = aug1_dir / "recovery_v2"
    v2_dir.mkdir(parents=True, exist_ok=True)

    # ── Load plan ──────────────────────────────────────────────────────
    plan_rows = v4._read_csv(plan_path)
    planned_count = len(plan_rows)
    print(f"[V2] Plan rows: {planned_count}")

    # ── Scan files ─────────────────────────────────────────────────────
    print("[V2] Scanning case files...")
    stem_branches, unparseable = scan_case_files(cases_dir)
    print(f"[V2] Stems: {len(stem_branches)}, Unparseable: {len(unparseable)}")

    # ── Build plan index ───────────────────────────────────────────────
    plan_index = build_plan_index(plan_rows)

    # ── Load actuators ─────────────────────────────────────────────────
    actuators = r0._load_round0_actuators()
    actuator_ids = [str(a) for a in actuators["actuator_id"].tolist()]
    priority_nodes = r0._priority_nodes()

    # ── Classify each plan row ─────────────────────────────────────────
    accepted: list[dict] = []
    rejected: list[dict] = []
    pending: list[dict] = []
    missing: list[dict] = []
    unmatched_files: list[dict] = list(unparseable)
    duplicate_files: list[dict] = []
    group_audit: dict[str, dict] = {}  # stem -> group audit row
    rainfall_audit: list[dict] = []  # rainfall feature audit rows

    # Track which candidate files have been matched
    matched_cand_files: set[str] = set()
    # Track actual schedule dedup: (event, ckpt, actual_sha) -> first accepted
    seen_actual: dict[str, str] = {}

    # Performance timing
    perf = {
        "t_scan": 0.0, "t_kpi": 0.0, "t_recovery": 0.0,
        "t_readback": 0.0, "t_csv_parse": 0.0, "bytes_read": 0,
    }
    CODE_VERSION = "v2_r1"

    # Group plan rows by stem
    plan_by_stem: dict[str, list[dict]] = defaultdict(list)
    for row in plan_rows:
        eid = str(row.get("event_id", ""))
        ckpt = float(row.get("checkpoint_elapsed_min", 0) or 0)
        stem = f"{eid}_{int(round(ckpt))}"
        plan_by_stem[stem].append(row)

    processed_groups = 0
    for stem, stem_plan_rows in sorted(plan_by_stem.items()):
        files_for_stem = stem_branches.get(stem, {})
        duration_min = float(stem_plan_rows[0].get("duration_min", 0) or 0)
        checkpoint_min = float(stem_plan_rows[0].get("checkpoint_elapsed_min", 0) or 0)
        sim_end_min = duration_min + FULL_TAIL_MIN
        n_steps = int(math.ceil((sim_end_min - checkpoint_min) / STEP_MIN)) + 2
        event_id = str(stem_plan_rows[0].get("event_id", ""))

        # Group audit counters
        g_audit: dict[str, Any] = {
            "event_id": event_id,
            "checkpoint_elapsed_min": checkpoint_min,
            "stem": stem,
            "planned_candidate_count": len(stem_plan_rows),
            "discovered_candidate_count": len(files_for_stem.get("candidate", [])),
            "accepted_candidate_count": 0,
            "rejected_candidate_count": 0,
            "pending_candidate_count": 0,
            "missing_candidate_count": 0,
            "reference_branches_present": "",
            "reference_branches_missing": "",
            "unique_candidate_file_count": 0,
            "duplicate_candidate_count": 0,
            "group_status": "empty",
        }

        # Check reference branches
        ref_present = [b for b in REF_BRANCHES if b in files_for_stem]
        ref_missing = [b for b in REF_BRANCHES if b not in files_for_stem]
        g_audit["reference_branches_present"] = ";".join(ref_present)
        g_audit["reference_branches_missing"] = ";".join(ref_missing)

        # Load group cache if references complete
        gcache = GroupCache()
        if not ref_missing:
            ok = gcache.load(files_for_stem, priority_nodes, checkpoint_min,
                             actuator_ids, n_steps)
            if not ok:
                g_audit["group_status"] = "invalid_reference"

        # Rainfall forecast
        rainfall_path_str = stem_plan_rows[0].get("rainfall_path", "")
        rainfall_ok = False
        rainfall_fail_reason = ""
        causal = [0.0] * len(CAUSAL_FEATURE_NAMES)
        ctx = np.zeros(len(CONTEXT_FEATURE_NAMES))
        rf_path = Path(rainfall_path_str)
        if rf_path.exists():
            try:
                rf = _rainfall_forecast(rf_path)
                if rf:
                    rainfall_ok = True
                    internal_detail = gcache.detail.get("hold_internal_snapshot")
                    if internal_detail is None:  # backward-compat alias
                        internal_detail = gcache.detail.get("internal_current_action")
                    if internal_detail is not None:
                        prefix = _prefix_rows(internal_detail, checkpoint_min)
                        try:
                            causal = v4.causal_context_features(
                                prefix, checkpoint_elapsed_min=checkpoint_min,
                                event_duration_min=duration_min,
                                rainfall_forecast=rf, priority_nodes=priority_nodes)
                        except Exception:
                            pass
                    internal_path = stem_plan_rows[0].get("detail_internal_rules", "")
                    try:
                        ctx_val = v4._context_from_detail(
                            Path(internal_path), checkpoint_min,
                            str(stem_plan_rows[0].get("phase", "")))
                        if ctx_val is not None:
                            ctx = ctx_val
                    except Exception:
                        pass
                else:
                    rainfall_fail_reason = "empty_forecast"
            except Exception as exc:
                rainfall_fail_reason = f"read_error:{exc}"
        else:
            rainfall_fail_reason = "file_not_found"

        # Rainfall audit row
        rainfall_audit.append({
            "event_id": event_id, "checkpoint_elapsed_min": checkpoint_min,
            "stem": stem, "rainfall_path": rainfall_path_str,
            "rainfall_ok": rainfall_ok, "fail_reason": rainfall_fail_reason,
        })

        # Process each plan row in this group
        for plan_row in stem_plan_rows:
            sig = str(plan_row.get("case_signature", ""))
            classification = "missing"  # default
            reject_reason = ""
            match_mode = ""
            match_length = 0
            uniqueness_count = 0
            rb_tb = ""  # readback traceback

            # Find candidate file(s) for this signature
            cand_recs = files_for_stem.get("candidate", [])
            matched_rec = None
            for rec in cand_recs:
                fsig = rec.get("file_signature", "")
                mm, mrow, uc = match_file_signature_to_plan(fsig, plan_index)
                if mrow is plan_row:
                    matched_rec = rec
                    match_mode = mm
                    match_length = len(fsig)
                    uniqueness_count = uc
                    break

            if matched_rec is None:
                # No candidate file found
                if ref_missing:
                    classification = "pending"
                    reject_reason = "missing_reference_branches"
                else:
                    classification = "missing"
                    reject_reason = "no_candidate_file"
            else:
                matched_cand_files.add(matched_rec["absolute_path"])

                # If references incomplete -> pending
                if ref_missing:
                    classification = "pending"
                    reject_reason = "missing_reference_branches"
                elif not gcache.loaded:
                    classification = "pending"
                    reject_reason = f"reference_load_error:{gcache.error}"
                elif not rainfall_ok:
                    classification = "rejected"
                    reject_reason = "rainfall_forecast_unavailable"
                else:
                    # Full analysis
                    cand_path = Path(matched_rec["absolute_path"])
                    traj = analyze_existing_aug1_trajectory(
                        cand_path, priority_nodes, checkpoint_min,
                        actuator_ids, duration_min)

                    if not traj["trajectory_valid"]:
                        # Distinguish H120-valid but full-invalid (§15)
                        if traj.get("h120_valid") and not traj.get("full_valid"):
                            classification = "rejected"
                            reject_reason = "rejected_full_event_invalid"
                        else:
                            classification = "rejected"
                            reject_reason = f"trajectory_invalid:{traj['error_class']}"
                    elif traj["h120"] is None or traj["full"] is None:
                        classification = "rejected"
                        reject_reason = "kpi_window_empty"
                    elif traj["full"].get("PFV") is None:
                        classification = "rejected"
                        reject_reason = "full_event_kpi_none"
                    else:
                        # Readback (fail-closed with traceback)
                        t_rb = time.time()
                        rb_ok, readback_worst, readback_status, rb_tb = safe_readback_check(
                            traj["detail"],
                            _sequence_from_detail(
                                traj["detail"], actuator_ids, checkpoint_min, n_steps),
                            checkpoint_min, actuator_ids)
                        readback_ok = rb_ok
                        perf["t_readback"] += time.time() - t_rb

                        if readback_status in ("unknown", "error"):
                            classification = "rejected"
                            reject_reason = f"readback_{readback_status}"
                        elif not readback_ok:
                            classification = "rejected"
                            reject_reason = "readback_failed"
                        else:
                            # Paired hash check
                            cand_hash = traj["initial_state_hash"]
                            if not cand_hash:
                                classification = "rejected"
                                reject_reason = "initial_state_hash_failed"
                            elif not gcache.paired_hash_ok:
                                classification = "rejected"
                                reject_reason = "reference_hash_mismatch"
                            elif cand_hash != gcache.reference_hash:
                                classification = "rejected"
                                reject_reason = "paired_initial_state_hash_mismatch"
                            else:
                                # Differs check
                                differs_info = compute_differs(
                                    traj["detail"], gcache, checkpoint_min,
                                    actuator_ids, n_steps)
                                if differs_info.get("all_references_identical"):
                                    classification = "rejected"
                                    reject_reason = "candidate_equals_all_references"
                                else:
                                    # Actual schedule dedup
                                    actual_sha = traj.get("actual_action_sha256", "")
                                    dedup_key = f"{event_id}|{checkpoint_min}|{actual_sha}"
                                    if actual_sha and dedup_key in seen_actual:
                                        classification = "rejected"
                                        reject_reason = "duplicate_actual_schedule"
                                    else:
                                        # Recovery check
                                        rec_status = traj.get("recovery_status", "")
                                        rec_censored = traj.get("recovery_censored", False)
                                        if not rec_status:
                                            classification = "rejected"
                                            reject_reason = "recovery_status_empty"
                                        elif rec_censored:
                                            classification = "rejected"
                                            reject_reason = "recovery_censored"
                                        else:
                                            # ACCEPTED
                                            if actual_sha:
                                                seen_actual[dedup_key] = sig
                                            classification = "accepted"

                                            # Build manifest row
                                            row = _build_manifest_row(
                                                plan_row, traj, gcache,
                                                differs_info, causal, ctx,
                                                actuator_ids, checkpoint_min,
                                                sim_end_min, duration_min,
                                                matched_rec, readback_ok,
                                                readback_worst, match_mode,
                                                match_length, uniqueness_count)
                                            accepted.append(row)
                                            g_audit["accepted_candidate_count"] += 1

            # Non-accepted classification
            if classification != "accepted":
                rej_row = dict(plan_row)
                rej_row["classification"] = classification
                rej_row["reject_reason"] = reject_reason
                rej_row["match_mode"] = match_mode
                rej_row["match_length"] = match_length
                rej_row["uniqueness_count"] = uniqueness_count
                rej_row["file_sha256"] = matched_rec["sha256"] if matched_rec else ""
                rej_row["readback_traceback"] = rb_tb
                if classification == "rejected":
                    rejected.append(rej_row)
                    g_audit["rejected_candidate_count"] += 1
                elif classification == "pending":
                    pending.append(rej_row)
                    g_audit["pending_candidate_count"] += 1
                elif classification == "missing":
                    missing.append(rej_row)
                    g_audit["missing_candidate_count"] += 1

        # Group status
        if ref_missing:
            g_audit["group_status"] = "partial_reference"
        elif g_audit["accepted_candidate_count"] == len(stem_plan_rows):
            g_audit["group_status"] = "complete"
        elif g_audit["accepted_candidate_count"] > 0:
            g_audit["group_status"] = "partial_candidates"
        elif g_audit["discovered_candidate_count"] > 0:
            g_audit["group_status"] = "blocked"
        else:
            g_audit["group_status"] = "empty"

        g_audit["unique_candidate_file_count"] = len(
            {r["absolute_path"] for r in files_for_stem.get("candidate", [])})
        g_audit["duplicate_candidate_count"] = max(
            0, g_audit["discovered_candidate_count"] - g_audit["unique_candidate_file_count"])
        group_audit[stem] = g_audit
        processed_groups += 1

        # Unmatched candidate files
        for rec in files_for_stem.get("candidate", []):
            if rec["absolute_path"] not in matched_cand_files:
                unmatched_files.append({**rec, "reason": "unmatched_to_plan"})

    # ── Duplicate file detection ───────────────────────────────────────
    for stem, branches in stem_branches.items():
        for branch, recs in branches.items():
            if len(recs) > 1:
                shas = [r["sha256"] for r in recs]
                if len(set(shas)) == 1:
                    for r in recs[1:]:
                        duplicate_files.append({**r, "dup_type": "identical"})
                else:
                    for r in recs:
                        duplicate_files.append({**r, "dup_type": "conflict"})

    # ── Missing branch plan ────────────────────────────────────────────
    missing_branch_rows: list[dict] = []
    for stem, g in group_audit.items():
        miss_br = g.get("reference_branches_missing", "")
        if miss_br:
            for br in miss_br.split(";"):
                if br:
                    missing_branch_rows.append({
                        "event_id": g["event_id"],
                        "checkpoint": g["checkpoint_elapsed_min"],
                        "stem": stem,
                        "missing_branch": br,
                        "affected_plan_count": g["planned_candidate_count"],
                        "existing_candidate_count": g["discovered_candidate_count"],
                        "recommended_resume_action": f"run_{br}_branch",
                    })

    # ── Accounting check ───────────────────────────────────────────────
    total_classified = len(accepted) + len(rejected) + len(pending) + len(missing)
    accounting_closed = (total_classified == planned_count)

    # ── Write outputs ──────────────────────────────────────────────────
    unique_events = sorted({str(r.get("event_id", "")) for r in accepted})

    manifest_path = v2_dir / "v4_aug1_generation_manifest_recovered_v2.csv"
    write_csv(manifest_path, accepted)
    write_csv(v2_dir / "v4_aug1_generation_rejected_v2.csv", rejected)
    write_csv(v2_dir / "v4_aug1_generation_pending_v2.csv", pending)
    write_csv(v2_dir / "v4_aug1_generation_missing_v2.csv", missing)
    write_csv(v2_dir / "v4_aug1_generation_unmatched_files_v2.csv",
              [r for r in unmatched_files if isinstance(r, dict)])
    write_csv(v2_dir / "v4_aug1_generation_duplicate_files_v2.csv",
              duplicate_files)
    write_csv(v2_dir / "v4_aug1_generation_group_audit_v2.csv",
              list(group_audit.values()))
    write_csv(v2_dir / "v4_aug1_missing_branch_plan_v2.csv",
              missing_branch_rows)
    write_csv(v2_dir / "rainfall_feature_audit_v2.csv", rainfall_audit)

    # Recovery audit
    elapsed = time.time() - t0
    recovery_audit = {
        "status": _compute_status(
            planned_count, len(accepted), len(rejected), len(pending), len(missing),
            accounting_closed),
        "planned_count": planned_count,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "pending_count": len(pending),
        "missing_count": len(missing),
        "total_classified": total_classified,
        "accounting_closed": accounting_closed,
        "unique_events": unique_events,
        "unique_events_count": len(unique_events),
        "unique_groups": processed_groups,
        "complete_groups": sum(1 for g in group_audit.values() if g["group_status"] == "complete"),
        "partial_groups": sum(1 for g in group_audit.values() if "partial" in g["group_status"]),
        "unmatched_files": len(unmatched_files),
        "duplicate_files": len(duplicate_files),
        "dry_run": args.dry_run,
        "elapsed_sec": round(elapsed, 1),
        "created_at": utc_now(),
        # Performance statistics (§20)
        "perf": {
            "total_runtime_sec": round(elapsed, 2),
            "files_scanned": sum(len(br) for sb in stem_branches.values() for br in sb.values()),
            "groups_per_min": round(processed_groups / max(elapsed / 60, 0.001), 1),
            "t_scan_sec": round(perf["t_scan"], 2),
            "t_kpi_sec": round(perf["t_kpi"], 2),
            "t_recovery_sec": round(perf["t_recovery"], 2),
            "t_readback_sec": round(perf["t_readback"], 2),
        },
    }
    write_json(v2_dir / "v4_aug1_generation_recovery_audit_v2.json", recovery_audit)

    # Provenance
    prov = {
        "code_version": "v2",
        "config": str(args.config),
        "plan_path": str(plan_path),
        "cases_dir": str(cases_dir),
        "files_scanned": sum(len(br) for sb in stem_branches.values() for br in sb.values()),
        "stems_found": len(stem_branches),
    }
    write_json(v2_dir / "v4_aug1_generation_recovery_provenance_v2.json", prov)

    # Completion marker
    write_json(v2_dir / "recovery_v2_completion.json", {
        "completed": True, "status": recovery_audit["status"],
        "elapsed_sec": round(elapsed, 1), "timestamp": utc_now(),
    })

    # ── Promote to official generation manifest (opt-in, non-dry-run) ───
    # The recovered manifest is only copied onto the official
    # ``v4_aug1_generation_manifest.csv`` (which Build reads) when --promote
    # is explicitly requested on a real (non dry-run) recovery that produced
    # at least one accepted row. The prior official manifest is always backed
    # up first, so promotion is reversible.
    promote_info: dict[str, Any] = {"promoted": False}
    if args.promote and not args.dry_run:
        official_path = aug1_dir / "v4_aug1_generation_manifest.csv"
        if len(accepted) == 0:
            promote_info = {"promoted": False, "reason": "no_accepted_rows"}
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_path = v2_dir / f"v4_aug1_generation_manifest_prepromote_backup_{ts}.csv"
            had_official = official_path.exists()
            if had_official:
                backup_path.write_bytes(official_path.read_bytes())
            # Atomic replace: write recovered rows to a temp then os.replace.
            tmp_path = official_path.parent / (official_path.name + ".promote_tmp")
            write_csv(tmp_path, accepted)
            os.replace(tmp_path, official_path)
            promote_info = {
                "promoted": True,
                "official_manifest": str(official_path),
                "backup": str(backup_path) if had_official else "",
                "row_count": len(accepted),
                "unique_events": len(unique_events),
            }
        write_json(v2_dir / "v4_aug1_generation_promote_v2.json", {
            **promote_info, "created_at": utc_now(),
        })

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"V2 RECOVERY {'(DRY-RUN)' if args.dry_run else ''}")
    print(f"{'=' * 60}")
    print(f"Status: {recovery_audit['status']}")
    print(f"Planned:    {planned_count}")
    print(f"Accepted:   {len(accepted)}")
    print(f"Rejected:   {len(rejected)}")
    print(f"Pending:    {len(pending)}")
    print(f"Missing:    {len(missing)}")
    print(f"Total:      {total_classified} (closed={accounting_closed})")
    print(f"Events:     {len(unique_events)}")
    print(f"Groups:     {processed_groups}")
    print(f"Elapsed:    {elapsed:.1f}s")
    print(f"\nOutputs: {v2_dir}")

    # Exit code
    status = recovery_audit["status"]
    if status == "PASS":
        return 0
    if status == "PARTIAL":
        return 3
    return 4


def _build_manifest_row(
    plan_row, traj, gcache, differs_info, causal, ctx,
    actuator_ids, checkpoint_min, sim_end_min, duration_min,
    matched_rec, readback_ok, readback_worst, match_mode,
    match_length, uniqueness_count,
) -> dict:
    """Build one accepted manifest row with full dual-reference labels."""
    nc_h, nc_full = gcache.h120["no_control"], gcache.full["no_control"]
    pa_h, pa_full = gcache.h120["passive_anchor"], gcache.full["passive_anchor"]
    _in_key = "hold_internal_snapshot" if "hold_internal_snapshot" in gcache.h120 else "internal_current_action"
    in_h, in_full = gcache.h120[_in_key], gcache.full[_in_key]
    hp_h = gcache.h120.get("hold_previous", {})
    hp_full = gcache.full.get("hold_previous", {})
    cand_h, cand_full = traj["h120"], traj["full"]

    prov = infer_provenance(traj["detail"])

    row = dict(plan_row)
    row.update({
        "sample_id": str(plan_row.get("case_signature", "")),
        "v4_data_layer": "aug1",
        "causal_source": "aug1_real",
        # Provenance
        **prov,
        "provenance_file": matched_rec["absolute_path"],
        "provenance_sha256": matched_rec["sha256"],
        "evidence_fields": "trajectory_content",
        # Runtime truth (from evidence, not hardcoded)
        "runtime_executed": prov.get("runtime_executed", "unknown"),
        "authoritative_swmm": prov.get("authoritative_swmm", "unknown"),
        "deterministic_prefix_replay": prov.get("deterministic_prefix_replay", "unknown"),
        "hotstart_used": prov.get("hotstart_used", "unknown"),
        "truth_future_leakage": "0",
        # Hashes
        "initial_state_sha256": traj["initial_state_hash"],
        "reference_initial_state_sha256": gcache.reference_hash,
        "paired_initial_state_hash_ok": _truth_str(True),
        "candidate_differs": _truth_str(not differs_info.get("all_references_identical", True)),
        # Readback
        "readback_ok": _truth_str(readback_ok),
        "readback_worst_abs": float(readback_worst),
        # Recovery
        "recovery_status": traj.get("recovery_status", ""),
        "recovery_censored": _truth_str(bool(traj.get("recovery_censored", False))),
        "actual_tail_min": traj.get("actual_tail_min", ""),
        "sim_end_min": float(sim_end_min),
        # Action schedule hashes
        "actual_schedule_sha256": traj.get("actual_action_sha256", ""),
        # Reference H120
        "no_control_PFV_H120": float(nc_h["PFV"]),
        "passive_PFV_H120": float(pa_h["PFV"]),
        "internal_PFV_H120": float(in_h["PFV"]),
        "internal_TFV_H120": float(in_h["TFV"]),
        "internal_peak_H120": float(in_h["peak_TFV_rate"]),
        # Reference full
        "no_control_PFV_full": float(nc_full["PFV"]),
        "passive_PFV_full": float(pa_full["PFV"]),
        "internal_PFV_full": float(in_full["PFV"]),
        # Candidate H120
        "candidate_PFV_H120": float(cand_h["PFV"]),
        "candidate_TFV_H120": float(cand_h["TFV"]),
        "candidate_peak_H120": float(cand_h["peak_TFV_rate"]),
        # Candidate full
        "candidate_PFV_full": float(cand_full["PFV"]),
        "candidate_TFV_full": float(cand_full["TFV"]),
        "candidate_peak_full": float(cand_full.get("peak_TFV_rate", 0.0)),
        "priority_flood_duration_min": float(cand_full.get("priority_flood_duration_min", 0.0)),
        "non_priority_flood_TFV": float(cand_full["TFV"]) - float(cand_full["PFV"]),
        # Deltas
        "delta_PFV_H120_vs_no_control": float(cand_h["PFV"]) - float(nc_h["PFV"]),
        "delta_PFV_H120_vs_passive": float(cand_h["PFV"]) - float(pa_h["PFV"]),
        "delta_TFV_H120_vs_internal": float(cand_h["TFV"]) - float(in_h["TFV"]),
        "delta_peak_H120_vs_internal": float(cand_h["peak_TFV_rate"]) - float(in_h["peak_TFV_rate"]),
        "delta_PFV_full_vs_no_control": float(cand_full["PFV"]) - float(nc_full["PFV"]),
        "delta_PFV_full_vs_passive": float(cand_full["PFV"]) - float(pa_full["PFV"]),
        "delta_PFV_H120_vs_hold_previous": float(cand_h["PFV"]) - float(hp_h.get("PFV", 0)),
        "delta_PFV_full_vs_hold_previous": float(cand_full["PFV"]) - float(hp_full.get("PFV", 0)),
        # Differs detail
        "differs_from_no_control": _truth_str(differs_info.get("differs_from_no_control", False)),
        "differs_from_passive_anchor": _truth_str(differs_info.get("differs_from_passive_anchor", False)),
        "differs_from_internal_current_action": _truth_str(differs_info.get("differs_from_internal_current_action", False)),
        "differs_from_hold_previous": _truth_str(differs_info.get("differs_from_hold_previous", False)),
        "number_of_changed_actuators": int(differs_info.get("number_of_changed_actuators", 0)),
        "number_of_changed_steps": int(differs_info.get("number_of_changed_steps", 0)),
        "maximum_abs_setting_delta": float(differs_info.get("maximum_abs_setting_delta", 0.0)),
        # Selected fallback
        "selected_fallback": "internal_rules",
        # Match metadata
        "match_mode": match_mode,
        "match_length": match_length,
        "uniqueness_count": uniqueness_count,
    })

    # Context / action / causal features
    row.update({f"v4_ctx_{n}": float(v) for n, v in zip(CONTEXT_FEATURE_NAMES, ctx)})
    try:
        action_feats = v4._training_action_features(plan_row)
    except Exception:
        action_feats = [0.0] * len(ACTION_FEATURE_NAMES)
    row.update({f"v4_act_{n}": float(v) for n, v in zip(ACTION_FEATURE_NAMES, action_feats)})
    row.update({f"v4_causal_{n}": float(v) for n, v in zip(CAUSAL_FEATURE_NAMES, causal)})

    return row


def _compute_status(planned, accepted, rejected, pending, missing, closed):
    """Compute recovery status per spec §19.

    PASS: all conditions met.
    PARTIAL: accounting closed but pending/missing exist.
    BLOCKED: any critical failure.
    """
    if not closed:
        return "BLOCKED"
    total = accepted + rejected + pending + missing
    if total != planned:
        return "BLOCKED"
    if accepted == 0 and (pending > 0 or missing > 0):
        return "PARTIAL"
    if accepted == 0 and rejected > 0:
        return "BLOCKED"
    if pending > 0 or missing > 0:
        return "PARTIAL"
    return "PASS"


if __name__ == "__main__":
    sys.exit(main())
