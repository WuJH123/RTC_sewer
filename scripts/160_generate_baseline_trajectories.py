#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from sewerrtc.contracts.prompt3a import OUT_ROOT, PROJECT_ROOT, read_csv, sha256_file, utc_now, write_csv, write_json
from sewerrtc.simulation.baseline_trajectory import validate_frozen_baseline_plan
from sewerrtc.simulation.pyswmm_runner import run_swmm_trajectory
from sewerrtc.simulation.swmm_event_builder import build_event_inp_from_plan, duration_from_event_id
from sewerrtc.simulation.trajectory_writer import write_trajectory_schema


ACTUATOR_CSV = PROJECT_ROOT / "data" / "project6_v8_storage_retrofit_assets.csv"
FACILITY_SEMANTICS_CSV = PROJECT_ROOT / "data" / "project6_v3_facility_semantics_36.csv"
MANAGED_IDS_TXT = PROJECT_ROOT / "data" / "project6_v8_storage_retrofit_control_enabled_ids.txt"
PRIORITY_NODES = PROJECT_ROOT / "data" / "project5_design" / "priority_pfv_core_nodes.txt"


def _priority_nodes() -> list[str]:
    return [line.strip() for line in PRIORITY_NODES.read_text(encoding="utf-8").splitlines() if line.strip()]


def _managed_facility_ids() -> list[str]:
    out: list[str] = []
    for raw in MANAGED_IDS_TXT.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _load_baseline_actuators() -> pd.DataFrame:
    """Return the full 36-facility contract, preserving retrofit asset fields when present."""
    assets = pd.read_csv(ACTUATOR_CSV) if ACTUATOR_CSV.exists() else pd.DataFrame()
    if not FACILITY_SEMANTICS_CSV.exists() or not MANAGED_IDS_TXT.exists():
        return assets
    managed = _managed_facility_ids()
    semantics = pd.read_csv(FACILITY_SEMANTICS_CSV)
    if "facility_id" not in semantics or len(managed) != 36:
        return assets
    base = semantics.rename(
        columns={
            "facility_id": "actuator_id",
            "actuator_type": "link_type",
            "storage_role": "storage_control_type",
        }
    ).copy()
    base["actuator_id"] = base["actuator_id"].astype(str)
    base = base[base["actuator_id"].isin(managed)].copy()
    base["_order"] = base["actuator_id"].map({aid: i for i, aid in enumerate(managed)})
    base = base.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    if not assets.empty and "actuator_id" in assets:
        asset_map = assets.set_index("actuator_id", drop=False)
        for i, row in base.iterrows():
            aid = str(row["actuator_id"])
            if aid not in asset_map.index:
                continue
            for col, value in asset_map.loc[aid].items():
                if col not in base.columns:
                    base[col] = ""
                if pd.notna(value) and str(value) != "":
                    base.at[i, col] = value
    defaults = {
        "link_type": "",
        "control_enabled": True,
        "near_storage": False,
        "storage_control_type": "none",
        "fail_safe_setting": 1.0,
    }
    for col, value in defaults.items():
        if col not in base:
            base[col] = value
        base[col] = base[col].fillna(value)
    return base


def _select_rows(rows: list[dict[str, str]], max_events: int, policy_filter: str) -> list[dict[str, str]]:
    policies = {p.strip() for p in policy_filter.split(",") if p.strip()}
    if policies:
        rows = [row for row in rows if row.get("policy_id") in policies]
    if max_events > 0:
        selected: list[str] = []
        for row in rows:
            event_id = row.get("event_id", "")
            if event_id and event_id not in selected:
                selected.append(event_id)
            if len(selected) >= max_events:
                break
        rows = [row for row in rows if row.get("event_id", "") in set(selected)]
    return rows


def _job_from_row(row: dict[str, str], out_dir: Path, resume: bool, skip_existing: bool) -> dict[str, Any]:
    event_id = row["event_id"]
    policy_id = row["policy_id"]
    detail = out_dir / "details" / event_id / f"{event_id}__{policy_id}.csv"
    return {
        "row": row,
        "out_dir": str(out_dir),
        "detail": str(detail),
        "resume": resume,
        "skip_existing": skip_existing,
    }


def _load_existing_manifest(out_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = out_dir / "baseline_trajectory_manifest.csv"
    if not manifest.exists():
        return {}
    return {row.get("trajectory_id", ""): row for row in read_csv(manifest) if row.get("trajectory_id")}


def _result_from_existing(row: dict[str, str], out_dir: Path, detail: Path, previous: dict[str, dict[str, Any]]) -> dict[str, Any]:
    trajectory_id = row["trajectory_id"]
    if trajectory_id in previous:
        restored = dict(previous[trajectory_id])
        restored["status"] = restored.get("status") or "skipped_existing"
        restored["skip_source"] = "previous_manifest"
        return restored
    event_id = row["event_id"]
    recovery_path = out_dir / "recovery" / event_id / f"{trajectory_id}__recovery.json"
    checkpoint_manifest = out_dir / "checkpoints" / event_id / f"{trajectory_id}__checkpoint_manifest.csv"
    result: dict[str, Any] = {
        "status": "skipped_existing",
        "skip_source": "reconstructed_from_files",
        "trajectory_id": trajectory_id,
        "event_id": event_id,
        "policy_id": row["policy_id"],
        "detail_file": str(detail),
        "rainfall_path": row.get("rainfall_path", ""),
        "rainfall_series_sha256": row.get("rainfall_series_sha256", ""),
        "network_sha256": row.get("network_sha256", ""),
        "truth_controller_separation_required": row.get("truth_controller_separation_required", ""),
        "recovery_contract_path": str(recovery_path) if recovery_path.exists() else "",
        "recovery_contract_sha256": sha256_file(recovery_path) if recovery_path.exists() else "",
        "checkpoint_manifest_file": str(checkpoint_manifest) if checkpoint_manifest.exists() else "",
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest) if checkpoint_manifest.exists() else "",
        "checkpoint_count": len(read_csv(checkpoint_manifest)) if checkpoint_manifest.exists() else 0,
    }
    if recovery_path.exists():
        try:
            recovery = json.loads(recovery_path.read_text(encoding="utf-8-sig"))
            result.update(
                {
                    "recovery_criteria_met": recovery.get("recovery_criteria_met"),
                    "recovery_censored": recovery.get("recovery_censored"),
                    "actual_tail_min": recovery.get("actual_tail_min"),
                    "tail_termination_reason": recovery.get("tail_termination_reason"),
                }
            )
        except json.JSONDecodeError:
            result["status"] = "failed_existing_recovery_json_decode"
    return result


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    out_dir = Path(job["out_dir"])
    detail = Path(job["detail"])
    if (job["resume"] or job["skip_existing"]) and detail.exists() and detail.stat().st_size > 0:
        return _result_from_existing(row, out_dir, detail, job.get("previous_manifest", {}))
    built = build_event_inp_from_plan(row, out_dir)
    actuators = _load_baseline_actuators()
    result = run_swmm_trajectory(
        built["event_inp"],
        row["policy_id"],
        actuators,
        _priority_nodes(),
        detail,
        row["event_id"],
        int(built["duration_min"]),
        control_step_sec=300,
        seed=2026,
        max_steps=0,
        simulation_duration_min=int(built["simulation_duration_min"]),
        recession_min=int(built["tail_min"]),
        pump_control_mode="continuous",
        variable_speed_pump_ids=["add350.1"],
        trajectory_id=row["trajectory_id"],
        runtime_output_root=out_dir,
    )
    result.update(
        {
            "status": "completed",
            "trajectory_id": row["trajectory_id"],
            "event_inp": built["event_inp"],
            "rainfall_path": row["rainfall_path"],
            "rainfall_series_sha256": row["rainfall_series_sha256"],
            "network_sha256": row["network_sha256"],
            "truth_controller_separation_required": row["truth_controller_separation_required"],
        }
    )
    return result


def _write_failures(path: Path, failures: list[dict[str, Any]]) -> Path:
    return write_csv(path, failures, ["trajectory_id", "event_id", "policy_id", "status", "failure_reason"])


def _write_outputs(
    out_dir: Path,
    plan_path: Path,
    rows: list[dict[str, str]],
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    args: argparse.Namespace,
    started: float,
    *,
    interrupted: bool = False,
) -> dict[str, Path]:
    write_trajectory_schema(out_dir / "trajectory_schema.json")
    manifest_path = write_csv(out_dir / "baseline_trajectory_manifest.csv", results)
    recovery_rows = []
    checkpoint_rows = []
    for item in results:
        recovery_path = Path(str(item.get("recovery_contract_path", "")))
        if recovery_path.exists():
            try:
                recovery_rows.append(json.loads(recovery_path.read_text(encoding="utf-8-sig")))
            except json.JSONDecodeError:
                recovery_rows.append({"trajectory_id": item.get("trajectory_id", ""), "status": "failed_recovery_json_decode"})
        checkpoint_manifest = Path(str(item.get("checkpoint_manifest_file", "")))
        if checkpoint_manifest.exists():
            checkpoint_rows.extend(read_csv(checkpoint_manifest))
    recovery_audit_path = write_csv(out_dir / "baseline_recovery_audit.csv", recovery_rows)
    checkpoint_audit_path = write_csv(out_dir / "baseline_checkpoint_audit.csv", checkpoint_rows)
    failures_path = _write_failures(out_dir / "baseline_trajectory_failures.csv", failures)
    status_path = write_csv(
        out_dir / "baseline_trajectory_status.csv",
        [
            {
                "trajectory_id": row.get("trajectory_id", ""),
                "event_id": row.get("event_id", ""),
                "policy_id": row.get("policy_id", ""),
                "selected": True,
                "completed_or_skipped": any(result.get("trajectory_id") == row.get("trajectory_id") for result in results),
            }
            for row in rows
        ],
        ["trajectory_id", "event_id", "policy_id", "selected", "completed_or_skipped"],
    )
    completed = [row for row in results if row.get("status") in {"completed", "skipped_existing"} or row.get("detail_file")]
    quality = {
        "status": "interrupted" if interrupted else "completed" if not failures and len(completed) == len(rows) else "partial" if completed else "failed",
        "created_at": utc_now(),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "selected_trajectory_count": len(rows),
        "completed_trajectory_count": len(completed),
        "failure_count": len(failures),
        "interrupted": interrupted,
        "workers": args.workers,
        "process_id": os.getpid(),
        "wall_time_sec": time.time() - started,
        "truth_controller_separation_required": True,
        "outputs": {
            "manifest": str(manifest_path),
            "recovery_audit": str(recovery_audit_path),
            "checkpoint_audit": str(checkpoint_audit_path),
            "failures": str(failures_path),
            "status": str(status_path),
        },
        "visible_state_step_sec": 300,
        "rtc_decision_interval_sec": 600,
        "control_action_spans_visible_state_steps": 2,
        "recovery_audit_rows": len(recovery_rows),
        "checkpoint_rows": len(checkpoint_rows),
    }
    write_json(out_dir / "trajectory_quality_report.json", quality)
    write_json(out_dir / "baseline_trajectory_generation_report.json", quality)
    return {
        "manifest": manifest_path,
        "recovery_audit": recovery_audit_path,
        "checkpoint_audit": checkpoint_audit_path,
        "failures": failures_path,
        "status": status_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate baseline trajectories from the frozen Prompt3A baseline plan.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", default=str(OUT_ROOT / "baseline_trajectories" / "baseline_trajectory_plan.csv"))
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "baseline_trajectories"))
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--policy-filter", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tail-min", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--refresh-existing-only", action="store_true", help="Rebuild manifest/audits from existing detail, recovery, checkpoint and hot-start files without launching SWMM.")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    plan_path = Path(args.plan)
    valid, validation = validate_frozen_baseline_plan(plan_path)
    if not valid:
        write_json(out_dir / "baseline_trajectory_generation_report.json", validation)
        print(json.dumps(validation, indent=2, ensure_ascii=False))
        return 6 if validation.get("status") == "contract_mismatch" else 3
    if (not ACTUATOR_CSV.exists() and not FACILITY_SEMANTICS_CSV.exists()) or not PRIORITY_NODES.exists():
        report = {"status": "blocked", "failure_reason": "actuator_or_priority_contract_missing", "actuator_csv": str(ACTUATOR_CSV), "priority_nodes": str(PRIORITY_NODES)}
        write_json(out_dir / "baseline_trajectory_generation_report.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 3
    rows = _select_rows(read_csv(plan_path), args.max_events, args.policy_filter)
    if not rows:
        report = {"status": "blocked", "failure_reason": "no_plan_rows_selected", "plan": str(plan_path)}
        write_json(out_dir / "baseline_trajectory_generation_report.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 3
    if args.tail_min != 180 and any(str(row.get("tail_min", "")) != str(args.tail_min) for row in rows):
        report = {"status": "contract_mismatch", "failure_reason": "tail_min_override_not_consistent_with_plan_contract", "requested_tail_min": args.tail_min}
        write_json(out_dir / "baseline_trajectory_generation_report.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 6
    started = time.time()
    jobs = [_job_from_row(row, out_dir, args.resume, args.skip_existing) for row in rows]
    previous = _load_existing_manifest(out_dir)
    for job in jobs:
        job["previous_manifest"] = previous
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if args.refresh_existing_only:
        for job in jobs:
            detail = Path(job["detail"])
            if detail.exists() and detail.stat().st_size > 0:
                results.append(_result_from_existing(job["row"], out_dir, detail, previous))
        _write_outputs(out_dir, plan_path, rows, results, failures, args, started)
        quality = json.loads((out_dir / "trajectory_quality_report.json").read_text(encoding="utf-8-sig"))
        quality["status"] = "completed" if int(quality.get("completed_trajectory_count") or 0) == len(rows) else "partial"
        quality["refresh_existing_only"] = True
        quality["missing_existing_trajectory_count"] = len(rows) - int(quality.get("completed_trajectory_count") or 0)
        write_json(out_dir / "trajectory_quality_report.json", quality)
        write_json(out_dir / "baseline_trajectory_generation_report.json", quality)
        print(json.dumps(quality, indent=2, ensure_ascii=False))
        return 0 if quality["status"] == "completed" else 3
    executor: ProcessPoolExecutor | None = None
    try:
        if args.workers <= 1:
            for job in jobs:
                try:
                    results.append(_run_job(job))
                    _write_outputs(out_dir, plan_path, rows, results, failures, args, started)
                except Exception as exc:  # noqa: BLE001
                    failures.append({"trajectory_id": job["row"].get("trajectory_id", ""), "event_id": job["row"].get("event_id", ""), "policy_id": job["row"].get("policy_id", ""), "status": "failed", "failure_reason": str(exc)})
                    _write_outputs(out_dir, plan_path, rows, results, failures, args, started)
        else:
            executor = ProcessPoolExecutor(max_workers=int(args.workers))
            future_map = {executor.submit(_run_job, job): job for job in jobs}
            for future in as_completed(future_map):
                job = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append({"trajectory_id": job["row"].get("trajectory_id", ""), "event_id": job["row"].get("event_id", ""), "policy_id": job["row"].get("policy_id", ""), "status": "failed", "failure_reason": str(exc)})
                _write_outputs(out_dir, plan_path, rows, results, failures, args, started)
            executor.shutdown(wait=True)
            executor = None
    except KeyboardInterrupt:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        _write_outputs(out_dir, plan_path, rows, results, failures, args, started, interrupted=True)
        report = {
            "status": "interrupted",
            "completed_trajectory_count": len(results),
            "failure_count": len(failures),
            "selected_trajectory_count": len(rows),
            "exit_code": 130,
            "resume_command_required": True,
        }
        write_json(out_dir / "baseline_trajectory_generation_report.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 130
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        write_trajectory_schema(out_dir / "trajectory_schema.json")
    _write_outputs(out_dir, plan_path, rows, results, failures, args, started)
    quality = json.loads((out_dir / "trajectory_quality_report.json").read_text(encoding="utf-8-sig"))
    print(json.dumps(quality, indent=2, ensure_ascii=False))
    if failures:
        return 4
    if int(quality.get("completed_trajectory_count") or 0) != len(rows):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
