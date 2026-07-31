from __future__ import annotations

import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc._project_root import PROJECT_ROOT
from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.simulation.pyswmm_runner import run_swmm_trajectory
from sewerrtc.state.gat_audit import GAT_CANDIDATES, load_checkpoint, metadata_from_checkpoint, sha256_file
from sewerrtc.state.gat_independent_validation import EXPECTED_SR0P15_SHA256


RETROFIT_INP = PROJECT_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"
ACTUATOR_IDS = PROJECT_ROOT / "data" / "project6_v8_storage_retrofit_control_enabled_ids.txt"
FACILITY_SEMANTICS = PROJECT_ROOT / "data" / "project6_v3_facility_semantics_36.csv"
PRIORITY_NODES = PROJECT_ROOT / "outputs" / "design" / "priority_nodes.txt"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _event_duration_min(event_id: str, fallback: int = 180) -> int:
    match = re.search(r"_D(\d+)_", str(event_id))
    return int(match.group(1)) if match else int(fallback)


def _storm_family(event_id: str) -> str:
    match = re.match(r"^(T[^_]+)_D\d+_(.+)$", str(event_id))
    return f"{match.group(1)}_{match.group(2)}" if match else str(event_id)


def _sha_if_file(value: str) -> str:
    path = Path(str(value or ""))
    return sha256_file(path) if str(value or "").strip() and path.exists() and path.is_file() else ""


def _actuators() -> pd.DataFrame:
    ids = [line.strip() for line in ACTUATOR_IDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    type_by_id: dict[str, str] = {}
    if FACILITY_SEMANTICS.exists():
        sem = pd.read_csv(FACILITY_SEMANTICS)
        id_col = "facility_id" if "facility_id" in sem.columns else ("actuator_id" if "actuator_id" in sem.columns else "")
        type_col = "actuator_type" if "actuator_type" in sem.columns else ("facility_type" if "facility_type" in sem.columns else "")
        if id_col and type_col:
            type_by_id = sem.set_index(id_col)[type_col].astype(str).str.lower().to_dict()
    return pd.DataFrame(
        {
            "actuator_id": ids,
            "link_type": [type_by_id.get(aid, "pump" if aid in {"add350.1", "ADD301.2", "ADD301.3"} else "orifice") for aid in ids],
        }
    )


def _priority_nodes() -> list[str]:
    if PRIORITY_NODES.exists():
        return [line.strip() for line in PRIORITY_NODES.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []


def _sr0p15_metadata() -> tuple[dict[str, Any], Path]:
    for name, ratio, path in GAT_CANDIDATES:
        if name == "sr0p15":
            loaded = load_checkpoint(name, ratio, path)
            if loaded.checkpoint is None:
                raise RuntimeError(f"sr0p15 checkpoint load failed: {loaded.load_error}")
            actual = sha256_file(path)
            if actual != EXPECTED_SR0P15_SHA256:
                raise RuntimeError(f"sr0p15 checkpoint hash mismatch: {actual}")
            return metadata_from_checkpoint(loaded), path
    raise RuntimeError("sr0p15 checkpoint is not configured")


def _job(job: dict[str, Any]) -> dict[str, Any]:
    policy = str(job["policy_id"])
    strip_controls = policy == "no_control"
    mutate_inp_for_event(
        job["network_path"],
        job["rainfall_path"],
        job["event_inp"],
        int(job["simulation_duration_min"]),
        strip_controls=strip_controls,
    )
    row = run_swmm_trajectory(
        job["event_inp"],
        policy,
        pd.DataFrame(job["actuators"]),
        list(job["priority_nodes"]),
        job["detail_file"],
        job["event_id"],
        int(job["duration_min"]),
        int(job["control_step_sec"]),
        int(job["seed"]),
        max_steps=int(job["max_steps"]),
        simulation_duration_min=int(job["simulation_duration_min"]),
        recession_min=int(job["tail_min"]),
        pump_control_mode="binary_unless_verified",
        variable_speed_pump_ids=["add350.1"],
    )
    return {"status": "completed", **row}


def _selected_plan_rows(plan_path: Path, max_events: int) -> list[dict[str, str]]:
    rows = [row for row in _read_csv(plan_path) if row.get("rainfall_path") and Path(row["rainfall_path"]).exists()]
    seen: set[str] = set()
    selected: list[dict[str, str]] = []
    for row in rows:
        event = str(row.get("event_id", ""))
        family = str(row.get("storm_family_id", ""))
        if not event or event in seen:
            continue
        seen.add(event)
        selected.append(row)
        if max_events and len(selected) >= int(max_events):
            break
    return selected


def generate_holdout_trajectories(
    *,
    plan_path: Path,
    out_dir: Path,
    max_events: int = 0,
    policies: list[str] | None = None,
    workers: int = 1,
    tail_min: int = 180,
    control_step_sec: int = 600,
    max_steps: int = 0,
    resume: bool = False,
) -> dict[str, Any]:
    policies = policies or ["no_control"]
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = out_dir / "trajectories"
    inp_dir = out_dir / "event_inp"
    cache_dir = out_dir / "sr0p15_cache"
    detail_dir.mkdir(parents=True, exist_ok=True)
    inp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected_plan_rows(plan_path, max_events)
    if not selected:
        _write_json(out_dir / "gat_holdout_generation_report.json", {"status": "blocked", "failure_reason": "no_planned_holdout_rows", "plan_path": str(plan_path)})
        return {"status": "blocked", "exit_code": 3}

    actuators = _actuators()
    priority = _priority_nodes()
    jobs: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    seed = 20260717
    for row in selected:
        duration = _event_duration_min(row["event_id"])
        sim_duration = duration + int(tail_min)
        for policy in policies:
            detail = detail_dir / f"{row['event_id']}__{policy}_detail.csv"
            schedule.append(
                {
                    "event_id": row["event_id"],
                    "storm_family_id": row.get("storm_family_id") or _storm_family(row["event_id"]),
                    "policy_id": policy,
                    "rainfall_path": row["rainfall_path"],
                    "detail_file": str(detail),
                    "duration_min": duration,
                    "simulation_duration_min": sim_duration,
                }
            )
            if resume and detail.exists():
                continue
            jobs.append(
                {
                    "network_path": str(RETROFIT_INP),
                    "rainfall_path": row["rainfall_path"],
                    "event_inp": str(inp_dir / f"{row['event_id']}__{policy}.inp"),
                    "detail_file": str(detail),
                    "event_id": row["event_id"],
                    "policy_id": policy,
                    "duration_min": duration,
                    "tail_min": int(tail_min),
                    "simulation_duration_min": sim_duration,
                    "control_step_sec": int(control_step_sec),
                    "max_steps": int(max_steps),
                    "seed": seed + len(jobs),
                    "actuators": actuators.to_dict(orient="records"),
                    "priority_nodes": priority,
                }
            )
    _write_csv(out_dir / "gat_independent_holdout_trajectory_schedule.csv", schedule)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if workers <= 1:
        for item in jobs:
            try:
                rows.append(_job(item))
            except Exception as exc:
                failures.append({"event_id": item["event_id"], "policy_id": item["policy_id"], "error": repr(exc)})
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {pool.submit(_job, item): item for item in jobs}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    failures.append({"event_id": item["event_id"], "policy_id": item["policy_id"], "error": repr(exc)})
    existing_details = [r for r in schedule if Path(r["detail_file"]).exists()]
    summary_rows = rows + [
        {"event_id": r["event_id"], "policy_id": r["policy_id"], "detail_file": r["detail_file"], "recovered_existing_detail": True}
        for r in existing_details
        if not any(str(x.get("detail_file", "")) == r["detail_file"] for x in rows)
    ]
    _write_csv(out_dir / "gat_independent_holdout_summary.csv", summary_rows)
    _write_csv(out_dir / "gat_independent_holdout_failures.csv", failures)
    report = {
        "status": "completed" if existing_details else "blocked",
        "planned_events": len(selected),
        "planned_policy_trajectories": len(schedule),
        "pending_jobs_run": len(jobs),
        "completed_or_existing_trajectories": len(existing_details),
        "failures": len(failures),
        "detail_dir": str(detail_dir),
        "created_at": _now(),
    }
    _write_json(out_dir / "gat_holdout_generation_report.json", report)
    return report


def build_sr0p15_holdout_cache(
    *,
    holdout_dir: Path,
    manifest_path: Path,
    time_stride: int = 1,
    min_history_min: int = 60,
) -> dict[str, Any]:
    metadata, checkpoint_path = _sr0p15_metadata()
    node_ids = [str(x) for x in metadata.get("node_ids") or []]
    sensor_ids = set(str(x) for x in metadata.get("sensor_ids") or [])
    if not node_ids or not sensor_ids:
        raise RuntimeError("sr0p15 metadata is missing node_ids or sensor_ids")

    detail_files = sorted((holdout_dir / "trajectories").glob("*_detail.csv"))
    schedule_rows = _read_csv(holdout_dir / "gat_independent_holdout_trajectory_schedule.csv")
    schedule_by_event = {row.get("event_id", ""): row for row in schedule_rows}
    state_rows: list[np.ndarray] = []
    rain_rows: list[list[float]] = []
    sources: list[str] = []
    event_ids: list[str] = []
    policy_ids: list[str] = []
    elapsed_rows: list[float] = []
    rejected: list[dict[str, Any]] = []
    for detail in detail_files:
        df = pd.read_csv(detail)
        h_cols = [f"h:{node}" for node in node_ids]
        missing = [col for col in h_cols if col not in df.columns]
        if missing:
            rejected.append({"detail_file": str(detail), "reason": "missing_gat_node_depth_columns", "missing_count": len(missing)})
            continue
        if "rainfall_mm_h" not in df.columns or "elapsed_min" not in df.columns:
            rejected.append({"detail_file": str(detail), "reason": "missing_rainfall_or_elapsed"})
            continue
        event = str(df["event_id"].iloc[0]) if "event_id" in df.columns and len(df) else detail.stem.split("__", 1)[0]
        policy = str(df["policy_id"].iloc[0]) if "policy_id" in df.columns and len(df) else "unknown"
        elapsed = pd.to_numeric(df["elapsed_min"], errors="coerce").fillna(-1).to_numpy(float)
        valid_idx = np.flatnonzero(elapsed >= float(min_history_min))
        valid_idx = valid_idx[:: max(1, int(time_stride))]
        if len(valid_idx) == 0:
            rejected.append({"detail_file": str(detail), "reason": "no_rows_after_min_history"})
            continue
        state = df[h_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
        rain = pd.to_numeric(df["rainfall_mm_h"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        for idx in valid_idx:
            state_rows.append(state[int(idx)])
            rain_rows.append([float(rain[int(idx)])])
            sources.append(f"{detail.name}:{int(idx)}:gat_independent_holdout")
            event_ids.append(event)
            policy_ids.append(policy)
            elapsed_rows.append(float(elapsed[int(idx)]))
    cache_path = holdout_dir / "sr0p15_cache" / "gat_independent_holdout_sr0p15_cache.npz"
    if not state_rows:
        _write_csv(holdout_dir / "gat_independent_holdout_cache_rejected.csv", rejected)
        _write_json(holdout_dir / "gat_independent_holdout_cache_report.json", {"status": "blocked", "failure_reason": "no_valid_holdout_cache_samples", "rejected": rejected})
        return {"status": "blocked", "exit_code": 3, "cache_path": str(cache_path)}
    np.savez_compressed(
        cache_path,
        state=np.asarray(state_rows, dtype=np.float32),
        rain=np.asarray(rain_rows, dtype=np.float32),
        event_ids=np.asarray(event_ids, dtype=object),
        policy_ids=np.asarray(policy_ids, dtype=object),
        sources=np.asarray(sources, dtype=object),
        elapsed_min=np.asarray(elapsed_rows, dtype=np.float32),
        node_cols=np.asarray([f"h:{node}" for node in node_ids], dtype=object),
        sensor_ids=np.asarray(sorted(sensor_ids), dtype=object),
    )
    _write_csv(holdout_dir / "gat_independent_holdout_cache_rejected.csv", rejected)
    cache_sha = sha256_file(cache_path)
    by_event: dict[str, dict[str, Any]] = {}
    for event, policy in zip(event_ids, policy_ids):
        rec = by_event.setdefault(
            event,
            {
                "holdout_id": f"gat_independent_{len(by_event):04d}",
                "event_id": event,
                "storm_family_id": _storm_family(event),
                "source_project": "Project6_generated_from_independent_rainfall_plan",
                "split": "gat_independent_holdout",
                "rainfall_path": schedule_by_event.get(event, {}).get("rainfall_path", ""),
                "rainfall_file_sha256": _sha_if_file(schedule_by_event.get(event, {}).get("rainfall_path", "")),
                "rainfall_series_sha256": _sha_if_file(schedule_by_event.get(event, {}).get("rainfall_path", "")),
                "trajectory_path": str(holdout_dir / "trajectories"),
                "trajectory_sha256": "",
                "network_path": str(RETROFIT_INP),
                "network_sha256": sha256_file(RETROFIT_INP),
                "cache_path": str(cache_path),
                "cache_sha256": cache_sha,
                "timestamp_range": "elapsed_min>=60",
                "node_truth_path": str(cache_path),
                "sensor_input_path": str(cache_path),
                "sample_count": 0,
                "full_node_truth_available": "true",
                "sr0p15_sensor_available": "true",
                "timestamps_available": "true",
                "has_60min_history": "true",
                "high_water_support": "",
                "eligibility_evidence": str(holdout_dir / "gat_holdout_generation_report.json"),
                "exclusion_audit_hash": "",
                "policies": set(),
            },
        )
        rec["sample_count"] = int(rec["sample_count"]) + 1
        rec["policies"].add(policy)
    manifest_rows = []
    for rec in by_event.values():
        rec = dict(rec)
        rec["policies"] = ",".join(sorted(rec["policies"]))
        manifest_rows.append(rec)
    _write_csv(manifest_path, manifest_rows)
    report = {
        "status": "completed",
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha,
        "sample_count": len(state_rows),
        "event_count": len(by_event),
        "storm_family_count": len({row["storm_family_id"] for row in manifest_rows}),
        "node_count": len(node_ids),
        "sensor_count": len(sensor_ids),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest_path": str(manifest_path),
        "created_at": _now(),
    }
    _write_json(holdout_dir / "gat_independent_holdout_cache_report.json", report)
    return report


def generate_and_build_holdout(
    *,
    plan_path: Path,
    gat_dir: Path,
    max_events: int = 0,
    policies: list[str] | None = None,
    workers: int = 1,
    tail_min: int = 180,
    control_step_sec: int = 600,
    max_steps: int = 0,
    time_stride: int = 1,
    resume: bool = False,
) -> tuple[int, dict[str, Path]]:
    holdout_dir = gat_dir / "independent_holdout" / "generated_trajectories"
    manifest_path = gat_dir / "gat_independent_validation_manifest.csv"
    generation = generate_holdout_trajectories(
        plan_path=plan_path,
        out_dir=holdout_dir,
        max_events=max_events,
        policies=policies,
        workers=workers,
        tail_min=tail_min,
        control_step_sec=control_step_sec,
        max_steps=max_steps,
        resume=resume,
    )
    outputs = {
        "generation_report": holdout_dir / "gat_holdout_generation_report.json",
        "summary": holdout_dir / "gat_independent_holdout_summary.csv",
        "failures": holdout_dir / "gat_independent_holdout_failures.csv",
        "schedule": holdout_dir / "gat_independent_holdout_trajectory_schedule.csv",
        "cache_report": holdout_dir / "gat_independent_holdout_cache_report.json",
        "cache": holdout_dir / "sr0p15_cache" / "gat_independent_holdout_sr0p15_cache.npz",
        "manifest": manifest_path,
    }
    if generation.get("status") != "completed":
        return 3, outputs
    cache_report = build_sr0p15_holdout_cache(
        holdout_dir=holdout_dir,
        manifest_path=manifest_path,
        time_stride=time_stride,
    )
    return (0 if cache_report.get("status") == "completed" else 3), outputs
