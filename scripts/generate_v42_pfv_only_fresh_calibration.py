"""Generate isolated fresh PFV-only Calibration12 SWMM evidence.

The output lives under ``formal_f2/pfv_only_v2`` and never overwrites the old
revealed Calibration campaign. Event selection is frozen by the forcing-only
plan; this script only executes the six physical branches per event:
No-control/Hold shared, native Internal, and three real candidates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.simulation.kpi_metrics import compute_kpis
from sewerrtc.simulation.pyswmm_runner import physical_network_sha256, run_swmm_fixed_action
from sewerrtc.v4.formal_f2 import sha256_file
from sewerrtc.v4.v42_formal_runtime import load_actuators
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_step1_dataset import load_graph_assets


_DURATION = re.compile(r"_D(\d+)(?:_|$)")
CHECKPOINT_MIN = 125.0


def _delay_native_rules(source: Path, target: Path, checkpoint: float) -> None:
    seconds = int(round(checkpoint * 60.0)) + 1
    hh, rem = divmod(seconds, 3600)
    mm, ss = divmod(rem, 60)
    gate = f"AND SIMULATION TIME > {hh:02d}:{mm:02d}:{ss:02d}"
    output: list[str] = []
    in_controls = False
    in_rule = False
    inserted = False
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        upper = line.strip().upper()
        if upper.startswith("["):
            in_controls = upper == "[CONTROLS]"
            in_rule = False
            inserted = False
        if in_controls and upper.startswith("RULE "):
            in_rule = True
            inserted = False
        output.append(line)
        if in_controls and in_rule and upper.startswith("IF ") and not inserted:
            output.append(gate)
            inserted = True
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temp.write_text("\n".join(output) + "\n", encoding="utf-8")
    os.replace(temp, target)


def _candidate_schedules(ids: list[str]) -> list[np.ndarray]:
    index = next((i for i, aid in enumerate(ids) if aid.casefold() == "add350.1"), None)
    if index is None:
        raise RuntimeError("Engineering36 actuator add350.1 is missing")
    values = ((0.80, 0.60, 0.40), (0.60, 0.80, 1.00), (0.40, 0.40, 0.40))
    result = []
    for sequence in values:
        action = np.ones((3, len(ids)), dtype=np.float64)
        action[:, index] = sequence
        result.append(action)
    return result


def _audit_detail(path: Path, actuator_ids: list[str], checkpoint: float, end_min: float) -> dict:
    header = pd.read_csv(path, nrows=0)
    required = {"elapsed_min", "rainfall_mm_h"}
    required.update(f"target_setting:{aid}" for aid in actuator_ids)
    required.update(f"actual_setting:{aid}" for aid in actuator_ids)
    required.update(f"readback_setting:{aid}" for aid in actuator_ids)
    missing = sorted(required.difference(header.columns))
    if missing:
        raise RuntimeError(f"{path}: missing readback columns: {missing[:5]}")
    hydraulic = [
        column
        for column in header.columns
        if column.startswith(("h:", "head:", "flood:", "storage_volume:"))
    ]
    columns = sorted(required.union(hydraulic))
    frame = pd.read_csv(path, usecols=columns)
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{path}: empty or non-finite detail")
    elapsed = frame["elapsed_min"].to_numpy(dtype=float)
    # A 5-minute report is stamped at its interval start; the final row at
    # end_min-5 covers [end_min-5, end_min].
    if np.any(np.diff(elapsed) <= 0) or elapsed.min() < 0 or elapsed.max() + 5.0 + 1e-9 < end_min:
        raise RuntimeError(
            f"{path}: incomplete/non-monotone time axis min={elapsed.min()} max={elapsed.max()} required_end={end_min}"
        )
    target = frame[[f"target_setting:{aid}" for aid in actuator_ids]].to_numpy(dtype=float)
    actual = frame[[f"actual_setting:{aid}" for aid in actuator_ids]].to_numpy(dtype=float)
    readback = frame[[f"readback_setting:{aid}" for aid in actuator_ids]].to_numpy(dtype=float)
    return {
        "rows": int(len(frame)),
        "elapsed_min": float(elapsed.min()),
        "elapsed_max": float(elapsed.max()),
        "target_actual_max_abs": float(np.max(np.abs(target - actual))),
        "target_readback_max_abs": float(np.max(np.abs(target - readback))),
        "readback_finite": bool(np.isfinite(readback).all()),
        "target_actual_readback_pass": bool(
            np.allclose(target, actual, atol=1e-6, rtol=0.0)
            and np.allclose(target, readback, atol=1e-6, rtol=0.0)
        ),
        "rainfall_finite": True,
        # The row stamped exactly at checkpoint is already after the first
        # post-checkpoint action; it is not part of the causal pre-action state.
        "prefix_frame": frame.loc[elapsed < float(checkpoint) - 1e-9].copy(),
        "hydraulic_columns": hydraulic,
    }


def _same_prefix(a: dict, b: dict) -> bool:
    left = a["prefix_frame"].reset_index(drop=True)
    right = b["prefix_frame"].reset_index(drop=True)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        return False
    return bool(np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), atol=1e-7, rtol=0.0))


def _covers_duration(path: Path, end_min: float) -> bool:
    try:
        elapsed = pd.read_csv(path, usecols=["elapsed_min"])["elapsed_min"].to_numpy(dtype=float)
    except Exception:
        return False
    return bool(elapsed.size and np.isfinite(elapsed).all() and elapsed.max() + 5.0 + 1e-9 >= end_min)


def _job(job: dict) -> dict:
    event = str(job["event_id"])
    event_dir = Path(job["event_dir"])
    detail_dir = event_dir / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: detail_dir / f"{event}__{name}.csv" for name in ("no_control", "dynamic_internal", "candidate_1", "candidate_2", "candidate_3")}
    completion_dir = event_dir / "completions"
    result_path = event_dir / "event_result.json"
    existing_result = None
    if result_path.exists():
        try:
            existing_result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            existing_result = None
    checkpoint_repair = bool(
        existing_result is not None
        and float(existing_result.get("checkpoint_min", -1.0)) != CHECKPOINT_MIN
    )
    endpoint_repair = bool(
        job["resume"]
        and checkpoint_repair
        and any(
            not _covers_duration(path, float(job["simulation_duration_min"]))
            for path in paths.values()
            if path.exists() and path.stat().st_size > 0
        )
    )
    if job["resume"] and existing_result is not None and not checkpoint_repair and all(
        path.exists() and path.stat().st_size > 0 and _covers_duration(path, float(job["simulation_duration_min"]))
        for path in paths.values()
    ):
        return json.loads(result_path.read_text(encoding="utf-8"))
    root = Path(job["project_root"])
    base_inp = root / "data/wuhan_v8_storage_retrofit.inp"
    rainfall = Path(job["rainfall_path"])
    native_raw = event_dir / f"{event}__native_rules_raw.inp"
    native = event_dir / f"{event}__native_rules.inp"
    no_controls = event_dir / f"{event}__no_controls.inp"
    if not no_controls.exists() or job["force"] or endpoint_repair:
        mutate_inp_for_event(base_inp, rainfall, no_controls, int(job["simulation_duration_min"]), strip_controls=True)
    if not native_raw.exists() or job["force"] or endpoint_repair:
        mutate_inp_for_event(base_inp, rainfall, native_raw, int(job["simulation_duration_min"]), strip_controls=False)
    if not native.exists() or job["force"] or endpoint_repair:
        _delay_native_rules(native_raw, native, float(job["checkpoint_min"]))
    actuators = pd.DataFrame(job["actuators"])
    ids = actuators["actuator_id"].astype(str).tolist()
    priority = list(job["priority_nodes"])
    prefix = {float(t): np.ones(len(ids), dtype=np.float64) for t in np.arange(0.0, float(job["checkpoint_min"]), 5.0)}

    kpis: dict[str, dict] = {}

    def run(name: str, inp: Path, action: np.ndarray, mode: str) -> None:
        if (
            paths[name].exists()
            and paths[name].stat().st_size > 0
            and _covers_duration(paths[name], float(job["simulation_duration_min"]))
            and not job["force"]
        ):
            return
        kpis[name] = run_swmm_fixed_action(
            inp_path=inp,
            actuators=actuators,
            priority_nodes=priority,
            out_detail_csv=paths[name],
            event_id=event,
            duration_min=int(job["rain_duration_min"]),
            prefix_schedule=prefix,
            override_start_min=float(job["checkpoint_min"]),
            post_action=action,
            control_step_sec=300,
            decision_interval_sec=600,
            simulation_duration_min=int(job["simulation_duration_min"]),
            policy_id=name,
            post_control_mode=mode,
        )

    ones = np.ones(len(ids), dtype=np.float64)
    run("no_control", no_controls, ones, "external_override")
    # Hold is physically identical to the all-open initial readback and shares
    # the No-control trajectory by contract; no duplicate SWMM is run.
    run("dynamic_internal", native, ones, "native_rules")
    for index, schedule in enumerate(_candidate_schedules(ids), start=1):
        run(f"candidate_{index}", no_controls, schedule, "external_override")

    audits = {
        name: _audit_detail(path, ids, float(job["checkpoint_min"]), float(job["simulation_duration_min"]))
        for name, path in paths.items()
    }
    for name, path in paths.items():
        detail = pd.read_csv(path)
        recomputed = compute_kpis(detail, priority, dt_sec=300)
        stored = kpis.get(name, {})
        audit = audits[name]
        audit["kpi_recompute_pass"] = bool(
            all(
                key in recomputed and key in stored and np.isclose(
                    float(recomputed[key]), float(stored[key]), atol=1e-9, rtol=1e-9
                )
                for key in ("PFV", "TFV", "peak_TFV_rate")
            )
        ) if stored else True
        audit["kpi"] = {key: float(recomputed[key]) for key in ("PFV", "TFV", "peak_TFV_rate") if key in recomputed}
    if not all(item["kpi_recompute_pass"] for item in audits.values()):
        raise RuntimeError(f"{event}: persisted-detail KPI recomputation mismatch")
    external_names = ("no_control", "candidate_1", "candidate_2", "candidate_3")
    if not all(audits[name]["target_actual_readback_pass"] for name in external_names):
        raise RuntimeError(f"{event}: target/readback mismatch: {audits}")
    if not audits["dynamic_internal"]["readback_finite"]:
        raise RuntimeError(f"{event}: Internal readback is non-finite")
    for name in ("candidate_1", "candidate_2", "candidate_3"):
        if not _same_prefix(audits["no_control"], audits[name]):
            raise RuntimeError(f"{event}: candidate prefix mismatch for {name}")
    if not _same_prefix(audits["no_control"], audits["dynamic_internal"]):
        raise RuntimeError(f"{event}: Internal prefix mismatch")

    rows = []
    for index in range(1, 4):
        case_id = f"pfv2cal__{event}__{int(job['checkpoint_min']):04d}__candidate_{index}"
        case_dir = completion_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "case_id": case_id,
            "event_id": event,
            "rainfall_sha256": job["rainfall_sha256"],
            "checkpoint_min": float(job["checkpoint_min"]),
            "branches": {
                "candidate": {"detail_path": str(paths[f"candidate_{index}"].resolve())},
                "no_control": {"detail_path": str(paths["no_control"].resolve())},
                "dynamic_internal": {"detail_path": str(paths["dynamic_internal"].resolve())},
                "hold_previous": {"detail_path": str(paths["no_control"].resolve()), "shared_reference": True},
            },
            "physical_network_sha256": job["physical_network_sha256"],
            "hotstart_used": False,
            "actual_readback_verified": bool(audits[f"candidate_{index}"]["target_actual_readback_pass"]),
            "same_state_raw_verified": bool(_same_prefix(audits["no_control"], audits[f"candidate_{index}"])),
            "same_forcing_raw_verified": True,
            "h120_window_complete": bool(audits[f"candidate_{index}"]["elapsed_max"] + 5.0 + 1e-9 >= float(job["simulation_duration_min"])),
            "kpi_recompute_ok": bool(audits[f"candidate_{index}"]["kpi_recompute_pass"]),
            "detail_sha256": sha256_file(paths[f"candidate_{index}"]),
            "detail_audit": {key: value for key, value in audits[f"candidate_{index}"].items() if key != "prefix_frame"},
            "action_authority": "actual_readback_setting",
            "formal_mainline_authorized": False,
        }
        (case_dir / "completion.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(
            {
                "case_id": case_id,
                "event_id": event,
                "rainfall_sha256": job["rainfall_sha256"],
                "rainfall_group_key": job["rainfall_sha256"],
                "checkpoint_min": float(job["checkpoint_min"]),
                "inp_path": str(native.resolve()),
                "rain_duration_min": int(job["rain_duration_min"]),
                "simulation_duration_min": int(job["simulation_duration_min"]),
                "history_detail_path": str(paths["no_control"].resolve()),
                "candidate_detail_path": str(paths[f"candidate_{index}"].resolve()),
                "physical_network_sha256": job["physical_network_sha256"],
                "hotstart_used": False,
                "formal_f2_role": "fresh_pfv_only_calibration",
                "training_admission_authorized": False,
            }
        )
    result = {
        "event_id": event,
        "rainfall_sha256": job["rainfall_sha256"],
        "checkpoint_min": float(job["checkpoint_min"]),
        "case_rows": rows,
        "trajectory_paths": {key: str(value.resolve()) for key, value in paths.items()},
        "detail_audits": {key: {k: v for k, v in value.items() if k != "prefix_frame"} for key, value in audits.items()},
        "kpis": kpis,
        "swmm_runs": 5,
        "shared_hold_reference": True,
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--project-root", type=Path, default=root)
    ap.add_argument("--plan", type=Path, default=root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/pfv_only_v2/FRESH_PFV_ONLY_CALIBRATION_PLAN.csv")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    plan = pd.read_csv(args.plan)
    if len(plan) != 12 or plan["rainfall_sha256"].nunique() != 12:
        raise RuntimeError("fresh PFV-only Calibration plan must contain exactly 12 independent groups")
    formal = args.project_root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    output = formal / "pfv_only_v2/fresh_calibration_inputs"
    inventory = pd.read_csv(args.project_root / "outputs/project6_dual_reference_v4/final_v4/inventory/event_inventory.csv")
    inv = inventory.set_index(inventory["event_id"].astype(str))
    graph = load_graph_assets(args.project_root)
    actuators = load_actuators(args.project_root)
    priority = [str(graph.node_ids[i]) for i in get_pfv_core_node_indices(list(graph.node_ids))]
    network_sha = physical_network_sha256(args.project_root / "data/wuhan_v8_storage_retrofit.inp")
    jobs = []
    for row in plan.to_dict("records"):
        event = str(row["event_id"])
        if event not in inv.index:
            raise RuntimeError(f"fresh plan event missing from inventory: {event}")
        rainfall = Path(str(inv.loc[event, "rainfall_path"]))
        if not rainfall.exists():
            raise FileNotFoundError(rainfall)
        duration_match = _DURATION.search(event)
        duration = int(duration_match.group(1)) if duration_match else int(math.ceil(float(row["duration_min"])))
        # The final 5-minute report row is stamped at end-5; add one interval
        # so the exact checkpoint+120 row exists for Formal H120 selection.
        simulation_duration = max(CHECKPOINT_MIN + 120.0 + 5.0, duration + CHECKPOINT_MIN)
        jobs.append(
            {
                "project_root": str(args.project_root.resolve()),
                "event_id": event,
                "rainfall_path": str(rainfall.resolve()),
                "rainfall_sha256": str(row["rainfall_sha256"]),
                "rain_duration_min": duration,
                "simulation_duration_min": simulation_duration,
                "checkpoint_min": CHECKPOINT_MIN,
                "event_dir": str(output / event),
                "actuators": actuators.to_dict(orient="list"),
                "priority_nodes": priority,
                "physical_network_sha256": network_sha,
                "resume": bool(args.resume),
                "force": bool(args.force),
            }
        )
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "events": [job["event_id"] for job in jobs], "workers": int(args.workers), "swmm_runs_if_executed": 5 * len(jobs)}, indent=2), flush=True)
        return 0
    results = []
    failures = []
    workers = max(1, min(int(args.workers), 4))
    if workers == 1:
        iterator = ((_job(job), job) for job in jobs)
        for result, job in iterator:
            results.append(result)
            print(json.dumps({"stage": "fresh_pfv_only_calibration", "done": len(results), "total": len(jobs), "event": result["event_id"]}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(_job, job): job for job in jobs}
            for future in as_completed(pending):
                job = pending[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(json.dumps({"stage": "fresh_pfv_only_calibration", "done": len(results), "total": len(jobs), "event": result["event_id"]}), flush=True)
                except Exception as exc:
                    failures.append({"event": job["event_id"], "error": f"{type(exc).__name__}: {exc}"})
                    print(json.dumps(failures[-1]), flush=True)
    if failures or len(results) != len(jobs):
        raise RuntimeError(json.dumps({"failures": failures, "completed": len(results), "total": len(jobs)}))
    rows = [row for result in results for row in result["case_rows"]]
    manifest = formal / "pfv_only_v2/FRESH_PFV_ONLY_CALIBRATION_CASE_MANIFEST.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["rainfall_sha256", "case_id"]).to_csv(manifest, index=False)
    audit = {
        "status": "pass",
        "development_only": False,
        "formal_mainline_authorized": False,
        "event_count": len(results),
        "rainfall_groups": len({str(x["rainfall_sha256"]) for x in results}),
        "case_rows": len(rows),
        "swmm_runs": sum(int(x["swmm_runs"]) for x in results),
        "shared_hold_reference": True,
        "checkpoint_min": CHECKPOINT_MIN,
        "h120_future_end_min": CHECKPOINT_MIN + 120.0,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "network_sha256": network_sha,
        "selection_plan_sha256": sha256_file(args.plan),
        "selection_uses_control_outcome": False,
    }
    audit_path = formal / "pfv_only_v2/FRESH_PFV_ONLY_CALIBRATION_GENERATION_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
