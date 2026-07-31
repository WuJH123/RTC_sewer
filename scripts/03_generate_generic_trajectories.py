from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from pandas.errors import EmptyDataError

from sewerrtc.control.actuator_scope import select_actuators_for_scope
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.simulation.kpi_metrics import compute_kpis
from sewerrtc.simulation.pyswmm_runner import run_swmm_trajectory


def _safe_read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _run_job(job: dict) -> dict:
    row = run_swmm_trajectory(
        job["event_inp"],
        job["policy"],
        job["actuators"],
        job["priority_nodes"],
        job["detail"],
        job["event_id"],
        job["duration_min"],
        job["control_step_sec"],
        job["seed"],
        max_steps=job["max_steps"],
        simulation_duration_min=job["simulation_duration_min"],
        recession_min=job["recession_min"],
        pump_control_mode=job.get("pump_control_mode", "continuous"),
        variable_speed_pump_ids=job.get("variable_speed_pump_ids"),
    )
    return {"ok": True, "row": row, "event_id": job["event_id"], "policy": job["policy"]}


def _recover_existing_detail(job: dict) -> dict:
    detail = pd.read_csv(job["detail"])
    row = compute_kpis(detail, job["priority_nodes"], dt_sec=int(job["control_step_sec"]))
    row.update(
        {
            "event_id": job["event_id"],
            "policy_id": job["policy"],
            "duration_min": job["duration_min"],
            "rain_duration_min": job["duration_min"],
            "recession_min": job["recession_min"],
            "simulation_duration_min": job["simulation_duration_min"],
            "detail_file": job["detail"],
            "rows": len(detail),
            "wall_time_sec": 0.0,
            "recovered_from_detail": True,
        }
    )
    return row


def _balanced_policies_for_event(
    all_policies: list[str], core_policies: list[str], exploration_per_event: int, event_index: int
) -> list[str]:
    core = [p for p in all_policies if p in set(core_policies)]
    exploration = [p for p in all_policies if p not in set(core)]
    k = min(len(exploration), max(0, int(exploration_per_event)))
    if not exploration or k >= len(exploration):
        return core + exploration
    start = (int(event_index) * k) % len(exploration)
    selected = [exploration[(start + j) % len(exploration)] for j in range(k)]
    return core + selected


def _scope_manifest(actuator_scope: str, actuators: pd.DataFrame) -> dict:
    """Describe the action schema that produced a trajectory bank."""
    actuator_ids = actuators["actuator_id"].astype(str).tolist()
    return {
        "actuator_scope": str(actuator_scope),
        "actuator_count": int(len(actuator_ids)),
        "actuator_ids": actuator_ids,
    }


def _validate_or_write_scope_manifest(out_root: Path, manifest: dict, resume: bool) -> None:
    """Prevent resume from mixing trajectory details with incompatible actions."""
    path = out_root / "action_scope_manifest.json"
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unreadable trajectory action-scope manifest: {path}") from exc
        if previous != manifest and resume:
            raise ValueError(
                "Trajectory bank action schema differs from the requested controller scope. "
                "Use a distinct outputs.data_bank_train directory or regenerate without --resume. "
                f"existing={previous.get('actuator_scope')}:{previous.get('actuator_count')} "
                f"requested={manifest['actuator_scope']}:{manifest['actuator_count']}"
            )
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--mode", choices=["debug", "full"], default="debug")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-trajectories", type=int, default=-1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_root = ensure_dir(cfg_path(cfg, "outputs.data_bank_train"))
    traj_dir = ensure_dir(out_root / "trajectories")
    inp_dir = ensure_dir(out_root / "event_inp")
    rain_table = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv")
    policies = list(cfg["trajectory"]["debug_policies"] if args.mode == "debug" else cfg["trajectory"]["full_policies"])
    core_policies = list(cfg["trajectory"].get("full_core_policies", policies))
    exploration_per_event = int(cfg["trajectory"].get("full_exploration_policies_per_event", len(policies)))
    max_traj = args.max_trajectories
    if max_traj < 0:
        max_traj = int(cfg["trajectory"]["debug_max_trajectories"] if args.mode == "debug" else cfg["trajectory"]["full_max_trajectories"])
    actuators = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    actuator_scope = str((cfg.get("controller", {}) or {}).get("actuator_scope", "existing_rtc"))
    actuators = select_actuators_for_scope(actuators, actuator_scope)
    if actuators.empty:
        raise ValueError(f"No actuators available for controller.actuator_scope={actuator_scope}")
    _validate_or_write_scope_manifest(out_root, _scope_manifest(actuator_scope, actuators), args.resume)
    priority_nodes = (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines()
    summary_path = out_root / "summary.csv"
    failed_path = out_root / "failed_trajectories.csv"
    existing = _safe_read_csv(summary_path)
    existing_keys = set()
    if not existing.empty and {"event_id", "policy_id"}.issubset(existing.columns):
        existing_keys = set(zip(existing["event_id"].astype(str), existing["policy_id"].astype(str)))
    rows, failures, jobs = [], [], []
    count = 0
    schedule_rows = []
    for event_index, (_, ev) in enumerate(rain_table.iterrows()):
        event_policies = (
            policies
            if args.mode == "debug"
            else _balanced_policies_for_event(policies, core_policies, exploration_per_event, event_index)
        )
        event_inp = inp_dir / f"{ev['event_id']}__no_controls.inp"
        if not event_inp.exists():
            mutate_inp_for_event(
                cfg_path(cfg, "network.inp"),
                ev["rainfall_csv"],
                event_inp,
                int(ev["simulation_duration_min"]),
                strip_controls=True,
            )
        for policy in event_policies:
            if max_traj and count >= max_traj:
                break
            detail = traj_dir / f"{ev['event_id']}__{policy}_detail.csv"
            schedule_rows.append({"event_id": str(ev["event_id"]), "policy_id": str(policy), "detail_file": str(detail)})
            job = (
                {
                    "event_inp": str(event_inp),
                    "policy": str(policy),
                    "actuators": actuators,
                    "priority_nodes": priority_nodes,
                    "detail": str(detail),
                    "event_id": str(ev["event_id"]),
                    "duration_min": int(ev["duration_min"]),
                    "recession_min": int(ev.get("recession_min", cfg["experiment"]["recession_min"])),
                    "simulation_duration_min": int(ev["simulation_duration_min"]),
                    "control_step_sec": int(cfg["experiment"]["control_step_sec"]),
                    "seed": int(cfg["experiment"]["random_seed"]) + count,
                    "max_steps": int(args.max_steps),
                    "pump_control_mode": str((cfg.get("controller", {}) or {}).get("pump_control_mode", "continuous")),
                    "variable_speed_pump_ids": list((cfg.get("controller", {}) or {}).get("variable_speed_pump_ids", [])),
                }
            )
            if args.resume and detail.exists():
                if (str(ev["event_id"]), str(policy)) in existing_keys:
                    continue
                try:
                    rows.append(_recover_existing_detail(job))
                    print(f"[trajectory] recovered {ev['event_id']} {policy} from existing detail")
                    count += 1
                    continue
                except Exception as exc:
                    print(f"[trajectory] could not recover {ev['event_id']} {policy}: {exc}; rerunning")
            jobs.append(job)
            count += 1
        if max_traj and count >= max_traj:
            break
    pd.DataFrame(schedule_rows).to_csv(out_root / "trajectory_schedule.csv", index=False)
    workers = max(1, int(args.workers))
    print(f"[trajectory] pending={len(jobs)} workers={workers}")
    if workers <= 1:
        for job in jobs:
            try:
                result = _run_job(job)
                rows.append(result["row"])
                print(f"[trajectory] ok {result['event_id']} {result['policy']}")
            except Exception as exc:
                failures.append({"event_id": job["event_id"], "policy_id": job["policy"], "error": repr(exc)})
                print(f"[trajectory] failed {job['event_id']} {job['policy']}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_job, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                job = futs[fut]
                done += 1
                try:
                    result = fut.result()
                    rows.append(result["row"])
                    print(f"[trajectory] done={done}/{len(jobs)} ok {result['event_id']} {result['policy']}")
                except Exception as exc:
                    failures.append({"event_id": job["event_id"], "policy_id": job["policy"], "error": repr(exc)})
                    print(f"[trajectory] done={done}/{len(jobs)} failed {job['event_id']} {job['policy']}: {exc}")
    new_summary = pd.DataFrame(rows)
    summary = pd.concat([existing, new_summary], ignore_index=True)
    if not summary.empty and {"event_id", "policy_id"}.issubset(summary.columns):
        summary = summary.drop_duplicates(subset=["event_id", "policy_id"], keep="last")
        summary.to_csv(summary_path, index=False)
    old_failures = _safe_read_csv(failed_path)
    all_failures = pd.concat([old_failures, pd.DataFrame(failures)], ignore_index=True)
    all_failures.to_csv(failed_path, index=False)
    report = {
        "mode": args.mode,
        "pending_jobs": len(jobs),
        "new_trajectories": len(rows),
        "summary_rows_total": int(len(summary)),
        "new_failures": len(failures),
        "failures_total": int(len(all_failures)),
        "workers": workers,
        "scheduled_trajectories": int(len(schedule_rows)),
        "core_policies": core_policies if args.mode == "full" else [],
        "exploration_policies_per_event": exploration_per_event if args.mode == "full" else 0,
        "out": str(out_root),
    }
    (out_root / "data_generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
