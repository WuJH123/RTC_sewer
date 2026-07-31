from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
from pathlib import Path
import shutil
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from sewerrtc.experiments.targeted_joint_pairs import action_window, materialize_candidate
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.simulation.pyswmm_runner import run_swmm_no_control_action_ablation


def _profiles(spec: dict[str, Any]) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    delta_sequences = {
        str(key): [float(v) for v in value]
        for key, value in (spec.get("signed_profiles", {}) or {}).items()
    }
    target_sequences = {
        str(key): [float(v) for v in value]
        for key, value in (spec.get("target_profiles", {}) or {}).items()
    }
    return delta_sequences, target_sequences


def _run_case(job: dict[str, Any]) -> dict[str, Any]:
    source = Path(job["event_inp"])
    case_inp = Path(job["case_inp"])
    case_inp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, case_inp)
    spec = json.loads(job["executed_action_sequence"])
    reference_frame = pd.read_csv(job["reference_detail"])
    action_ids = job["actuators"]["actuator_id"].astype(str).tolist()
    reference_sequence = action_window(
        reference_frame,
        action_ids=action_ids,
        start_min=float(job["override_start_min"]),
        horizon_steps=int(job["horizon_steps"]),
    )
    runtime_sequence = materialize_candidate(reference_sequence, action_ids=action_ids, specification=spec)
    expected = np.asarray(json.loads(job["expected_action_sequence"]), dtype=np.float32)
    if tuple(runtime_sequence.shape) != tuple(expected.shape):
        raise ValueError("runtime sequence shape differs from manifest")
    delta_sequences, target_sequences = _profiles(spec)
    target_ids = sorted(set(delta_sequences) | set(target_sequences))
    if not target_ids:
        raise ValueError("no action profiles to execute")
    result = run_swmm_no_control_action_ablation(
        case_inp,
        job["actuators"],
        job["priority_nodes"],
        job["reference_detail"],
        job["candidate_detail"],
        job["event_id"],
        int(job["duration_min"]),
        float(job["override_start_min"]),
        int(job["horizon_steps"]),
        target_ids[0],
        0.0,
        control_step_sec=int(job["dt_sec"]),
        max_steps=0,
        override_delta_sequence=delta_sequences,
        override_target_sequence=target_sequences,
        policy_id=str(job["policy_id"]),
        cleanup_swmm_artifacts=True,
    )
    result.update(
        {
            "pair_id": str(job["pair_id"]),
            "case_id": str(job["case_id"]),
            "branch": str(job["branch"]),
            "event_role": str(job["event_role"]),
            "split": str(job["split"]),
            "phase": str(job["phase"]),
            "core_template_id": str(job["core_template_id"]),
            "residual_mode": str(job["residual_mode"]),
            "reference_detail": str(job["reference_detail"]),
            "candidate_detail": str(job["candidate_detail"]),
            "executed_action_sequence": str(job["executed_action_sequence"]),
        }
    )
    return result


def _inp_path(cfg: dict[str, Any]) -> Path:
    for dotted in ("inp_path", "network.inp", "swmm.inp", "input.inp"):
        try:
            return cfg_path(cfg, dotted)
        except Exception:
            continue
    raise KeyError("could not resolve INP path from config; tried inp_path, network.inp, swmm.inp, input.inp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Core26 and Core26+Residual10 same-state paired SWMM cases.")
    parser.add_argument("--config", default="configs/wuhan_project6_36_hierarchical_eventbudget_h120_v2.yaml")
    parser.add_argument("--manifest", default="outputs/project6_36_residual10_core_paired_h120_v1/paired_plan/residual10_core_paired_manifest.csv")
    parser.add_argument("--out-dir", default="outputs/project6_36_residual10_core_paired_h120_v1/paired_cases")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-logical-pairs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    manifest_path = root / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    manifest = pd.read_csv(manifest_path)
    if args.max_logical_pairs:
        keep = manifest["pair_id"].drop_duplicates().head(int(args.max_logical_pairs)).tolist()
        manifest = manifest[manifest["pair_id"].isin(set(keep))].copy()
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    details = ensure_dir(out / "details")
    inp_dir = ensure_dir(out / "event_inp")
    case_inp_dir = ensure_dir(out / "case_inp")
    results_path = out / "paired_candidate_results.csv"
    failures_path = out / "failures.csv"
    existing = pd.read_csv(results_path) if results_path.exists() and results_path.stat().st_size else pd.DataFrame()
    completed = set(existing.get("case_id", pd.Series(dtype=str)).astype(str)) if args.resume else set()
    rain = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv").set_index("event_id")
    actuators = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    enabled = set(cfg_path(cfg, "network.control_enabled_actuator_ids_file").read_text(encoding="utf-8").split())
    actuators = actuators[actuators["actuator_id"].astype(str).isin(enabled)].copy()
    priority_nodes = [
        line.strip()
        for line in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    jobs: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        if str(row.case_id) in completed:
            continue
        event = rain.loc[str(row.event_id)]
        event_inp = inp_dir / f"{row.event_id}.inp"
        if not event_inp.exists():
            mutate_inp_for_event(
                _inp_path(cfg),
                Path(event["rainfall_csv"]),
                event_inp,
                simulation_duration_min=int(event["simulation_duration_min"]),
            )
        expected_sequence = row.materialized_core26_action_sequence if str(row.branch) == "A" else row.materialized_candidate_action_sequence
        jobs.append(
            {
                "pair_id": str(row.pair_id),
                "case_id": str(row.case_id),
                "branch": str(row.branch),
                "event_id": str(row.event_id),
                "event_role": str(row.event_role),
                "split": str(row.split),
                "phase": str(row.phase),
                "core_template_id": str(row.core_template_id),
                "residual_mode": str(row.residual_mode),
                "duration_min": int(event["duration_min"]),
                "horizon_steps": int(row.horizon_steps),
                "override_start_min": float(row.override_start_min),
                "dt_sec": int(cfg["experiment"]["control_step_sec"]),
                "event_inp": str(event_inp),
                "case_inp": str(case_inp_dir / f"{row.case_id}.inp"),
                "reference_detail": str(row.reference_detail),
                "candidate_detail": str(details / f"{row.case_id}.csv"),
                "executed_action_sequence": str(row.executed_action_sequence),
                "expected_action_sequence": str(expected_sequence),
                "policy_id": "retrofit_core26_v6" if str(row.branch) == "A" else "retrofit_core26_plus_residual10",
                "actuators": actuators,
                "priority_nodes": priority_nodes,
            }
        )
    report = {
        "manifest": str(manifest_path),
        "logical_pairs": int(manifest["pair_id"].nunique()),
        "physical_cases": int(len(manifest)),
        "pending_jobs": int(len(jobs)),
        "resume_completed": int(len(completed)),
        "out_dir": str(out),
    }
    (out / "execution_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not jobs:
        return
    rows = existing.to_dict("records") if len(existing) else []
    failures: list[dict[str, Any]] = []
    done = 0
    with futures.ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        future_map = {pool.submit(_run_case, job): job for job in jobs}
        for future in futures.as_completed(future_map):
            job = future_map[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({"case_id": str(job["case_id"]), "pair_id": str(job["pair_id"]), "error": repr(exc)})
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"[residual10_pair] done={done}/{len(jobs)} failures={len(failures)}", flush=True)
                pd.DataFrame(rows).to_csv(results_path, index=False)
                if failures:
                    pd.DataFrame(failures).to_csv(failures_path, index=False)
    pd.DataFrame(rows).to_csv(results_path, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
