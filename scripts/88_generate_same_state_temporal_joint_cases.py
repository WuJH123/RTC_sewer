from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.control.actuator_scope import select_actuators_for_scope
from sewerrtc.experiments.targeted_joint_pairs import action_window, materialize_candidate, sequence_diagnostics
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.simulation.pyswmm_runner import run_swmm_no_control_action_ablation


def _profiles(spec: dict) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    actuators = list(spec.get("actuators", []))
    delta_sequences: dict[str, list[float]] = {}
    target_sequences: dict[str, list[float]] = {}
    if "signed_profiles" in spec:
        delta_sequences = {str(key): list(map(float, value)) for key, value in spec["signed_profiles"].items()}
    if "signed_profile" in spec:
        delta_sequences[str(actuators[0])] = list(map(float, spec["signed_profile"]))
    if "target_profile" in spec:
        target_sequences[str(actuators[0])] = list(map(float, spec["target_profile"]))
    if "target_profiles" in spec:
        target_sequences = {str(key): list(map(float, value)) for key, value in spec["target_profiles"].items()}
    if not delta_sequences and not target_sequences:
        raise ValueError(f"candidate lacks an explicit temporal profile: {spec}")
    return delta_sequences, target_sequences


def _run(job: dict) -> dict:
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
    materialized = materialize_candidate(
        reference_sequence,
        action_ids=action_ids,
        specification=spec,
    )
    expected = job.get("materialized_candidate_action_sequence")
    if isinstance(expected, str) and expected.strip():
        expected_array = np.asarray(json.loads(expected), dtype=np.float32)
        if expected_array.shape != materialized.shape or not np.allclose(expected_array, materialized, atol=1.0e-6):
            raise ValueError("runtime candidate does not match preflight materialized [H,36] sequence")
    delta_sequences, target_sequences = _profiles(spec)
    targets = sorted(set(delta_sequences) | set(target_sequences))
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
        targets[0],
        0.0,
        control_step_sec=int(job["dt_sec"]),
        override_delta_sequence=delta_sequences,
        override_target_sequence=target_sequences,
        policy_id="same_state_temporal_joint_candidate",
        cleanup_swmm_artifacts=True,
    )
    result.update({key: value for key, value in job.items() if key != "actuators"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--manifest", default="outputs/project6_36_temporal_joint_v1/paired_plan/joint_action_case_manifest.csv")
    parser.add_argument("--reference-bank", default="outputs/data_bank_train_v8_storage_variablepump/trajectories")
    parser.add_argument("--out-dir", default="outputs/project6_36_temporal_joint_v1/paired_cases")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=387)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-noop-filter", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    manifest = pd.read_csv(root / args.manifest if not Path(args.manifest).is_absolute() else args.manifest)
    candidates = manifest[manifest["branch"].astype(str).eq("B")].copy()
    if not candidates["executed_action_sequence"].map(
        lambda value: json.loads(value).get("sequence_semantics") == "relative_to_same_state_no_control_reference"
    ).all():
        raise ValueError("manifest contains unverified or ambiguous action semantics")
    if int(args.max_cases) > 0:
        candidates = candidates.head(int(args.max_cases)).copy()
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    details = ensure_dir(out / "details")
    inp_dir = ensure_dir(out / "event_inp")
    case_inp_dir = ensure_dir(out / "case_inp")
    results_path = out / "paired_candidate_results.csv"
    failures_path = out / "failures.csv"
    existing = pd.read_csv(results_path) if results_path.exists() and results_path.stat().st_size else pd.DataFrame()
    completed = set(existing.get("case_id", pd.Series(dtype=str)).astype(str)) if args.resume else set()
    rain = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv").set_index("event_id")
    audit = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    actuators = select_actuators_for_scope(audit, "control_enabled")
    if len(actuators) != 36:
        raise ValueError(f"paired execution requires 36 canonical actuators, got {len(actuators)}")
    priority_nodes = [item.strip() for item in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text().splitlines() if item.strip()]
    reference_bank = root / args.reference_bank if not Path(args.reference_bank).is_absolute() else Path(args.reference_bank)
    jobs = []
    preflight_rejections = []
    for row in candidates.itertuples(index=False):
        if str(row.case_id) in completed:
            continue
        event = rain.loc[str(row.event_id)]
        reference = reference_bank / f"{row.event_id}__no_control_detail.csv"
        if not reference.exists():
            raise FileNotFoundError(f"Missing same-network No-control reference: {reference}")
        if args.preflight_noop_filter:
            reference_frame = pd.read_csv(reference)
            reference_sequence = action_window(
                reference_frame,
                action_ids=actuators["actuator_id"].astype(str).tolist(),
                start_min=(
                    float(row.override_start_min)
                    if hasattr(row, "override_start_min") and pd.notna(row.override_start_min)
                    else float(event.duration_min) * float(row.split_timestamp_fraction)
                ),
                horizon_steps=int((cfg.get("controller", {}) or {}).get("horizon_steps", 6)),
            )
            specification = json.loads(row.executed_action_sequence)
            candidate_sequence = materialize_candidate(
                reference_sequence,
                action_ids=actuators["actuator_id"].astype(str).tolist(),
                specification=specification,
            )
            diagnostics = sequence_diagnostics(
                candidate_sequence,
                reference_sequence,
                action_ids=actuators["actuator_id"].astype(str).tolist(),
                binary_pump_ids=set(((cfg.get("controller", {}) or {}).get("temporal_joint", {}) or {}).get("candidate_search", {}).get("binary_pump_ids", [])),
                minimum_effective_delta=0.02,
            )
            if not diagnostics["valid"]:
                preflight_rejections.append({"case_id": row.case_id, "reason": diagnostics["reason"]})
                continue
        event_inp = inp_dir / f"{row.event_id}__no_controls.inp"
        if not event_inp.exists():
            mutate_inp_for_event(
                cfg_path(cfg, "network.inp"), event.rainfall_csv, event_inp,
                int(event.simulation_duration_min), strip_controls=True,
            )
        jobs.append({
            **row._asdict(),
            "duration_min": int(event.duration_min),
            "override_start_min": (
                float(row.override_start_min)
                if hasattr(row, "override_start_min") and pd.notna(row.override_start_min)
                else float(event.duration_min) * float(row.split_timestamp_fraction)
            ),
            "horizon_steps": int((cfg.get("controller", {}) or {}).get("horizon_steps", 6)),
            "dt_sec": int(cfg["experiment"]["control_step_sec"]),
            "event_inp": str(event_inp),
            "case_inp": str(case_inp_dir / f"{row.case_id}.inp"),
            "reference_detail": str(reference),
            "candidate_detail": str(details / f"{row.case_id}.csv"),
            "actuators": actuators,
            "priority_nodes": priority_nodes,
        })
    report = {
        "logical_paired_experiments": int(candidates["pair_id"].nunique()),
        "candidate_swmm_jobs": len(jobs),
        "reused_reference_checkpoints": int(manifest[manifest["branch"].eq("A")]["execution_case_id"].nunique()),
        "expected_new_physical_cases": len(jobs),
        "resume_completed": len(completed),
        "action_count": len(actuators),
        "out_dir": str(out),
        "preflight_rejected_cases": len(preflight_rejections),
    }
    pd.DataFrame(preflight_rejections).to_csv(out / "preflight_rejections.csv", index=False)
    (out / "execution_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        return
    rows: list[dict] = []
    failures: list[dict] = []
    workers = max(1, int(args.workers))
    for wave_start in range(0, len(jobs), workers):
        wave = jobs[wave_start : wave_start + workers]
        with ProcessPoolExecutor(max_workers=len(wave)) as pool:
            futures = {pool.submit(_run, job): job for job in wave}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    failures.append({"case_id": job["case_id"], "error": repr(exc)})
        prior = pd.read_csv(results_path) if results_path.exists() and results_path.stat().st_size else existing
        pd.concat([prior, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("case_id", keep="last").to_csv(results_path, index=False)
        rows.clear()
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"[paired_temporal_joint] done={min(wave_start + len(wave), len(jobs))}/{len(jobs)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
