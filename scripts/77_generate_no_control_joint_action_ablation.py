from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.control.actuator_scope import select_actuators_for_scope
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.simulation.pyswmm_runner import run_swmm_no_control_action_ablation


def _load_single_ablation_module():
    path = Path(__file__).with_name("76_generate_no_control_single_actuator_ablation.py")
    spec = importlib.util.spec_from_file_location("single_ablation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SINGLE = _load_single_ablation_module()


def _joint_case_id(event_id: str, elapsed_min: float, targets: dict[str, float], hold_steps: int) -> str:
    payload = "|".join(f"{aid}={targets[aid]:.6f}" for aid in sorted(targets))
    return hashlib.sha1(f"{event_id}|{elapsed_min:.6f}|{hold_steps}|{payload}".encode("utf-8")).hexdigest()[:16]


def _eligible_rows(reliable: pd.DataFrame, *, allow_pilot_evidence: bool = False) -> pd.DataFrame:
    work = reliable.copy()
    for col in ("repair_safe_frac", "tfv_improved_frac", "peak_safe_frac", "events", "rows"):
        work[col] = pd.to_numeric(work.get(col, 0.0), errors="coerce").fillna(0.0)
    eligibility = (
        work["repair_safe_frac"].ge(0.70)
        & work["tfv_improved_frac"].ge(0.55)
        & work["peak_safe_frac"].ge(0.80)
    )
    if allow_pilot_evidence:
        # A targeted pairwise scan is expressly collecting the missing joint
        # evidence.  One locally observed safe component is enough to propose
        # a *research case*, never an online control whitelist.
        eligibility &= work["events"].ge(1) & work["rows"].ge(1)
    else:
        eligibility &= work["events"].ge(2) & work["rows"].ge(2)
    return work[eligibility].copy()


def _joint_action_designs(
    reference_action: np.ndarray,
    action_ids: list[str],
    reliable: pd.DataFrame,
    *,
    max_group_size: int,
    max_combinations: int,
    allow_pilot_evidence: bool = False,
) -> list[dict]:
    """Build bounded joint absolute targets from direction-safe facilities."""
    eligible = _eligible_rows(reliable, allow_pilot_evidence=allow_pilot_evidence)
    if eligible.empty:
        return []
    index = {str(aid): i for i, aid in enumerate(action_ids)}
    eligible = eligible[eligible["actuator_id"].astype(str).isin(index)].copy()
    if eligible.empty:
        return []
    eligible["action_amplitude"] = pd.to_numeric(eligible.get("action_amplitude", 0.05), errors="coerce").fillna(0.05)
    eligible = eligible.sort_values(
        ["repair_safe_frac", "tfv_improved_frac", "peak_safe_frac", "action_amplitude"],
        ascending=[False, False, False, True],
    ).drop_duplicates(["actuator_id", "action_direction"], keep="first")
    records = eligible.to_dict(orient="records")
    designs: list[dict] = []
    max_size = min(max(2, int(max_group_size)), len(records))
    for size in range(2, max_size + 1):
        for group in itertools.combinations(records, size):
            targets: dict[str, float] = {}
            deltas: dict[str, float] = {}
            for row in group:
                aid = str(row["actuator_id"])
                baseline = float(reference_action[index[aid]])
                magnitude = abs(float(row.get("action_amplitude", 0.05)))
                sign = 1.0 if str(row.get("action_direction", "")).lower() == "increase" else -1.0
                target = float(np.clip(baseline + sign * magnitude, 0.0, 1.0))
                if abs(target - baseline) <= 1.0e-9:
                    continue
                targets[aid] = target
                deltas[aid] = target - baseline
            if len(targets) != size:
                continue
            designs.append(
                {
                    "target_settings": targets,
                    "effective_deltas": deltas,
                    "joint_size": size,
                    "joint_signature": "+".join(sorted(targets)),
                }
            )
            if len(designs) >= int(max_combinations):
                return designs
    return designs


def _parse_explicit_pair_specs(value: str) -> list[tuple[tuple[str, float], tuple[str, float]]]:
    """Parse ``pump:+,orifice:-;...`` into an offline-only research design."""
    pairs: list[tuple[tuple[str, float], tuple[str, float]]] = []
    for spec in (part.strip() for part in str(value or "").split(";") if part.strip()):
        entries = [item.strip() for item in spec.split(",") if item.strip()]
        if len(entries) != 2:
            raise ValueError(f"Each explicit pair must contain exactly two entries: {spec!r}")
        pair: list[tuple[str, float]] = []
        for entry in entries:
            if ":" not in entry:
                raise ValueError(f"Explicit pair entry needs a sign (+ or -): {entry!r}")
            actuator_id, token = (part.strip() for part in entry.rsplit(":", 1))
            if token not in {"+", "-"}:
                raise ValueError(f"Explicit pair sign must be + or -: {entry!r}")
            pair.append((actuator_id, 1.0 if token == "+" else -1.0))
        if pair[0][0] == pair[1][0]:
            raise ValueError(f"An explicit pair cannot repeat an actuator: {spec!r}")
        pairs.append((pair[0], pair[1]))
    return pairs


def _explicit_pair_action_designs(
    reference_action: np.ndarray,
    action_ids: list[str],
    pair_specs: list[tuple[tuple[str, float], tuple[str, float]]],
    amplitudes: list[float],
) -> list[dict]:
    """Build two-facility absolute targets from a no-control reference state."""
    index = {str(aid): i for i, aid in enumerate(action_ids)}
    designs: list[dict] = []
    for pair in pair_specs:
        if any(aid not in index for aid, _ in pair):
            continue
        for first_amp, second_amp in itertools.product(amplitudes, repeat=2):
            targets: dict[str, float] = {}
            deltas: dict[str, float] = {}
            for (aid, sign), amplitude in zip(pair, (first_amp, second_amp)):
                baseline = float(reference_action[index[aid]])
                target = float(np.clip(baseline + sign * abs(float(amplitude)), 0.0, 1.0))
                if abs(target - baseline) <= 1.0e-9:
                    break
                targets[aid] = target
                deltas[aid] = target - baseline
            if len(targets) == 2:
                designs.append({
                    "target_settings": targets,
                    "effective_deltas": deltas,
                    "joint_size": 2,
                    "joint_signature": "+".join(sorted(targets)),
                    "action_design": "explicit_pair_amplitude_grid",
                })
    return designs


def _run_joint(job: dict) -> dict:
    # SWMM derives .out/.rpt paths from the INP path.  A shared event INP lets
    # concurrent cases overwrite one another's native artifacts, so every
    # worker receives a private copy before opening a Simulation.
    source_inp = Path(job["event_inp"])
    case_inp = Path(job["case_inp"])
    case_inp.parent.mkdir(parents=True, exist_ok=True)
    if not case_inp.exists():
        shutil.copy2(source_inp, case_inp)
    first = sorted(job["target_settings"])[0]
    result = run_swmm_no_control_action_ablation(
        case_inp, job["actuators"], job["priority_nodes"], job["reference_detail"],
        job["candidate_detail"], job["event_id"], job["duration_min"], job["elapsed_min"],
        job["hold_steps"], first, 0.0, control_step_sec=job["dt_sec"],
        override_targets=job["target_settings"],
        policy_id="no_control_joint_action_ablation",
        cleanup_swmm_artifacts=True,
    )
    result.update(job)
    return result


def _dynamic_rows_for_event(
    table: pd.DataFrame,
    *,
    pattern: str,
    phase: str,
    max_action_amplitude: float,
) -> pd.DataFrame:
    work = table.copy()
    if "pattern" in work:
        matched = work[work["pattern"].astype(str).eq(str(pattern))]
        if not matched.empty:
            work = matched
    if "phase" in work:
        matched = work[work["phase"].astype(str).eq(str(phase))]
        if not matched.empty:
            work = matched
    work["action_amplitude"] = pd.to_numeric(work.get("action_amplitude", 0.05), errors="coerce").fillna(0.05)
    return work[work["action_amplitude"].le(float(max_action_amplitude))].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan_project6.yaml")
    ap.add_argument("--event-ids", default="")
    ap.add_argument("--max-events", type=int, default=10)
    ap.add_argument("--actuator-ids", default="", help="Comma-separated actuator subset for a predeclared targeted pairwise design.")
    ap.add_argument("--samples-per-phase", type=int, default=1)
    ap.add_argument("--max-group-size", type=int, default=4)
    ap.add_argument("--max-combinations-per-phase", type=int, default=24)
    ap.add_argument("--max-action-amplitude", type=float, default=0.10)
    ap.add_argument("--hold-steps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-details", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Build and report cases without starting SWMM.")
    ap.add_argument(
        "--explicit-pair-specs",
        default="",
        help="Offline-only pairs, e.g. 'ADD301.2:+,cc006.1:-;ADD301.2:+,dwxh.2:-'.",
    )
    ap.add_argument(
        "--pair-amplitudes",
        default="0.10,0.20",
        help="Amplitude grid for --explicit-pair-specs; each pair gets the Cartesian product.",
    )
    ap.add_argument("--reliability", default="")
    ap.add_argument("--allow-pilot-evidence", action="store_true", help="Permit one-event local-safe components for targeted joint data collection only.")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    out = ensure_dir(Path(args.out_dir) if args.out_dir else root / "outputs" / "ablation_all109")
    explicit_pair_specs = _parse_explicit_pair_specs(args.explicit_pair_specs)
    pair_amplitudes = SINGLE._parse_levels(args.pair_amplitudes, [0.10, 0.20])
    reliability_path = Path(args.reliability) if args.reliability else out / "actuator_dynamic_reliability.csv"
    if explicit_pair_specs:
        reliable = pd.DataFrame()
    else:
        if not reliability_path.exists():
            raise FileNotFoundError(f"Missing dynamic single-actuator reliability table: {reliability_path}")
        reliable = pd.read_csv(reliability_path)
    requested_actuators = {value.strip() for value in args.actuator_ids.split(",") if value.strip()}
    if requested_actuators and not explicit_pair_specs:
        reliable = reliable[reliable["actuator_id"].astype(str).isin(requested_actuators)].copy()
    details_dir = ensure_dir(out / "joint_details")
    inp_dir = ensure_dir(out / "event_inp")
    case_inp_dir = ensure_dir(out / "case_inp")
    results_path = out / "joint_action_ablation_results.csv"
    exact_path = out / "exact_no_control_action_effect_dataset.csv"
    existing = pd.read_csv(results_path) if results_path.exists() and results_path.stat().st_size else pd.DataFrame()
    existing_ids = set(existing.get("case_id", pd.Series(dtype=str)).astype(str))

    rain = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv")
    requested = [x.strip() for x in args.event_ids.split(",") if x.strip()]
    if requested:
        rain = rain[rain["event_id"].astype(str).isin(requested)].copy()
    if int(args.max_events) > 0:
        rain = rain.head(int(args.max_events)).copy()
    audit = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    actuators = select_actuators_for_scope(audit, str((cfg.get("controller", {}) or {}).get("actuator_scope", "existing_plus_retrofit")))
    action_ids = actuators["actuator_id"].astype(str).tolist()
    priority_nodes = [x.strip() for x in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    bank = cfg_path(cfg, "outputs.data_bank_train")
    trajectory_dir = bank / "trajectories"
    network_out = Path((cfg.get("outputs", {}) or {}).get("network", "outputs/network"))
    if not network_out.is_absolute():
        network_out = root / network_out
    influence_path = network_out / "priority_to_actuators.csv"
    if not influence_path.exists():
        influence_path = network_out / "priority_to_actuator_candidates.csv"
    influence = pd.read_csv(influence_path) if influence_path.exists() else None
    dt_sec = int(cfg["experiment"]["control_step_sec"])
    horizon_steps = int((cfg.get("controller", {}) or {}).get("horizon_steps", 6))
    gat_cache_dir = cfg_path(cfg, "outputs.gat_features")

    jobs = []
    for event in rain.itertuples(index=False):
        event_id = str(event.event_id)
        reference_detail = trajectory_dir / f"{event_id}__no_control_detail.csv"
        if not reference_detail.exists():
            raise FileNotFoundError(f"Missing No-control trajectory: {reference_detail}")
        event_inp = inp_dir / f"{event_id}__no_controls.inp"
        if not event_inp.exists():
            mutate_inp_for_event(cfg_path(cfg, "network.inp"), event.rainfall_csv, event_inp, int(event.simulation_duration_min), strip_controls=True)
        reference = pd.read_csv(reference_detail)
        for elapsed_min, phase in SINGLE._risk_times(reference, int(event.duration_min), args.samples_per_phase):
            reference_action = np.asarray(
                [SINGLE._reference_action_setting(reference, elapsed_min, aid) for aid in action_ids], dtype=float
            )
            if explicit_pair_specs:
                action_designs = _explicit_pair_action_designs(
                    reference_action, action_ids, explicit_pair_specs, pair_amplitudes
                )
            else:
                local_reliable = _dynamic_rows_for_event(
                    reliable, pattern=str(event.pattern), phase=phase,
                    max_action_amplitude=float(args.max_action_amplitude),
                )
                action_designs = _joint_action_designs(
                    reference_action, action_ids, local_reliable,
                    max_group_size=int(args.max_group_size),
                    max_combinations=int(args.max_combinations_per_phase),
                    allow_pilot_evidence=bool(args.allow_pilot_evidence),
                )
            for design in action_designs:
                case_id = _joint_case_id(event_id, elapsed_min, design["target_settings"], int(args.hold_steps))
                if args.resume and case_id in existing_ids:
                    continue
                jobs.append(
                    {
                        "case_id": case_id,
                        "event_id": event_id,
                        "duration_min": int(event.duration_min),
                        "pattern": str(event.pattern),
                        "phase": phase,
                        "elapsed_min": float(elapsed_min),
                        "hold_steps": int(args.hold_steps),
                        "dt_sec": dt_sec,
                        "event_inp": str(event_inp),
                        "case_inp": str(case_inp_dir / f"{case_id}.inp"),
                        "reference_detail": str(reference_detail),
                        "candidate_detail": str(details_dir / f"{case_id}.csv"),
                        "target_settings": design["target_settings"],
                        "effective_deltas": design["effective_deltas"],
                        "joint_size": design["joint_size"],
                        "joint_signature": design["joint_signature"],
                        "action_design": design.get("action_design", "joint_direction_safe"),
                        "actuators": actuators,
                        "priority_nodes": priority_nodes,
                    }
                )

    if args.dry_run:
        preview = {
            "events": int(rain["event_id"].nunique()),
            "jobs": int(len(jobs)),
            "unique_joint_signatures": int(len({str(job["joint_signature"]) for job in jobs})),
            "phase_counts": pd.Series([str(job["phase"]) for job in jobs]).value_counts().to_dict(),
            "pattern_counts": pd.Series([str(job["pattern"]) for job in jobs]).value_counts().to_dict(),
            "out_dir": str(out),
            "case_isolation": True,
            "explicit_pair_mode": bool(explicit_pair_specs),
            "pair_amplitudes": pair_amplitudes if explicit_pair_specs else [],
        }
        (out / "joint_ablation_dry_run.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")
        print(json.dumps(preview, indent=2))
        return

    rows, exact_rows, failures = [], [], []
    workers = max(1, int(args.workers))

    def flush_checkpoint(completed: int) -> None:
        if not rows and not exact_rows:
            return
        prior_results = pd.read_csv(results_path) if results_path.exists() and results_path.stat().st_size else existing
        pd.concat([prior_results, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(
            "case_id", keep="last"
        ).to_csv(results_path, index=False)
        base = pd.read_csv(exact_path) if exact_path.exists() and exact_path.stat().st_size else pd.DataFrame()
        pd.concat([base, pd.DataFrame(exact_rows)], ignore_index=True).drop_duplicates(
            "case_id", keep="last"
        ).to_csv(exact_path, index=False)
        rows.clear()
        exact_rows.clear()
        print(f"[joint_ablation] done={completed}/{len(jobs)} failures={len(failures)}", flush=True)

    print(f"[joint_ablation] jobs={len(jobs)} workers={workers} wave_size={workers}", flush=True)
    completed = 0
    # PySWMM holds native resources after a Simulation is closed.  Keeping one
    # case per process and recreating the pool after every wave avoids hangs
    # while retaining full parallel SWMM execution.
    for wave_start in range(0, len(jobs), workers):
        wave = jobs[wave_start: wave_start + workers]
        with ProcessPoolExecutor(max_workers=len(wave)) as pool:
            futures = {pool.submit(_run_joint, job): job for job in wave}
            for future in as_completed(futures):
                completed += 1
                job = futures[future]
                try:
                    result = future.result()
                    result["actuator_id"] = result["joint_signature"]
                    result["action_direction"] = "joint"
                    result["action_delta"] = float(max(abs(v) for v in result["effective_deltas"].values()))
                    result["action_amplitude"] = float(result["action_delta"])
                    result["amplitude_tier"] = SINGLE._amplitude_tier(result["action_amplitude"])
                    result["action_design"] = str(result.get("action_design", "joint_direction_safe"))
                    exact = SINGLE._exact_effect_row(
                        result, priority_nodes, actuators, influence, horizon_steps, dt_sec, gat_cache_dir
                    )
                    exact["joint_size"] = int(result["joint_size"])
                    exact["joint_signature"] = str(result["joint_signature"])
                    exact["target_settings"] = json.dumps(result["target_settings"], sort_keys=True)
                    exact["effect_label_mode"] = "exact_no_control_joint_replay_counterfactual"
                    exact_rows.append(exact)
                    rows.append(result)
                    if not args.keep_details:
                        Path(result["candidate_detail"]).unlink(missing_ok=True)
                except Exception as exc:
                    failures.append({"case_id": job["case_id"], "error": repr(exc)})
                if completed % 20 == 0:
                    flush_checkpoint(completed)
        # A completed wave is a valid resume boundary, even if it has fewer
        # than 20 cases.
        flush_checkpoint(completed)

    combined = pd.read_csv(exact_path) if exact_path.exists() and exact_path.stat().st_size else pd.DataFrame()
    joint_rows = combined[combined.get("effect_label_mode", pd.Series(dtype=str)).astype(str).eq("exact_no_control_joint_replay_counterfactual")].copy()
    if not joint_rows.empty:
        report = SINGLE._summarize_reliability(joint_rows, ["pattern", "phase", "joint_size", "joint_signature"], cfg)
    else:
        report = pd.DataFrame()
    report.to_csv(out / "joint_dynamic_reliability.csv", index=False)
    pd.DataFrame(failures).to_csv(out / "joint_failures.csv", index=False)
    summary = {
        "events": int(rain["event_id"].nunique()),
        "new_jobs": int(len(jobs)),
        "failures": int(len(failures)),
        "effect_dataset": str(exact_path),
        "joint_reliability": str(out / "joint_dynamic_reliability.csv"),
        "action_semantics": "absolute_from_no_control_reference",
        "effect_label_mode": "exact_no_control_joint_replay_counterfactual",
        "action_design": "explicit_pair_amplitude_grid" if explicit_pair_specs else "joint_direction_safe",
    }
    (out / "joint_ablation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
