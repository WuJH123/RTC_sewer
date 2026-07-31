from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.control.actuator_scope import select_actuators_for_scope
from sewerrtc.control.horizon_rollout import build_horizon_samples_from_detail
from sewerrtc.data.gat_feature_cache import gat_feature_cache_path
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.simulation.pyswmm_runner import run_swmm_no_control_action_ablation


def _case_id(event_id: str, actuator_id: str, elapsed_min: float, delta: float, steps: int) -> str:
    raw = f"{event_id}|{actuator_id}|{elapsed_min:.6f}|{delta:.6f}|{steps}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _parse_levels(text: str, fallback: list[float], *, allow_zero: bool = False) -> list[float]:
    values = []
    for raw in str(text or "").replace(";", ",").split(","):
        try:
            value = abs(float(raw.strip()))
        except ValueError:
            continue
        if (value > 1.0e-9 or allow_zero) and value not in values:
            values.append(value)
    return values or [abs(float(x)) for x in fallback if abs(float(x)) > 1.0e-9 or allow_zero]


def _amplitude_tier(value: float) -> str:
    return f"d_{abs(float(value)):0.3f}".replace(".", "p")


def _effective_action_designs(
    reference_setting: float,
    *,
    delta_levels: list[float],
    absolute_levels: list[float],
) -> list[dict]:
    """Return non-zero, bounded absolute settings around a reference action."""
    reference = float(np.clip(reference_setting, 0.0, 1.0))
    by_target: dict[float, dict] = {}

    def add(target: float, design: str) -> None:
        target = float(np.clip(target, 0.0, 1.0))
        delta = float(target - reference)
        if abs(delta) <= 1.0e-9:
            return
        key = round(target, 8)
        # A physical target is unique. Prefer an explicit absolute-grid label
        # when it duplicates a residual design.
        if key in by_target and by_target[key]["action_design"] == "absolute":
            return
        by_target[key] = {
            "reference_setting": reference,
            "target_setting": target,
            "effective_delta": delta,
            "action_direction": "increase" if delta > 0.0 else "decrease",
            "action_amplitude": abs(delta),
            "amplitude_tier": _amplitude_tier(delta),
            "action_design": design,
        }

    for level in delta_levels:
        add(reference - abs(float(level)), "relative")
        add(reference + abs(float(level)), "relative")
    for level in absolute_levels:
        add(float(level), "absolute")
    return sorted(by_target.values(), key=lambda row: (row["target_setting"], row["action_design"]))


def _reference_action_setting(reference: pd.DataFrame, elapsed_min: float, actuator_id: str) -> float:
    col = f"a:{actuator_id}"
    if col not in reference:
        return 1.0
    times = pd.to_numeric(reference["elapsed_min"], errors="coerce").fillna(0.0).to_numpy(float)
    idx = int(np.argmin(np.abs(times - float(elapsed_min))))
    return float(np.clip(pd.to_numeric(reference.iloc[idx][col], errors="coerce"), 0.0, 1.0))


def _run_case(job: dict) -> dict:
    source_inp = Path(job["event_inp"])
    case_inp = Path(job["case_inp"])
    case_inp.parent.mkdir(parents=True, exist_ok=True)
    # SWMM derives its .out/.rpt names from the input path. A shared event INP
    # therefore corrupts concurrent PySWMM jobs even when their detail CSVs
    # differ. Each counterfactual must own its input basename.
    if not case_inp.exists():
        shutil.copy2(source_inp, case_inp)
    result = run_swmm_no_control_action_ablation(
        case_inp,
        job["actuators"],
        job["priority_nodes"],
        job["reference_detail"],
        job["candidate_detail"],
        job["event_id"],
        job["duration_min"],
        job["elapsed_min"],
        job["hold_steps"],
        job["actuator_id"],
        job["delta"],
        control_step_sec=job["dt_sec"],
        target_setting=job.get("target_setting"),
        cleanup_swmm_artifacts=True,
    )
    result.update(
        {
            "case_id": job["case_id"],
            "pattern": job["pattern"],
            "phase": job["phase"],
            "candidate_detail": job["candidate_detail"],
            "reference_setting": job["reference_setting"],
            "target_setting": job["target_setting"],
            "effective_delta": job["delta"],
            "action_amplitude": job["action_amplitude"],
            "amplitude_tier": job["amplitude_tier"],
            "action_design": job["action_design"],
        }
    )
    return result


def _risk_times(
    reference: pd.DataFrame,
    duration_min: int,
    samples_per_phase: int,
    phases: tuple[str, ...] = ("pre_peak", "peak", "recession"),
) -> list[tuple[float, str]]:
    frame = reference.copy()
    frame["elapsed_min"] = pd.to_numeric(frame["elapsed_min"], errors="coerce")
    flood_cols = [c for c in frame if c.startswith("flood:")]
    frame["_risk"] = (
        frame[flood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        if flood_cols
        else 0.0
    )
    if "phase" not in frame:
        frame["phase"] = np.where(
            frame["elapsed_min"] < 0.35 * duration_min,
            "pre_peak",
            np.where(frame["elapsed_min"] <= duration_min, "peak", "recession"),
        )
    selected: list[tuple[float, str]] = []
    for phase in phases:
        group = frame[frame["phase"].astype(str).eq(phase)].copy()
        if group.empty:
            continue
        group = group.iloc[:-7] if len(group) > 7 else group.iloc[:1]
        top = group.nlargest(max(1, int(samples_per_phase)), "_risk")
        selected.extend((float(row.elapsed_min), phase) for row in top.itertuples())
    return selected


def _apply_cached_gat_state(row: pd.Series, reference_detail: Path, cache_dir: Path) -> pd.Series:
    cache_path = gat_feature_cache_path(cache_dir, reference_detail)
    if not cache_path.exists():
        row["state_feature_source"] = "no_control_swmm_state"
        return row
    cache = np.load(cache_path, allow_pickle=False)
    idx = int(row["row_index"])
    for col in (
        "current_depth_mean", "current_depth_p95", "current_depth_max",
        "priority_depth_mean", "priority_depth_max",
    ):
        row[col] = float(np.asarray(cache[col])[idx])
    priority_max = np.asarray(cache["priority_depth_max"], dtype=float)
    row["priority_depth_trend"] = float(priority_max[idx] - priority_max[max(0, idx - 2)])
    row["state_feature_source"] = "gat_reconstructed_sparse_sensors"
    return row


def _exact_effect_row(
    result: dict,
    priority_nodes: list[str],
    actuators: pd.DataFrame,
    influence: pd.DataFrame | None,
    horizon_steps: int,
    dt_sec: int,
    gat_cache_dir: Path,
) -> dict:
    candidate = Path(result["candidate_detail"])
    reference = Path(result["reference_detail_file"])
    samples = build_horizon_samples_from_detail(
        candidate,
        priority_nodes,
        horizon_steps=horizon_steps,
        history_steps=3,
        dt_sec=dt_sec,
        stride=1,
        actuators=actuators,
        priority_to_actuators=influence,
        reference_detail_path=reference,
    )
    if samples.empty:
        raise ValueError(f"No horizon samples generated for ablation {result['case_id']}")
    idx = int(np.argmin(np.abs(samples["elapsed_min"].to_numpy(float) - float(result["override_start_min"]))))
    row = samples.iloc[idx].copy()
    row = _apply_cached_gat_state(row, reference, gat_cache_dir)
    row["case_id"] = result["case_id"]
    row["actuator_id"] = result["actuator_id"]
    row["action_delta"] = float(result["action_delta"])
    row["action_direction"] = result["action_direction"]
    row["pattern"] = result["pattern"]
    row["phase"] = result["phase"]
    for column in ("reference_setting", "target_setting", "effective_delta", "action_amplitude", "amplitude_tier", "action_design"):
        row[column] = result.get(column, np.nan)
    row["effect_label_mode"] = "exact_no_control_replay_counterfactual"
    return row.to_dict()


def _summarize_reliability(exact: pd.DataFrame, group_cols: list[str], cfg: dict) -> pd.DataFrame:
    if exact.empty:
        return pd.DataFrame()
    gate_cfg = ((cfg.get("evaluation", {}) or {}).get("no_control_repair_gate", {}) or {})
    pfv_abs = float(gate_cfg.get("no_control_pfv_noninferiority_abs", 100.0))
    pfv_frac = float(gate_cfg.get("no_control_pfv_noninferiority_frac", 0.005))
    peak_frac = float((cfg.get("controller", {}) or {}).get("peak_tolerance_frac", 0.01))
    work = exact.copy()
    fallback_amplitude = (
        pd.to_numeric(work["action_delta"], errors="coerce").fillna(0.0).abs()
        if "action_delta" in work
        else pd.Series(0.0, index=work.index)
    )
    if "action_amplitude" not in work:
        work["action_amplitude"] = fallback_amplitude
    if "amplitude_tier" not in work:
        work["amplitude_tier"] = work["action_amplitude"].map(_amplitude_tier)
    if "action_design" not in work:
        work["action_design"] = "relative"
    work["pfv_noninferior"] = work["effect_PFV_H"] <= np.maximum(
        pfv_abs, pfv_frac * work["reference_PFV_H"].clip(lower=0.0)
    )
    work["tfv_improved"] = work["effect_TFV_H"] < 0.0
    work["peak_safe"] = work["effect_peak_TFV_rate_H"] <= np.maximum(
        0.5, peak_frac * work["reference_peak_TFV_rate_H"].clip(lower=0.0)
    )
    work["repair_safe"] = work["pfv_noninferior"] & work["tfv_improved"] & work["peak_safe"]
    work["action_changed"] = pd.to_numeric(work["sequence_delta_abs_max"], errors="coerce").fillna(0.0) > 1.0e-6
    work = work[work["action_changed"]].copy()
    if work.empty:
        return pd.DataFrame()
    return (
        work.groupby(group_cols, as_index=False)
        .agg(
            events=("event_id", "nunique"),
            rows=("case_id", "nunique"),
            pfv_noninferior_frac=("pfv_noninferior", "mean"),
            tfv_improved_frac=("tfv_improved", "mean"),
            peak_safe_frac=("peak_safe", "mean"),
            repair_safe_frac=("repair_safe", "mean"),
            effect_PFV_mean=("effect_PFV_H", "mean"),
            effect_TFV_mean=("effect_TFV_H", "mean"),
            effect_peak_mean=("effect_peak_TFV_rate_H", "mean"),
        )
        .sort_values(["repair_safe_frac", "tfv_improved_frac", "peak_safe_frac"], ascending=False)
    )


def _dynamic_reliability(exact: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return _summarize_reliability(
        exact,
        ["pattern", "phase", "actuator_id", "action_direction", "action_amplitude", "amplitude_tier", "action_design"],
        cfg,
    )


def _normalise_exact_schema(exact: pd.DataFrame) -> pd.DataFrame:
    if exact.empty:
        return exact
    out = exact.copy()
    fallback_amplitude = (
        pd.to_numeric(out["action_delta"], errors="coerce").fillna(0.0).abs()
        if "action_delta" in out
        else pd.Series(0.0, index=out.index)
    )
    if "action_amplitude" not in out:
        out["action_amplitude"] = fallback_amplitude
    else:
        out["action_amplitude"] = pd.to_numeric(out["action_amplitude"], errors="coerce").fillna(fallback_amplitude)
    if "amplitude_tier" not in out:
        out["amplitude_tier"] = out["action_amplitude"].map(_amplitude_tier)
    else:
        out["amplitude_tier"] = out["amplitude_tier"].fillna(out["action_amplitude"].map(_amplitude_tier))
    if "action_design" not in out:
        out["action_design"] = "relative"
    else:
        out["action_design"] = out["action_design"].fillna("relative")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan_project6.yaml")
    ap.add_argument("--event-ids", default="")
    ap.add_argument("--max-events", type=int, default=10)
    ap.add_argument("--max-actuators", type=int, default=0)
    ap.add_argument("--actuator-ids", default="", help="Comma-separated explicit actuator subset for targeted experiments.")
    ap.add_argument("--samples-per-phase", type=int, default=1)
    ap.add_argument("--phases", default="pre_peak,peak,recession")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--delta-levels", default="", help="Relative action amplitudes, e.g. 0.05,0.10,0.20,0.40.")
    ap.add_argument("--absolute-levels", default="", help="Absolute target settings, e.g. 0,0.25,0.50,0.75,1.")
    ap.add_argument("--hold-steps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-details", action="store_true")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    out = ensure_dir(Path(args.out_dir) if args.out_dir else root / "outputs" / "ablation_all109")
    details_dir = ensure_dir(out / "details")
    inp_dir = ensure_dir(out / "event_inp")
    case_inp_dir = ensure_dir(out / "case_inp")
    results_path = out / "single_actuator_ablation_results.csv"
    exact_path = out / "exact_no_control_action_effect_dataset.csv"
    existing = pd.read_csv(results_path) if results_path.exists() and results_path.stat().st_size else pd.DataFrame()
    existing_ids = set(existing.get("case_id", pd.Series(dtype=str)).astype(str))
    if exact_path.exists() and exact_path.stat().st_size:
        prior_exact = _normalise_exact_schema(pd.read_csv(exact_path))
        prior_exact.to_csv(exact_path, index=False)
        existing_ids.update(prior_exact.get("case_id", pd.Series(dtype=str)).astype(str))

    rain = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv")
    requested = [x.strip() for x in args.event_ids.split(",") if x.strip()]
    if requested:
        rain = rain[rain["event_id"].astype(str).isin(requested)].copy()
    if int(args.max_events) > 0:
        rain = rain.head(int(args.max_events)).copy()
    audit = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    scope = str((cfg.get("controller", {}) or {}).get("actuator_scope", "existing_plus_retrofit"))
    actuators = select_actuators_for_scope(audit, scope)
    requested_actuators = {value.strip() for value in args.actuator_ids.split(",") if value.strip()}
    if requested_actuators:
        actuators = actuators[actuators["actuator_id"].astype(str).isin(requested_actuators)].copy()
        missing = sorted(requested_actuators - set(actuators["actuator_id"].astype(str)))
        if missing:
            raise ValueError(f"Requested actuators are not available in scope={scope}: {missing}")
    if int(args.max_actuators) > 0:
        actuators = actuators.head(int(args.max_actuators)).copy()
    priority_nodes = [x.strip() for x in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    bank = cfg_path(cfg, "outputs.data_bank_train")
    trajectory_dir = bank / "trajectories"
    network_out = Path((cfg.get("outputs", {}) or {}).get("network", "outputs/network"))
    if not network_out.is_absolute():
        network_out = root / network_out
    influence_path = network_out / "priority_to_actuator_candidates.csv"
    influence = pd.read_csv(influence_path) if influence_path.exists() else None
    gat_cache_dir = cfg_path(cfg, "outputs.gat_features")
    dt_sec = int(cfg["experiment"]["control_step_sec"])
    horizon_steps = int((cfg.get("controller", {}) or {}).get("horizon_steps", 6))
    delta_levels = _parse_levels(args.delta_levels, [args.delta])
    absolute_levels = _parse_levels(args.absolute_levels, [], allow_zero=True) if str(args.absolute_levels).strip() else []

    jobs = []
    phases = tuple(x.strip() for x in args.phases.split(",") if x.strip())
    for event in rain.itertuples(index=False):
        event_id = str(event.event_id)
        reference_detail = trajectory_dir / f"{event_id}__no_control_detail.csv"
        if not reference_detail.exists():
            raise FileNotFoundError(f"Missing No-control trajectory for ablation: {reference_detail}")
        event_inp = inp_dir / f"{event_id}__no_controls.inp"
        if not event_inp.exists():
            mutate_inp_for_event(
                cfg_path(cfg, "network.inp"),
                event.rainfall_csv,
                event_inp,
                int(event.simulation_duration_min),
                strip_controls=True,
            )
        reference = pd.read_csv(reference_detail)
        for elapsed_min, phase in _risk_times(reference, int(event.duration_min), args.samples_per_phase, phases):
            for actuator_id in actuators["actuator_id"].astype(str):
                reference_setting = _reference_action_setting(reference, elapsed_min, actuator_id)
                designs = _effective_action_designs(
                    reference_setting,
                    delta_levels=delta_levels,
                    absolute_levels=absolute_levels,
                )
                for design in designs:
                    delta = float(design["effective_delta"])
                    case_id = _case_id(event_id, actuator_id, elapsed_min, delta, args.hold_steps)
                    if args.resume and case_id in existing_ids:
                        continue
                    jobs.append(
                        {
                            "case_id": case_id,
                            "event_id": event_id,
                            "duration_min": int(event.duration_min),
                            "pattern": str(event.pattern),
                            "phase": phase,
                            "elapsed_min": elapsed_min,
                            "actuator_id": actuator_id,
                            "delta": delta,
                            "reference_setting": float(design["reference_setting"]),
                            "target_setting": float(design["target_setting"]),
                            "action_amplitude": float(design["action_amplitude"]),
                            "amplitude_tier": str(design["amplitude_tier"]),
                            "action_design": str(design["action_design"]),
                            "hold_steps": int(args.hold_steps),
                            "dt_sec": dt_sec,
                            "event_inp": str(event_inp),
                            "case_inp": str(case_inp_dir / f"{case_id}.inp"),
                            "reference_detail": str(reference_detail),
                            "candidate_detail": str(details_dir / f"{case_id}.csv"),
                            "actuators": actuators,
                            "priority_nodes": priority_nodes,
                        }
                    )

    rows = []
    exact_rows = []
    failures = []
    workers = max(1, int(args.workers))

    def flush_checkpoint(completed: int) -> None:
        if not rows and not exact_rows:
            return
        prior_results = pd.read_csv(results_path) if results_path.exists() and results_path.stat().st_size else existing
        merged = pd.concat([prior_results, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("case_id", keep="last")
        merged.to_csv(results_path, index=False)
        prior_exact = pd.read_csv(exact_path) if exact_path.exists() and exact_path.stat().st_size else pd.DataFrame()
        pd.concat([prior_exact, pd.DataFrame(exact_rows)], ignore_index=True).drop_duplicates("case_id", keep="last").to_csv(exact_path, index=False)
        rows.clear()
        exact_rows.clear()
        print(f"[all109_ablation] done={completed}/{len(jobs)} failures={len(failures)}", flush=True)

    print(f"[all109_ablation] jobs={len(jobs)} workers={workers} wave_size={workers}", flush=True)
    completed = 0
    # PySWMM can retain native resources after several consecutive Simulation
    # objects in one spawned worker. Recreate the worker pool after each wave
    # so every process owns at most one SWMM case. This is slower to spawn but
    # prevents silent hangs and preserves real 16-way SWMM parallelism.
    for wave_start in range(0, len(jobs), workers):
        wave = jobs[wave_start: wave_start + workers]
        with ProcessPoolExecutor(max_workers=len(wave)) as pool:
            futures = {pool.submit(_run_case, job): job for job in wave}
            for future in as_completed(futures):
                completed += 1
                job = futures[future]
                try:
                    result = future.result()
                    exact_rows.append(
                        _exact_effect_row(result, priority_nodes, actuators, influence, horizon_steps, dt_sec, gat_cache_dir)
                    )
                    rows.append(result)
                    if not args.keep_details:
                        Path(result["candidate_detail"]).unlink(missing_ok=True)
                except Exception as exc:
                    failures.append({"case_id": job["case_id"], "error": repr(exc)})
                if completed % 20 == 0:
                    flush_checkpoint(completed)
        # A completed wave is a safe restart boundary even when it contains
        # fewer than 20 cases.
        flush_checkpoint(completed)

    exact = _normalise_exact_schema(pd.read_csv(exact_path)) if exact_path.exists() and exact_path.stat().st_size else pd.DataFrame()
    if not exact.empty:
        exact.to_csv(exact_path, index=False)
    online_max_delta = float((cfg.get("controller", {}) or {}).get("online_reliability_max_delta", 0.10))
    online_exact = exact[pd.to_numeric(exact.get("action_amplitude", pd.Series(dtype=float)), errors="coerce").fillna(0.0) <= online_max_delta].copy()
    global_summary = _summarize_reliability(online_exact, ["actuator_id", "action_direction"], cfg)
    pattern_phase = _summarize_reliability(online_exact, ["pattern", "phase", "actuator_id", "action_direction"], cfg)
    dynamic = _dynamic_reliability(exact, cfg)
    global_summary.to_csv(out / "actuator_direction_reliability.csv", index=False)
    pattern_phase.to_csv(out / "actuator_direction_reliability_by_pattern_phase.csv", index=False)
    dynamic.to_csv(out / "actuator_dynamic_reliability.csv", index=False)
    pd.DataFrame(failures).to_csv(out / "failures.csv", index=False)
    report = {
        "events": int(rain["event_id"].nunique()),
        "actuators": int(len(actuators)),
        "new_jobs": int(len(jobs)),
        "total_exact_effect_rows": int(len(exact)),
        "failures": int(len(failures)),
        "action_semantics": "absolute_from_no_control_reference",
        "effect_label_mode": "exact_no_control_replay_counterfactual",
        "delta_levels": delta_levels,
        "absolute_levels": absolute_levels,
        "online_reliability_max_delta": online_max_delta,
        "results": str(results_path),
        "exact_effect_dataset": str(exact_path),
        "reliability": str(out / "actuator_direction_reliability.csv"),
        "reliability_by_pattern_phase": str(out / "actuator_direction_reliability_by_pattern_phase.csv"),
        "dynamic_reliability": str(out / "actuator_dynamic_reliability.csv"),
    }
    (out / "ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
