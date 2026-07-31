from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.models.residual_value import build_residual_feature_dict
from sewerrtc.simulation.kpi_metrics import compute_kpis
from sewerrtc.simulation.pyswmm_runner import run_swmm_residual_override_trajectory, run_swmm_trajectory


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _kpis_from_detail(path: Path, priority: list[str], dt_sec: int, policy_id: str, event_id: str, duration_min: int) -> dict:
    detail = pd.read_csv(path)
    row = compute_kpis(detail, priority, dt_sec)
    row.update({"event_id": event_id, "policy_id": policy_id, "duration_min": int(duration_min), "detail_file": str(path), "rows": len(detail)})
    return row


def _priority_depth_max(row: pd.Series, priority: list[str]) -> float:
    vals = []
    for n in priority:
        c = f"h:{n}"
        if c in row:
            try:
                vals.append(float(row[c]))
            except Exception:
                pass
    return float(np.nanmax(vals)) if vals else 0.0


def _build_case_features(
    detail_row: pd.Series,
    actuators: pd.DataFrame,
    priority: list[str],
    deltas: dict[str, float],
) -> dict:
    nominal = []
    candidate = []
    for aid in actuators["actuator_id"].astype(str):
        v = float(detail_row.get(f"a:{aid}", 1.0))
        d = float(deltas.get(aid, 0.0))
        nominal.append(v)
        candidate.append(np.clip(v + d, 0.0, 1.0))
    return build_residual_feature_dict(
        actuators,
        np.asarray(nominal, dtype=np.float32),
        np.asarray(candidate, dtype=np.float32),
        str(detail_row.get("phase", "")),
        float(detail_row.get("rainfall_mm_h", 0.0)),
        _priority_depth_max(detail_row, priority),
        float(detail_row.get("elapsed_min", 0.0)),
    )


def _parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_float_list(text: str) -> list[float]:
    vals = []
    for x in _parse_csv_list(text):
        try:
            vals.append(float(x))
        except Exception:
            pass
    return vals


def _delta_tier(delta: float) -> str:
    d = abs(float(delta))
    if d <= 0.080001:
        return "small"
    if d <= 0.160001:
        return "medium"
    return "large"


def _case_templates(
    actuators: pd.DataFrame,
    max_delta: float,
    template_names: set[str] | None = None,
) -> list[tuple[str, dict[str, float]]]:
    a = actuators.copy()
    aid = a["actuator_id"].astype(str)
    typ = a.get("link_type", pd.Series("", index=a.index)).fillna("").astype(str)
    role = a.get("storage_control_type", pd.Series("", index=a.index)).fillna("").astype(str)
    near_storage = a.get("near_storage", pd.Series(False, index=a.index)).fillna(False).astype(bool)
    masks = {
        "storage_outlet_retain": (role == "storage_outlet", -abs(max_delta)),
        "storage_outlet_release": (role == "storage_outlet", abs(max_delta)),
        "storage_inlet_restrict": (role == "storage_inlet", -abs(max_delta)),
        "storage_inlet_open": (role == "storage_inlet", abs(max_delta)),
        "pump_throttle": (typ == "pump", -abs(max_delta)),
        "pump_boost": (typ == "pump", abs(max_delta)),
        "storage_all_retain": (near_storage & (typ != "pump"), -abs(max_delta)),
        "storage_all_release": (near_storage & (typ != "pump"), abs(max_delta)),
    }
    out = []
    for name, (mask, delta) in masks.items():
        if template_names is not None and name not in template_names:
            continue
        ids = aid[mask].tolist()
        if ids:
            out.append((name, {x: float(delta) for x in ids}))
    return out


def _balanced_limit_jobs(jobs: list[dict], max_cases: int) -> list[dict]:
    if not max_cases or len(jobs) <= max_cases:
        return jobs
    buckets: dict[tuple[str, str, str], list[dict]] = {}
    for job in jobs:
        meta = job.get("meta", {})
        key = (
            str(meta.get("event_id", "")),
            str(meta.get("template_name", "")),
            str(meta.get("residual_delta_tier", _delta_tier(float(meta.get("residual_delta", 0.0))))),
        )
        buckets.setdefault(key, []).append(job)
    selected: list[dict] = []
    keys = sorted(buckets)
    while len(selected) < max_cases and keys:
        next_keys = []
        for key in keys:
            bucket = buckets.get(key, [])
            if bucket and len(selected) < max_cases:
                selected.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def _run_case(job: dict) -> dict:
    actuators = pd.read_csv(job["actuator_csv"])
    kpi = run_swmm_residual_override_trajectory(
        job["event_inp_native"],
        actuators,
        job["priority"],
        job["detail_file"],
        job["event_id"],
        int(job["duration_min"]),
        float(job["override_start_min"]),
        int(job["override_steps"]),
        job["override_deltas"],
        int(job["control_step_sec"]),
        int(job["seed"]),
        int(job["max_steps"]),
        simulation_duration_min=int(job["simulation_duration_min"]),
        recession_min=int(job["recession_min"]),
    )
    return {**job["meta"], **kpi}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--mode", choices=["debug", "formal"], default="debug")
    ap.add_argument("--max-events", type=int, default=8)
    ap.add_argument("--max-cases", type=int, default=160)
    ap.add_argument(
        "--target-total-cases",
        type=int,
        default=0,
        help=(
            "Resume-aware target total sample count. If set, the script only "
            "runs enough new cases to bring residual_counterfactual_results.csv "
            "up to this many unique case_id rows."
        ),
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-internal-pfv", type=float, default=100.0)
    ap.add_argument("--max-delta", type=float, default=0.08)
    ap.add_argument(
        "--delta-list",
        default="",
        help="Optional comma-separated residual deltas. If set, overrides --max-delta for case generation, e.g. 0.04,0.08.",
    )
    ap.add_argument(
        "--templates",
        default="storage_inlet_restrict,storage_outlet_retain,storage_all_retain,pump_throttle",
        help="Comma-separated residual action templates to generate. Default keeps only templates aligned with the current controller.",
    )
    ap.add_argument("--override-steps", type=int, default=3)
    ap.add_argument("--risk-rows-per-event", type=int, default=8)
    ap.add_argument(
        "--min-action-change",
        type=float,
        default=1e-6,
        help="Skip cases where clipping means the candidate action is effectively unchanged.",
    )
    ap.add_argument(
        "--selection-mode",
        choices=["round_robin", "sequential"],
        default="round_robin",
        help="How to enforce --max-cases after candidate jobs are built.",
    )
    ap.add_argument("--tfv-guard-pct", type=float, default=0.005)
    ap.add_argument("--peak-guard-pct", type=float, default=0.010)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    priority = (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines()
    actuators = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    rain_table = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv")
    out_root = ensure_dir(cfg_path(cfg, "outputs.closed_loop") / "internal_residual_counterfactuals")
    inp_dir = ensure_dir(out_root / "event_inp")
    base_dir = ensure_dir(out_root / "internal_rules")
    detail_dir = ensure_dir(out_root / "details")
    rows_path = out_root / "residual_counterfactual_results.csv"
    completed = set()
    existing = _safe_read(rows_path)
    if args.resume and not existing.empty and "case_id" in existing:
        completed = set(existing["case_id"].astype(str))

    jobs = []
    baseline_rows = []
    for _, ev in rain_table.iterrows():
        event_id = str(ev["event_id"])
        event_inp_native = inp_dir / f"{event_id}__with_controls.inp"
        if not event_inp_native.exists():
            mutate_inp_for_event(cfg_path(cfg, "network.inp"), ev["rainfall_csv"], event_inp_native, int(ev["simulation_duration_min"]), strip_controls=False)
        internal_detail = base_dir / f"{event_id}__internal_rules_detail.csv"
        if args.resume and internal_detail.exists():
            base_row = _kpis_from_detail(internal_detail, priority, int(cfg["experiment"]["control_step_sec"]), "internal_rules", event_id, int(ev["duration_min"]))
        else:
            base_row = run_swmm_trajectory(
                event_inp_native,
                "internal_rules",
                actuators,
                priority,
                internal_detail,
                event_id,
                int(ev["duration_min"]),
                int(cfg["experiment"]["control_step_sec"]),
                int(cfg["experiment"]["random_seed"]),
                args.max_steps,
                simulation_duration_min=int(ev["simulation_duration_min"]),
                recession_min=int(cfg["experiment"]["recession_min"]),
            )
        baseline_rows.append(base_row)
    base_df = pd.DataFrame(baseline_rows)
    base_df.to_csv(out_root / "internal_baseline_results.csv", index=False)
    selected_base = base_df[pd.to_numeric(base_df["PFV"], errors="coerce").fillna(0.0) > float(args.min_internal_pfv)].sort_values("PFV", ascending=False)
    if args.max_events:
        selected_base = selected_base.head(int(args.max_events))
    template_names = set(_parse_csv_list(args.templates)) if args.templates else None
    delta_values = _parse_float_list(args.delta_list) or [float(args.max_delta)]
    skipped_no_change = 0
    raw_job_count = 0
    for _, brow in selected_base.iterrows():
        event_id = str(brow["event_id"])
        ev = rain_table[rain_table["event_id"].astype(str) == event_id].iloc[0]
        internal_detail = Path(str(brow["detail_file"]))
        detail = pd.read_csv(internal_detail)
        flood_cols = [f"flood:{n}" for n in priority if f"flood:{n}" in detail]
        if flood_cols:
            detail["_priority_rate"] = detail[flood_cols].fillna(0.0).sum(axis=1)
        else:
            detail["_priority_rate"] = 0.0
        detail["_priority_depth_max"] = detail.apply(lambda r: _priority_depth_max(r, priority), axis=1)
        risk_rows = detail[(detail["_priority_rate"] > 0) | (detail["_priority_depth_max"] >= 1.0)].copy()
        if risk_rows.empty:
            risk_rows = detail.sort_values("_priority_depth_max", ascending=False).head(max(1, int(args.risk_rows_per_event)))
        else:
            risk_rows = risk_rows.sort_values(["_priority_rate", "_priority_depth_max"], ascending=False).head(max(1, int(args.risk_rows_per_event)))
        for _, r in risk_rows.iterrows():
            for delta_value in delta_values:
                templates = _case_templates(actuators, float(delta_value), template_names=template_names)
                for template_name, deltas in templates:
                    raw_job_count += 1
                    case_id = f"{event_id}__t{float(r['elapsed_min']):.1f}__{template_name}__d{float(delta_value):.2f}".replace(".", "p")
                    if case_id in completed:
                        continue
                    feat = _build_case_features(r, actuators, priority, deltas)
                    if float(feat.get("feat_delta_abs_max", 0.0)) <= float(args.min_action_change):
                        skipped_no_change += 1
                        continue
                    meta = {
                        "case_id": case_id,
                        "event_id": event_id,
                        "template_name": template_name,
                        "candidate_scope": "all",
                        "residual_delta": float(delta_value),
                        "residual_delta_tier": _delta_tier(float(delta_value)),
                        "override_start_min": float(r["elapsed_min"]),
                        "override_steps": int(args.override_steps),
                        "feat_hold_steps": float(args.override_steps),
                        "baseline_TFV": float(brow["TFV"]),
                        "baseline_PFV": float(brow["PFV"]),
                        "baseline_peak_TFV_rate": float(brow["peak_TFV_rate"]),
                        **feat,
                    }
                    jobs.append(
                        {
                            "event_inp_native": str(inp_dir / f"{event_id}__with_controls.inp"),
                            "actuator_csv": str(cfg_path(cfg, "outputs.audit") / "actuator_table.csv"),
                            "priority": priority,
                            "detail_file": str(detail_dir / f"{case_id}_detail.csv"),
                            "event_id": event_id,
                            "duration_min": int(ev["duration_min"]),
                            "simulation_duration_min": int(ev["simulation_duration_min"]),
                            "recession_min": int(cfg["experiment"]["recession_min"]),
                            "control_step_sec": int(cfg["experiment"]["control_step_sec"]),
                            "seed": int(cfg["experiment"]["random_seed"]),
                            "max_steps": int(args.max_steps),
                            "override_start_min": float(r["elapsed_min"]),
                            "override_steps": int(args.override_steps),
                            "override_deltas": deltas,
                            "meta": meta,
                        }
                    )

    planned_jobs = jobs
    existing_unique_cases = int(existing["case_id"].astype(str).nunique()) if (not existing.empty and "case_id" in existing) else 0
    if int(args.target_total_cases) > 0:
        remaining_to_target = max(0, int(args.target_total_cases) - existing_unique_cases)
        if remaining_to_target <= 0:
            print(
                f"[residual_cf] target_total_cases={args.target_total_cases} already satisfied "
                f"by existing_unique_cases={existing_unique_cases}; no new SWMM jobs needed."
            )
            jobs = []
        elif int(args.max_cases) <= 0 or int(args.max_cases) > remaining_to_target:
            args.max_cases = remaining_to_target
    if args.selection_mode == "round_robin":
        jobs = _balanced_limit_jobs(jobs, int(args.max_cases))
    elif args.max_cases:
        jobs = jobs[: int(args.max_cases)]

    print(
        f"[residual_cf] selected_events={len(selected_base)} raw_jobs={raw_job_count} "
        f"eligible_jobs={len(planned_jobs)} jobs={len(jobs)} "
        f"skipped_no_change={skipped_no_change} resume_completed={len(completed)} "
        f"templates={','.join(sorted(template_names or [])) or 'all'} deltas={','.join(map(str, delta_values))}"
    )
    new_rows = []
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futs = [ex.submit(_run_case, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                row["delta_TFV"] = float(row["TFV"]) - float(row["baseline_TFV"])
                row["delta_PFV"] = float(row["PFV"]) - float(row["baseline_PFV"])
                row["delta_peak"] = float(row["peak_TFV_rate"]) - float(row["baseline_peak_TFV_rate"])
                row["y_pfv_improve"] = int(row["delta_PFV"] < 0)
                row["y_safe"] = int(row["delta_TFV"] <= 0 and row["delta_peak"] <= 0)
                row["tfv_guard"] = float(args.tfv_guard_pct) * float(row["baseline_TFV"])
                row["peak_guard"] = float(args.peak_guard_pct) * float(row["baseline_peak_TFV_rate"])
                row["y_safe_guarded"] = int(row["delta_TFV"] <= row["tfv_guard"] and row["delta_peak"] <= row["peak_guard"])
                new_rows.append(row)
                if i % 10 == 0 or i == len(jobs):
                    print(f"[residual_cf] done={i}/{len(jobs)}")
    out_df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if not existing.empty else pd.DataFrame(new_rows)
    if not out_df.empty:
        out_df = out_df.drop_duplicates("case_id", keep="last")
        out_df.to_csv(rows_path, index=False)
    report = {
        "out_dir": str(out_root),
        "baseline_events": int(len(base_df)),
        "selected_events": int(len(selected_base)),
        "raw_jobs": int(raw_job_count),
        "eligible_jobs_before_limit": int(len(planned_jobs)),
        "skipped_no_change": int(skipped_no_change),
        "new_cases": int(len(new_rows)),
        "total_cases": int(len(out_df)),
        "existing_unique_cases_before_run": int(existing_unique_cases),
        "target_total_cases": int(args.target_total_cases),
        "templates": sorted(template_names or []),
        "delta_values": delta_values,
        "delta_tiers": sorted({_delta_tier(x) for x in delta_values}),
        "risk_rows_per_event": int(args.risk_rows_per_event),
        "results": str(rows_path),
    }
    (out_root / "residual_counterfactual_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
