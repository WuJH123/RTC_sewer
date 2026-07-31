from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.control.candidate_generator import (
    candidate_metadata_features,
    generate_labeled_candidates,
    parse_candidate_label,
    _slug,
)
from sewerrtc.graph.graph_builder import khop_nodes
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


def _as_name_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.split(",")
    else:
        raw = list(value)
    out: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _as_int_tuple(value, default: tuple[int, ...] = (1, 2)) -> tuple[int, ...]:
    vals = []
    for item in _as_name_list(value):
        try:
            iv = int(float(item))
        except Exception:
            continue
        if iv > 0 and iv not in vals:
            vals.append(iv)
    return tuple(vals) or default


def _as_scope_map(value) -> dict[str, list[str]]:
    if not value:
        return {}
    out = {}
    for key, scopes in dict(value).items():
        out[str(key).strip()] = _as_name_list(scopes)
    return {k: v for k, v in out.items() if k and v}


def _candidate_label_allowed(
    label: str,
    allowed_templates: list[str],
    blocked_templates: list[str],
    allowed_scopes_by_template: dict[str, list[str]],
    max_delta: float,
) -> bool:
    meta = parse_candidate_label(label)
    template = str(meta.get("template", ""))
    scope = str(meta.get("scope", ""))
    try:
        delta = abs(float(meta.get("delta", 0.0) or 0.0))
    except Exception:
        delta = 0.0
    if allowed_templates and template not in set(allowed_templates):
        return False
    if template in set(blocked_templates):
        return False
    allowed_scopes = allowed_scopes_by_template.get(template) or allowed_scopes_by_template.get("*") or []
    if allowed_scopes and scope not in set(allowed_scopes):
        return False
    if delta > float(max_delta) + 1e-9:
        return False
    return True


def _load_high_risk_event_ids(cfg: dict, explicit_path: str, default_limit: int) -> list[str]:
    path = Path(explicit_path) if explicit_path else cfg_path(cfg, "project_root") / "outputs" / "evaluation_project5_priority_zone" / "project5_priority_event_table.csv"
    table = _safe_read(path)
    if table.empty or "event_id" not in table:
        return []
    if "event_risk_class" in table:
        table = table[table["event_risk_class"].astype(str).eq("high_risk_event")].copy()
    if "internal_project5_priority_PFV" in table:
        table["_rank_pfv"] = pd.to_numeric(table["internal_project5_priority_PFV"], errors="coerce").fillna(0.0)
        table = table.sort_values("_rank_pfv", ascending=False)
    ids = table["event_id"].astype(str).drop_duplicates().tolist()
    if default_limit:
        ids = ids[: int(default_limit)]
    return ids


def _select_key_window_rows(hist: pd.DataFrame, samples_per_phase: int) -> pd.DataFrame:
    if hist.empty or "event_id" not in hist:
        return hist
    phase_col = "phase" if "phase" in hist else None
    if phase_col is None:
        return hist.head(max(0, int(samples_per_phase)) * max(1, hist["event_id"].nunique()))
    phases = ["pre_peak", "peak", "recession"]
    work = hist[hist[phase_col].astype(str).isin(phases)].copy()
    if work.empty:
        return hist
    for col in ["_candidate_delta_pfv", "_candidate_safe_prob", "_candidate_peak_nonworse_prob", "elapsed_min"]:
        if col in work:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "_candidate_passes_all" in work:
        work["_pass_sort"] = work["_candidate_passes_all"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    else:
        work["_pass_sort"] = 0
    sort_cols = ["_pass_sort"]
    ascending = [False]
    if "_candidate_delta_pfv" in work:
        sort_cols.append("_candidate_delta_pfv")
        ascending.append(True)
    if "_candidate_safe_prob" in work:
        sort_cols.append("_candidate_safe_prob")
        ascending.append(False)
    if "elapsed_min" in work:
        sort_cols.append("elapsed_min")
        ascending.append(True)
    work = work.sort_values(sort_cols, ascending=ascending)
    selected = []
    n = max(1, int(samples_per_phase))
    for _, group in work.groupby(["event_id", phase_col], sort=False):
        selected.append(group.head(n))
    if not selected:
        return work.iloc[0:0].copy()
    out = pd.concat(selected, ignore_index=True, sort=False)
    return out.drop(columns=[c for c in ["_pass_sort"] if c in out], errors="ignore")


def _priority_depth_max(row: pd.Series, priority: list[str]) -> float:
    vals = []
    for nid in priority:
        col = f"h:{nid}"
        if col in row:
            vals.append(pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0])
    vals = [float(v) for v in vals if np.isfinite(v)]
    return float(max(vals)) if vals else 0.0


def _nearest_detail_row(detail: pd.DataFrame, elapsed_min: float) -> pd.Series | None:
    if detail.empty or "elapsed_min" not in detail:
        return None
    t = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    idx = (t - float(elapsed_min)).abs().idxmin()
    return detail.loc[idx]


def _event_duration(event_id: str, rain_table: pd.DataFrame) -> tuple[pd.Series, int]:
    r = rain_table[rain_table["event_id"].astype(str) == str(event_id)]
    if r.empty:
        raise KeyError(f"event_id not found in rainfall table: {event_id}")
    ev = r.iloc[0]
    return ev, int(ev["duration_min"])


def _prepare_native_inp(cfg: dict, event_id: str, ev: pd.Series, target_dir: Path) -> Path:
    target = target_dir / f"{event_id}__with_controls.inp"
    if not target.exists():
        mutate_inp_for_event(
            cfg_path(cfg, "network.inp"),
            ev["rainfall_csv"],
            target,
            int(ev["simulation_duration_min"]),
            strip_controls=False,
        )
    return target


def _internal_detail_path(source_root: Path, residual_root: Path, event_id: str) -> Path:
    candidates = [
        source_root / "baselines" / "internal_rules" / f"{event_id}__internal_rules_detail.csv",
        residual_root / "internal_rules" / f"{event_id}__internal_rules_detail.csv",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return candidates[0]


def _build_job(
    cfg: dict,
    source_root: Path,
    residual_root: Path,
    rain_table: pd.DataFrame,
    actuators: pd.DataFrame,
    priority: list[str],
    node_order: list[str],
    priority_upstream_nodes: set[str],
    priority_downstream_nodes: set[str],
    hist_row: pd.Series,
    max_delta: float,
    override_steps: int,
    max_steps: int,
    candidate_hold_steps: tuple[int, ...],
    allowed_templates: list[str],
    blocked_templates: list[str],
    allowed_scopes_by_template: dict[str, list[str]],
) -> dict | None:
    event_id = str(hist_row["event_id"])
    label = str(hist_row.get("selected_candidate_label", "")).strip()
    if not label or label == "nominal":
        return None
    meta_label = parse_candidate_label(label)
    ev, duration_min = _event_duration(event_id, rain_table)
    event_inp = _prepare_native_inp(cfg, event_id, ev, ensure_dir(residual_root / "event_inp"))
    internal_detail = _internal_detail_path(source_root, residual_root, event_id)
    if not internal_detail.exists():
        internal_row = run_swmm_trajectory(
            event_inp,
            "internal_rules",
            actuators,
            priority,
            internal_detail,
            event_id,
            duration_min,
            int(cfg["experiment"]["control_step_sec"]),
            int(cfg["experiment"]["random_seed"]),
            max_steps,
            simulation_duration_min=int(ev["simulation_duration_min"]),
            recession_min=int(cfg["experiment"]["recession_min"]),
        )
    else:
        internal_detail_df = pd.read_csv(internal_detail)
        internal_row = compute_kpis(internal_detail_df, priority, int(cfg["experiment"]["control_step_sec"]))
        internal_row.update({"event_id": event_id, "policy_id": "internal_rules", "detail_file": str(internal_detail)})
    detail = pd.read_csv(internal_detail)
    elapsed_min = float(hist_row["elapsed_min"])
    drow = _nearest_detail_row(detail, elapsed_min)
    if drow is None:
        return None
    actuator_ids = actuators["actuator_id"].astype(str).tolist()
    nominal = np.asarray([float(drow.get(f"a:{aid}", 1.0)) for aid in actuator_ids], dtype=np.float32)
    phase = str(drow.get("phase", hist_row.get("phase", "")))
    rainfall = float(drow.get("rainfall_mm_h", hist_row.get("rainfall_mm_h", 0.0)))
    state = np.asarray([float(drow.get(f"h:{nid}", 0.0)) for nid in node_order], dtype=np.float32)
    candidates = generate_labeled_candidates(
        nominal,
        actuators,
        phase,
        max_delta=float(max_delta),
        include_nominal=False,
        state=state,
        priority_upstream_nodes=priority_upstream_nodes,
        priority_downstream_nodes=priority_downstream_nodes,
        hold_steps=candidate_hold_steps,
        allowed_templates=allowed_templates,
        blocked_templates=blocked_templates,
        allowed_scopes_by_template=allowed_scopes_by_template,
    )
    chosen = None
    for cand_label, action in candidates:
        if cand_label == label:
            chosen = action
            break
    if chosen is None:
        return None
    delta = np.asarray(chosen, dtype=np.float32) - nominal
    if float(np.nanmax(np.abs(delta))) <= 1e-6:
        return None
    override_deltas = {
        aid: float(delta[i])
        for i, aid in enumerate(actuator_ids)
        if abs(float(delta[i])) > 1e-6
    }
    feat = build_residual_feature_dict(
        actuators,
        nominal,
        chosen,
        phase,
        rainfall,
        _priority_depth_max(drow, priority),
        elapsed_min,
    )
    feat.update(candidate_metadata_features(label))
    hold_steps = int(meta_label.get("hold_steps", override_steps) or override_steps)
    case_id = f"onpolicy__{source_root.name}__{event_id}__t{elapsed_min:.1f}__{_slug(label)}".replace(".", "p")
    detail_hash = hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:16]
    return {
        "event_inp_native": str(event_inp),
        "actuator_csv": str(cfg_path(cfg, "outputs.audit") / "actuator_table.csv"),
        "priority": priority,
        "detail_file": str(residual_root / "details" / f"onpolicy_{detail_hash}_detail.csv"),
        "event_id": event_id,
        "duration_min": duration_min,
        "simulation_duration_min": int(ev["simulation_duration_min"]),
        "recession_min": int(cfg["experiment"]["recession_min"]),
        "control_step_sec": int(cfg["experiment"]["control_step_sec"]),
        "seed": int(cfg["experiment"]["random_seed"]),
        "max_steps": int(max_steps),
        "override_start_min": elapsed_min,
        "override_steps": hold_steps,
        "override_deltas": override_deltas,
        "meta": {
            "case_id": case_id,
            "detail_hash": detail_hash,
            "event_id": event_id,
            "template_name": label,
            "candidate_scope": str(meta_label.get("scope", "unknown")),
            "residual_delta": float(np.nanmax(np.abs(delta))),
            "candidate_label_delta": float(meta_label.get("delta", np.nan))
            if np.isfinite(float(meta_label.get("delta", np.nan)))
            else float(np.nanmax(np.abs(delta))),
            "residual_delta_tier": "small" if np.nanmax(np.abs(delta)) <= 0.080001 else "medium",
            "override_start_min": elapsed_min,
            "override_steps": hold_steps,
            "source_run_tag": source_root.name,
            "source_history_file": str(hist_row.get("_history_file", "")),
            "source_candidate_rank": int(hist_row.get("_candidate_rank", -1)),
            "source_candidate_passes_all": bool(hist_row.get("_candidate_passes_all", False)),
            "source_candidate_rejection_reason": str(hist_row.get("_candidate_rejection_reason", "")),
            "baseline_TFV": float(internal_row.get("TFV", 0.0)),
            "baseline_PFV": float(internal_row.get("PFV", 0.0)),
            "baseline_peak_TFV_rate": float(internal_row.get("peak_TFV_rate", 0.0)),
            **feat,
        },
    }


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


def _load_on_policy_rows(
    source_root: Path,
    allowed_templates: list[str],
    blocked_templates: list[str],
    allowed_scopes_by_template: dict[str, list[str]],
    max_delta: float,
) -> pd.DataFrame:
    history_rows = []
    for path in sorted((source_root / "proposed").glob("*__controller_history.csv")):
        df = _safe_read(path)
        if df.empty:
            continue
        df["_history_file"] = str(path)
        history_rows.append(df)
    if not history_rows:
        return pd.DataFrame()
    all_rows = pd.concat(history_rows, ignore_index=True, sort=False)
    expanded = []
    if "topk_candidates_json" in all_rows:
        for _, row in all_rows.iterrows():
            raw = row.get("topk_candidates_json", "")
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                candidates = json.loads(raw)
            except Exception:
                continue
            if not isinstance(candidates, list):
                continue
            for rank, cand in enumerate(candidates):
                label = str(cand.get("candidate_label", "")).strip()
                if not label or label == "nominal":
                    continue
                if not _candidate_label_allowed(
                    label,
                    allowed_templates,
                    blocked_templates,
                    allowed_scopes_by_template,
                    max_delta,
                ):
                    continue
                r = row.copy()
                r["selected_candidate_label"] = label
                r["_candidate_rank"] = int(rank)
                r["_candidate_passes_all"] = bool(cand.get("passes_all", False))
                r["_candidate_rejection_reason"] = str(cand.get("rejection_reason", ""))
                for key, val in cand.items():
                    if key == "candidate_label":
                        continue
                    r[f"_candidate_{key}"] = val
                expanded.append(r)
    if expanded:
        out = pd.DataFrame(expanded)
    else:
        out = all_rows.copy()
        if "fallback_to_nominal" in out:
            fallback = out["fallback_to_nominal"].astype(str).str.lower().isin(["true", "1", "yes"])
            out = out[~fallback]
        if "selected_candidate_label" in out:
            label = out["selected_candidate_label"].fillna("").astype(str)
            out = out[label.ne("") & label.ne("nominal")]
            out = out[
                out["selected_candidate_label"].map(
                    lambda x: _candidate_label_allowed(
                        str(x),
                        allowed_templates,
                        blocked_templates,
                        allowed_scopes_by_template,
                        max_delta,
                    )
                )
            ].copy()
    if out.empty:
        return out
    keep_cols = ["event_id", "elapsed_min", "selected_candidate_label"]
    out = out.drop_duplicates([c for c in keep_cols if c in out.columns]).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--mode", choices=["debug", "formal"], default="debug")
    ap.add_argument("--source-run-tag", default="native_shield_safe010")
    ap.add_argument("--max-events", type=int, default=25)
    ap.add_argument("--samples-per-phase", type=int, default=4)
    ap.add_argument("--event-table", default="")
    ap.add_argument("--include-non-high-risk", action="store_true")
    ap.add_argument("--allowed-templates", default="")
    ap.add_argument("--blocked-templates", default="")
    ap.add_argument("--dry-run-plan", action="store_true")
    ap.add_argument("--max-cases", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-delta", type=float, default=0.08)
    ap.add_argument("--override-steps", type=int, default=3)
    ap.add_argument("--priority-khop", type=int, default=3)
    ap.add_argument("--append-to-residual", action="store_true", default=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    shield_cfg = ((cfg.get("intervention_policy", {}) or {}).get("project5_safety_shield", {}) or {})
    allowed_templates = _as_name_list(args.allowed_templates) or _as_name_list(shield_cfg.get("allowed_templates"))
    blocked_templates = _as_name_list(args.blocked_templates) or _as_name_list(shield_cfg.get("blocked_templates"))
    allowed_scopes_by_template = _as_scope_map(shield_cfg.get("allowed_scopes_by_template"))
    candidate_hold_steps = _as_int_tuple(shield_cfg.get("candidate_hold_steps"), default=(1, 2))
    if float(args.max_delta) <= 0:
        args.max_delta = float(shield_cfg.get("max_candidate_delta", 0.08))
    source_root = cfg_path(cfg, "outputs.closed_loop") / args.mode / args.source_run_tag
    if not source_root.exists():
        raise FileNotFoundError(f"Missing source closed-loop run: {source_root}")
    residual_root = ensure_dir(cfg_path(cfg, "outputs.closed_loop") / "internal_residual_counterfactuals")
    rows_path = residual_root / "residual_counterfactual_results.csv"
    existing = _safe_read(rows_path)
    completed = set(existing["case_id"].astype(str)) if args.resume and not existing.empty and "case_id" in existing else set()

    hist = _load_on_policy_rows(
        source_root,
        allowed_templates=allowed_templates,
        blocked_templates=blocked_templates,
        allowed_scopes_by_template=allowed_scopes_by_template,
        max_delta=float(args.max_delta),
    )
    if hist.empty:
        report = {
            "source_run": str(source_root),
            "jobs": 0,
            "new_cases": 0,
            "reason": "no non-fallback SafetyShield-allowed selected actions found in controller history",
            "allowed_templates": allowed_templates,
            "blocked_templates": blocked_templates,
            "allowed_scopes_by_template": allowed_scopes_by_template,
        }
        (residual_root / "on_policy_counterfactual_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    rain_table = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv")
    if not args.include_non_high_risk:
        high_risk_ids = _load_high_risk_event_ids(cfg, args.event_table, int(args.max_events))
        if high_risk_ids:
            hist = hist[hist["event_id"].astype(str).isin(high_risk_ids)].copy()
        else:
            print("[on_policy_cf] no Project5 high-risk event table found; keeping all source events")
    elif args.max_events:
        keep = hist["event_id"].astype(str).drop_duplicates().head(int(args.max_events)).tolist()
        hist = hist[hist["event_id"].astype(str).isin(keep)].copy()
    hist = _select_key_window_rows(hist, int(args.samples_per_phase))
    if args.max_cases:
        hist = hist.head(int(args.max_cases))

    actuators = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    node_table = pd.read_csv(cfg_path(cfg, "outputs.audit") / "node_table.csv")
    link_table = pd.read_csv(cfg_path(cfg, "outputs.audit") / "link_table.csv")
    node_order = node_table["node_id"].astype(str).tolist()
    priority = (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines()
    priority_upstream_nodes = khop_nodes(link_table, priority, k=int(args.priority_khop), direction="upstream")
    priority_downstream_nodes = khop_nodes(link_table, priority, k=int(args.priority_khop), direction="downstream")
    ensure_dir(residual_root / "job_plans")
    hist.to_csv(residual_root / "job_plans" / f"on_policy_project5_selection_{args.mode}_{args.source_run_tag}.csv", index=False)
    jobs = []
    skipped = 0
    for _, row in hist.iterrows():
        job = _build_job(
            cfg,
            source_root,
            residual_root,
            rain_table,
            actuators,
            priority,
            node_order,
            priority_upstream_nodes,
            priority_downstream_nodes,
            row,
            float(args.max_delta),
            int(args.override_steps),
            int(args.max_steps),
            candidate_hold_steps,
            allowed_templates,
            blocked_templates,
            allowed_scopes_by_template,
        )
        if job is None:
            skipped += 1
            continue
        if job["meta"]["case_id"] in completed:
            skipped += 1
            continue
        jobs.append(job)

    print(f"[on_policy_cf] source={source_root} selected_rows={len(hist)} jobs={len(jobs)} skipped={skipped}")
    if jobs:
        pd.DataFrame([j["meta"] for j in jobs]).to_csv(
            residual_root / "job_plans" / f"on_policy_project5_jobs_{args.mode}_{args.source_run_tag}.csv",
            index=False,
        )
    if args.dry_run_plan:
        report = {
            "source_run": str(source_root),
            "selected_history_rows": int(len(hist)),
            "jobs": int(len(jobs)),
            "skipped": int(skipped),
            "dry_run_plan": True,
            "allowed_templates": allowed_templates,
            "blocked_templates": blocked_templates,
            "allowed_scopes_by_template": allowed_scopes_by_template,
            "candidate_hold_steps": list(candidate_hold_steps),
            "selection_file": str(residual_root / "job_plans" / f"on_policy_project5_selection_{args.mode}_{args.source_run_tag}.csv"),
            "job_plan_file": str(residual_root / "job_plans" / f"on_policy_project5_jobs_{args.mode}_{args.source_run_tag}.csv"),
        }
        (residual_root / "on_policy_counterfactual_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
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
                row["tfv_guard"] = 0.005 * float(row["baseline_TFV"])
                row["peak_guard"] = 0.010 * float(row["baseline_peak_TFV_rate"])
                row["y_safe"] = int(row["delta_TFV"] <= 0 and row["delta_peak"] <= 0)
                row["y_safe_guarded"] = int(row["delta_TFV"] <= row["tfv_guard"] and row["delta_peak"] <= row["peak_guard"])
                new_rows.append(row)
                if i % 10 == 0 or i == len(jobs):
                    print(f"[on_policy_cf] done={i}/{len(jobs)}")

    out_df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True, sort=False) if not existing.empty else pd.DataFrame(new_rows)
    if not out_df.empty:
        out_df = out_df.drop_duplicates("case_id", keep="last")
        out_df.to_csv(rows_path, index=False)
    report = {
        "source_run": str(source_root),
        "selected_history_rows": int(len(hist)),
        "jobs": int(len(jobs)),
        "skipped": int(skipped),
        "new_cases": int(len(new_rows)),
        "total_residual_cases": int(len(out_df)),
        "results": str(rows_path),
        "allowed_templates": allowed_templates,
        "blocked_templates": blocked_templates,
        "allowed_scopes_by_template": allowed_scopes_by_template,
        "candidate_hold_steps": list(candidate_hold_steps),
    }
    (residual_root / "on_policy_counterfactual_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
