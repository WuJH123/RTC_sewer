from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.evaluation.policy_sets import normalize_policy_id


def _read_priority_nodes(cfg: dict) -> list[str]:
    path = cfg_path(cfg, "network.priority_nodes_file")
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _event_phase(elapsed_min: float, duration_min: float) -> str:
    duration = max(1.0, float(duration_min or 1.0))
    elapsed = float(elapsed_min)
    if elapsed < 0.50 * duration:
        return "pre_peak"
    if elapsed <= duration:
        return "storm_peak"
    if elapsed <= duration + 60.0:
        return "early_recession"
    return "late_recession"


def _prepare_detail(frame: pd.DataFrame, priority_nodes: list[str]) -> dict:
    if frame.empty or "elapsed_min" not in frame:
        return {
            "times": np.asarray([], dtype=float),
            "priority_cum": np.asarray([0.0], dtype=float),
            "total_cum": np.asarray([0.0], dtype=float),
            "total_rate": np.asarray([], dtype=float),
            "action_cols": [],
            "actions": np.zeros((0, 0), dtype=float),
            "action_index": {},
        }
    work = frame.copy()
    work["elapsed_min"] = pd.to_numeric(work["elapsed_min"], errors="coerce")
    work = work.dropna(subset=["elapsed_min"]).sort_values("elapsed_min").reset_index(drop=True)
    flood_cols = [c for c in work.columns if c.startswith("flood:")]
    pr_cols = [f"flood:{n}" for n in priority_nodes if f"flood:{n}" in work.columns]
    times = work["elapsed_min"].to_numpy(float)
    if flood_cols:
        total_rate = work[flood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(float)
    else:
        total_rate = np.zeros(len(work), dtype=float)
    if pr_cols:
        priority_rate = work[pr_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(float)
    else:
        priority_rate = np.zeros(len(work), dtype=float)
    action_cols = [c for c in work.columns if c.startswith("a:")]
    actions = (
        work[action_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        if action_cols
        else np.zeros((len(work), 0), dtype=float)
    )
    return {
        "times": times,
        "priority_cum": np.concatenate([[0.0], np.cumsum(priority_rate)]),
        "total_cum": np.concatenate([[0.0], np.cumsum(total_rate)]),
        "total_rate": total_rate,
        "action_cols": action_cols,
        "actions": actions,
        "action_index": {col: i for i, col in enumerate(action_cols)},
    }


def _horizon_metrics(prepared: dict, elapsed_min: float, horizon_steps: int, dt_sec: int) -> dict:
    times = prepared.get("times", np.asarray([], dtype=float))
    if len(times) == 0:
        return {"PFV_H": 0.0, "TFV_H": 0.0, "peak_TFV_rate_H": 0.0}
    start_idx = int(np.searchsorted(times, float(elapsed_min), side="left")) + 1
    end_idx = min(len(times), start_idx + int(horizon_steps))
    if start_idx >= end_idx:
        return {"PFV_H": 0.0, "TFV_H": 0.0, "peak_TFV_rate_H": 0.0}
    priority_cum = prepared.get("priority_cum", np.asarray([0.0], dtype=float))
    total_cum = prepared.get("total_cum", np.asarray([0.0], dtype=float))
    total_rate = prepared.get("total_rate", np.asarray([], dtype=float))
    return {
        "PFV_H": float((priority_cum[end_idx] - priority_cum[start_idx]) * int(dt_sec)),
        "TFV_H": float((total_cum[end_idx] - total_cum[start_idx]) * int(dt_sec)),
        "peak_TFV_rate_H": float(np.max(total_rate[start_idx:end_idx])) if end_idx > start_idx else 0.0,
    }


def _action_delta_features(candidate: dict, reference: dict, elapsed_min: float) -> dict:
    cand_cols = candidate.get("action_cols", [])
    ref_index = reference.get("action_index", {})
    action_cols = [c for c in cand_cols if c in ref_index]
    if not action_cols:
        return {"action_l1": 0.0, "changed_actuators": 0, "target_actuators": ""}
    cand_times = candidate.get("times", np.asarray([], dtype=float))
    ref_times = reference.get("times", np.asarray([], dtype=float))
    if len(cand_times) == 0 or len(ref_times) == 0:
        return {"action_l1": 0.0, "changed_actuators": 0, "target_actuators": ""}
    ci = int(np.argmin(np.abs(cand_times - float(elapsed_min))))
    ri = int(np.argmin(np.abs(ref_times - float(elapsed_min))))
    cand_index = candidate.get("action_index", {})
    c = np.asarray([candidate["actions"][ci, cand_index[col]] for col in action_cols], dtype=float)
    r = np.asarray([reference["actions"][ri, ref_index[col]] for col in action_cols], dtype=float)
    delta = c - r
    changed = [action_cols[i][2:] for i, v in enumerate(delta) if abs(float(v)) > 1e-6]
    return {
        "action_l1": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "changed_actuators": int(len(changed)),
        "target_actuators": ",".join(changed[:32]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--event-policy-metrics", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--horizon-steps", type=int, default=6)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pfv-tolerance-frac", type=float, default=0.01)
    ap.add_argument("--pfv-tolerance-abs", type=float, default=0.0)
    ap.add_argument("--peak-tolerance-frac", type=float, default=0.05)
    ap.add_argument("--tfv-required-reduction-frac", type=float, default=0.005)
    ap.add_argument("--policies", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    priority_nodes = _read_priority_nodes(cfg)
    dt_sec = int(cfg["experiment"]["control_step_sec"])
    metrics = _read_csv(args.event_policy_metrics)
    if metrics.empty:
        raise FileNotFoundError(f"Missing or empty event policy metrics: {args.event_policy_metrics}")
    required = {"event_id", "policy_id", "detail_file"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"event policy metrics missing columns: {sorted(missing)}")
    metrics = metrics.copy()
    metrics["policy_id"] = metrics["policy_id"].map(normalize_policy_id)
    event_meta = {}
    rain_table_path = cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv"
    if rain_table_path.exists():
        rain_table = pd.read_csv(rain_table_path)
        if "event_id" in rain_table:
            event_meta = rain_table.set_index(rain_table["event_id"].astype(str)).to_dict(orient="index")
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    if not policies:
        policies = sorted(p for p in metrics["policy_id"].astype(str).unique() if p != "no_control")
    else:
        policies = [normalize_policy_id(p) for p in policies]

    no_rows = metrics[metrics["policy_id"].astype(str).eq("no_control")][["event_id", "detail_file"]].copy()
    no_detail_by_event = {str(r.event_id): str(r.detail_file) for _, r in no_rows.iterrows()}
    detail_cache: dict[str, pd.DataFrame] = {}
    prepared_cache: dict[str, dict] = {}

    def detail(path: str) -> pd.DataFrame:
        if path not in detail_cache:
            detail_cache[path] = _read_csv(path)
        return detail_cache[path]

    def prepared(path: str) -> dict:
        if path not in prepared_cache:
            prepared_cache[path] = _prepare_detail(detail(path), priority_nodes)
        return prepared_cache[path]

    rows: list[dict] = []
    for _, row in metrics.iterrows():
        event_id = str(row.get("event_id", ""))
        policy_id = str(row.get("policy_id", ""))
        if policy_id == "no_control" or policy_id not in policies:
            continue
        no_path = no_detail_by_event.get(event_id, "")
        cand_path = str(row.get("detail_file", ""))
        if not no_path or not cand_path:
            continue
        ref = prepared(no_path)
        cand = prepared(cand_path)
        if len(ref.get("times", [])) == 0 or len(cand.get("times", [])) == 0:
            continue
        meta = event_meta.get(event_id, {})
        duration_min = float(meta.get("duration_min", row.get("duration_min", 0.0)) or 0.0)
        times = np.asarray(cand["times"], dtype=float)
        for elapsed in times[:: max(1, int(args.stride))]:
            cm = _horizon_metrics(cand, elapsed, int(args.horizon_steps), dt_sec)
            rm = _horizon_metrics(ref, elapsed, int(args.horizon_steps), dt_sec)
            pfv_tol = max(float(args.pfv_tolerance_abs), float(args.pfv_tolerance_frac) * float(rm["PFV_H"]))
            peak_tol = float(args.peak_tolerance_frac) * float(rm["peak_TFV_rate_H"])
            tfv_req = float(args.tfv_required_reduction_frac) * float(rm["TFV_H"])
            delta_pfv = float(cm["PFV_H"] - rm["PFV_H"])
            delta_tfv = float(cm["TFV_H"] - rm["TFV_H"])
            delta_peak = float(cm["peak_TFV_rate_H"] - rm["peak_TFV_rate_H"])
            feats = _action_delta_features(cand, ref, float(elapsed))
            rows.append(
                {
                    "event_id": event_id,
                    "rain_id": str(meta.get("rain_id", "")),
                    "duration_min": duration_min,
                    "pattern": str(meta.get("pattern", "")),
                    "event_phase": _event_phase(float(elapsed), duration_min),
                    "policy_id": policy_id,
                    "elapsed_min": float(elapsed),
                    "candidate_detail_file": cand_path,
                    "no_control_detail_file": no_path,
                    "candidate_PFV_H": cm["PFV_H"],
                    "no_control_PFV_H": rm["PFV_H"],
                    "delta_PFV_vs_no": delta_pfv,
                    "candidate_TFV_H": cm["TFV_H"],
                    "no_control_TFV_H": rm["TFV_H"],
                    "delta_TFV_vs_no": delta_tfv,
                    "candidate_peak_H": cm["peak_TFV_rate_H"],
                    "no_control_peak_H": rm["peak_TFV_rate_H"],
                    "delta_peak_vs_no": delta_peak,
                    "pfv_tolerance": pfv_tol,
                    "tfv_required_reduction": tfv_req,
                    "peak_tolerance": peak_tol,
                    "pfv_noninferior": bool(delta_pfv <= pfv_tol),
                    "tfv_improved": bool(delta_tfv <= -tfv_req),
                    "peak_safe": bool(delta_peak <= peak_tol),
                    "repair_safe_label": bool(delta_pfv <= pfv_tol and delta_tfv <= -tfv_req and delta_peak <= peak_tol),
                    **feats,
                }
            )

    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else cfg_path(cfg, "outputs.diagnostics") / "no_control_repair_supervision")
    data = pd.DataFrame(rows)
    out_csv = out_dir / "no_control_repair_supervision.csv"
    data.to_csv(out_csv, index=False, encoding="utf-8-sig")
    actuator_summary = pd.DataFrame()
    if not data.empty and "target_actuators" in data:
        exploded = []
        for _, row in data.iterrows():
            targets = [x.strip() for x in str(row.get("target_actuators", "")).split(",") if x.strip()]
            if not targets:
                targets = ["__hold_or_no_action__"]
            for actuator_id in targets:
                item = row.to_dict()
                item["actuator_id"] = actuator_id
                exploded.append(item)
        if exploded:
            exp = pd.DataFrame(exploded)
            actuator_summary = (
                exp.groupby("actuator_id", as_index=False)
                .agg(
                    rows=("repair_safe_label", "size"),
                    events=("event_id", "nunique"),
                    policies=("policy_id", lambda x: ",".join(sorted(set(map(str, x))))),
                    repair_safe_frac=("repair_safe_label", "mean"),
                    pfv_noninferior_frac=("pfv_noninferior", "mean"),
                    tfv_improved_frac=("tfv_improved", "mean"),
                    peak_safe_frac=("peak_safe", "mean"),
                    mean_delta_PFV_vs_no=("delta_PFV_vs_no", "mean"),
                    mean_delta_TFV_vs_no=("delta_TFV_vs_no", "mean"),
                    mean_delta_peak_vs_no=("delta_peak_vs_no", "mean"),
                )
                .sort_values(["repair_safe_frac", "rows"], ascending=[False, False])
            )
    actuator_summary_csv = out_dir / "no_control_repair_target_actuator_summary.csv"
    actuator_summary.to_csv(actuator_summary_csv, index=False, encoding="utf-8-sig")
    pattern_phase_summary = pd.DataFrame()
    if not data.empty and "target_actuators" in data:
        exploded = []
        for _, row in data.iterrows():
            targets = [x.strip() for x in str(row.get("target_actuators", "")).split(",") if x.strip()]
            if not targets:
                targets = ["__hold_or_no_action__"]
            for actuator_id in targets:
                item = row.to_dict()
                item["actuator_id"] = actuator_id
                exploded.append(item)
        if exploded:
            exp = pd.DataFrame(exploded)
            pattern_phase_summary = (
                exp.groupby(["actuator_id", "pattern", "event_phase"], as_index=False)
                .agg(
                    rows=("repair_safe_label", "size"),
                    events=("event_id", "nunique"),
                    policies=("policy_id", lambda x: ",".join(sorted(set(map(str, x))))),
                    repair_safe_frac=("repair_safe_label", "mean"),
                    pfv_noninferior_frac=("pfv_noninferior", "mean"),
                    tfv_improved_frac=("tfv_improved", "mean"),
                    peak_safe_frac=("peak_safe", "mean"),
                    mean_delta_PFV_vs_no=("delta_PFV_vs_no", "mean"),
                    mean_delta_TFV_vs_no=("delta_TFV_vs_no", "mean"),
                    mean_delta_peak_vs_no=("delta_peak_vs_no", "mean"),
                )
                .sort_values(["pattern", "event_phase", "repair_safe_frac", "rows"], ascending=[True, True, False, False])
            )
    pattern_phase_summary_csv = out_dir / "no_control_repair_target_actuator_by_pattern_phase.csv"
    pattern_phase_summary.to_csv(pattern_phase_summary_csv, index=False, encoding="utf-8-sig")
    report = {
        "event_policy_metrics": str(args.event_policy_metrics),
        "rows": int(len(data)),
        "events": int(data["event_id"].nunique()) if not data.empty else 0,
        "policies": sorted(data["policy_id"].astype(str).unique().tolist()) if not data.empty else [],
        "pfv_noninferior_frac": float(data["pfv_noninferior"].mean()) if not data.empty else None,
        "tfv_improved_frac": float(data["tfv_improved"].mean()) if not data.empty else None,
        "peak_safe_frac": float(data["peak_safe"].mean()) if not data.empty else None,
        "repair_safe_frac": float(data["repair_safe_label"].mean()) if not data.empty else None,
        "output_csv": str(out_csv),
        "target_actuator_summary_csv": str(actuator_summary_csv),
        "target_actuator_pattern_phase_summary_csv": str(pattern_phase_summary_csv),
    }
    (out_dir / "no_control_repair_supervision_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
