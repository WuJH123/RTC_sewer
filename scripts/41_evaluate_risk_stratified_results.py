from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.evaluation.risk_stratified import (
    RiskThresholds,
    build_event_table,
    build_risk_stratified_comparison,
    build_water_research_tables,
    recompute_priority_kpis_for_results,
    summarize_risk_strata,
)
from sewerrtc.io.priority_config import combined_priority_depth_nodes, configured_priority_nodes, priority_config_summary
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _run_root(cfg: dict, mode: str, run_tag: str) -> Path:
    root = cfg_path(cfg, "outputs.closed_loop") / mode
    if run_tag:
        root = root / run_tag
    return root


def _read_histories(root: Path) -> pd.DataFrame:
    rows = []
    proposed_dir = root / "proposed"
    if not proposed_dir.exists():
        return pd.DataFrame()
    for path in sorted(proposed_dir.glob("*__controller_history.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        if "event_id" not in df:
            df["event_id"] = path.name.split("__")[0]
        df["history_file"] = str(path)
        rows.append(df)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _history_diagnostic_tables(history: pd.DataFrame, event_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if history.empty:
        empty = pd.DataFrame()
        return {
            "intervention_reason_summary": empty,
            "action_uncertainty_summary": empty,
            "stage_action_usage_by_phase": empty,
        }
    hist = history.copy()
    if "event_risk_class" not in hist:
        hist = hist.merge(event_table[["event_id", "event_risk_class"]], on="event_id", how="left")
    for col, default in [
        ("event_risk_class", "unknown"),
        ("phase", "unknown"),
        ("intervention_reason", "unknown"),
        ("selected_candidate_label", ""),
    ]:
        if col not in hist:
            hist[col] = default
        hist[col] = hist[col].fillna(default).astype(str)
    if "fallback_to_nominal" not in hist:
        hist["fallback_to_nominal"] = True
    hist["fallback_to_nominal"] = hist["fallback_to_nominal"].fillna(True).astype(bool)
    hist["is_intervention"] = ~hist["fallback_to_nominal"]
    reason = (
        hist.groupby(["event_risk_class", "intervention_reason"], dropna=False)
        .agg(
            steps=("event_id", "size"),
            events=("event_id", "nunique"),
            intervention_steps=("is_intervention", "sum"),
            fallback_steps=("fallback_to_nominal", "sum"),
        )
        .reset_index()
    )
    reason["intervention_step_frac"] = reason["intervention_steps"] / reason["steps"].clip(lower=1)
    for c in [
        "delta_PFV_p50",
        "delta_PFV_p90",
        "delta_TFV_p90",
        "delta_peak_p90",
        "uncertainty_score",
        "predicted_pfv_gain",
    ]:
        if c not in hist:
            hist[c] = float("nan")
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    if "uncertainty_gate_pass" not in hist:
        hist["uncertainty_gate_pass"] = False
    hist["uncertainty_gate_pass"] = hist["uncertainty_gate_pass"].fillna(False).astype(bool)
    uncertainty = (
        hist.groupby("event_risk_class", dropna=False)
        .agg(
            steps=("event_id", "size"),
            intervention_steps=("is_intervention", "sum"),
            uncertainty_gate_pass_rate=("uncertainty_gate_pass", "mean"),
            mean_delta_PFV_p50=("delta_PFV_p50", "mean"),
            mean_delta_PFV_p90=("delta_PFV_p90", "mean"),
            mean_delta_TFV_p90=("delta_TFV_p90", "mean"),
            mean_delta_peak_p90=("delta_peak_p90", "mean"),
            mean_uncertainty_score=("uncertainty_score", "mean"),
            mean_predicted_pfv_gain=("predicted_pfv_gain", "mean"),
        )
        .reset_index()
    )
    phase = (
        hist.groupby(["event_risk_class", "phase", "selected_candidate_label"], dropna=False)
        .agg(
            steps=("event_id", "size"),
            events=("event_id", "nunique"),
            intervention_steps=("is_intervention", "sum"),
        )
        .reset_index()
        .sort_values(["event_risk_class", "phase", "intervention_steps", "steps"], ascending=[True, True, False, False])
    )
    return {
        "intervention_reason_summary": reason,
        "action_uncertainty_summary": uncertainty,
        "stage_action_usage_by_phase": phase,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--mode", choices=["debug", "formal"], default="formal")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--baseline-policy", default="internal_rules")
    ap.add_argument("--event-table", default="")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = _run_root(cfg, args.mode, args.run_tag)
    proposed_path = root / "proposed_results.csv"
    baseline_path = root / "baseline_results.csv"
    if not proposed_path.exists():
        raise FileNotFoundError(f"Missing proposed results: {proposed_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline results: {baseline_path}")
    proposed = pd.read_csv(proposed_path)
    baseline = pd.read_csv(baseline_path)
    priority_nodes = configured_priority_nodes(cfg)
    depth_nodes = combined_priority_depth_nodes(cfg)
    thresholds = RiskThresholds.from_config(cfg)
    baseline = recompute_priority_kpis_for_results(
        baseline,
        priority_nodes,
        dt_sec=int(thresholds.control_step_sec),
        keep_original=True,
    )
    proposed = recompute_priority_kpis_for_results(
        proposed,
        priority_nodes,
        dt_sec=int(thresholds.control_step_sec),
        keep_original=True,
    )
    if args.event_table:
        event_table = pd.read_csv(args.event_table)
    else:
        event_table = build_event_table(
            baseline,
            priority_nodes,
            thresholds,
            baseline_policy=args.baseline_policy,
            depth_nodes=depth_nodes,
        )
    history = _read_histories(root)
    comp = build_risk_stratified_comparison(
        proposed,
        baseline,
        event_table,
        history,
        baseline_policy=args.baseline_policy,
    )
    summary = summarize_risk_strata(comp)
    tables = build_water_research_tables(comp, summary)
    history_tables = _history_diagnostic_tables(history, event_table)

    out_dir = Path(args.out_dir) if args.out_dir else cfg_path(cfg, "project_root") / "outputs" / "evaluation"
    out_dir = ensure_dir(out_dir)
    event_table.to_csv(out_dir / "risk_stratified_event_table.csv", index=False)
    comp.to_csv(out_dir / "risk_stratified_comparison.csv", index=False)
    summary.to_csv(out_dir / "water_research_risk_stratified_table.csv", index=False)
    summary.to_csv(out_dir / "risk_stratified_metrics_by_class.csv", index=False)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    for name, table in history_tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    high = tables.get("high_risk_success_table", pd.DataFrame())
    low = tables.get("low_risk_false_intervention_table", pd.DataFrame())
    high.to_csv(out_dir / "high_risk_event_metrics.csv", index=False)
    low.to_csv(out_dir / "low_risk_false_intervention_report.csv", index=False)

    high_pfv_mean = None
    high_tfv_mean = None
    high_peak_mean = None
    low_false = None
    if not summary.empty:
        h = summary[summary["event_risk_class"].eq("high_risk_event")]
        if not h.empty:
            high_pfv_mean = float(h["PFV_mean_reduction_pct"].iloc[0])
            high_tfv_mean = float(h["TFV_mean_reduction_pct"].iloc[0])
            high_peak_mean = float(h["peak_TFV_rate_mean_reduction_pct"].iloc[0])
        l = summary[summary["event_risk_class"].eq("low_risk_event")]
        if not l.empty and "false_intervention_rate" in l:
            low_false = float(l["false_intervention_rate"].iloc[0])
    result = {
        "mode": args.mode,
        "run_tag": args.run_tag,
        "baseline_policy": args.baseline_policy,
        "events": int(len(event_table)),
        "risk_class_counts": {str(k): int(v) for k, v in event_table["event_risk_class"].value_counts().to_dict().items()},
        "priority": priority_config_summary(cfg),
        "high_risk_PFV_mean_reduction_pct": high_pfv_mean,
        "high_risk_TFV_mean_reduction_pct": high_tfv_mean,
        "high_risk_peak_mean_reduction_pct": high_peak_mean,
        "low_risk_false_intervention_rate": low_false,
        "outputs": {
            "event_table": str(out_dir / "risk_stratified_event_table.csv"),
            "main_table": str(out_dir / "water_research_main_table.csv"),
            "risk_table": str(out_dir / "water_research_risk_stratified_table.csv"),
            "low_risk": str(out_dir / "low_risk_false_intervention_table.csv"),
            "high_risk": str(out_dir / "high_risk_success_table.csv"),
            "intervention_reason_summary": str(out_dir / "intervention_reason_summary.csv"),
            "action_uncertainty_summary": str(out_dir / "action_uncertainty_summary.csv"),
            "stage_action_usage_by_phase": str(out_dir / "stage_action_usage_by_phase.csv"),
        },
    }
    (out_dir / "risk_stratified_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
