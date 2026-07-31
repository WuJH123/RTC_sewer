from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.evaluation.risk_stratified import RiskThresholds, classify_event
from sewerrtc.evaluation.policy_sets import diagnostic_policy_ids, normalize_policy_id, paper_policy_ids, split_policy_set_frames
from sewerrtc.io.priority_config import (
    combined_priority_depth_nodes,
    configured_priority_nodes,
    configured_priority_sentinel_nodes,
    priority_config_summary,
    read_node_list,
)
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


POLICY_LABELS = {
    "proposed_gat_mpc": "Proposed-GAT-MPC",
    "proposed_temporal_joint_36": "Proposed 36-facility actual-repair MPC",
    "proposed_hierarchical_v8_residual_36": "Proposed 36-facility actual-repair MPC",
    "official_mpc": "Published pystorms Beta MPC",
    "internal_rules": "Internal SWMM rules",
    "proposed_native_shield": "Proposed-NativeShield",
    "no_control": "No-control (stripped native rules)",
    "all_open": "All-open supplementary diagnostic",
    "random_safe": "Random-safe supplementary diagnostic",
    "auto_rbc": "Auto-RBC external rule baseline",
    "efd_static": "Wuhan-EFD-like static supplementary diagnostic",
    "efd_storage_priority": "Wuhan-EFD-like storage-priority supplementary diagnostic",
}

POLICY_ORDER = [
    "proposed_gat_mpc",
    "proposed_temporal_joint_36",
    "proposed_hierarchical_v8_residual_36",
    "official_mpc",
    "internal_rules",
    "proposed_native_shield",
    "no_control",
    "all_open",
    "random_safe",
    "auto_rbc",
    "efd_static",
    "efd_storage_priority",
]


def _infer_proposed_policy_id(run_dir: Path, proposed: pd.DataFrame) -> str:
    report_path = run_dir / "closed_loop_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            controller = str(report.get("proposed_controller", "")).strip()
            if controller == "generic_gat_mpc":
                return "proposed_gat_mpc"
            if controller == "temporal_joint_36":
                return "proposed_temporal_joint_36"
            if controller == "native_shield":
                return "proposed_native_shield"
        except Exception:
            pass
    if "policy_id" in proposed:
        raw = {str(x).strip() for x in proposed["policy_id"].dropna().tolist()}
        if raw & {"proposed_gat_mpc", "generic_gat_mpc"}:
            return "proposed_gat_mpc"
        if raw & {"proposed_temporal_joint_36", "proposed_hierarchical_v8_residual_36"}:
            return "proposed_temporal_joint_36"
        if raw & {"proposed_native_shield", "native_shield"}:
            return "proposed_native_shield"
        normalised = {normalize_policy_id(x) for x in raw}
        if "proposed_gat_mpc" in normalised:
            return "proposed_gat_mpc"
    return "proposed_native_shield"


def _default_priority_file(cfg: dict, project_root: Path) -> Path:
    configured = cfg.get("network", {}).get("priority_nodes_file", "")
    if configured:
        p = cfg_path(cfg, "network.priority_nodes_file")
        if p.exists():
            return p
    local = project_root / "data" / "project2_design" / "priority_zone_nodes.txt"
    if local.exists():
        return local
    raise FileNotFoundError(
        "Project6 priority nodes must be configured or stored inside Project6; "
        f"missing network.priority_nodes_file and {local}"
    )


def _read_nodes(path: Path) -> list[str]:
    return read_node_list(path)


def _safe_reduction(base: pd.Series, value: pd.Series, eps: float = 1e-9) -> pd.Series:
    base_num = pd.to_numeric(base, errors="coerce")
    value_num = pd.to_numeric(value, errors="coerce")
    out = pd.Series(np.nan, index=base_num.index, dtype=float)
    mask = base_num.abs() > eps
    out.loc[mask] = (base_num.loc[mask] - value_num.loc[mask]) / base_num.loc[mask] * 100.0
    out.loc[(~mask) & (value_num.abs() <= eps)] = 0.0
    return out


def _attach_event_risk_class(event_policy: pd.DataFrame, event_table: pd.DataFrame) -> pd.DataFrame:
    """Attach one baseline-defined risk class to every policy/event row."""
    cols = ["event_id", "event_risk_class", "is_near_zero_pfv"]
    risk = event_table[[c for c in cols if c in event_table]].drop_duplicates("event_id")
    work = event_policy.drop(columns=[c for c in cols[1:] if c in event_policy], errors="ignore")
    return work.merge(risk, on="event_id", how="left", validate="many_to_one")


def _depth_metrics(detail: pd.DataFrame, depth_nodes: list[str], dt_sec: int, exposure_depth_m: float = 0.20) -> dict:
    hcols = [f"h:{n}" for n in depth_nodes if f"h:{n}" in detail.columns]
    missing = [n for n in depth_nodes if f"h:{n}" not in detail.columns]
    if not hcols:
        return {
            "project5_depth_safety_peak_depth": 0.0,
            "project5_depth_safety_exposure_min": 0.0,
            "project5_depth_safety_nodes_present": 0,
            "project5_depth_safety_nodes_missing": len(missing),
            "missing_depth_safety_nodes": ",".join(missing),
        }
    depths = detail[hcols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row_peak = depths.max(axis=1)
    return {
        "project5_depth_safety_peak_depth": float(row_peak.max()) if len(row_peak) else 0.0,
        "project5_depth_safety_exposure_min": float((row_peak >= float(exposure_depth_m)).sum() * dt_sec / 60.0),
        "project5_depth_safety_nodes_present": len(hcols),
        "project5_depth_safety_nodes_missing": len(missing),
        "missing_depth_safety_nodes": ",".join(missing),
    }


def _recompute_priority_metrics(
    detail_file: str,
    priority_nodes: list[str],
    dt_sec: int,
    depth_nodes: list[str] | None = None,
    exposure_depth_m: float = 0.20,
) -> dict:
    path = Path(str(detail_file))
    if not path.exists():
        return {
            "project5_priority_PFV": np.nan,
            "project5_priority_duration_min": np.nan,
            "project5_priority_peak_rate": np.nan,
            "project5_priority_nodes_present": 0,
            "project5_priority_nodes_missing": len(priority_nodes),
            "missing_priority_nodes": ",".join(priority_nodes),
            **_depth_metrics(pd.DataFrame(), depth_nodes or [], dt_sec, exposure_depth_m),
        }
    header = pd.read_csv(path, nrows=0)
    header_cols = set(header.columns)
    pr_cols = [f"flood:{n}" for n in priority_nodes if f"flood:{n}" in header_cols]
    depth_cols = [f"h:{n}" for n in (depth_nodes or priority_nodes) if f"h:{n}" in header_cols]
    missing = [n for n in priority_nodes if f"flood:{n}" not in header_cols]
    usecols = pr_cols + [c for c in depth_cols if c not in pr_cols]
    detail = pd.read_csv(path, usecols=usecols) if usecols else pd.DataFrame()
    depth = _depth_metrics(detail, depth_nodes or priority_nodes, dt_sec, exposure_depth_m)
    if not pr_cols:
        return {
            "project5_priority_PFV": 0.0,
            "project5_priority_duration_min": 0.0,
            "project5_priority_peak_rate": 0.0,
            "project5_priority_nodes_present": 0,
            "project5_priority_nodes_missing": len(missing),
            "missing_priority_nodes": ",".join(missing),
            **depth,
        }
    arr = detail[pr_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    rate = arr.sum(axis=1)
    return {
        "project5_priority_PFV": float(rate.sum() * dt_sec),
        "project5_priority_duration_min": float((rate > 1e-9).sum() * dt_sec / 60.0),
        "project5_priority_peak_rate": float(rate.max()) if len(rate) else 0.0,
        "project5_priority_nodes_present": len(pr_cols),
        "project5_priority_nodes_missing": len(missing),
        "missing_priority_nodes": ",".join(missing),
        **depth,
    }


def _format_float(value: float, ndigits: int = 2) -> str:
    try:
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return ""


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--priority-file", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--dt-sec", type=int, default=300)
    args = ap.parse_args()

    cfg = load_config(args.config)
    project_root = cfg_path(cfg, "project_root")
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        reuse_run = cfg.get("reuse_sources", {}).get("project4_formal_run", "")
        run_dir = Path(reuse_run) if reuse_run else cfg_path(cfg, "outputs.closed_loop") / "formal" / "formal_native_shield_horizon_lowrisk_v2"
    priority_file = Path(args.priority_file) if args.priority_file else _default_priority_file(cfg, project_root)
    out_dir = ensure_dir(
        Path(args.out_dir)
        if args.out_dir
        else project_root / "outputs" / "evaluation_project5_priority_zone"
    )

    priority_nodes = _read_nodes(priority_file) if args.priority_file else configured_priority_nodes(cfg)
    sentinel_nodes = configured_priority_sentinel_nodes(cfg)
    depth_nodes = combined_priority_depth_nodes(cfg)
    thresholds = RiskThresholds.from_config(cfg)
    baseline_path = run_dir / "baseline_results.csv"
    proposed_path = run_dir / "proposed_results.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline results: {baseline_path}")
    if not proposed_path.exists():
        raise FileNotFoundError(f"Missing proposed results: {proposed_path}")
    baseline = pd.read_csv(baseline_path)
    proposed = pd.read_csv(proposed_path)
    proposed = proposed.copy()
    proposed["policy_id"] = _infer_proposed_policy_id(run_dir, proposed)
    all_results = pd.concat([baseline, proposed], ignore_index=True, sort=False)
    configured_order = paper_policy_ids(cfg) + diagnostic_policy_ids(cfg)
    policy_order = configured_order + [p for p in POLICY_ORDER if p not in configured_order]
    all_results = all_results[all_results["policy_id"].astype(str).isin(policy_order)].copy()

    rows = []
    for _, row in all_results.iterrows():
        metrics = _recompute_priority_metrics(
            str(row.get("detail_file", "")),
            priority_nodes,
            int(args.dt_sec),
            depth_nodes,
            thresholds.high_risk_exposure_depth_m,
        )
        policy = str(row.get("policy_id", ""))
        rows.append(
            {
                "event_id": str(row.get("event_id", "")),
                "duration_min": int(row.get("duration_min", 0) or 0),
                "policy_id": policy,
                "policy_label": POLICY_LABELS.get(policy, policy),
                "TFV": float(row.get("TFV", np.nan)),
                "peak_TFV_rate": float(row.get("peak_TFV_rate", np.nan)),
                "action_changes": float(row.get("action_changes", np.nan)),
                "detail_file": str(row.get("detail_file", "")),
                **metrics,
            }
        )
    event_policy = pd.DataFrame(rows)
    order = {p: i for i, p in enumerate(policy_order)}
    event_policy["policy_order"] = event_policy["policy_id"].map(order).fillna(999).astype(int)
    event_policy = event_policy.sort_values(["event_id", "policy_order"]).drop(columns=["policy_order"])

    internal_event_policy = event_policy[event_policy["policy_id"].eq("internal_rules")].copy()
    internal = internal_event_policy[
        [
            "event_id",
            "project5_priority_PFV",
            "project5_priority_duration_min",
            "project5_priority_peak_rate",
            "TFV",
            "peak_TFV_rate",
        ]
    ].rename(
        columns={
            "project5_priority_PFV": "internal_project5_priority_PFV",
            "project5_priority_duration_min": "internal_project5_priority_duration_min",
            "project5_priority_peak_rate": "internal_project5_priority_peak_rate",
            "TFV": "internal_TFV",
            "peak_TFV_rate": "internal_peak_TFV_rate",
        }
    )
    paired = event_policy.merge(internal, on="event_id", how="left")
    paired["project5_priority_PFV_reduction_pct_vs_internal"] = _safe_reduction(
        paired["internal_project5_priority_PFV"], paired["project5_priority_PFV"]
    )
    paired["project5_priority_duration_reduction_min_vs_internal"] = (
        paired["internal_project5_priority_duration_min"] - paired["project5_priority_duration_min"]
    )
    paired["TFV_reduction_pct_vs_internal"] = _safe_reduction(paired["internal_TFV"], paired["TFV"])
    paired["peak_TFV_rate_reduction_pct_vs_internal"] = _safe_reduction(
        paired["internal_peak_TFV_rate"], paired["peak_TFV_rate"]
    )

    event_table = internal.copy()
    depth_cols = [
        "project5_depth_safety_peak_depth",
        "project5_depth_safety_exposure_min",
        "project5_depth_safety_nodes_present",
        "project5_depth_safety_nodes_missing",
        "missing_depth_safety_nodes",
    ]
    if not internal_event_policy.empty:
        depth_by_event = internal_event_policy.set_index("event_id")
        for col in depth_cols:
            if col in depth_by_event:
                event_table[col] = event_table["event_id"].map(depth_by_event[col])
    event_table["internal_PFV"] = event_table["internal_project5_priority_PFV"]
    event_table["internal_TFV"] = event_table["internal_TFV"]
    event_table["internal_peak_TFV_rate"] = event_table["internal_peak_TFV_rate"]
    event_table["event_risk_class"] = event_table.apply(lambda r: classify_event(r, thresholds), axis=1)
    event_table["is_near_zero_pfv"] = event_table["internal_project5_priority_PFV"].abs() <= thresholds.near_zero_pfv_epsilon
    event_policy = _attach_event_risk_class(event_policy, event_table)
    paired = paired.merge(event_table[["event_id", "event_risk_class", "is_near_zero_pfv"]], on="event_id", how="left")

    summary_rows = []
    for scope_name, sub in [("all_events", paired)] + [
        (str(k), g) for k, g in paired.groupby("event_risk_class", dropna=False)
    ]:
        for policy, g in sub.groupby("policy_id", dropna=False):
            g = g.copy()
            summary_rows.append(
                {
                    "scope": scope_name,
                    "policy_id": policy,
                    "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                    "n_events": int(g["event_id"].nunique()),
                    "PFV_mean": float(pd.to_numeric(g["project5_priority_PFV"], errors="coerce").mean()),
                    "PFV_median": float(pd.to_numeric(g["project5_priority_PFV"], errors="coerce").median()),
                    "PFV_reduction_mean_pct_vs_internal": float(
                        pd.to_numeric(g["project5_priority_PFV_reduction_pct_vs_internal"], errors="coerce").mean()
                    ),
                    "PFV_reduction_median_pct_vs_internal": float(
                        pd.to_numeric(g["project5_priority_PFV_reduction_pct_vs_internal"], errors="coerce").median()
                    ),
                    "near_zero_PFV_frac_le_100m3": float(
                        pd.to_numeric(g["project5_priority_PFV"], errors="coerce").fillna(0.0).abs().le(100.0).mean()
                    ),
                    "priority_duration_mean_min": float(
                        pd.to_numeric(g["project5_priority_duration_min"], errors="coerce").mean()
                    ),
                    "priority_duration_median_min": float(
                        pd.to_numeric(g["project5_priority_duration_min"], errors="coerce").median()
                    ),
                    "priority_duration_reduction_mean_min_vs_internal": float(
                        pd.to_numeric(g["project5_priority_duration_reduction_min_vs_internal"], errors="coerce").mean()
                    ),
                    "TFV_reduction_mean_pct_vs_internal": float(
                        pd.to_numeric(g["TFV_reduction_pct_vs_internal"], errors="coerce").mean()
                    ),
                    "peak_reduction_mean_pct_vs_internal": float(
                        pd.to_numeric(g["peak_TFV_rate_reduction_pct_vs_internal"], errors="coerce").mean()
                    ),
                    "TFV_worse_frac": float((pd.to_numeric(g["TFV"], errors="coerce") > pd.to_numeric(g["internal_TFV"], errors="coerce") + 1e-6).mean()),
                    "peak_worse_frac": float((pd.to_numeric(g["peak_TFV_rate"], errors="coerce") > pd.to_numeric(g["internal_peak_TFV_rate"], errors="coerce") + 1e-6).mean()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary["policy_order"] = summary["policy_id"].map(order).fillna(999).astype(int)
    summary = summary.sort_values(["scope", "policy_order"]).drop(columns=["policy_order"])
    event_policy_main, event_policy_diagnostic = split_policy_set_frames(event_policy, cfg)
    paired_main, paired_diagnostic = split_policy_set_frames(paired, cfg)
    summary_main, summary_diagnostic = split_policy_set_frames(summary, cfg)

    metadata = {
        "priority_source": str(priority_file),
        "priority": priority_config_summary(cfg),
        "priority_node_count": len(priority_nodes),
        "priority_nodes": priority_nodes,
        "sentinel_node_count": len(sentinel_nodes),
        "sentinel_nodes": sentinel_nodes,
        "depth_safety_nodes": depth_nodes,
        "run_dir": str(run_dir),
        "dt_sec": int(args.dt_sec),
        "events": int(event_table["event_id"].nunique()),
        "risk_class_counts": {str(k): int(v) for k, v in event_table["event_risk_class"].value_counts().to_dict().items()},
        "outputs": {
            "event_policy_metrics": str(out_dir / "project5_priority_event_policy_metrics.csv"),
            "event_policy_metrics_main": str(out_dir / "project5_priority_event_policy_metrics_main.csv"),
            "event_policy_metrics_diagnostic": str(out_dir / "project5_priority_event_policy_metrics_diagnostic.csv"),
            "paired_metrics": str(out_dir / "project5_priority_paired_metrics.csv"),
            "paired_metrics_main": str(out_dir / "project5_priority_paired_metrics_main.csv"),
            "paired_metrics_diagnostic": str(out_dir / "project5_priority_paired_metrics_diagnostic.csv"),
            "policy_summary": str(out_dir / "project5_priority_policy_summary.csv"),
            "policy_summary_main": str(out_dir / "project5_priority_policy_summary_main.csv"),
            "policy_summary_diagnostic": str(out_dir / "project5_priority_policy_summary_diagnostic.csv"),
            "event_table": str(out_dir / "project5_priority_event_table.csv"),
            "summary_markdown": str(out_dir / "project5_priority_summary.md"),
        },
        "paper_policy_set": paper_policy_ids(cfg),
        "diagnostic_policy_set": diagnostic_policy_ids(cfg),
    }

    event_policy.to_csv(out_dir / "project5_priority_event_policy_metrics.csv", index=False, encoding="utf-8-sig")
    event_policy_main.to_csv(out_dir / "project5_priority_event_policy_metrics_main.csv", index=False, encoding="utf-8-sig")
    event_policy_diagnostic.to_csv(out_dir / "project5_priority_event_policy_metrics_diagnostic.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(out_dir / "project5_priority_paired_metrics.csv", index=False, encoding="utf-8-sig")
    paired_main.to_csv(out_dir / "project5_priority_paired_metrics_main.csv", index=False, encoding="utf-8-sig")
    paired_diagnostic.to_csv(out_dir / "project5_priority_paired_metrics_diagnostic.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "project5_priority_policy_summary.csv", index=False, encoding="utf-8-sig")
    summary_main.to_csv(out_dir / "project5_priority_policy_summary_main.csv", index=False, encoding="utf-8-sig")
    summary_diagnostic.to_csv(out_dir / "project5_priority_policy_summary_diagnostic.csv", index=False, encoding="utf-8-sig")
    event_table.to_csv(out_dir / "project5_priority_event_table.csv", index=False, encoding="utf-8-sig")
    (out_dir / "project5_priority_recalc_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Project5 Priority-Zone Recalculation",
        "",
        f"Priority source: `{priority_file}`",
        f"Priority PFV-core nodes: {len(priority_nodes)}",
        f"Depth/surcharge sentinel nodes: {len(sentinel_nodes)}",
        f"Run directory: `{run_dir}`",
        f"Main paper policy set: `{', '.join(paper_policy_ids(cfg))}`",
        f"Supplementary diagnostic policy set: `{', '.join(diagnostic_policy_ids(cfg))}`",
        "Baseline note: supplementary diagnostic policies other than Internal SWMM rules were run on stripped-native-control INPs.",
        f"Risk class counts: {metadata['risk_class_counts']}",
        "",
    ]
    for scope in ["all_events", "high_risk_event", "medium_risk_event", "low_risk_event"]:
        sub = summary_main[summary_main["scope"].eq(scope)].copy()
        if sub.empty:
            continue
        compact = sub[
            [
                "policy_label",
                "n_events",
                "PFV_mean",
                "PFV_reduction_mean_pct_vs_internal",
                "near_zero_PFV_frac_le_100m3",
                "priority_duration_mean_min",
                "TFV_reduction_mean_pct_vs_internal",
                "peak_reduction_mean_pct_vs_internal",
                "TFV_worse_frac",
                "peak_worse_frac",
            ]
        ].copy()
        for col in compact.columns:
            if col == "policy_label":
                continue
            if col == "n_events":
                compact[col] = compact[col].map(lambda x: str(int(x)))
            else:
                compact[col] = compact[col].map(lambda x: _format_float(x, 2))
        md.extend([f"## {scope}", "", _markdown_table(compact, list(compact.columns)), ""])
    (out_dir / "project5_priority_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
