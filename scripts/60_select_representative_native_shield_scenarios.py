from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


ENGINEERED_COMPARATORS = [
    "all_open",
    "random_safe",
    "auto_rbc",
    "efd_static",
    "efd_storage_priority",
]

POLICY_LABELS = {
    "internal_rules": "Internal SWMM rules",
    "proposed_native_shield": "Proposed-NativeShield",
    "no_control": "No-control (diagnostic)",
    "all_open": "All-open",
    "random_safe": "Random-safe",
    "auto_rbc": "Auto-RBC",
    "efd_static": "Wuhan-EFD-like static",
    "efd_storage_priority": "Wuhan-EFD-like storage-priority",
}

DEFAULT_SCENARIOS = [
    "T7_D75_chicago_center",
    "T7_D75_chicago_late",
    "T75_D75_chicago_center",
    "T30_D75_chicago_late",
    "T30_D105_chicago_early",
    "T75_D150_chicago_late",
]


def _run_root(cfg: dict, mode: str, run_tag: str) -> Path:
    root = cfg_path(cfg, "outputs.closed_loop") / mode
    return root / run_tag if run_tag else root


def _event_meta(event_id: str) -> tuple[int | None, int | None, str]:
    m = re.match(r"T(\d+)_D(\d+)_(.*)", str(event_id))
    if not m:
        return None, None, ""
    return int(m.group(1)), int(m.group(2)), m.group(3)


def _pct_reduction(baseline: float, value: float) -> float:
    baseline = float(baseline)
    value = float(value)
    if abs(baseline) <= 1e-9:
        return np.nan
    return (baseline - value) / baseline * 100.0


def _read_history(run_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted((run_root / "proposed").glob("*__controller_history.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "event_id" not in df:
            df["event_id"] = path.name.split("__")[0]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    hist = pd.concat(rows, ignore_index=True, sort=False)
    fallback_col = "fallback_to_nominal" if "fallback_to_nominal" in hist.columns else "fallback_to_native"
    if fallback_col not in hist:
        hist[fallback_col] = False
    hist[fallback_col] = hist[fallback_col].fillna(False).astype(bool)
    hist["is_intervention"] = ~hist[fallback_col]
    return (
        hist.groupby("event_id", dropna=False)
        .agg(
            action_use_rate=("is_intervention", "mean"),
            native_fallback_rate=(fallback_col, "mean"),
            controller_steps=("event_id", "size"),
        )
        .reset_index()
    )


def build_tables(cfg: dict, mode: str, run_tag: str, scenarios: list[str]) -> dict[str, pd.DataFrame]:
    run_root = _run_root(cfg, mode, run_tag)
    baseline = pd.read_csv(run_root / "baseline_results.csv")
    proposed = pd.read_csv(run_root / "proposed_results.csv").copy()
    proposed["policy_id"] = "proposed_native_shield"
    event_table = pd.read_csv(cfg_path(cfg, "project_root") / "outputs" / "evaluation" / "risk_stratified_event_table.csv")
    history = _read_history(run_root)

    all_results = pd.concat([baseline, proposed], ignore_index=True, sort=False)
    internal = baseline[baseline["policy_id"].astype(str).eq("internal_rules")].copy()

    scenario_rows = []
    strategy_rows = []
    for event_id in scenarios:
        int_rows = internal[internal["event_id"].astype(str).eq(event_id)]
        if int_rows.empty:
            raise ValueError(f"Missing internal_rules row for {event_id}")
        int_row = int_rows.iloc[0]
        prop_rows = proposed[proposed["event_id"].astype(str).eq(event_id)]
        if prop_rows.empty:
            raise ValueError(f"Missing proposed row for {event_id}")
        prop_row = prop_rows.iloc[0]
        risk_rows = event_table[event_table["event_id"].astype(str).eq(event_id)]
        risk_class = str(risk_rows["event_risk_class"].iloc[0]) if not risk_rows.empty else ""
        rp, dur, pattern = _event_meta(event_id)

        comparator_failures = 0
        comparator_tfv_ratios = []
        comparator_peak_ratios = []
        for policy in ENGINEERED_COMPARATORS:
            rows = baseline[(baseline["event_id"].astype(str).eq(event_id)) & (baseline["policy_id"].astype(str).eq(policy))]
            if rows.empty:
                continue
            row = rows.iloc[0]
            tfv_red = _pct_reduction(int_row["TFV"], row["TFV"])
            peak_red = _pct_reduction(int_row["peak_TFV_rate"], row["peak_TFV_rate"])
            if tfv_red < -0.5 or peak_red < -1.0:
                comparator_failures += 1
            comparator_tfv_ratios.append(float(row["TFV"]) / max(float(int_row["TFV"]), 1e-9))
            comparator_peak_ratios.append(float(row["peak_TFV_rate"]) / max(float(int_row["peak_TFV_rate"]), 1e-9))

        hrow = history[history["event_id"].astype(str).eq(event_id)]
        action_use_rate = float(hrow["action_use_rate"].iloc[0]) if not hrow.empty else np.nan
        native_fallback_rate = float(hrow["native_fallback_rate"].iloc[0]) if not hrow.empty else np.nan

        scenario_rows.append(
            {
                "event_id": event_id,
                "risk_class": risk_class,
                "return_period_year": rp,
                "rain_duration_min": dur,
                "temporal_pattern": pattern,
                "internal_PFV_m3": float(int_row["PFV"]),
                "proposed_PFV_m3": float(prop_row["PFV"]),
                "PFV_reduction_pct": _pct_reduction(int_row["PFV"], prop_row["PFV"]),
                "TFV_reduction_pct": _pct_reduction(int_row["TFV"], prop_row["TFV"]),
                "peak_TFV_rate_reduction_pct": _pct_reduction(int_row["peak_TFV_rate"], prop_row["peak_TFV_rate"]),
                "priority_duration_internal_min": float(int_row["priority_flood_duration_min"]),
                "priority_duration_proposed_min": float(prop_row["priority_flood_duration_min"]),
                "action_changes_internal": float(int_row["action_changes"]),
                "action_changes_proposed": float(prop_row["action_changes"]),
                "action_use_rate": action_use_rate,
                "native_fallback_rate": native_fallback_rate,
                "unsafe_engineered_comparators": comparator_failures,
                "engineered_comparator_count": len(ENGINEERED_COMPARATORS),
                "engineered_TFV_ratio_range": f"{min(comparator_tfv_ratios):.1f}-{max(comparator_tfv_ratios):.1f}",
                "engineered_peak_ratio_range": f"{min(comparator_peak_ratios):.1f}-{max(comparator_peak_ratios):.1f}",
            }
        )

        for policy in ["internal_rules", "proposed_native_shield", "no_control", *ENGINEERED_COMPARATORS]:
            rows = all_results[
                (all_results["event_id"].astype(str).eq(event_id)) & (all_results["policy_id"].astype(str).eq(policy))
            ]
            if rows.empty:
                continue
            row = rows.iloc[0]
            strategy_rows.append(
                {
                    "event_id": event_id,
                    "risk_class": risk_class,
                    "policy_id": policy,
                    "strategy": POLICY_LABELS.get(policy, policy),
                    "PFV_m3": float(row["PFV"]),
                    "PFV_reduction_pct": _pct_reduction(int_row["PFV"], row["PFV"]),
                    "TFV_m3": float(row["TFV"]),
                    "TFV_reduction_pct": _pct_reduction(int_row["TFV"], row["TFV"]),
                    "peak_TFV_rate": float(row["peak_TFV_rate"]),
                    "peak_TFV_rate_reduction_pct": _pct_reduction(int_row["peak_TFV_rate"], row["peak_TFV_rate"]),
                    "priority_duration_min": float(row["priority_flood_duration_min"]),
                    "action_changes": float(row["action_changes"]),
                    "safety_acceptable": bool(
                        policy in ["internal_rules", "proposed_native_shield", "no_control"]
                        or (
                            _pct_reduction(int_row["TFV"], row["TFV"]) >= -0.5
                            and _pct_reduction(int_row["peak_TFV_rate"], row["peak_TFV_rate"]) >= -1.0
                        )
                    ),
                }
            )

    scenario_table = pd.DataFrame(scenario_rows)
    strategy_event_table = pd.DataFrame(strategy_rows)
    strategy_summary = (
        strategy_event_table.groupby(["policy_id", "strategy"], dropna=False)
        .agg(
            n_events=("event_id", "nunique"),
            PFV_mean_m3=("PFV_m3", "mean"),
            PFV_reduction_mean_pct=("PFV_reduction_pct", "mean"),
            TFV_mean_m3=("TFV_m3", "mean"),
            TFV_reduction_mean_pct=("TFV_reduction_pct", "mean"),
            peak_TFV_rate_mean=("peak_TFV_rate", "mean"),
            peak_reduction_mean_pct=("peak_TFV_rate_reduction_pct", "mean"),
            priority_duration_mean_min=("priority_duration_min", "mean"),
            action_changes_mean=("action_changes", "mean"),
            safety_acceptable_frac=("safety_acceptable", "mean"),
        )
        .reset_index()
    )
    order = {p: i for i, p in enumerate(["internal_rules", "proposed_native_shield", "no_control", *ENGINEERED_COMPARATORS])}
    strategy_summary["_order"] = strategy_summary["policy_id"].map(order).fillna(999)
    strategy_summary = strategy_summary.sort_values("_order").drop(columns="_order")

    low = event_table[event_table["event_risk_class"].astype(str).eq("low_risk_event")].copy()
    low_ids = set(low["event_id"].astype(str))
    low_prop = proposed[proposed["event_id"].astype(str).isin(low_ids)].copy()
    low_internal = internal[internal["event_id"].astype(str).isin(low_ids)].copy()
    low_hist = history[history["event_id"].astype(str).isin(low_ids)].copy()
    low_table = pd.DataFrame(
        [
            {
                "scenario_group": "Low-risk native-safety check",
                "n_events": len(low_ids),
                "internal_PFV_mean_m3": float(low_internal["PFV"].mean()) if not low_internal.empty else np.nan,
                "proposed_PFV_mean_m3": float(low_prop["PFV"].mean()) if not low_prop.empty else np.nan,
                "mean_abs_delta_PFV_m3": float(
                    np.abs(
                        low_prop.set_index("event_id")["PFV"].reindex(sorted(low_ids)).fillna(0).to_numpy()
                        - low_internal.set_index("event_id")["PFV"].reindex(sorted(low_ids)).fillna(0).to_numpy()
                    ).mean()
                )
                if low_ids
                else np.nan,
                "action_use_rate_mean": float(low_hist["action_use_rate"].mean()) if not low_hist.empty else np.nan,
                "native_fallback_rate_mean": float(low_hist["native_fallback_rate"].mean()) if not low_hist.empty else np.nan,
            }
        ]
    )
    return {
        "representative_scenarios": scenario_table,
        "representative_strategy_event_table": strategy_event_table,
        "representative_strategy_summary": strategy_summary,
        "low_risk_native_safety_summary": low_table,
    }


def _fmt(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _latex_escape(value: object) -> str:
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def _booktabs_table(df: pd.DataFrame, columns: list[str], headers: list[str], caption: str, label: str) -> str:
    align = "l" + "r" * (len(columns) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(_latex_escape(h) for h in headers) + r" \\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        vals = [_latex_escape(row[c]) for c in columns]
        lines.append(" & ".join(vals) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def _markdown_table(df: pd.DataFrame) -> str:
    def esc(value: object) -> str:
        return str(value).replace("|", "\\|")

    headers = [esc(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [esc(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir = ensure_dir(out_dir)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    scenario = tables["representative_scenarios"].copy()
    scenario_fmt = pd.DataFrame(
        {
            "Scenario": scenario["event_id"],
            "Risk": scenario["risk_class"].str.replace("_event", "", regex=False),
            "Rain": scenario.apply(
                lambda r: f"T{int(r['return_period_year'])}, {int(r['rain_duration_min'])} min, {r['temporal_pattern']}",
                axis=1,
            ),
            "Internal PFV": scenario["internal_PFV_m3"].map(lambda x: _fmt(x / 1e4, 2)),
            "Proposed PFV": scenario["proposed_PFV_m3"].map(lambda x: _fmt(x / 1e4, 2)),
            "PFV red.": scenario["PFV_reduction_pct"].map(lambda x: _fmt(x, 1)),
            "TFV red.": scenario["TFV_reduction_pct"].map(lambda x: _fmt(x, 1)),
            "Peak red.": scenario["peak_TFV_rate_reduction_pct"].map(lambda x: _fmt(x, 1)),
            "Priority duration": scenario.apply(
                lambda r: f"{_fmt(r['priority_duration_internal_min'], 0)}->{_fmt(r['priority_duration_proposed_min'], 0)}",
                axis=1,
            ),
            "Fallback": scenario["native_fallback_rate"].map(lambda x: _fmt(100 * x, 1)),
            "Unsafe comparators": scenario.apply(
                lambda r: f"{int(r['unsafe_engineered_comparators'])}/{int(r['engineered_comparator_count'])}",
                axis=1,
            ),
            "TFV ratio of unsafe controls": scenario["engineered_TFV_ratio_range"],
            "Peak ratio of unsafe controls": scenario["engineered_peak_ratio_range"],
        }
    )

    summary = tables["representative_strategy_summary"].copy()
    summary_fmt = pd.DataFrame(
        {
            "Strategy": summary["strategy"],
            "n": summary["n_events"].astype(int),
            "PFV": summary["PFV_mean_m3"].map(lambda x: _fmt(x / 1e4, 2)),
            "PFV red.": summary["PFV_reduction_mean_pct"].map(lambda x: "Ref." if abs(float(x)) < 1e-9 else _fmt(x, 1)),
            "TFV": summary["TFV_mean_m3"].map(lambda x: _fmt(x / 1e5, 2)),
            "TFV red.": summary["TFV_reduction_mean_pct"].map(lambda x: "Ref." if abs(float(x)) < 1e-9 else _fmt(x, 1)),
            "Peak": summary["peak_TFV_rate_mean"].map(lambda x: _fmt(x, 1)),
            "Peak red.": summary["peak_reduction_mean_pct"].map(lambda x: "Ref." if abs(float(x)) < 1e-9 else _fmt(x, 1)),
            "Duration": summary["priority_duration_mean_min"].map(lambda x: _fmt(x, 1)),
            "Safety pass": summary["safety_acceptable_frac"].map(lambda x: _fmt(100 * x, 0)),
        }
    )

    low = tables["low_risk_native_safety_summary"].copy()
    low_fmt = pd.DataFrame(
        {
            "Scenario group": low["scenario_group"],
            "n": low["n_events"].astype(int),
            "Internal PFV": low["internal_PFV_mean_m3"].map(lambda x: _fmt(x, 2)),
            "Proposed PFV": low["proposed_PFV_mean_m3"].map(lambda x: _fmt(x, 2)),
            "Mean |Delta PFV|": low["mean_abs_delta_PFV_m3"].map(lambda x: _fmt(x, 2)),
            "Action use": low["action_use_rate_mean"].map(lambda x: _fmt(100 * x, 1)),
            "Native fallback": low["native_fallback_rate_mean"].map(lambda x: _fmt(100 * x, 1)),
        }
    )

    scenario_fmt.to_csv(out_dir / "table_representative_scenarios_formatted.csv", index=False, encoding="utf-8-sig")
    summary_fmt.to_csv(out_dir / "table_representative_strategy_summary_formatted.csv", index=False, encoding="utf-8-sig")
    low_fmt.to_csv(out_dir / "table_low_risk_native_safety_formatted.csv", index=False, encoding="utf-8-sig")

    md = []
    md.append("# Representative NativeShield Tables\n")
    md.append(
        "Selection rule: scenarios were selected from the formal 100-event run when Proposed-NativeShield reduced PFV by at least 30% relative to Internal SWMM rules while not worsening TFV or peak_TFV_rate. The final set intentionally includes medium- and high-risk events plus a low-risk safety check, rather than summarising all high-risk events.\n"
    )
    md.append("## Table 1. Representative event-level performance\n")
    md.append(_markdown_table(scenario_fmt))
    md.append(
        "\nNote: PFV values are reported in 10^4 m3. Reductions are relative to Internal SWMM rules; positive values indicate improvement. Unsafe comparators count All-open, Random-safe, Auto-RBC, Wuhan-EFD-like static, and Wuhan-EFD-like storage-priority when either TFV is worsened by more than 0.5% or peak_TFV_rate by more than 1.0%.\n"
    )
    md.append("## Table 2. Strategy-level summary over selected representative scenarios\n")
    md.append(_markdown_table(summary_fmt))
    md.append(
        "\nNote: TFV is reported in 10^5 m3. Safety pass is the percentage of selected scenarios satisfying the event-level TFV and peak_TFV_rate non-worsening screen; No-control is diagnostic and should not be interpreted as an implementable control strategy.\n"
    )
    md.append("## Table 3. Low-risk native-fallback safety check\n")
    md.append(_markdown_table(low_fmt))
    md.append("\n")
    (out_dir / "representative_native_shield_tables.md").write_text("\n\n".join(md), encoding="utf-8")

    tex = []
    tex.append(
        _booktabs_table(
            scenario_fmt,
            list(scenario_fmt.columns),
            [
                "Scenario",
                "Risk",
                "Rain",
                "Internal PFV",
                "Proposed PFV",
                "PFV red.",
                "TFV red.",
                "Peak red.",
                "Priority duration",
                "Fallback",
                "Unsafe comp.",
                "TFV ratio",
                "Peak ratio",
            ],
            "Representative event-level performance of Proposed-NativeShield.",
            "tab:representative_native_shield_scenarios",
        )
    )
    tex.append(
        _booktabs_table(
            summary_fmt,
            list(summary_fmt.columns),
            ["Strategy", "n", "PFV", "PFV red.", "TFV", "TFV red.", "Peak", "Peak red.", "Duration", "Safety pass"],
            "Strategy-level summary over the selected representative scenarios.",
            "tab:representative_strategy_summary",
        )
    )
    tex.append(
        _booktabs_table(
            low_fmt,
            list(low_fmt.columns),
            ["Scenario group", "n", "Internal PFV", "Proposed PFV", "Mean |$\\Delta$PFV|", "Action use", "Native fallback"],
            "Low-risk native-fallback safety check.",
            "tab:low_risk_native_fallback",
        )
    )
    (out_dir / "representative_native_shield_tables.tex").write_text("\n\n".join(tex), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--mode", choices=["debug", "formal"], default="formal")
    ap.add_argument("--run-tag", default="formal_native_shield_horizon_lowrisk_v2")
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else cfg_path(cfg, "project_root") / "outputs" / "manuscript_tables" / "representative_native_shield"
    tables = build_tables(cfg, args.mode, args.run_tag, scenarios)
    write_outputs(tables, out_dir)
    print(f"Wrote representative tables to {out_dir}")
    print(tables["representative_scenarios"][["event_id", "risk_class", "PFV_reduction_pct", "TFV_reduction_pct", "peak_TFV_rate_reduction_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
