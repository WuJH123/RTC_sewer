from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.control.candidate_generator import parse_candidate_label
from sewerrtc.evaluation.evaluate_closed_loop import compare_to_baseline, gate_summary, summarize_by_policy
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _read_controller_histories(run_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    proposed_dir = run_root / "proposed"
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


def _selected_action_rows(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "selected_candidate_label" not in history:
        return pd.DataFrame()
    hist = history.copy()
    fallback_col = "fallback_to_nominal" if "fallback_to_nominal" in hist else "fallback_to_native"
    if fallback_col in hist:
        fallback = hist[fallback_col].fillna(False).astype(bool)
        hist = hist[~fallback].copy()
    hist["selected_candidate_label"] = hist["selected_candidate_label"].fillna("").astype(str)
    hist = hist[hist["selected_candidate_label"].ne("")].copy()
    if hist.empty:
        return hist
    parsed = hist["selected_candidate_label"].map(parse_candidate_label)
    hist["template_name"] = [str(x.get("template", "unknown")) for x in parsed]
    hist["candidate_scope"] = [str(x.get("scope", "all")) for x in parsed]
    hist["candidate_delta"] = [float(x.get("delta", 0.0) or 0.0) for x in parsed]
    hist["hold_steps"] = [int(x.get("hold_steps", 1) or 1) for x in parsed]
    for c in [
        "selected_pred_delta_pfv",
        "selected_pred_delta_tfv",
        "selected_pred_delta_peak",
        "selected_safe_prob",
        "selected_peak_nonworse_prob",
    ]:
        if c not in hist:
            hist[c] = float("nan")
    return hist


def _selected_action_summary(history: pd.DataFrame) -> pd.DataFrame:
    selected = _selected_action_rows(history)
    if selected.empty:
        return pd.DataFrame()
    if "phase" not in selected:
        selected["phase"] = "unknown"
    group_cols = ["template_name", "candidate_scope", "candidate_delta", "hold_steps"]
    summary = (
        selected.groupby(group_cols, dropna=False)
        .agg(
            selected_count=("selected_candidate_label", "size"),
            events=("event_id", "nunique"),
            phase_peak_count=("phase", lambda s: int((s.astype(str) == "peak").sum())),
            phase_pre_peak_count=("phase", lambda s: int((s.astype(str) == "pre_peak").sum())),
            phase_recession_count=("phase", lambda s: int((s.astype(str) == "recession").sum())),
            mean_pred_delta_pfv=(
                "selected_pred_delta_pfv",
                lambda s: float(pd.to_numeric(s, errors="coerce").mean()) if len(s) else float("nan"),
            ),
            mean_pred_delta_tfv=(
                "selected_pred_delta_tfv",
                lambda s: float(pd.to_numeric(s, errors="coerce").mean()) if len(s) else float("nan"),
            ),
            mean_pred_delta_peak=(
                "selected_pred_delta_peak",
                lambda s: float(pd.to_numeric(s, errors="coerce").mean()) if len(s) else float("nan"),
            ),
            mean_safe_prob=(
                "selected_safe_prob",
                lambda s: float(pd.to_numeric(s, errors="coerce").mean()) if len(s) else float("nan"),
            ),
            mean_peak_nonworse_prob=(
                "selected_peak_nonworse_prob",
                lambda s: float(pd.to_numeric(s, errors="coerce").mean()) if len(s) else float("nan"),
            ),
        )
        .reset_index()
        .sort_values(["selected_count", "events"], ascending=False)
    )
    return summary


def _failure_action_attribution(failures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    selected = _selected_action_rows(history)
    if failures.empty or selected.empty:
        return pd.DataFrame()
    attrs = []
    for _, row in failures.iterrows():
        event_id = str(row.get("event_id", ""))
        ev = selected[selected["event_id"].astype(str).eq(event_id)].copy()
        if ev.empty:
            continue
        grouped = (
            ev.groupby(["template_name", "candidate_scope", "candidate_delta", "hold_steps", "phase"], dropna=False)
            .agg(
                selected_count=("selected_candidate_label", "size"),
                mean_pred_delta_pfv=(
                    "selected_pred_delta_pfv",
                    lambda s: float(pd.to_numeric(s, errors="coerce").mean()) if len(s) else float("nan"),
                ),
                mean_pred_delta_tfv=(
                    "selected_pred_delta_tfv",
                    lambda s: float(pd.to_numeric(s, errors="coerce").mean()) if len(s) else float("nan"),
                ),
                mean_pred_delta_peak=(
                    "selected_pred_delta_peak",
                    lambda s: float(pd.to_numeric(s, errors="coerce").mean()) if len(s) else float("nan"),
                ),
                mean_safe_prob=(
                    "selected_safe_prob",
                    lambda s: float(pd.to_numeric(s, errors="coerce").mean()) if len(s) else float("nan"),
                ),
                mean_peak_nonworse_prob=(
                    "selected_peak_nonworse_prob",
                    lambda s: float(pd.to_numeric(s, errors="coerce").mean()) if len(s) else float("nan"),
                ),
            )
            .reset_index()
        )
        for _, g in grouped.iterrows():
            attrs.append(
                {
                    "baseline_policy": row.get("baseline_policy", ""),
                    "event_id": event_id,
                    "duration_min": row.get("duration_min", float("nan")),
                    "failure_reason": row.get("failure_reason", ""),
                    "PFV_reduction_pct": row.get("PFV_reduction_pct", float("nan")),
                    "TFV_reduction_pct": row.get("TFV_reduction_pct", float("nan")),
                    "peak_TFV_rate_reduction_pct": row.get("peak_TFV_rate_reduction_pct", float("nan")),
                    **g.to_dict(),
                }
            )
    if not attrs:
        return pd.DataFrame()
    out = pd.DataFrame(attrs)
    return out.sort_values(["baseline_policy", "event_id", "selected_count"], ascending=[True, True, False])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--mode", choices=["debug", "formal"], default="debug")
    ap.add_argument("--run-tag", default="", help="Optional run subdirectory used by scripts/08_run_closed_loop.py.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "outputs.closed_loop") / args.mode
    if args.run_tag:
        root = root / args.run_tag
    proposed = pd.read_csv(root / "proposed_results.csv")
    baseline = pd.read_csv(root / "baseline_results.csv")
    history = _read_controller_histories(root)
    comps = []
    for policy, bdf in baseline.groupby("policy_id"):
        c = compare_to_baseline(proposed, bdf.copy())
        c.insert(0, "baseline_policy", policy)
        comps.append(c)
    comp = pd.concat(comps, ignore_index=True) if comps else pd.DataFrame()
    out = cfg_path(cfg, "outputs.diagnostics") / args.mode
    if args.run_tag:
        out = out / args.run_tag
    out = ensure_dir(out)
    comp.to_csv(out / "strategy_comparison.csv", index=False)
    summarize_by_policy(comp).to_csv(out / "strategy_policy_summary.csv", index=False)
    if not comp.empty:
        fail = comp.copy()
        for c in ["PFV_reduction_pct", "TFV_reduction_pct", "peak_TFV_rate_reduction_pct"]:
            if c not in fail:
                fail[c] = 0.0
            fail[c] = pd.to_numeric(fail[c], errors="coerce").fillna(0.0)
        fail["failure_reason"] = ""
        fail.loc[fail["PFV_reduction_pct"] < 0, "failure_reason"] += "PFV_worse;"
        fail.loc[fail["TFV_reduction_pct"] < -0.5, "failure_reason"] += "TFV_guard_failed;"
        fail.loc[fail["peak_TFV_rate_reduction_pct"] < -1.0, "failure_reason"] += "peak_guard_failed;"
        fail = fail[fail["failure_reason"].astype(str).str.len() > 0].copy()
        if not fail.empty:
            cols = [
                c
                for c in [
                    "baseline_policy",
                    "event_id",
                    "duration_min",
                    "failure_reason",
                    "PFV_reduction_pct",
                    "TFV_reduction_pct",
                    "peak_TFV_rate_reduction_pct",
                    "PFV_proposed",
                    "PFV_baseline",
                    "TFV_proposed",
                    "TFV_baseline",
                    "peak_TFV_rate_proposed",
                    "peak_TFV_rate_baseline",
                    "fallback_rate_proposed",
                    "action_changes_proposed",
                    "detail_file_proposed",
                    "detail_file_baseline",
                ]
                if c in fail.columns
            ]
            fail[cols].sort_values(
                ["baseline_policy", "failure_reason", "PFV_reduction_pct"],
                ascending=[True, True, True],
            ).to_csv(out / "failure_case_diagnostics.csv", index=False)
            attribution = _failure_action_attribution(fail, history)
            if not attribution.empty:
                attribution.to_csv(out / "failure_action_attribution.csv", index=False)
    selected_summary = _selected_action_summary(history)
    if not selected_summary.empty:
        selected_summary.to_csv(out / "selected_action_diagnostics.csv", index=False)
    speed_cols = [
        c
        for c in [
            "baseline_policy",
            "event_id",
            "duration_min",
            "wall_time_sec_proposed",
            "wall_time_sec_baseline",
            "proposed_wall_time_sec_per_step",
            "baseline_wall_time_sec_per_step",
            "wall_time_ratio_proposed_vs_baseline",
            "action_changes_proposed",
            "action_changes_baseline",
            "fallback_rate_proposed",
        ]
        if c in comp.columns
    ]
    if speed_cols:
        comp[speed_cols].to_csv(out / "runtime_and_stability_diagnostics.csv", index=False)
    gate_policy = "internal_rules" if "internal_rules" in set(comp.get("baseline_policy", [])) else "auto_rbc"
    gate_df = comp[comp["baseline_policy"] == gate_policy].copy() if "baseline_policy" in comp else comp
    gate = gate_summary(gate_df)
    gate["gate_baseline_policy"] = gate_policy
    gate["diagnostics_dir"] = str(out)
    (out / "gate_summary.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
