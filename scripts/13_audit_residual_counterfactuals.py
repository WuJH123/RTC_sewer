from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _tier_from_delta(delta: pd.Series, fallback: pd.Series | None = None) -> pd.Series:
    d_abs = pd.to_numeric(delta, errors="coerce").abs()
    if fallback is not None:
        fb = pd.to_numeric(fallback, errors="coerce").abs()
        d_abs = d_abs.fillna(fb)
        d_abs = d_abs.mask(d_abs <= 1e-9, fb)
    return pd.Series(
        np.where(d_abs <= 0.080001, "small", np.where(d_abs <= 0.160001, "medium", "large")),
        index=delta.index,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "outputs.closed_loop") / "internal_residual_counterfactuals"
    path = root / "residual_counterfactual_results.csv"
    out = ensure_dir(cfg_path(cfg, "outputs.diagnostics") / "residual_counterfactuals")
    if not path.exists():
        raise FileNotFoundError(f"Missing residual counterfactual file: {path}")
    df = pd.read_csv(path)
    for c in [
        "delta_PFV",
        "delta_TFV",
        "delta_peak",
        "PFV",
        "TFV",
        "peak_TFV_rate",
        "baseline_PFV",
        "baseline_TFV",
        "baseline_peak_TFV_rate",
        "tfv_guard",
        "peak_guard",
        "residual_delta",
        "override_start_min",
        "feat_delta_abs_max",
    ]:
        if c in df:
            df[c] = _num(df[c])
    if "template_name" not in df:
        df["template_name"] = df.get("template", "unknown")
    if "residual_delta" not in df:
        df["residual_delta"] = np.nan
    fallback_delta = df["feat_delta_abs_max"] if "feat_delta_abs_max" in df else None
    inferred_tier = _tier_from_delta(df["residual_delta"], fallback_delta)
    if "residual_delta_tier" not in df:
        df["residual_delta_tier"] = inferred_tier
    else:
        tier = df["residual_delta_tier"].fillna("").astype(str).str.strip()
        df["residual_delta_tier"] = tier.mask(tier.eq(""), inferred_tier)
    default_tfv_guard = float(cfg["experiment"].get("tfv_guard_pct", 0.005)) * pd.to_numeric(
        df.get("baseline_TFV", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    default_peak_guard = float(cfg["experiment"].get("peak_guard_pct", 0.010)) * pd.to_numeric(
        df.get("baseline_peak_TFV_rate", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    if "tfv_guard" not in df:
        df["tfv_guard"] = default_tfv_guard
    else:
        df["tfv_guard"] = pd.to_numeric(df["tfv_guard"], errors="coerce").fillna(default_tfv_guard)
    if "peak_guard" not in df:
        df["peak_guard"] = default_peak_guard
    else:
        df["peak_guard"] = pd.to_numeric(df["peak_guard"], errors="coerce").fillna(default_peak_guard)
    df["pfv_improve"] = df["delta_PFV"] < 0
    df["pfv_worse"] = df["delta_PFV"] > 0
    df["safe_guarded"] = (df["delta_TFV"] <= df["tfv_guard"]) & (df["delta_peak"] <= df["peak_guard"])
    df["pfv_improve_safe"] = df["pfv_improve"] & df["safe_guarded"]
    df["nonzero_action"] = df.get("feat_delta_abs_max", pd.Series(1.0, index=df.index)).fillna(0.0) > 1e-6

    overall = {
        "rows": int(len(df)),
        "events": int(df["event_id"].nunique()) if "event_id" in df else 0,
        "templates": int(df["template_name"].nunique()),
        "nonzero_action_rows": int(df["nonzero_action"].sum()),
        "pfv_improve_n": int(df["pfv_improve"].sum()),
        "pfv_worse_n": int(df["pfv_worse"].sum()),
        "pfv_zero_n": int((~df["pfv_improve"] & ~df["pfv_worse"]).sum()),
        "safe_guarded_n": int(df["safe_guarded"].sum()),
        "pfv_improve_safe_n": int(df["pfv_improve_safe"].sum()),
        "pfv_improve_frac": float(df["pfv_improve"].mean()) if len(df) else 0.0,
        "pfv_worse_frac": float(df["pfv_worse"].mean()) if len(df) else 0.0,
        "pfv_improve_safe_frac": float(df["pfv_improve_safe"].mean()) if len(df) else 0.0,
        "median_delta_PFV": float(df["delta_PFV"].median()) if len(df) else 0.0,
        "mean_delta_PFV": float(df["delta_PFV"].mean()) if len(df) else 0.0,
        "p05_delta_PFV": float(df["delta_PFV"].quantile(0.05)) if len(df) else 0.0,
        "p95_delta_PFV": float(df["delta_PFV"].quantile(0.95)) if len(df) else 0.0,
        "mean_delta_TFV": float(df["delta_TFV"].mean()) if len(df) else 0.0,
        "mean_delta_peak": float(df["delta_peak"].mean()) if len(df) else 0.0,
    }
    pd.DataFrame([overall]).to_csv(out / "residual_counterfactual_overall.csv", index=False)
    (out / "residual_counterfactual_overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    by_template = (
        df.groupby(["template_name", "residual_delta_tier", "residual_delta"], dropna=False)
        .agg(
            n=("delta_PFV", "size"),
            events=("event_id", "nunique"),
            pfv_improve_frac=("pfv_improve", "mean"),
            pfv_worse_frac=("pfv_worse", "mean"),
            safe_guarded_frac=("safe_guarded", "mean"),
            pfv_improve_safe_frac=("pfv_improve_safe", "mean"),
            median_delta_PFV=("delta_PFV", "median"),
            mean_delta_PFV=("delta_PFV", "mean"),
            mean_delta_TFV=("delta_TFV", "mean"),
            mean_delta_peak=("delta_peak", "mean"),
            action_nonzero_frac=("nonzero_action", "mean"),
        )
        .reset_index()
        .sort_values(["pfv_improve_safe_frac", "mean_delta_PFV"], ascending=[False, True])
    )
    by_template.to_csv(out / "residual_counterfactual_by_template.csv", index=False)

    by_tier = (
        df.groupby(["residual_delta_tier"], dropna=False)
        .agg(
            n=("delta_PFV", "size"),
            events=("event_id", "nunique"),
            templates=("template_name", "nunique"),
            pfv_improve_frac=("pfv_improve", "mean"),
            pfv_worse_frac=("pfv_worse", "mean"),
            safe_guarded_frac=("safe_guarded", "mean"),
            pfv_improve_safe_frac=("pfv_improve_safe", "mean"),
            median_delta_PFV=("delta_PFV", "median"),
            mean_delta_PFV=("delta_PFV", "mean"),
            mean_delta_TFV=("delta_TFV", "mean"),
            mean_delta_peak=("delta_peak", "mean"),
        )
        .reset_index()
        .sort_values("residual_delta_tier")
    )
    by_tier.to_csv(out / "residual_counterfactual_by_delta_tier.csv", index=False)

    top_cols = [
        c
        for c in [
            "event_id",
            "template_name",
            "residual_delta_tier",
            "residual_delta",
            "override_start_min",
            "delta_PFV",
            "delta_TFV",
            "delta_peak",
            "baseline_PFV",
            "PFV",
            "baseline_TFV",
            "TFV",
            "detail_file",
        ]
        if c in df.columns
    ]
    top = df[df["pfv_improve_safe"]].sort_values("delta_PFV").head(int(args.top_n))[top_cols]
    top.to_csv(out / "top_safe_beneficial_residual_actions.csv", index=False)
    # Missing-data plan: templates with good PFV signal but few safe samples
    plan = by_template[(by_template["n"] < 80) | (by_template["events"] < 10)].copy()
    plan["recommended_action"] = "add_targeted_counterfactuals_for_this_template_delta"
    plan.to_csv(out / "residual_counterfactual_gap_plan.csv", index=False)
    print(json.dumps(overall, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
