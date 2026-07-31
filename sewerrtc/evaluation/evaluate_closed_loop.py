from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_reduction_pct(baseline: pd.Series, proposed: pd.Series, eps: float = 1e-6) -> pd.Series:
    """Percent reduction with undefined zero-baseline cases represented as NaN.

    A baseline PFV of zero followed by a positive proposed PFV is a real
    degradation, but its percent reduction is mathematically undefined. The old
    1e-9 denominator created 1e15-sized artifacts that polluted summaries and
    significance tests. Worse-fraction is handled separately from absolute
    deltas, so NaN here is the clean statistical representation.
    """
    b = pd.to_numeric(baseline, errors="coerce")
    p = pd.to_numeric(proposed, errors="coerce")
    out = pd.Series(np.nan, index=b.index, dtype=float)
    mask = b.abs() > float(eps)
    out.loc[mask] = (b.loc[mask] - p.loc[mask]) / b.loc[mask] * 100.0
    both_zero = (~mask) & (p.abs() <= float(eps))
    out.loc[both_zero] = 0.0
    return out


def compare_to_baseline(proposed: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    key = ["event_id", "duration_min"]
    base = baseline.rename(columns={c: f"{c}_baseline" for c in baseline.columns if c not in key})
    prop = proposed.rename(columns={c: f"{c}_proposed" for c in proposed.columns if c not in key})
    df = prop.merge(base, on=key, how="left")
    for m in ["TFV", "PFV", "peak_TFV_rate", "priority_flood_duration_min"]:
        p, b = f"{m}_proposed", f"{m}_baseline"
        if p in df and b in df:
            df[f"{m}_delta"] = pd.to_numeric(df[p], errors="coerce") - pd.to_numeric(df[b], errors="coerce")
            df[f"{m}_reduction_pct"] = _safe_reduction_pct(df[b], df[p])
    if "wall_time_sec_proposed" in df and "wall_time_sec_baseline" in df:
        df["wall_time_ratio_proposed_vs_baseline"] = df["wall_time_sec_proposed"] / df["wall_time_sec_baseline"].clip(lower=1e-9)
        df["wall_time_delta_sec"] = df["wall_time_sec_proposed"] - df["wall_time_sec_baseline"]
    if "rows_proposed" in df and "wall_time_sec_proposed" in df:
        df["proposed_wall_time_sec_per_step"] = df["wall_time_sec_proposed"] / df["rows_proposed"].clip(lower=1)
    if "rows_baseline" in df and "wall_time_sec_baseline" in df:
        df["baseline_wall_time_sec_per_step"] = df["wall_time_sec_baseline"] / df["rows_baseline"].clip(lower=1)
    if "action_changes_proposed" in df and "action_changes_baseline" in df:
        df["action_changes_delta"] = df["action_changes_proposed"] - df["action_changes_baseline"]
    return df


def gate_summary(comp: pd.DataFrame | dict) -> dict:
    if isinstance(comp, dict):
        comp = pd.DataFrame([comp])
    if comp.empty:
        return {"passed": False, "reason": "empty comparison"}
    pfv_series = pd.to_numeric(comp.get("PFV_reduction_pct", pd.Series([0])), errors="coerce")
    tfv_series = pd.to_numeric(comp.get("TFV_reduction_pct", pd.Series([0])), errors="coerce")
    peak_series = pd.to_numeric(comp.get("peak_TFV_rate_reduction_pct", pd.Series([0])), errors="coerce")
    pfv_med = float(pfv_series.median(skipna=True)) if pfv_series.notna().any() else 0.0
    tfv_mean = float(tfv_series.mean(skipna=True)) if tfv_series.notna().any() else 0.0
    peak_mean = float(peak_series.mean(skipna=True)) if peak_series.notna().any() else 0.0
    pfv_delta = pd.to_numeric(comp.get("PFV_delta", pd.Series([0])), errors="coerce").fillna(0.0)
    tfv_delta = pd.to_numeric(comp.get("TFV_delta", pd.Series([0])), errors="coerce").fillna(0.0)
    peak_delta = pd.to_numeric(comp.get("peak_TFV_rate_delta", pd.Series([0])), errors="coerce").fillna(0.0)
    pfv_worse = float((pfv_delta > 1e-6).mean())
    tfv_worse = float((tfv_delta > 1e-6).mean())
    peak_worse = float((peak_delta > 1e-6).mean())
    if "PFV_worse_frac" in comp and len(comp) == 1:
        pfv_worse = float(pd.to_numeric(comp["PFV_worse_frac"], errors="coerce").iloc[0])
    if "TFV_worse_frac" in comp and len(comp) == 1:
        tfv_worse = float(pd.to_numeric(comp["TFV_worse_frac"], errors="coerce").iloc[0])
    if "peak_worse_frac" in comp and len(comp) == 1:
        peak_worse = float(pd.to_numeric(comp["peak_worse_frac"], errors="coerce").iloc[0])
    time_per_step = float(comp.get("proposed_wall_time_sec_per_step", pd.Series([float("nan")])).mean())
    fallback_rate = float(comp.get("fallback_rate_proposed", pd.Series([float("nan")])).mean())
    thresholds = {
        "PFV_median_reduction_pct": 0.3,
        "PFV_worse_frac": 0.34,
        "TFV_mean_reduction_pct": -0.5,
        "peak_TFV_rate_mean_reduction_pct": -1.0,
        "TFV_worse_frac": 0.34,
        "peak_worse_frac": 0.34,
    }
    reasons = []
    if not (pfv_med > thresholds["PFV_median_reduction_pct"]):
        reasons.append(f"PFV_median_reduction_pct {pfv_med:.3f} <= {thresholds['PFV_median_reduction_pct']:.3f}")
    if not (pfv_worse <= thresholds["PFV_worse_frac"]):
        reasons.append(f"PFV_worse_frac {pfv_worse:.3f} > {thresholds['PFV_worse_frac']:.3f}")
    if not (tfv_mean >= thresholds["TFV_mean_reduction_pct"]):
        reasons.append(f"TFV_mean_reduction_pct {tfv_mean:.3f} < {thresholds['TFV_mean_reduction_pct']:.3f}")
    if not (peak_mean >= thresholds["peak_TFV_rate_mean_reduction_pct"]):
        reasons.append(
            f"peak_TFV_rate_mean_reduction_pct {peak_mean:.3f} < {thresholds['peak_TFV_rate_mean_reduction_pct']:.3f}"
        )
    if not (tfv_worse <= thresholds["TFV_worse_frac"]):
        reasons.append(f"TFV_worse_frac {tfv_worse:.3f} > {thresholds['TFV_worse_frac']:.3f}")
    if not (peak_worse <= thresholds["peak_worse_frac"]):
        reasons.append(f"peak_worse_frac {peak_worse:.3f} > {thresholds['peak_worse_frac']:.3f}")
    passed = not reasons
    return {
        "passed": bool(passed),
        "reasons": reasons,
        "gate_thresholds": thresholds,
        "PFV_median_reduction_pct": pfv_med,
        "PFV_worse_frac": pfv_worse,
        "TFV_worse_frac": tfv_worse,
        "peak_worse_frac": peak_worse,
        "TFV_mean_reduction_pct": tfv_mean,
        "peak_TFV_rate_mean_reduction_pct": peak_mean,
        "proposed_wall_time_sec_per_step_mean": time_per_step,
        "fallback_rate_mean": fallback_rate,
    }


def summarize_by_policy(comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if comp.empty:
        return pd.DataFrame()
    group_cols = ["baseline_policy"] if "baseline_policy" in comp else [None]
    iterator = comp.groupby("baseline_policy") if "baseline_policy" in comp else [(None, comp)]
    for policy, sub in iterator:
        row = {
            "baseline_policy": policy or "baseline",
            "n_events": int(len(sub)),
        }
        for m in ["PFV", "TFV", "peak_TFV_rate", "priority_flood_duration_min"]:
            c = f"{m}_reduction_pct"
            if c in sub:
                vals = pd.to_numeric(sub[c], errors="coerce")
                row[f"{m}_mean_reduction_pct"] = float(vals.mean(skipna=True)) if vals.notna().any() else float("nan")
                row[f"{m}_median_reduction_pct"] = float(vals.median(skipna=True)) if vals.notna().any() else float("nan")
                delta = pd.to_numeric(sub.get(f"{m}_delta", pd.Series(0, index=sub.index)), errors="coerce").fillna(0.0)
                row[f"{m}_worse_frac"] = float((delta > 1e-6).mean())
                row[f"{m}_undefined_pct_frac"] = float(vals.isna().mean())
        for c in [
            "wall_time_sec_proposed",
            "wall_time_sec_baseline",
            "proposed_wall_time_sec_per_step",
            "baseline_wall_time_sec_per_step",
            "wall_time_ratio_proposed_vs_baseline",
            "action_changes_proposed",
            "action_changes_baseline",
            "fallback_rate_proposed",
        ]:
            if c in sub:
                row[f"{c}_mean"] = float(pd.to_numeric(sub[c], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)
