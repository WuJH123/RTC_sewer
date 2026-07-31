from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate_closed_loop import compare_to_baseline


@dataclass(frozen=True)
class RiskThresholds:
    low_pfv_threshold: float = 1000.0
    high_pfv_threshold: float = 20000.0
    near_zero_pfv_epsilon: float = 100.0
    low_priority_depth_threshold_m: float = 0.02
    high_priority_depth_threshold_m: float = 0.20
    high_risk_exposure_depth_m: float = 0.20
    high_risk_exposure_time_min: float = 30.0
    control_step_sec: int = 300

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "RiskThresholds":
        r = cfg.get("risk_stratification", {}) or {}
        exp = cfg.get("experiment", {}) or {}
        return cls(
            low_pfv_threshold=float(r.get("low_pfv_threshold", 1000.0)),
            high_pfv_threshold=float(r.get("high_pfv_threshold", 20000.0)),
            near_zero_pfv_epsilon=float(r.get("near_zero_pfv_epsilon", 100.0)),
            low_priority_depth_threshold_m=float(r.get("low_priority_depth_threshold_m", 0.02)),
            high_priority_depth_threshold_m=float(r.get("high_priority_depth_threshold_m", 0.20)),
            high_risk_exposure_depth_m=float(r.get("high_risk_exposure_depth_m", 0.20)),
            high_risk_exposure_time_min=float(r.get("high_risk_exposure_time_min", 30.0)),
            control_step_sec=int(exp.get("control_step_sec", 300)),
        )


def _num(s: pd.Series | Any, default: float = 0.0) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").fillna(default)
    return pd.Series([default])


def _safe_reduction(baseline: pd.Series, proposed: pd.Series, eps: float = 1e-6) -> pd.Series:
    b = pd.to_numeric(baseline, errors="coerce")
    p = pd.to_numeric(proposed, errors="coerce")
    out = pd.Series(np.nan, index=b.index, dtype=float)
    mask = b.abs() > float(eps)
    out.loc[mask] = (b.loc[mask] - p.loc[mask]) / b.loc[mask] * 100.0
    out.loc[(~mask) & (p.abs() <= float(eps))] = 0.0
    return out


def _priority_depth_metrics(detail_file: str | Path, depth_nodes: list[str], thresholds: RiskThresholds) -> dict[str, float]:
    path = Path(str(detail_file))
    if not path.exists():
        return {
            "internal_priority_peak_depth": float("nan"),
            "internal_high_risk_exposure_time_min": 0.0,
        }
    try:
        header = pd.read_csv(path, nrows=0)
        hcols = [f"h:{n}" for n in depth_nodes if f"h:{n}" in header.columns]
        detail = pd.read_csv(path, usecols=hcols) if hcols else pd.DataFrame()
    except Exception:
        return {
            "internal_priority_peak_depth": float("nan"),
            "internal_high_risk_exposure_time_min": 0.0,
        }
    if not hcols:
        return {
            "internal_priority_peak_depth": float("nan"),
            "internal_high_risk_exposure_time_min": 0.0,
        }
    depths = detail[hcols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    peak = float(depths.max().max())
    exposure_steps = int((depths.max(axis=1) >= thresholds.high_risk_exposure_depth_m).sum())
    exposure_min = exposure_steps * thresholds.control_step_sec / 60.0
    return {
        "internal_priority_peak_depth": peak,
        "internal_high_risk_exposure_time_min": float(exposure_min),
    }


def recompute_priority_kpis_for_results(
    results: pd.DataFrame,
    priority_nodes: list[str],
    dt_sec: int = 300,
    keep_original: bool = True,
) -> pd.DataFrame:
    """Recompute priority-dependent KPI columns from detail files.

    Project5 changes the priority-zone definition to the Project2
    waterlogging/bottleneck nodes. Any PFV already stored in Project4 summary
    CSVs belongs to the old structural top-k priority zone, so it must not be
    used after changing the target zone.
    """

    out = results.copy()
    if keep_original:
        for col in ["PFV", "priority_flood_duration_min"]:
            if col in out and f"original_{col}" not in out:
                out[f"original_{col}"] = out[col]
    rows: list[dict[str, float]] = []
    for _, row in out.iterrows():
        detail_file = Path(str(row.get("detail_file", "")))
        if not detail_file.exists():
            rows.append(
                {
                    "PFV": float("nan"),
                    "priority_flood_duration_min": float("nan"),
                    "priority_peak_flood_rate": float("nan"),
                    "priority_nodes_present": 0.0,
                    "priority_nodes_missing": float(len(priority_nodes)),
                }
            )
            continue
        header = pd.read_csv(detail_file, nrows=0)
        pr_cols = [f"flood:{n}" for n in priority_nodes if f"flood:{n}" in header.columns]
        missing = [n for n in priority_nodes if f"flood:{n}" not in header.columns]
        detail = pd.read_csv(detail_file, usecols=pr_cols) if pr_cols else pd.DataFrame()
        if pr_cols:
            rate = detail[pr_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float).sum(axis=1)
            pfv = float(rate.sum() * int(dt_sec))
            priority_peak = float(rate.max()) if len(rate) else 0.0
            priority_duration = float((rate > 1e-9).sum() * int(dt_sec) / 60.0)
        else:
            pfv = 0.0
            priority_peak = 0.0
            priority_duration = 0.0
        rows.append(
            {
                "PFV": pfv,
                "priority_flood_duration_min": priority_duration,
                "priority_peak_flood_rate": priority_peak,
                "priority_nodes_present": float(len(pr_cols)),
                "priority_nodes_missing": float(len(missing)),
            }
        )
    recalculated = pd.DataFrame(rows, index=out.index)
    for col in recalculated.columns:
        out[col] = recalculated[col]
    return out


def classify_event(row: pd.Series, thresholds: RiskThresholds) -> str:
    pfv = float(row.get("internal_PFV", 0.0) or 0.0)
    # Classification must be anchored to the baseline priority flood volume.
    # The detail CSV h:* columns are hydraulic states and may include absolute
    # node depth/elevation-like values; using a raw 0.2 m threshold here made
    # every Wuhan event "high risk" and prevented low-risk false-intervention
    # diagnostics from working. Depth/exposure columns remain in the event table
    # as secondary diagnostics, but PFV defines the event risk stratum.
    if pfv >= thresholds.high_pfv_threshold:
        return "high_risk_event"
    if pfv <= thresholds.low_pfv_threshold or pfv <= thresholds.near_zero_pfv_epsilon:
        return "low_risk_event"
    return "medium_risk_event"


def build_event_table(
    baseline_results: pd.DataFrame,
    priority_nodes: list[str],
    thresholds: RiskThresholds,
    baseline_policy: str = "internal_rules",
    recompute_priority_kpis: bool = True,
    depth_nodes: list[str] | None = None,
) -> pd.DataFrame:
    base = baseline_results.copy()
    if recompute_priority_kpis:
        base = recompute_priority_kpis_for_results(
            base,
            priority_nodes,
            dt_sec=int(thresholds.control_step_sec),
            keep_original=True,
        )
    if "policy_id" in base:
        internal = base[base["policy_id"].astype(str).eq(baseline_policy)].copy()
    else:
        internal = base.copy()
    if internal.empty:
        raise ValueError(f"No baseline rows found for policy_id={baseline_policy}")
    rows = []
    depth_metric_nodes = depth_nodes if depth_nodes is not None else priority_nodes
    for _, row in internal.iterrows():
        metrics = _priority_depth_metrics(row.get("detail_file", ""), depth_metric_nodes, thresholds)
        rows.append(
            {
                "event_id": str(row.get("event_id", "")),
                "duration_min": int(row.get("duration_min", 0) or 0),
                "internal_policy_id": baseline_policy,
                "internal_PFV": float(row.get("PFV", 0.0) or 0.0),
                "internal_TFV": float(row.get("TFV", 0.0) or 0.0),
                "internal_peak_TFV_rate": float(row.get("peak_TFV_rate", 0.0) or 0.0),
                "internal_detail_file": str(row.get("detail_file", "")),
                **metrics,
            }
        )
    table = pd.DataFrame(rows)
    table["event_risk_class"] = table.apply(lambda r: classify_event(r, thresholds), axis=1)
    table["is_near_zero_pfv"] = table["internal_PFV"].abs() <= thresholds.near_zero_pfv_epsilon
    return table.sort_values(["event_risk_class", "internal_PFV", "event_id"], ascending=[True, False, True])


def _merge_history_features(comp: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = comp.copy()
    if history.empty or "event_id" not in history:
        out["action_use_rate"] = np.nan
        out["pump_action_count"] = 0
        out["storage_action_count"] = 0
        out["native_fallback_rate"] = out.get("fallback_rate_proposed", np.nan)
        out["unnecessary_action_count"] = 0
        return out
    hist = history.copy()
    fallback_col = "fallback_to_nominal" if "fallback_to_nominal" in hist else "fallback_to_native"
    if fallback_col not in hist:
        hist[fallback_col] = True
    hist[fallback_col] = hist[fallback_col].fillna(True).astype(bool)
    if "selected_candidate_label" not in hist.columns:
        hist["selected_candidate_label"] = pd.Series("", index=hist.index, dtype="object")
    else:
        hist["selected_candidate_label"] = hist["selected_candidate_label"].fillna("").astype(str)
    hist["is_intervention"] = ~hist[fallback_col]
    hist["pump_action"] = hist["selected_candidate_label"].str.contains("pump", case=False, na=False) & hist["is_intervention"]
    hist["storage_action"] = (
        hist["selected_candidate_label"].str.contains("storage|inlet|release|retain", case=False, regex=True, na=False)
        & hist["is_intervention"]
    )
    agg = (
        hist.groupby("event_id", as_index=False)
        .agg(
            controller_steps=("event_id", "size"),
            intervention_steps=("is_intervention", "sum"),
            native_fallback_rate=(fallback_col, "mean"),
            pump_action_count=("pump_action", "sum"),
            storage_action_count=("storage_action", "sum"),
        )
    )
    agg["action_use_rate"] = agg["intervention_steps"] / agg["controller_steps"].clip(lower=1)
    out = out.merge(agg, on="event_id", how="left")
    for c in ["action_use_rate", "native_fallback_rate"]:
        out[c] = pd.to_numeric(out.get(c), errors="coerce")
    for c in ["pump_action_count", "storage_action_count", "intervention_steps"]:
        out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0).astype(int)
    out["unnecessary_action_count"] = np.where(
        out["event_risk_class"].eq("low_risk_event"),
        out["intervention_steps"],
        0,
    )
    return out


def build_risk_stratified_comparison(
    proposed_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    event_table: pd.DataFrame,
    history: pd.DataFrame,
    baseline_policy: str = "internal_rules",
) -> pd.DataFrame:
    base = baseline_results.copy()
    if "policy_id" in base:
        base = base[base["policy_id"].astype(str).eq(baseline_policy)].copy()
    comp = compare_to_baseline(proposed_results.copy(), base.copy())
    comp["baseline_policy"] = baseline_policy
    comp = comp.merge(event_table[["event_id", "event_risk_class", "is_near_zero_pfv", "internal_PFV"]], on="event_id", how="left")
    for metric in ["PFV", "TFV", "peak_TFV_rate"]:
        p = f"{metric}_proposed"
        b = f"{metric}_baseline"
        if p in comp and b in comp:
            comp[f"absolute_delta_{metric}"] = pd.to_numeric(comp[p], errors="coerce") - pd.to_numeric(comp[b], errors="coerce")
            comp[f"{metric}_reduction_pct_near_zero_safe"] = _safe_reduction(comp[b], comp[p], eps=100.0 if metric == "PFV" else 1e-6)
    comp = _merge_history_features(comp, history)
    comp["low_risk_false_intervention"] = comp["event_risk_class"].eq("low_risk_event") & (comp["action_use_rate"].fillna(0.0) > 0.0)
    comp["induced_PFV_worsen"] = comp["absolute_delta_PFV"].fillna(0.0) > 1e-6
    comp["induced_TFV_worsen"] = comp["absolute_delta_TFV"].fillna(0.0) > 1e-6
    comp["induced_peak_worsen"] = comp["absolute_delta_peak_TFV_rate"].fillna(0.0) > 1e-6
    return comp


def summarize_risk_strata(comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if comp.empty:
        return pd.DataFrame()
    for risk_class, sub in comp.groupby("event_risk_class", dropna=False):
        row = {
            "event_risk_class": risk_class,
            "n_events": int(len(sub)),
            "PFV_mean_reduction_pct": float(pd.to_numeric(sub["PFV_reduction_pct"], errors="coerce").mean(skipna=True)),
            "PFV_median_reduction_pct": float(pd.to_numeric(sub["PFV_reduction_pct"], errors="coerce").median(skipna=True)),
            "absolute_delta_PFV_mean": float(pd.to_numeric(sub["absolute_delta_PFV"], errors="coerce").mean(skipna=True)),
            "absolute_delta_PFV_median": float(pd.to_numeric(sub["absolute_delta_PFV"], errors="coerce").median(skipna=True)),
            "PFV_worse_frac": float(pd.to_numeric(sub["absolute_delta_PFV"], errors="coerce").fillna(0.0).gt(1e-6).mean()),
            "TFV_mean_reduction_pct": float(pd.to_numeric(sub["TFV_reduction_pct"], errors="coerce").mean(skipna=True)),
            "peak_TFV_rate_mean_reduction_pct": float(
                pd.to_numeric(sub["peak_TFV_rate_reduction_pct"], errors="coerce").mean(skipna=True)
            ),
            "TFV_worse_frac": float(pd.to_numeric(sub["absolute_delta_TFV"], errors="coerce").fillna(0.0).gt(1e-6).mean()),
            "peak_worse_frac": float(pd.to_numeric(sub["absolute_delta_peak_TFV_rate"], errors="coerce").fillna(0.0).gt(1e-6).mean()),
            "action_use_rate": float(pd.to_numeric(sub["action_use_rate"], errors="coerce").mean(skipna=True)),
            "native_fallback_rate": float(pd.to_numeric(sub["native_fallback_rate"], errors="coerce").mean(skipna=True)),
            "pump_action_count": int(pd.to_numeric(sub["pump_action_count"], errors="coerce").fillna(0).sum()),
            "storage_action_count": int(pd.to_numeric(sub["storage_action_count"], errors="coerce").fillna(0).sum()),
            "near_zero_pfv_frac": float(sub["is_near_zero_pfv"].fillna(False).astype(bool).mean()),
        }
        if str(risk_class) == "low_risk_event":
            row.update(
                {
                    "false_intervention_rate": float(sub["low_risk_false_intervention"].fillna(False).astype(bool).mean()),
                    "induced_PFV_worsen_count": int(sub["induced_PFV_worsen"].fillna(False).astype(bool).sum()),
                    "induced_TFV_worsen_count": int(sub["induced_TFV_worsen"].fillna(False).astype(bool).sum()),
                    "induced_peak_worsen_count": int(sub["induced_peak_worsen"].fillna(False).astype(bool).sum()),
                    "mean_absolute_delta_PFV": float(pd.to_numeric(sub["absolute_delta_PFV"], errors="coerce").abs().mean(skipna=True)),
                    "unnecessary_action_count": int(pd.to_numeric(sub["unnecessary_action_count"], errors="coerce").fillna(0).sum()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("event_risk_class")


def build_water_research_tables(comp: pd.DataFrame, summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    high = comp[comp["event_risk_class"].eq("high_risk_event")].copy()
    low = comp[comp["event_risk_class"].eq("low_risk_event")].copy()
    main_rows = []
    for label, sub in [("all_events_diagnostic", comp), ("high_risk_main", high), ("low_risk_safety", low)]:
        if sub.empty:
            continue
        main_rows.append(
            {
                "analysis_scope": label,
                "n_events": int(len(sub)),
                "PFV_mean_reduction_pct": float(pd.to_numeric(sub["PFV_reduction_pct"], errors="coerce").mean(skipna=True)),
                "PFV_median_reduction_pct": float(pd.to_numeric(sub["PFV_reduction_pct"], errors="coerce").median(skipna=True)),
                "absolute_delta_PFV_mean": float(pd.to_numeric(sub["absolute_delta_PFV"], errors="coerce").mean(skipna=True)),
                "PFV_worse_frac": float(pd.to_numeric(sub["absolute_delta_PFV"], errors="coerce").fillna(0.0).gt(1e-6).mean()),
                "TFV_mean_reduction_pct": float(pd.to_numeric(sub["TFV_reduction_pct"], errors="coerce").mean(skipna=True)),
                "peak_TFV_rate_mean_reduction_pct": float(
                    pd.to_numeric(sub["peak_TFV_rate_reduction_pct"], errors="coerce").mean(skipna=True)
                ),
                "action_use_rate": float(pd.to_numeric(sub["action_use_rate"], errors="coerce").mean(skipna=True)),
                "native_fallback_rate": float(pd.to_numeric(sub["native_fallback_rate"], errors="coerce").mean(skipna=True)),
            }
        )
    low_table = pd.DataFrame()
    if not low.empty:
        low_table = pd.DataFrame(
            [
                {
                    "n_low_risk_events": int(len(low)),
                    "false_intervention_rate": float(low["low_risk_false_intervention"].fillna(False).astype(bool).mean()),
                    "induced_PFV_worsen_count": int(low["induced_PFV_worsen"].fillna(False).astype(bool).sum()),
                    "induced_TFV_worsen_count": int(low["induced_TFV_worsen"].fillna(False).astype(bool).sum()),
                    "induced_peak_worsen_count": int(low["induced_peak_worsen"].fillna(False).astype(bool).sum()),
                    "mean_absolute_delta_PFV": float(pd.to_numeric(low["absolute_delta_PFV"], errors="coerce").abs().mean(skipna=True)),
                    "native_fallback_rate": float(pd.to_numeric(low["native_fallback_rate"], errors="coerce").mean(skipna=True)),
                    "unnecessary_action_count": int(pd.to_numeric(low["unnecessary_action_count"], errors="coerce").fillna(0).sum()),
                }
            ]
        )
    high_table = pd.DataFrame()
    if not high.empty:
        high_table = high[
            [
                "event_id",
                "duration_min",
                "PFV_baseline",
                "PFV_proposed",
                "PFV_reduction_pct",
                "absolute_delta_PFV",
                "TFV_reduction_pct",
                "peak_TFV_rate_reduction_pct",
                "action_use_rate",
                "native_fallback_rate",
                "pump_action_count",
                "storage_action_count",
            ]
        ].copy()
    return {
        "water_research_main_table": pd.DataFrame(main_rows),
        "water_research_risk_stratified_table": summary.copy(),
        "low_risk_false_intervention_table": low_table,
        "high_risk_success_table": high_table,
    }
