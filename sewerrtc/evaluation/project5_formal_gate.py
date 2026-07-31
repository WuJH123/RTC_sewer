from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .policy_sets import normalize_policy_id

PROPOSED_POLICIES = ("proposed_gat_mpc", "proposed_native_shield")


@dataclass(frozen=True)
class Project5GateThresholds:
    min_high_risk_events: int = 20
    min_pfv_mean_reduction_pct: float = 10.0
    min_pfv_median_reduction_pct: float = 10.0
    max_tfv_worse_frac: float = 0.10
    max_peak_worse_frac: float = 0.10
    max_action_change_ratio: float = 1.20
    min_pfv_direction_accuracy: float = 0.70
    min_safe_precision: float = 0.80
    min_peak_direction_accuracy: float = 0.80
    min_guard_event_coverage: int = 10
    near_zero_pfv_epsilon: float = 100.0
    required_baseline_policies: tuple[str, ...] = ("no_control", "internal_rules", "efd_storage_priority", "auto_rbc")


def _num(s: pd.Series | float | int, default: float = np.nan):
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").reset_index(drop=True)
    try:
        return float(s)
    except Exception:
        return default


def _reduction_pct(base: pd.Series, value: pd.Series) -> pd.Series:
    base = pd.to_numeric(base, errors="coerce").reset_index(drop=True)
    value = pd.to_numeric(value, errors="coerce").reset_index(drop=True)
    if len(value) != len(base):
        value = value.reindex(range(len(base)))
    out = pd.Series(np.nan, index=base.index, dtype=float)
    mask = base.abs() > 1e-9
    out.loc[mask] = (base.loc[mask] - value.loc[mask]) / base.loc[mask] * 100.0
    out.loc[(~mask) & (value.abs() <= 1e-9)] = 0.0
    return out


def _action_change_ratio(proposed: pd.Series, baseline: pd.Series) -> float:
    proposed_mean = float(pd.to_numeric(proposed, errors="coerce").mean())
    baseline_mean = float(pd.to_numeric(baseline, errors="coerce").mean())
    if not np.isfinite(proposed_mean) or not np.isfinite(baseline_mean):
        return np.nan
    if abs(baseline_mean) <= 1e-9:
        if abs(proposed_mean) <= 1e-9:
            return 1.0
        return np.nan
    return proposed_mean / baseline_mean


def _col(work: pd.DataFrame, col: str) -> pd.Series:
    if col in work:
        return pd.to_numeric(work[col], errors="coerce").reset_index(drop=True)
    return pd.Series(np.nan, index=range(len(work)), dtype=float)


def _best_training_row(report: pd.DataFrame) -> dict:
    if report.empty:
        return {}
    work = report.copy()
    for col in ["score", "PFV_direction_accuracy", "safe_precision", "peak_direction_accuracy"]:
        if col in work:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "score" in work and work["score"].notna().any():
        return work.sort_values("score", ascending=True).iloc[0].to_dict()
    quality = (
        work.get("PFV_direction_accuracy", pd.Series(0.0, index=work.index)).fillna(0.0)
        + work.get("safe_precision", pd.Series(0.0, index=work.index)).fillna(0.0)
        + work.get("peak_direction_accuracy", pd.Series(0.0, index=work.index)).fillna(0.0)
    )
    return work.loc[quality.idxmax()].to_dict()


def _guard_event_coverage(guard: pd.DataFrame) -> tuple[int, int]:
    if guard.empty:
        return 0, 0
    work = guard.copy()
    if "empirical_allow" not in work:
        return 0, 0
    allowed = work["empirical_allow"].astype(str).str.lower().isin(["true", "1", "yes", "allow", "allowed"])
    allowed_rows = int(allowed.sum())
    if allowed_rows == 0:
        return 0, 0
    sub = work.loc[allowed].copy()
    if "event_id" in sub:
        return int(sub["event_id"].astype(str).nunique()), allowed_rows
    if "events" in sub:
        events = pd.to_numeric(sub["events"], errors="coerce").fillna(0.0)
        return int(events.max()) if len(events) else 0, allowed_rows
    return 0, allowed_rows


def _normalised_high_risk(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    work = paired.copy()
    if "policy_id" in work:
        work["policy_id"] = work["policy_id"].map(normalize_policy_id)
    if "event_risk_class" in work:
        work = work[work["event_risk_class"].astype(str).eq("high_risk_event")].copy()
    return work


def _proposed_policy_id(work: pd.DataFrame) -> str:
    if work.empty or "policy_id" not in work:
        return "proposed_gat_mpc"
    present = set(work["policy_id"].astype(str))
    for policy in PROPOSED_POLICIES:
        if policy in present:
            return policy
    return "proposed_gat_mpc"


def _metric_col(work: pd.DataFrame, preferred: str, fallback: str) -> str:
    return preferred if preferred in work else fallback


def _paired_high_risk(paired: pd.DataFrame) -> pd.DataFrame:
    work = _normalised_high_risk(paired)
    if work.empty:
        return work
    proposed_policy = _proposed_policy_id(work)

    if "policy_id" in work and {"internal_rules", proposed_policy}.issubset(set(work["policy_id"])):
        internal_cols = ["event_id", "project5_priority_PFV", "TFV", "peak_TFV_rate", "action_changes"]
        internal = work[work["policy_id"].eq("internal_rules")][[c for c in internal_cols if c in work]].copy()
        rename = {
            "project5_priority_PFV": "internal_project5_priority_PFV",
            "TFV": "internal_TFV",
            "peak_TFV_rate": "internal_peak_TFV_rate",
            "action_changes": "internal_action_changes",
        }
        internal = internal.rename(columns=rename)
        proposed = work[work["policy_id"].eq(proposed_policy)].copy()
        internal = internal[[c for c in internal.columns if c == "event_id" or c not in proposed.columns]]
        if len(internal.columns) > 1:
            return proposed.merge(internal, on="event_id", how="left")
        return proposed
    if "policy_id" in work:
        work = work[work["policy_id"].eq(proposed_policy)].copy()
    return work


def _baseline_comparisons(
    work: pd.DataFrame,
    proposed_policy: str,
    baseline_policies: Sequence[str] | None = None,
    near_zero_pfv_epsilon: float = 100.0,
) -> list[dict]:
    if work.empty or "policy_id" not in work or "event_id" not in work:
        return []
    pfv_col = _metric_col(work, "project5_priority_PFV", "PFV")
    if pfv_col not in work:
        return []
    proposed_cols = ["event_id", pfv_col, "TFV", "peak_TFV_rate", "action_changes"]
    proposed = work[work["policy_id"].eq(proposed_policy)][[c for c in proposed_cols if c in work]].copy()
    if proposed.empty:
        return []
    proposed = proposed.rename(
        columns={
            pfv_col: "proposed_PFV",
            "TFV": "proposed_TFV",
            "peak_TFV_rate": "proposed_peak_TFV_rate",
            "action_changes": "proposed_action_changes",
        }
    )
    rows = []
    if baseline_policies is None:
        policies = sorted(p for p in work["policy_id"].astype(str).unique() if p != proposed_policy)
    else:
        policies = [normalize_policy_id(p) for p in baseline_policies if normalize_policy_id(p) != proposed_policy]
    for baseline_policy in policies:
        baseline = work[work["policy_id"].eq(baseline_policy)][[c for c in proposed_cols if c in work]].copy()
        if baseline.empty:
            continue
        baseline = baseline.rename(
            columns={
                pfv_col: "baseline_PFV",
                "TFV": "baseline_TFV",
                "peak_TFV_rate": "baseline_peak_TFV_rate",
                "action_changes": "baseline_action_changes",
            }
        )
        merged = proposed.merge(baseline, on="event_id", how="inner")
        if merged.empty:
            continue
        baseline_pfv = pd.to_numeric(merged["baseline_PFV"], errors="coerce")
        near_zero = baseline_pfv.abs() <= float(near_zero_pfv_epsilon)
        pfv_stat = merged.loc[~near_zero].copy()
        pfv_red = _reduction_pct(pfv_stat["baseline_PFV"], pfv_stat["proposed_PFV"]) if not pfv_stat.empty else pd.Series(dtype=float)
        tfv_worse = (
            pd.to_numeric(merged.get("proposed_TFV", pd.Series(np.nan, index=merged.index)), errors="coerce")
            > pd.to_numeric(merged.get("baseline_TFV", pd.Series(np.nan, index=merged.index)), errors="coerce") + 1e-6
        )
        peak_worse = (
            pd.to_numeric(merged.get("proposed_peak_TFV_rate", pd.Series(np.nan, index=merged.index)), errors="coerce")
            > pd.to_numeric(merged.get("baseline_peak_TFV_rate", pd.Series(np.nan, index=merged.index)), errors="coerce") + 1e-6
        )
        proposed_actions = pd.to_numeric(
            merged.get("proposed_action_changes", pd.Series(np.nan, index=merged.index)),
            errors="coerce",
        )
        baseline_actions = pd.to_numeric(
            merged.get("baseline_action_changes", pd.Series(np.nan, index=merged.index)),
            errors="coerce",
        )
        rows.append(
            {
                "baseline_policy": baseline_policy,
                "paired_events": int(merged["event_id"].nunique()),
                "PFV_percent_stat_events": int(pfv_stat["event_id"].nunique()) if "event_id" in pfv_stat else int(len(pfv_stat)),
                "near_zero_reference_events": int(merged.loc[near_zero, "event_id"].nunique()),
                "near_zero_reference_event_ids": sorted(merged.loc[near_zero, "event_id"].astype(str).unique().tolist()),
                "near_zero_pfv_epsilon": float(near_zero_pfv_epsilon),
                "PFV_mean_reduction_pct": float(pfv_red.mean(skipna=True)),
                "PFV_median_reduction_pct": float(pfv_red.median(skipna=True)),
                "TFV_worse_frac": float(tfv_worse.mean()) if len(merged) else np.nan,
                "peak_worse_frac": float(peak_worse.mean()) if len(merged) else np.nan,
                "proposed_action_changes_mean": float(proposed_actions.mean()) if len(merged) else np.nan,
                "baseline_action_changes_mean": float(baseline_actions.mean()) if len(merged) else np.nan,
                "action_change_ratio": _action_change_ratio(proposed_actions, baseline_actions),
            }
        )
    return rows


def evaluate_project5_gate(
    paired: pd.DataFrame,
    residual_training_report: pd.DataFrame,
    empirical_guard: pd.DataFrame,
    thresholds: Project5GateThresholds | None = None,
) -> dict:
    th = thresholds or Project5GateThresholds()
    required_baselines = tuple(normalize_policy_id(p) for p in th.required_baseline_policies)
    high_risk = _normalised_high_risk(paired)
    proposed_policy = _proposed_policy_id(high_risk)
    prop = _paired_high_risk(high_risk)
    comparisons = _baseline_comparisons(
        high_risk,
        proposed_policy,
        baseline_policies=required_baselines,
        near_zero_pfv_epsilon=float(th.near_zero_pfv_epsilon),
    )
    reasons: list[str] = []
    high_risk_filter_applied = bool(not paired.empty and "event_risk_class" in paired)
    if not paired.empty and "event_risk_class" not in paired:
        reasons.append("event_risk_class missing; high-risk-only gate cannot be applied")
    n_events = int(prop["event_id"].nunique()) if "event_id" in prop else 0
    if n_events < th.min_high_risk_events:
        reasons.append(f"high-risk paired events {n_events} < {th.min_high_risk_events}")
    present_policies = set(high_risk["policy_id"].astype(str)) if "policy_id" in high_risk else set()
    missing_baselines = [p for p in required_baselines if p not in present_policies]
    for policy in missing_baselines:
        reasons.append(f"required high-risk baseline policy missing: {policy}")

    pfv_base_col = "internal_project5_priority_PFV"
    pfv_col = "project5_priority_PFV"
    if pfv_base_col not in prop and "internal_PFV" in prop:
        pfv_base_col = "internal_PFV"
    if pfv_col not in prop and "PFV" in prop:
        pfv_col = "PFV"
    pfv_red = _reduction_pct(_col(prop, pfv_base_col), _col(prop, pfv_col))
    pfv_mean = float(pfv_red.mean(skipna=True)) if len(pfv_red) else np.nan
    pfv_median = float(pfv_red.median(skipna=True)) if len(pfv_red) else np.nan
    if comparisons:
        pfv_means = [float(row["PFV_mean_reduction_pct"]) for row in comparisons if np.isfinite(row["PFV_mean_reduction_pct"])]
        pfv_medians = [float(row["PFV_median_reduction_pct"]) for row in comparisons if np.isfinite(row["PFV_median_reduction_pct"])]
        pfv_mean = float(min(pfv_means)) if pfv_means else np.nan
        pfv_median = float(min(pfv_medians)) if pfv_medians else np.nan
        tfv_worse = float(max(row["TFV_worse_frac"] for row in comparisons))
        peak_worse = float(max(row["peak_worse_frac"] for row in comparisons))
    else:
        tfv = _col(prop, "TFV")
        tfv_base = _col(prop, "internal_TFV")
        peak = _col(prop, "peak_TFV_rate")
        peak_base = _col(prop, "internal_peak_TFV_rate")
        tfv_worse = float((tfv > tfv_base + 1e-6).mean()) if len(prop) else np.nan
        peak_worse = float((peak > peak_base + 1e-6).mean()) if len(prop) else np.nan
    if not np.isfinite(pfv_mean) or pfv_mean <= th.min_pfv_mean_reduction_pct:
        reasons.append(f"PFV mean reduction {pfv_mean:.3f} <= {th.min_pfv_mean_reduction_pct:.3f}")
    if not np.isfinite(pfv_median) or pfv_median <= th.min_pfv_median_reduction_pct:
        reasons.append(f"PFV median reduction {pfv_median:.3f} <= {th.min_pfv_median_reduction_pct:.3f}")
    if not np.isfinite(tfv_worse) or tfv_worse > th.max_tfv_worse_frac:
        reasons.append(f"TFV_worse_frac {tfv_worse:.3f} > {th.max_tfv_worse_frac:.3f}")
    if not np.isfinite(peak_worse) or peak_worse > th.max_peak_worse_frac:
        reasons.append(f"peak_worse_frac {peak_worse:.3f} > {th.max_peak_worse_frac:.3f}")
    for row in comparisons:
        if int(row.get("PFV_percent_stat_events", 0) or 0) == 0:
            reasons.append(f"{proposed_policy} vs {row['baseline_policy']} has no non-near-zero reference PFV events for percent statistics")
        if row["PFV_mean_reduction_pct"] <= th.min_pfv_mean_reduction_pct:
            reasons.append(
                f"{proposed_policy} vs {row['baseline_policy']} PFV mean reduction "
                f"{row['PFV_mean_reduction_pct']:.3f} <= {th.min_pfv_mean_reduction_pct:.3f}"
            )
        if row["PFV_median_reduction_pct"] <= th.min_pfv_median_reduction_pct:
            reasons.append(
                f"{proposed_policy} vs {row['baseline_policy']} PFV median reduction "
                f"{row['PFV_median_reduction_pct']:.3f} <= {th.min_pfv_median_reduction_pct:.3f}"
            )
        if row["TFV_worse_frac"] > th.max_tfv_worse_frac:
            reasons.append(
                f"{proposed_policy} vs {row['baseline_policy']} TFV_worse_frac "
                f"{row['TFV_worse_frac']:.3f} > {th.max_tfv_worse_frac:.3f}"
            )
        if row["peak_worse_frac"] > th.max_peak_worse_frac:
            reasons.append(
                f"{proposed_policy} vs {row['baseline_policy']} peak_worse_frac "
                f"{row['peak_worse_frac']:.3f} > {th.max_peak_worse_frac:.3f}"
            )

    action_ratios = [
        float(row["action_change_ratio"])
        for row in comparisons
        if np.isfinite(float(row.get("action_change_ratio", np.nan)))
    ]
    action_ratio = float(max(action_ratios)) if action_ratios else np.nan
    if np.isfinite(action_ratio) and action_ratio > th.max_action_change_ratio:
        reasons.append(f"action change ratio vs dynamic baseline {action_ratio:.3f} > {th.max_action_change_ratio:.3f}")

    legacy_residual_gate_applied = proposed_policy == "proposed_native_shield"
    best = _best_training_row(residual_training_report) if legacy_residual_gate_applied else {}
    pfv_dir = float(best.get("PFV_direction_accuracy", 0.0) or 0.0) if legacy_residual_gate_applied else float("nan")
    safe_precision = float(best.get("safe_precision", 0.0) or 0.0) if legacy_residual_gate_applied else float("nan")
    peak_dir = float(best.get("peak_direction_accuracy", 0.0) or 0.0) if legacy_residual_gate_applied else float("nan")
    if legacy_residual_gate_applied:
        if pfv_dir < th.min_pfv_direction_accuracy:
            reasons.append(f"PFV direction accuracy {pfv_dir:.3f} < {th.min_pfv_direction_accuracy:.3f}")
        if safe_precision < th.min_safe_precision:
            reasons.append(f"safe precision {safe_precision:.3f} < {th.min_safe_precision:.3f}")
        if peak_dir < th.min_peak_direction_accuracy:
            reasons.append(f"peak direction accuracy {peak_dir:.3f} < {th.min_peak_direction_accuracy:.3f}")

    guard_events, allowed_guard_rows = _guard_event_coverage(empirical_guard) if legacy_residual_gate_applied else (0, 0)
    if legacy_residual_gate_applied and guard_events < th.min_guard_event_coverage:
        reasons.append(f"allowed empirical guard coverage {guard_events} < {th.min_guard_event_coverage} events")

    return {
        "passed": bool(len(reasons) == 0),
        "reasons": reasons,
        "proposed_policy": proposed_policy,
        "legacy_residual_gate_applied": legacy_residual_gate_applied,
        "baseline_comparisons": comparisons,
        "required_baseline_policies": list(required_baselines),
        "missing_baseline_policies": missing_baselines,
        "high_risk_filter_applied": high_risk_filter_applied,
        "near_zero_pfv_epsilon": float(th.near_zero_pfv_epsilon),
        "n_high_risk_paired_events": n_events,
        "PFV_mean_reduction_pct": pfv_mean,
        "PFV_median_reduction_pct": pfv_median,
        "TFV_worse_frac": tfv_worse,
        "peak_worse_frac": peak_worse,
        "action_change_ratio_vs_compared_baselines": action_ratio,
        "action_change_ratio_vs_internal": action_ratio,
        "residual_best": best,
        "residual_PFV_direction_accuracy": pfv_dir,
        "residual_safe_precision": safe_precision,
        "residual_peak_direction_accuracy": peak_dir,
        "allowed_guard_event_coverage": guard_events,
        "allowed_guard_rows": allowed_guard_rows,
        "thresholds": th.__dict__,
    }
