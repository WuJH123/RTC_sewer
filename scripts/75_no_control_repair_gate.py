from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.evaluation.policy_sets import normalize_policy_id
from sewerrtc.io.project_paths import ensure_dir, load_config


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(series: pd.Series | object, index: pd.Index | None = None) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    if index is None:
        index = pd.RangeIndex(0)
    return pd.Series(series, index=index, dtype=float)


def _metric_col(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    if preferred in df:
        return preferred
    return fallback


def _pct_reduction(base: pd.Series, value: pd.Series) -> pd.Series:
    base = pd.to_numeric(base, errors="coerce")
    value = pd.to_numeric(value, errors="coerce")
    out = pd.Series(np.nan, index=base.index, dtype=float)
    mask = base.abs() > 1.0e-9
    out.loc[mask] = (base.loc[mask] - value.loc[mask]) / base.loc[mask] * 100.0
    out.loc[(~mask) & (value.abs() <= 1.0e-9)] = 0.0
    return out


def _return_period(event_id: object) -> str:
    match = re.match(r"^(T\d+)", str(event_id))
    return match.group(1) if match else ""


def _bootstrap_ci(values: np.ndarray, *, seed: int = 20260715, iterations: int = 2000) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"lower": None, "median": None, "upper": None, "samples": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(int(iterations), dtype=float)
    for i in range(int(iterations)):
        means[i] = float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
    return {
        "lower": float(np.quantile(means, 0.025)),
        "median": float(np.quantile(means, 0.500)),
        "upper": float(np.quantile(means, 0.975)),
        "samples": int(arr.size),
    }


def _norm_policy_list(value, default: list[str]) -> list[str]:
    if value is None:
        raw = default
    elif isinstance(value, str):
        raw = [x.strip() for x in value.split(",")]
    else:
        raw = [str(x).strip() for x in value]
    out: list[str] = []
    for item in raw:
        policy = normalize_policy_id(item)
        if policy and policy not in out:
            out.append(policy)
    return out


def _proposed_policy(work: pd.DataFrame) -> str:
    if "policy_id" not in work:
        return "proposed_gat_mpc"
    present = set(work["policy_id"].astype(str))
    for policy in ("proposed_temporal_joint_36", "proposed_gat_mpc", "proposed_native_shield"):
        if policy in present:
            return policy
    return "proposed_gat_mpc"


def _paired_comparison(
    work: pd.DataFrame,
    proposed_policy: str,
    baseline_policy: str,
    near_zero: float,
    pfv_noninferiority_abs: float,
    pfv_noninferiority_frac: float,
) -> dict:
    pfv_col = _metric_col(work, "project5_priority_PFV", "PFV")
    cols = ["event_id", pfv_col, "TFV", "peak_TFV_rate", "action_changes"]
    proposed = work[work["policy_id"].eq(proposed_policy)][[c for c in cols if c in work]].copy()
    baseline = work[work["policy_id"].eq(baseline_policy)][[c for c in cols if c in work]].copy()
    if proposed.empty or baseline.empty:
        return {
            "baseline_policy": baseline_policy,
            "paired_events": 0,
            "missing": True,
        }
    proposed = proposed.rename(
        columns={
            pfv_col: "proposed_PFV",
            "TFV": "proposed_TFV",
            "peak_TFV_rate": "proposed_peak",
            "action_changes": "proposed_action_changes",
        }
    )
    baseline = baseline.rename(
        columns={
            pfv_col: "baseline_PFV",
            "TFV": "baseline_TFV",
            "peak_TFV_rate": "baseline_peak",
            "action_changes": "baseline_action_changes",
        }
    )
    merged = proposed.merge(baseline, on="event_id", how="inner")
    if merged.empty:
        return {
            "baseline_policy": baseline_policy,
            "paired_events": 0,
            "missing": True,
        }
    baseline_pfv = _num(merged["baseline_PFV"])
    proposed_pfv = _num(merged["proposed_PFV"])
    baseline_tfv = _num(merged["baseline_TFV"])
    proposed_tfv = _num(merged["proposed_TFV"])
    baseline_peak = _num(merged["baseline_peak"])
    proposed_peak = _num(merged["proposed_peak"])
    nonzero = baseline_pfv.abs() > float(near_zero)
    pfv_reduction = _pct_reduction(baseline_pfv.loc[nonzero], proposed_pfv.loc[nonzero])
    tfv_reduction = _pct_reduction(baseline_tfv, proposed_tfv)
    peak_reduction = _pct_reduction(baseline_peak, proposed_peak)
    pfv_margin = float(pfv_noninferiority_abs) + float(pfv_noninferiority_frac) * baseline_pfv.clip(lower=0.0)
    pfv_noninferior = proposed_pfv <= baseline_pfv + pfv_margin + 1.0e-9
    out = {
        "baseline_policy": baseline_policy,
        "paired_events": int(merged["event_id"].nunique()),
        "non_near_zero_pfv_events": int(merged.loc[nonzero, "event_id"].nunique()),
        "near_zero_pfv_events": int(merged.loc[~nonzero, "event_id"].nunique()),
        "near_zero_event_ids": sorted(merged.loc[~nonzero, "event_id"].astype(str).unique().tolist()),
        "PFV_mean_reduction_pct": float(pfv_reduction.mean(skipna=True)) if len(pfv_reduction) else np.nan,
        "PFV_median_reduction_pct": float(pfv_reduction.median(skipna=True)) if len(pfv_reduction) else np.nan,
        "PFV_mean_delta": float((proposed_pfv - baseline_pfv).mean(skipna=True)),
        "PFV_worse_frac_zero_tol": float((proposed_pfv > baseline_pfv + 1.0e-6).mean()) if len(merged) else np.nan,
        "PFV_noninferiority_abs": float(pfv_noninferiority_abs),
        "PFV_noninferiority_frac": float(pfv_noninferiority_frac),
        "PFV_noninferiority_margin_mean": float(pfv_margin.mean(skipna=True)) if len(merged) else np.nan,
        "PFV_worse_frac_noninferiority": float((~pfv_noninferior).mean()) if len(merged) else np.nan,
        "PFV_noninferior_frac": float(pfv_noninferior.mean()) if len(merged) else np.nan,
        "TFV_mean_reduction_pct": float(tfv_reduction.mean(skipna=True)),
        "TFV_median_reduction_pct": float(tfv_reduction.median(skipna=True)),
        "TFV_reduction_bootstrap95_lower": _bootstrap_ci(tfv_reduction.to_numpy()).get("lower"),
        "TFV_improved_frac_zero_tol": float((proposed_tfv < baseline_tfv - 1.0e-6).mean()) if len(merged) else np.nan,
        "TFV_worse_frac_zero_tol": float((proposed_tfv > baseline_tfv + 1.0e-6).mean()) if len(merged) else np.nan,
        "peak_mean_reduction_pct": float(peak_reduction.mean(skipna=True)),
        "peak_median_reduction_pct": float(peak_reduction.median(skipna=True)),
        "peak_worse_frac_zero_tol": float((proposed_peak > baseline_peak + 1.0e-9).mean()) if len(merged) else np.nan,
        "proposed_TFV_mean": float(proposed_tfv.mean(skipna=True)),
        "baseline_TFV_mean": float(baseline_tfv.mean(skipna=True)),
        "proposed_PFV_mean": float(proposed_pfv.mean(skipna=True)),
        "baseline_PFV_mean": float(baseline_pfv.mean(skipna=True)),
        "proposed_peak_mean": float(proposed_peak.mean(skipna=True)),
        "baseline_peak_mean": float(baseline_peak.mean(skipna=True)),
        "missing": False,
    }
    return out


def _pfv_sensitivity(
    work: pd.DataFrame,
    proposed_policy: str,
    baseline_policy: str,
    abs_margin: float,
    fracs: list[float],
) -> list[dict]:
    pfv_col = _metric_col(work, "project5_priority_PFV", "PFV")
    proposed = work[work["policy_id"].eq(proposed_policy)][["event_id", pfv_col]].rename(columns={pfv_col: "proposed_PFV"})
    baseline = work[work["policy_id"].eq(baseline_policy)][["event_id", pfv_col]].rename(columns={pfv_col: "baseline_PFV"})
    merged = proposed.merge(baseline, on="event_id", how="inner")
    if merged.empty:
        return []
    out = []
    baseline_pfv = _num(merged["baseline_PFV"])
    proposed_pfv = _num(merged["proposed_PFV"])
    for frac in fracs:
        margin = float(abs_margin) + float(frac) * baseline_pfv.clip(lower=0.0)
        ok = proposed_pfv <= baseline_pfv + margin + 1.0e-9
        out.append(
            {
                "baseline_policy": baseline_policy,
                "pfv_abs_margin_m3": float(abs_margin),
                "pfv_rel_margin": float(frac),
                "events": int(len(merged)),
                "noninferior_frac": float(ok.mean()) if len(ok) else np.nan,
                "worse_frac_noninferiority": float((~ok).mean()) if len(ok) else np.nan,
            }
        )
    return out


def _thresholds(cfg: dict) -> dict:
    default = {
        "gate_profile": "legacy_high_risk",
        "risk_filter": "high_risk_event",
        "min_high_risk_events": 20,
        "near_zero_pfv_epsilon": 100.0,
        "no_control_max_pfv_worse_frac": 0.10,
        "no_control_max_pfv_mean_increase_pct": 2.0,
        "no_control_pfv_noninferiority_abs": 100.0,
        "no_control_pfv_noninferiority_frac": 0.02,
        "no_control_min_tfv_mean_reduction_pct": 5.0,
        "no_control_max_peak_worse_frac": 0.20,
        "pfv_sensitivity_fracs": [0.005, 0.01, 0.02],
        "calibration_min_events": 14,
        "calibration_max_pfv_mean_increase_pct": 2.0,
        "calibration_min_pfv_noninferior_frac": 0.85,
        "calibration_min_tfv_mean_reduction_pct": 5.0,
        "calibration_min_tfv_improved_frac": 0.60,
        "calibration_max_tfv_worse_frac": 0.20,
        "calibration_min_peak_mean_reduction_pct": 0.0,
        "calibration_max_peak_worse_frac": 0.20,
        "formal_primary_return_periods": ["T5", "T10", "T20", "T30", "T50"],
        "formal_stress_return_periods": ["T75", "T100"],
        "formal_min_primary_events": 20,
        "formal_max_pfv_mean_increase_pct": 2.0,
        "formal_min_pfv_noninferior_frac": 0.90,
        "formal_min_tfv_mean_reduction_pct": 8.0,
        "formal_require_tfv_bootstrap_lower_positive": True,
        "formal_min_tfv_improved_frac": 0.70,
        "formal_max_tfv_worse_frac": 0.15,
        "formal_min_peak_mean_reduction_pct": 0.0,
        "formal_max_peak_worse_frac": 0.15,
        "stress_max_pfv_mean_increase_pct": 2.0,
        "stress_max_pfv_worse_frac_noninferiority": 0.25,
        "stress_min_peak_mean_reduction_pct": 0.0,
        "stress_max_peak_worse_frac": 0.25,
        "superiority_min_pfv_mean_reduction_pct": 0.0,
        "superiority_min_tfv_mean_reduction_pct": 0.0,
        "enforce_superiority_policies": True,
        "superiority_policies": ["efd_storage_priority", "auto_rbc"],
        "diagnostic_compare_policies": ["internal_rules"],
    }
    gate_cfg = ((cfg.get("evaluation", {}) or {}).get("no_control_repair_gate", {}) or {})
    merged = dict(default)
    merged.update(gate_cfg)
    merged["superiority_policies"] = _norm_policy_list(
        merged.get("superiority_policies"),
        default["superiority_policies"],
    )
    merged["diagnostic_compare_policies"] = _norm_policy_list(
        merged.get("diagnostic_compare_policies"),
        default["diagnostic_compare_policies"],
    )
    return merged


def _no_control_reasons(row: dict, th: dict, prefix: str, *, require_tfv: bool) -> list[str]:
    reasons: list[str] = []
    pfv_mean = float(row.get("PFV_mean_reduction_pct", np.nan))
    pfv_increase = -pfv_mean if np.isfinite(pfv_mean) else np.inf
    max_pfv_increase = float(th[f"{prefix}_max_pfv_mean_increase_pct"])
    if pfv_increase > max_pfv_increase:
        reasons.append(f"{prefix}: PFV mean increase vs no_control {pfv_increase:.3f}% > {max_pfv_increase:.3f}%")
    min_pfv_noninferior = float(th[f"{prefix}_min_pfv_noninferior_frac"])
    if float(row.get("PFV_noninferior_frac", -np.inf)) < min_pfv_noninferior:
        reasons.append(
            f"{prefix}: PFV noninferior fraction vs no_control {float(row.get('PFV_noninferior_frac', -np.inf)):.3f} < "
            f"{min_pfv_noninferior:.3f}"
        )
    if require_tfv:
        min_tfv = float(th[f"{prefix}_min_tfv_mean_reduction_pct"])
        if float(row.get("TFV_mean_reduction_pct", -np.inf)) < min_tfv:
            reasons.append(
                f"{prefix}: TFV mean reduction vs no_control {float(row.get('TFV_mean_reduction_pct', -np.inf)):.3f}% < "
                f"{min_tfv:.3f}%"
            )
        min_improved = float(th[f"{prefix}_min_tfv_improved_frac"])
        if float(row.get("TFV_improved_frac_zero_tol", -np.inf)) < min_improved:
            reasons.append(
                f"{prefix}: TFV improved fraction vs no_control {float(row.get('TFV_improved_frac_zero_tol', -np.inf)):.3f} < "
                f"{min_improved:.3f}"
            )
        max_worse = float(th[f"{prefix}_max_tfv_worse_frac"])
        if float(row.get("TFV_worse_frac_zero_tol", np.inf)) > max_worse:
            reasons.append(
                f"{prefix}: TFV worse fraction vs no_control {float(row.get('TFV_worse_frac_zero_tol', np.inf)):.3f} > "
                f"{max_worse:.3f}"
            )
        if bool(th.get(f"{prefix}_require_tfv_bootstrap_lower_positive", False)):
            lower = row.get("TFV_reduction_bootstrap95_lower")
            if lower is None or not np.isfinite(float(lower)) or float(lower) <= 0.0:
                reasons.append(f"{prefix}: TFV bootstrap 95% lower bound {lower} <= 0")
    min_peak = float(th[f"{prefix}_min_peak_mean_reduction_pct"])
    if float(row.get("peak_mean_reduction_pct", -np.inf)) < min_peak:
        reasons.append(
            f"{prefix}: peak mean reduction vs no_control {float(row.get('peak_mean_reduction_pct', -np.inf)):.3f}% < "
            f"{min_peak:.3f}%"
        )
    max_peak_worse = float(th[f"{prefix}_max_peak_worse_frac"])
    if float(row.get("peak_worse_frac_zero_tol", np.inf)) > max_peak_worse:
        reasons.append(
            f"{prefix}: peak worse fraction vs no_control {float(row.get('peak_worse_frac_zero_tol', np.inf)):.3f} > "
            f"{max_peak_worse:.3f}"
        )
    return reasons


def _stress_reasons(row: dict, th: dict) -> list[str]:
    reasons: list[str] = []
    pfv_mean = float(row.get("PFV_mean_reduction_pct", np.nan))
    pfv_increase = -pfv_mean if np.isfinite(pfv_mean) else np.inf
    if pfv_increase > float(th["stress_max_pfv_mean_increase_pct"]):
        reasons.append(
            f"stress: PFV mean increase vs no_control {pfv_increase:.3f}% > "
            f"{float(th['stress_max_pfv_mean_increase_pct']):.3f}%"
        )
    if float(row.get("PFV_worse_frac_noninferiority", np.inf)) > float(th["stress_max_pfv_worse_frac_noninferiority"]):
        reasons.append(
            f"stress: PFV noninferiority worse fraction {float(row.get('PFV_worse_frac_noninferiority', np.inf)):.3f} > "
            f"{float(th['stress_max_pfv_worse_frac_noninferiority']):.3f}"
        )
    if float(row.get("peak_mean_reduction_pct", -np.inf)) < float(th["stress_min_peak_mean_reduction_pct"]):
        reasons.append(
            f"stress: peak mean reduction vs no_control {float(row.get('peak_mean_reduction_pct', -np.inf)):.3f}% < "
            f"{float(th['stress_min_peak_mean_reduction_pct']):.3f}%"
        )
    if float(row.get("peak_worse_frac_zero_tol", np.inf)) > float(th["stress_max_peak_worse_frac"]):
        reasons.append(
            f"stress: peak worse fraction vs no_control {float(row.get('peak_worse_frac_zero_tol', np.inf)):.3f} > "
            f"{float(th['stress_max_peak_worse_frac']):.3f}"
        )
    return reasons


def evaluate_repair_gate(metrics: pd.DataFrame, cfg: dict) -> dict:
    th = _thresholds(cfg)
    work = metrics.copy()
    if work.empty:
        return {"passed": False, "reasons": ["empty metrics table"], "thresholds": th}
    if "policy_id" not in work or "event_id" not in work:
        return {"passed": False, "reasons": ["metrics table missing policy_id or event_id"], "thresholds": th}
    risk_filter = str(th.get("risk_filter", "high_risk_event") or "high_risk_event")
    if "event_risk_class" not in work and risk_filter != "all":
        return {
            "passed": False,
            "reasons": ["event_risk_class missing"],
            "thresholds": th,
            "high_risk_filter_applied": False,
        }
    work["policy_id"] = work["policy_id"].map(normalize_policy_id)
    high_risk_filter_applied = False
    if risk_filter != "all":
        high = work[work["event_risk_class"].astype(str).eq(risk_filter)].copy()
        if high.empty:
            return {
                "passed": False,
                "reasons": [f"no {risk_filter} rows available"],
                "thresholds": th,
                "high_risk_filter_applied": False,
            }
        work = high
        high_risk_filter_applied = True
    proposed_policy = _proposed_policy(work)
    reasons: list[str] = []
    n_events = int(work.loc[work["policy_id"].eq(proposed_policy), "event_id"].nunique())
    profile = str(th.get("gate_profile", "legacy_high_risk") or "legacy_high_risk").lower()
    if profile == "legacy_high_risk" and n_events < int(th["min_high_risk_events"]):
        reasons.append(f"paired high-risk proposed events {n_events} < {int(th['min_high_risk_events'])}")
    near_zero = float(th["near_zero_pfv_epsilon"])
    baselines = ["no_control"]
    baselines.extend([p for p in th["superiority_policies"] if p not in baselines])
    baselines.extend([p for p in th["diagnostic_compare_policies"] if p not in baselines])
    pfv_noninferiority_abs = float(th.get("no_control_pfv_noninferiority_abs", 100.0))
    pfv_noninferiority_frac = float(th.get("no_control_pfv_noninferiority_frac", 0.005))
    comparisons = [
        _paired_comparison(
            work,
            proposed_policy,
            baseline,
            near_zero,
            pfv_noninferiority_abs,
            pfv_noninferiority_frac,
        )
        for baseline in baselines
    ]
    comp_by_policy = {row["baseline_policy"]: row for row in comparisons}
    no = comp_by_policy.get("no_control", {"missing": True, "baseline_policy": "no_control"})
    if no.get("missing"):
        reasons.append("required no_control paired comparison missing")
    elif profile == "calibration":
        if int(no.get("paired_events", 0) or 0) < int(th["calibration_min_events"]):
            reasons.append(
                f"calibration paired proposed events {int(no.get('paired_events', 0) or 0)} < "
                f"{int(th['calibration_min_events'])}"
            )
        reasons.extend(_no_control_reasons(no, th, "calibration", require_tfv=True))
    elif profile == "formal":
        event_return_period = work["event_id"].map(_return_period)
        primary_periods = {str(x) for x in th.get("formal_primary_return_periods", [])}
        stress_periods = {str(x) for x in th.get("formal_stress_return_periods", [])}
        primary = work[event_return_period.isin(primary_periods)].copy()
        stress = work[event_return_period.isin(stress_periods)].copy()
        primary_no = _paired_comparison(
            primary,
            proposed_policy,
            "no_control",
            near_zero,
            pfv_noninferiority_abs,
            pfv_noninferiority_frac,
        )
        stress_no = _paired_comparison(
            stress,
            proposed_policy,
            "no_control",
            near_zero,
            pfv_noninferiority_abs,
            pfv_noninferiority_frac,
        )
        if primary_no.get("missing"):
            reasons.append("formal primary no_control paired comparison missing")
        elif int(primary_no.get("paired_events", 0) or 0) < int(th["formal_min_primary_events"]):
            reasons.append(
                f"formal primary paired events {int(primary_no.get('paired_events', 0) or 0)} < "
                f"{int(th['formal_min_primary_events'])}"
            )
        else:
            reasons.extend(_no_control_reasons(primary_no, th, "formal", require_tfv=True))
        if not stress.empty and not stress_no.get("missing"):
            reasons.extend(_stress_reasons(stress_no, th))
    else:
        no_pfv_mean = float(no.get("PFV_mean_reduction_pct", np.nan))
        no_pfv_increase = -no_pfv_mean if np.isfinite(no_pfv_mean) else np.inf
        if no_pfv_increase > float(th["no_control_max_pfv_mean_increase_pct"]):
            reasons.append(
                f"PFV mean increase vs no_control {no_pfv_increase:.3f}% > "
                f"{float(th['no_control_max_pfv_mean_increase_pct']):.3f}%"
            )
        pfv_worse_noninferiority = float(no.get("PFV_worse_frac_noninferiority", np.inf))
        if pfv_worse_noninferiority > float(th["no_control_max_pfv_worse_frac"]):
            reasons.append(
                f"PFV_worse_frac_noninferiority vs no_control {pfv_worse_noninferiority:.3f} > "
                f"{float(th['no_control_max_pfv_worse_frac']):.3f}"
            )
        if float(no.get("TFV_mean_reduction_pct", -np.inf)) < float(th["no_control_min_tfv_mean_reduction_pct"]):
            reasons.append(
                f"TFV mean reduction vs no_control {float(no.get('TFV_mean_reduction_pct', -np.inf)):.3f}% < "
                f"{float(th['no_control_min_tfv_mean_reduction_pct']):.3f}%"
            )
        if float(no.get("peak_worse_frac_zero_tol", np.inf)) > float(th["no_control_max_peak_worse_frac"]):
            reasons.append(
                f"peak_worse_frac vs no_control {float(no.get('peak_worse_frac_zero_tol', np.inf)):.3f} > "
                f"{float(th['no_control_max_peak_worse_frac']):.3f}"
            )
    primary_comparisons: list[dict] = []
    stress_comparisons: list[dict] = []
    if profile == "formal":
        event_return_period = work["event_id"].map(_return_period)
        primary_periods = {str(x) for x in th.get("formal_primary_return_periods", [])}
        stress_periods = {str(x) for x in th.get("formal_stress_return_periods", [])}
        primary_work = work[event_return_period.isin(primary_periods)].copy()
        stress_work = work[event_return_period.isin(stress_periods)].copy()
        primary_comparisons = [
            _paired_comparison(primary_work, proposed_policy, baseline, near_zero, pfv_noninferiority_abs, pfv_noninferiority_frac)
            for baseline in baselines
        ]
        stress_comparisons = [
            _paired_comparison(stress_work, proposed_policy, baseline, near_zero, pfv_noninferiority_abs, pfv_noninferiority_frac)
            for baseline in baselines
        ]
    if bool(th.get("enforce_superiority_policies", True)):
        for policy in th["superiority_policies"]:
            row = comp_by_policy.get(policy, {"missing": True, "baseline_policy": policy})
            if row.get("missing"):
                reasons.append(f"required superiority baseline missing: {policy}")
                continue
            if int(row.get("non_near_zero_pfv_events", 0) or 0) == 0:
                reasons.append(f"{proposed_policy} vs {policy} has no non-near-zero PFV events")
            if float(row.get("PFV_mean_reduction_pct", -np.inf)) < float(th["superiority_min_pfv_mean_reduction_pct"]):
                reasons.append(
                    f"{proposed_policy} PFV mean reduction vs {policy} "
                    f"{float(row.get('PFV_mean_reduction_pct', -np.inf)):.3f}% < "
                    f"{float(th['superiority_min_pfv_mean_reduction_pct']):.3f}%"
                )
            if float(row.get("TFV_mean_reduction_pct", -np.inf)) < float(th["superiority_min_tfv_mean_reduction_pct"]):
                reasons.append(
                    f"{proposed_policy} TFV mean reduction vs {policy} "
                    f"{float(row.get('TFV_mean_reduction_pct', -np.inf)):.3f}% < "
                    f"{float(th['superiority_min_tfv_mean_reduction_pct']):.3f}%"
                )
    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "objective": "no_control_pfv_preserving_system_risk_repair",
        "proposed_policy": proposed_policy,
        "high_risk_filter_applied": high_risk_filter_applied,
        "paired_high_risk_events": n_events,
        "gate_profile": profile,
        "thresholds": th,
        "baseline_comparisons": comparisons,
        "primary_baseline_comparisons": primary_comparisons,
        "stress_baseline_comparisons": stress_comparisons,
        "pfv_sensitivity_vs_no_control": _pfv_sensitivity(
            work,
            proposed_policy,
            "no_control",
            pfv_noninferiority_abs,
            [float(x) for x in th.get("pfv_sensitivity_fracs", [0.005, 0.01, 0.02])],
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan_no_control_repair.yaml")
    ap.add_argument("--event-policy-metrics", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--gate-profile", choices=["legacy_high_risk", "calibration", "formal"], default="")
    ap.add_argument("--fail-on-block", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.gate_profile:
        cfg.setdefault("evaluation", {}).setdefault("no_control_repair_gate", {})["gate_profile"] = args.gate_profile
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else Path(cfg["project_root"]) / "outputs" / "evaluation_no_control_repair")
    metrics = _read_csv(args.event_policy_metrics)
    report = evaluate_repair_gate(metrics, cfg)
    report["event_policy_metrics"] = str(args.event_policy_metrics)
    out_json = out_dir / "no_control_repair_gate.json"
    out_md = out_dir / "no_control_repair_gate.md"
    out_csv = out_dir / "no_control_repair_baseline_comparisons.csv"
    out_primary_csv = out_dir / "no_control_repair_primary_baseline_comparisons.csv"
    out_stress_csv = out_dir / "no_control_repair_stress_baseline_comparisons.csv"
    out_sensitivity_csv = out_dir / "no_control_repair_pfv_margin_sensitivity.csv"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(report.get("baseline_comparisons", [])).to_csv(out_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(report.get("primary_baseline_comparisons", [])).to_csv(out_primary_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(report.get("stress_baseline_comparisons", [])).to_csv(out_stress_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(report.get("pfv_sensitivity_vs_no_control", [])).to_csv(out_sensitivity_csv, index=False, encoding="utf-8-sig")
    lines = [
        "# No-Control PFV-Preserving System-Risk Repair Gate",
        "",
        f"- Passed: `{report['passed']}`",
        f"- Proposed policy: `{report.get('proposed_policy', '')}`",
        f"- High-risk paired events: `{report.get('paired_high_risk_events', 0)}`",
        f"- Gate profile: `{report.get('gate_profile', '')}`",
        f"- Objective: `{report.get('objective', '')}`",
        "",
        "## Baseline Comparisons",
        "",
        "| Baseline | Events | PFV mean % | TFV mean % | Peak mean % | PFV worse zero | PFV worse NI | Peak worse frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("baseline_comparisons", []):
        lines.append(
            "| {baseline} | {events} | {pfv:.3f} | {tfv:.3f} | {peak:.3f} | {pfv_worse_zero:.3f} | {pfv_worse_ni:.3f} | {peak_worse:.3f} |".format(
                baseline=row.get("baseline_policy", ""),
                events=int(row.get("paired_events", 0) or 0),
                pfv=float(row.get("PFV_mean_reduction_pct", np.nan)),
                tfv=float(row.get("TFV_mean_reduction_pct", np.nan)),
                peak=float(row.get("peak_mean_reduction_pct", np.nan)),
                pfv_worse_zero=float(row.get("PFV_worse_frac_zero_tol", np.nan)),
                pfv_worse_ni=float(row.get("PFV_worse_frac_noninferiority", np.nan)),
                peak_worse=float(row.get("peak_worse_frac_zero_tol", np.nan)),
            )
        )
    if report.get("reasons"):
        lines.extend(["", "## Blocking Reasons", ""])
        lines.extend([f"- {reason}" for reason in report["reasons"]])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.fail_on_block and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
