from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.io.project_paths import ensure_dir, load_config


def _num(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _label_family(label: object) -> str:
    text = str(label or "")
    if text.startswith("engineered_"):
        text = text[len("engineered_"):]
    if text.startswith("tier2_binary_pump_"):
        return "tier2_binary_pump"
    if text.startswith("tier2_") and "RTC_IN" in text:
        return "tier2_storage_inlet"
    if text.startswith("tier2_") and "RTC_OUT" in text:
        return "tier2_storage_outlet"
    if text.startswith("tier1_ramp_dw3700.1"):
        return "tier1_dw3700"
    if "MH0200773_8" in text:
        return "v8_mh0200773_group"
    if "MSLBZW001_8" in text:
        return "v8_mslbzw001_group"
    if "HS2529198_8" in text:
        return "v8_hs2529198_group"
    return text.split("|", 1)[0].split("_0.", 1)[0]


def _return_rank(rain_id: object) -> int:
    try:
        return int(str(rain_id).replace("T", ""))
    except Exception:
        return 0


def _aggregate(group: pd.DataFrame, *, pfv_abs: float, pfv_rel: float) -> dict[str, object]:
    ref_pfv = _num(group["reference_PFV_H"])
    prop_pfv = _num(group["proposed_PFV_H"])
    tfv_delta = _num(group["true_TFV_delta"])
    peak_delta = _num(group["true_peak_delta"])
    margin = np.maximum(float(pfv_abs), float(pfv_rel) * np.maximum(0.0, ref_pfv.to_numpy(float)))
    pfv_noninferior = (prop_pfv.to_numpy(float) - ref_pfv.to_numpy(float)) <= margin
    tfv_improved = tfv_delta.to_numpy(float) < 0.0
    peak_safe = peak_delta.to_numpy(float) <= 0.0
    return {
        "n": int(len(group)),
        "n_events": int(group["event_id"].nunique()) if "event_id" in group else 0,
        "PFV_noninferior_frac": float(np.mean(pfv_noninferior)) if len(group) else np.nan,
        "TFV_improved_frac": float(np.mean(tfv_improved)) if len(group) else np.nan,
        "peak_safe_frac": float(np.mean(peak_safe)) if len(group) else np.nan,
        "mean_true_TFV_delta": float(tfv_delta.mean()) if len(group) else np.nan,
        "median_true_TFV_delta": float(tfv_delta.median()) if len(group) else np.nan,
        "mean_true_peak_delta": float(peak_delta.mean()) if len(group) else np.nan,
        "median_true_peak_delta": float(peak_delta.median()) if len(group) else np.nan,
        "mean_PFV_delta": float((prop_pfv - ref_pfv).mean()) if len(group) else np.nan,
    }


def _decision(row: pd.Series, args: argparse.Namespace) -> tuple[bool, str]:
    n = int(row.get("n", 0))
    n_events = int(row.get("n_events", 0))
    rain_rank = _return_rank(row.get("rain_id", ""))
    pattern = str(row.get("pattern", ""))
    family = str(row.get("label_family", ""))
    label = str(row.get("selected_sequence_label", ""))
    pfv_ok = float(row.get("PFV_noninferior_frac", 0.0)) >= float(args.min_pfv_noninferior_frac)
    tfv_ok = float(row.get("TFV_improved_frac", 0.0)) >= float(args.min_tfv_improved_frac)
    peak_ok = float(row.get("peak_safe_frac", 0.0)) >= float(args.min_peak_safe_frac)
    mean_tfv_ok = float(row.get("mean_true_TFV_delta", np.inf)) <= 0.0
    mean_peak_ok = float(row.get("mean_true_peak_delta", np.inf)) <= 0.0

    if n < int(args.min_count) or n_events < int(args.min_events):
        return False, "insufficient_support"
    if not pfv_ok:
        return False, "pfv_noninferiority_unreliable"
    if rain_rank in {5, 10} and ("early_then_restore" in label or family in {"tier1_dw3700", "v8_mh0200773_group"}):
        return False, "low_return_period_strong_template_blocked"
    if pattern in {"block", "double_peak"} and ("early_then_restore" in label or family in {"tier1_dw3700", "v8_mh0200773_group"}):
        if not (tfv_ok and peak_ok and mean_tfv_ok and mean_peak_ok):
            return False, "block_double_peak_strong_template_blocked"
    if family == "tier1_dw3700":
        if not (tfv_ok and peak_ok and mean_tfv_ok and mean_peak_ok):
            return False, "dw3700_requires_tfv_and_peak_safe"
    if family == "tier2_binary_pump":
        if not (tfv_ok and peak_ok and mean_tfv_ok and mean_peak_ok):
            return False, "binary_pump_requires_tfv_and_peak_safe"
    if family == "v8_mh0200773_group":
        if not (tfv_ok and peak_ok):
            return False, "mh0200773_requires_stratum_tfv_peak_reliability"
    if not peak_ok or not mean_peak_ok:
        return False, "peak_risk"
    if not tfv_ok or not mean_tfv_ok:
        return False, "tfv_not_reliable"
    return True, "allowed"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build stratum-specific template reliability rules from realized SWMM audit.")
    ap.add_argument("--config", default="configs/wuhan_project6_36_recovered_v8_groups_v1.yaml")
    ap.add_argument("--audit-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pfv-abs-margin-m3", type=float, default=-1.0)
    ap.add_argument("--pfv-rel-margin", type=float, default=-1.0)
    ap.add_argument("--min-count", type=int, default=8)
    ap.add_argument("--min-events", type=int, default=2)
    ap.add_argument("--min-pfv-noninferior-frac", type=float, default=0.90)
    ap.add_argument("--min-tfv-improved-frac", type=float, default=0.55)
    ap.add_argument("--min-peak-safe-frac", type=float, default=0.60)
    args = ap.parse_args()

    cfg = load_config(args.config)
    safety = (((cfg.get("controller", {}) or {}).get("temporal_joint", {}) or {}).get("safety", {}) or {})
    pfv_abs = float(args.pfv_abs_margin_m3 if args.pfv_abs_margin_m3 >= 0 else safety.get("pfv_abs_margin_m3", 100.0))
    pfv_rel = float(args.pfv_rel_margin if args.pfv_rel_margin >= 0 else safety.get("pfv_rel_margin", 0.02))

    audit = pd.read_csv(args.audit_csv)
    if audit.empty:
        raise ValueError(f"empty audit csv: {args.audit_csv}")
    required = {
        "selected_sequence_label", "rain_id", "pattern", "duration_min", "phase",
        "reference_PFV_H", "proposed_PFV_H", "true_TFV_delta", "true_peak_delta",
    }
    missing = required - set(audit.columns)
    if missing:
        raise KeyError(f"audit csv missing required columns: {sorted(missing)}")

    work = audit.copy()
    work = work[work["selected_sequence_label"].astype(str).str.len() > 0].copy()
    work["label_family"] = work["selected_sequence_label"].map(_label_family)
    out_dir = ensure_dir(Path(args.out_dir))

    group_cols = ["selected_sequence_label", "label_family", "rain_id", "pattern", "duration_min", "phase"]
    rows = []
    for keys, group in work.groupby(group_cols, dropna=False):
        record = dict(zip(group_cols, keys))
        record.update(_aggregate(group, pfv_abs=pfv_abs, pfv_rel=pfv_rel))
        allowed, reason = _decision(pd.Series(record), args)
        record["allowed"] = bool(allowed)
        record["reason"] = reason
        rows.append(record)
    summary = pd.DataFrame(rows).sort_values(
        ["allowed", "rain_id", "pattern", "duration_min", "phase", "selected_sequence_label"],
        ascending=[False, True, True, True, True, True],
    )

    family_rows = []
    family_cols = ["label_family", "rain_id", "pattern", "phase"]
    for keys, group in work.groupby(family_cols, dropna=False):
        record = dict(zip(family_cols, keys))
        record.update(_aggregate(group, pfv_abs=pfv_abs, pfv_rel=pfv_rel))
        allowed, reason = _decision(pd.Series(record), args)
        record["allowed"] = bool(allowed)
        record["reason"] = reason
        family_rows.append(record)
    family = pd.DataFrame(family_rows).sort_values(
        ["allowed", "rain_id", "pattern", "phase", "label_family"],
        ascending=[False, True, True, True, True],
    )

    coarse_rows = []
    coarse_specs = [
        ("family_rain_phase", ["label_family", "rain_id", "phase"]),
        ("family_rain", ["label_family", "rain_id"]),
        ("family_pattern_phase", ["label_family", "pattern", "phase"]),
    ]
    for match_level, cols in coarse_specs:
        for keys, group in work.groupby(cols, dropna=False):
            record = dict(zip(cols, keys))
            record.setdefault("rain_id", "")
            record.setdefault("pattern", "")
            record.setdefault("duration_min", "")
            record.setdefault("phase", "")
            record["selected_sequence_label"] = ""
            record["match_level"] = match_level
            record.update(_aggregate(group, pfv_abs=pfv_abs, pfv_rel=pfv_rel))
            allowed, reason = _decision(pd.Series(record), args)
            record["allowed"] = bool(allowed)
            record["reason"] = reason
            coarse_rows.append(record)
    coarse = pd.DataFrame(coarse_rows)

    summary_path = out_dir / "template_reliability_by_label_stratum.csv"
    family_path = out_dir / "template_reliability_by_family_stratum.csv"
    coarse_path = out_dir / "template_reliability_by_coarse_stratum.csv"
    rules_path = out_dir / "template_reliability_rules.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    family.to_csv(family_path, index=False, encoding="utf-8-sig")
    coarse.to_csv(coarse_path, index=False, encoding="utf-8-sig")
    # Controller rules use exact labels first and family rules as fallback.
    # Exact strata with insufficient support are diagnostic only; deploying
    # them as hard blocks would suppress every candidate in sparse cells.
    exact_rules = summary[summary["reason"].ne("insufficient_support")].copy()
    exact_rules["match_level"] = "label"
    family_rules = family[family["reason"].ne("insufficient_support")].copy()
    family_rules["selected_sequence_label"] = ""
    family_rules["duration_min"] = ""
    family_rules["match_level"] = "family"
    coarse_rules = coarse[coarse["reason"].ne("insufficient_support")].copy()
    coarse_rules["selected_sequence_label"] = ""
    if "duration_min" not in coarse_rules:
        coarse_rules["duration_min"] = ""
    coarse_rules["duration_min"] = ""
    common_cols = [
        "match_level", "selected_sequence_label", "label_family", "rain_id", "pattern",
        "duration_min", "phase", "allowed", "reason", "n", "n_events",
        "PFV_noninferior_frac", "TFV_improved_frac", "peak_safe_frac",
        "mean_true_TFV_delta", "mean_true_peak_delta",
    ]
    rules = pd.concat([exact_rules[common_cols], family_rules[common_cols], coarse_rules[common_cols]], ignore_index=True, sort=False)
    rules.to_csv(rules_path, index=False, encoding="utf-8-sig")
    report = {
        "audit_csv": str(args.audit_csv),
        "rows": int(len(work)),
        "unique_labels": int(work["selected_sequence_label"].nunique()),
        "pfv_abs_margin_m3": pfv_abs,
        "pfv_rel_margin": pfv_rel,
        "summary_csv": str(summary_path),
        "family_csv": str(family_path),
        "coarse_csv": str(coarse_path),
        "rules_csv": str(rules_path),
        "allowed_exact_rules": int(summary["allowed"].sum()) if not summary.empty else 0,
        "blocked_exact_rules": int((~summary["allowed"]).sum()) if not summary.empty else 0,
        "allowed_family_rules": int(family["allowed"].sum()) if not family.empty else 0,
        "blocked_family_rules": int((~family["allowed"]).sum()) if not family.empty else 0,
        "deployable_rules": int(len(rules)),
        "deployable_allowed_rules": int(rules["allowed"].sum()) if not rules.empty else 0,
        "deployable_blocked_rules": int((~rules["allowed"]).sum()) if not rules.empty else 0,
        "top_block_reasons": summary["reason"].value_counts().head(10).to_dict() if "reason" in summary else {},
    }
    (out_dir / "template_reliability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
