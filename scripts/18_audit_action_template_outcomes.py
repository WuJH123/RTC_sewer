from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.control.candidate_generator import parse_candidate_label
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _resolve_action_template_out_dir(diag_root: Path, mode: str, run_tag: str) -> Path:
    if str(run_tag).strip():
        return diag_root / mode / run_tag / "action_template_outcomes"
    return diag_root / "action_template_outcomes"


def _num(s: pd.Series | float | int, default: float = 0.0) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").fillna(default)
    return pd.Series([default])


def _tier_from_delta(delta: pd.Series) -> pd.Series:
    d = pd.to_numeric(delta, errors="coerce").abs().fillna(0.0)
    return pd.Series(
        np.where(d <= 0.080001, "small", np.where(d <= 0.160001, "medium", "large")),
        index=delta.index,
    )


def _normalise_residual_rows(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    if "template_name" not in out:
        out["template_name"] = out.get("template", "unknown")
    out["template_name"] = out["template_name"].fillna("unknown").astype(str)

    if "candidate_scope" not in out:
        out["candidate_scope"] = "all"
    out["candidate_scope"] = out["candidate_scope"].fillna("all").astype(str)

    if "residual_delta" not in out:
        out["residual_delta"] = np.nan
    out["residual_delta"] = pd.to_numeric(out["residual_delta"], errors="coerce")

    # On-policy rows may carry the complete candidate label in template_name
    # (e.g. "pump_boost|scope=...|d=..."). Parse those back to canonical
    # template/scope/delta/hold fields before grouping, otherwise the same
    # hydraulic action is fragmented into many one-off templates.
    label_series = pd.Series("", index=out.index, dtype=object)
    for c in ["selected_candidate_label", "candidate_label", "template_name"]:
        if c not in out:
            continue
        s = out[c].fillna("").astype(str)
        label_series = label_series.mask(~label_series.str.contains(r"\|scope=", regex=True, na=False), s)
    needs_parse = label_series.str.contains(r"\|scope=", regex=True, na=False)
    if needs_parse.any():
        parsed = label_series[needs_parse].map(parse_candidate_label)
        idx = out.index[needs_parse]
        out.loc[idx, "template_name"] = [str(x.get("template", "unknown")) for x in parsed]
        out.loc[idx, "candidate_scope"] = [str(x.get("scope", "all")) for x in parsed]
        parsed_delta = pd.Series([float(x.get("delta", 0.0) or 0.0) for x in parsed], index=idx)
        parsed_hold = pd.Series([int(x.get("hold_steps", 1) or 1) for x in parsed], index=idx)
        out.loc[idx[out.loc[idx, "residual_delta"].isna()], "residual_delta"] = parsed_delta.loc[
            out.loc[idx, "residual_delta"].isna()
        ]
        out.loc[idx, "_parsed_hold_steps"] = parsed_hold

    if "feat_delta_abs_max" in out:
        fb = pd.to_numeric(out["feat_delta_abs_max"], errors="coerce")
        out["residual_delta"] = out["residual_delta"].fillna(fb)
        out["residual_delta"] = out["residual_delta"].mask(out["residual_delta"].abs() <= 1e-9, fb)
    out["residual_delta"] = out["residual_delta"].fillna(0.0)

    if "residual_delta_tier" not in out:
        out["residual_delta_tier"] = _tier_from_delta(out["residual_delta"])
    else:
        tier = out["residual_delta_tier"].fillna("").astype(str).str.strip()
        out["residual_delta_tier"] = tier.mask(tier.eq(""), _tier_from_delta(out["residual_delta"]))

    if "override_steps" in out:
        out["hold_steps"] = pd.to_numeric(out["override_steps"], errors="coerce").fillna(1).clip(lower=1).round().astype(int)
    elif "feat_hold_steps" in out:
        out["hold_steps"] = pd.to_numeric(out["feat_hold_steps"], errors="coerce").fillna(1).clip(lower=1).round().astype(int)
    elif "_parsed_hold_steps" in out:
        out["hold_steps"] = pd.to_numeric(out["_parsed_hold_steps"], errors="coerce").fillna(1).clip(lower=1).round().astype(int)
    else:
        out["hold_steps"] = 1

    for c in ["delta_PFV", "delta_TFV", "delta_peak", "baseline_TFV", "baseline_peak_TFV_rate"]:
        out[c] = pd.to_numeric(out.get(c, pd.Series(np.nan, index=out.index)), errors="coerce")
    out = out.dropna(subset=["delta_PFV", "delta_TFV", "delta_peak"]).reset_index(drop=True)

    tfv_guard_default = float(cfg["experiment"].get("tfv_guard_pct", 0.005)) * out["baseline_TFV"].fillna(0.0)
    peak_guard_default = float(cfg["experiment"].get("peak_guard_pct", 0.010)) * out["baseline_peak_TFV_rate"].fillna(0.0)
    if "tfv_guard" in out:
        out["tfv_guard"] = pd.to_numeric(out["tfv_guard"], errors="coerce").fillna(tfv_guard_default)
    else:
        out["tfv_guard"] = tfv_guard_default
    if "peak_guard" in out:
        out["peak_guard"] = pd.to_numeric(out["peak_guard"], errors="coerce").fillna(peak_guard_default)
    else:
        out["peak_guard"] = peak_guard_default

    out["pfv_improve"] = out["delta_PFV"] < 0
    out["pfv_worse"] = out["delta_PFV"] > 0
    out["tfv_safe"] = out["delta_TFV"] <= out["tfv_guard"]
    out["peak_safe"] = out["delta_peak"] <= out["peak_guard"]
    out["safe_guarded"] = out["tfv_safe"] & out["peak_safe"]
    out["pfv_improve_safe"] = out["pfv_improve"] & out["safe_guarded"]
    return out


def _summarise_group(df: pd.DataFrame, keys: list[str], level: str, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for vals, g in df.groupby(keys, dropna=False):
        if not isinstance(vals, tuple):
            vals = (vals,)
        row = {k: v for k, v in zip(keys, vals)}
        n = int(len(g))
        events = int(g["event_id"].nunique()) if "event_id" in g else 0
        pfv_improve_safe_frac = float(g["pfv_improve_safe"].mean()) if n else 0.0
        pfv_worse_frac = float(g["pfv_worse"].mean()) if n else 1.0
        safe_guarded_frac = float(g["safe_guarded"].mean()) if n else 0.0
        peak_worse_frac = float((~g["peak_safe"]).mean()) if n else 1.0
        enough = n >= int(args.min_samples) and events >= int(args.min_events)
        allow = (
            enough
            and pfv_improve_safe_frac >= float(args.min_pfv_improve_safe_frac)
            and pfv_worse_frac <= float(args.max_pfv_worse_frac)
            and safe_guarded_frac >= float(args.min_safe_guarded_frac)
            and peak_worse_frac <= float(args.max_peak_worse_frac)
        )
        row.update(
            {
                "group_level": level,
                "n": n,
                "events": events,
                "pfv_improve_frac": float(g["pfv_improve"].mean()) if n else 0.0,
                "pfv_worse_frac": pfv_worse_frac,
                "safe_guarded_frac": safe_guarded_frac,
                "peak_worse_frac": peak_worse_frac,
                "pfv_improve_safe_frac": pfv_improve_safe_frac,
                "median_delta_PFV": float(g["delta_PFV"].median()) if n else 0.0,
                "mean_delta_PFV": float(g["delta_PFV"].mean()) if n else 0.0,
                "mean_delta_TFV": float(g["delta_TFV"].mean()) if n else 0.0,
                "mean_delta_peak": float(g["delta_peak"].mean()) if n else 0.0,
                "empirical_allow": bool(allow),
                "empirical_block_reason": ""
                if allow
                else ";".join(
                    [
                        r
                        for r, bad in [
                            ("few_samples", not enough),
                            ("low_pfv_improve_safe", pfv_improve_safe_frac < float(args.min_pfv_improve_safe_frac)),
                            ("high_pfv_worse", pfv_worse_frac > float(args.max_pfv_worse_frac)),
                            ("low_safe_guarded", safe_guarded_frac < float(args.min_safe_guarded_frac)),
                            ("high_peak_worse", peak_worse_frac > float(args.max_peak_worse_frac)),
                        ]
                        if bad
                    ]
                ),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in ["template_name", "candidate_scope", "residual_delta_tier", "hold_steps"]:
        if col not in out:
            out[col] = "*"
    return out


def _summarise_selected_actions(run_root: Path, out_dir: Path) -> None:
    if not run_root.exists():
        return
    hist_rows = []
    for path in sorted((run_root / "proposed").glob("*__controller_history.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df["history_file"] = str(path)
        hist_rows.append(df)
    if not hist_rows:
        return
    hist = pd.concat(hist_rows, ignore_index=True, sort=False)
    if "selected_candidate_label" not in hist:
        return
    selected = hist[hist["selected_candidate_label"].fillna("").astype(str).ne("")].copy()
    if selected.empty:
        return
    parsed = selected["selected_candidate_label"].map(parse_candidate_label)
    selected["template_name"] = [str(x.get("template", "unknown")) for x in parsed]
    selected["candidate_scope"] = [str(x.get("scope", "all")) for x in parsed]
    selected["residual_delta"] = [float(x.get("delta", 0.0) or 0.0) for x in parsed]
    selected["hold_steps"] = [int(x.get("hold_steps", 1) or 1) for x in parsed]
    selected["residual_delta_tier"] = _tier_from_delta(selected["residual_delta"])
    summary = (
        selected.groupby(["template_name", "candidate_scope", "residual_delta_tier", "hold_steps"], dropna=False)
        .agg(
            selected_count=("selected_candidate_label", "size"),
            events=("event_id", "nunique"),
            mean_pred_delta_pfv=("selected_pred_delta_pfv", lambda x: pd.to_numeric(x, errors="coerce").mean()),
            mean_pred_delta_tfv=("selected_pred_delta_tfv", lambda x: pd.to_numeric(x, errors="coerce").mean()),
            mean_pred_delta_peak=("selected_pred_delta_peak", lambda x: pd.to_numeric(x, errors="coerce").mean()),
            mean_safe_prob=("selected_safe_prob", lambda x: pd.to_numeric(x, errors="coerce").mean()),
            mean_peak_nonworse_prob=("selected_peak_nonworse_prob", lambda x: pd.to_numeric(x, errors="coerce").mean()),
        )
        .reset_index()
        .sort_values("selected_count", ascending=False)
    )
    summary.to_csv(out_dir / "selected_action_template_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--mode", choices=["debug", "formal"], default="debug")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--min-pfv-improve-safe-frac", type=float, default=0.25)
    ap.add_argument("--max-pfv-worse-frac", type=float, default=0.20)
    ap.add_argument("--min-safe-guarded-frac", type=float, default=0.45)
    ap.add_argument("--max-peak-worse-frac", type=float, default=0.35)
    args = ap.parse_args()

    cfg = load_config(args.config)
    residual_path = cfg_path(cfg, "outputs.closed_loop") / "internal_residual_counterfactuals" / "residual_counterfactual_results.csv"
    if not residual_path.exists():
        raise FileNotFoundError(f"Missing residual counterfactual results: {residual_path}")
    out_dir = ensure_dir(_resolve_action_template_out_dir(cfg_path(cfg, "outputs.diagnostics"), args.mode, args.run_tag))
    df = _normalise_residual_rows(pd.read_csv(residual_path), cfg)
    exact_keys = ["template_name", "candidate_scope", "residual_delta_tier", "hold_steps"]
    frames = [
        _summarise_group(df, exact_keys, "template_scope_tier_hold", args),
        _summarise_group(df, ["template_name", "candidate_scope", "residual_delta_tier"], "template_scope_tier", args),
        _summarise_group(df, ["template_name", "residual_delta_tier"], "template_tier", args),
        _summarise_group(df, ["template_name"], "template", args),
    ]
    guard = pd.concat(frames, ignore_index=True, sort=False)
    guard = guard.sort_values(
        ["empirical_allow", "pfv_improve_safe_frac", "pfv_worse_frac", "n"],
        ascending=[False, False, True, False],
    )
    guard_path = out_dir / "action_template_empirical_guard_table.csv"
    guard.to_csv(guard_path, index=False)

    by_phase = []
    if "phase" in df:
        phase_keys = ["template_name", "candidate_scope", "residual_delta_tier", "phase"]
        by_phase = [_summarise_group(df, phase_keys, "template_scope_tier_phase", args)]
        pd.concat(by_phase, ignore_index=True, sort=False).to_csv(out_dir / "action_template_by_phase.csv", index=False)

    run_root = cfg_path(cfg, "outputs.closed_loop") / args.mode
    if args.run_tag:
        run_root = run_root / args.run_tag
    _summarise_selected_actions(run_root, out_dir)

    report = {
        "residual_rows": int(len(df)),
        "templates": int(df["template_name"].nunique()),
        "guard_rows": int(len(guard)),
        "allowed_rows": int(guard["empirical_allow"].sum()),
        "guard_table": str(guard_path),
        "out_dir": str(out_dir),
        "mode": args.mode,
        "run_tag": args.run_tag,
        "criteria": {
            "min_samples": int(args.min_samples),
            "min_events": int(args.min_events),
            "min_pfv_improve_safe_frac": float(args.min_pfv_improve_safe_frac),
            "max_pfv_worse_frac": float(args.max_pfv_worse_frac),
            "min_safe_guarded_frac": float(args.min_safe_guarded_frac),
            "max_peak_worse_frac": float(args.max_peak_worse_frac),
        },
    }
    (out_dir / "action_template_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
