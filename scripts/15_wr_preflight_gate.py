from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config, resolve_gat_model_path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _best_training_row(
    df: pd.DataFrame,
    min_pfv_direction: float = 0.70,
    min_safe_precision: float = 0.80,
    min_peak_direction: float = 0.80,
) -> dict:
    if df.empty:
        return {}
    work = df.copy()
    for c in ["score", "PFV_direction_accuracy", "safe_precision", "peak_direction_accuracy"]:
        if c in work:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    gate_ok = (
        work.get("PFV_direction_accuracy", pd.Series(0, index=work.index)).fillna(0) >= float(min_pfv_direction)
    ) & (
        work.get("safe_precision", pd.Series(0, index=work.index)).fillna(0) >= float(min_safe_precision)
    ) & (
        work.get("peak_direction_accuracy", pd.Series(0, index=work.index)).fillna(0) >= float(min_peak_direction)
    )
    if gate_ok.any():
        subset = work.loc[gate_ok].copy()
        if "score" in subset and subset["score"].notna().any():
            row = subset.sort_values("score", ascending=True).iloc[0]
        else:
            score = (
                subset.get("PFV_direction_accuracy", pd.Series(0, index=subset.index)).fillna(0)
                + subset.get("safe_precision", pd.Series(0, index=subset.index)).fillna(0)
                + subset.get("peak_direction_accuracy", pd.Series(0, index=subset.index)).fillna(0)
            )
            row = subset.loc[score.idxmax()]
    elif "score" in work and work["score"].notna().any():
        row = work.sort_values("score", ascending=True).iloc[0]
    else:
        score = (
            work.get("PFV_direction_accuracy", pd.Series(0, index=work.index)).fillna(0)
            + work.get("safe_precision", pd.Series(0, index=work.index)).fillna(0)
            + work.get("peak_direction_accuracy", pd.Series(0, index=work.index)).fillna(0)
        )
        row = work.loc[score.idxmax()]
    return row.to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--mode", choices=["debug", "formal"], default="debug")
    ap.add_argument("--run-tag", default="", help="Optional run tag for output report placement.")
    ap.add_argument(
        "--controller-family",
        choices=["native_shield", "generic_clean"],
        default="native_shield",
        help="Preflight logic. native_shield requires residual action-value evidence; generic_clean requires GAT/surrogate models.",
    )
    ap.add_argument("--min-residual-rows", type=int, default=2000)
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--min-tier-rows", type=int, default=100)
    ap.add_argument("--min-pfv-improve-safe-frac", type=float, default=0.25)
    ap.add_argument("--min-pfv-direction", type=float, default=0.70)
    ap.add_argument("--min-safe-precision", type=float, default=0.80)
    ap.add_argument("--min-peak-direction", type=float, default=0.80)
    ap.add_argument("--require-training", action="store_true")
    ap.add_argument("--fail-on-block", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    diag = ensure_dir(cfg_path(cfg, "outputs.diagnostics"))
    out_dir = diag / args.mode
    if args.run_tag:
        out_dir = out_dir / args.run_tag
    out_dir = ensure_dir(out_dir)
    residual_dir = diag / "residual_counterfactuals"
    overall = _read_json(residual_dir / "residual_counterfactual_overall.json")
    tier = _read_csv(residual_dir / "residual_counterfactual_by_delta_tier.csv")
    train = _read_csv(diag / "residual_action_value_training_report.csv")
    group = _read_csv(diag / "residual_action_value_group_report.csv")
    best = _best_training_row(
        train,
        min_pfv_direction=args.min_pfv_direction,
        min_safe_precision=args.min_safe_precision,
        min_peak_direction=args.min_peak_direction,
    )

    reasons: list[str] = []
    model_dir = cfg_path(cfg, "outputs.models")
    gat_path = resolve_gat_model_path(cfg)
    surrogate_path = model_dir / "graph_surrogate_best.pt"
    if not surrogate_path.exists():
        surrogate_path = model_dir / "graph_surrogate.pt"
    if args.controller_family == "generic_clean":
        if not gat_path.exists():
            reasons.append(f"missing GAT model: {gat_path}")
        if not surrogate_path.exists():
            reasons.append(f"missing graph surrogate model: {surrogate_path}")
        report = {
            "mode": args.mode,
            "run_tag": args.run_tag,
            "controller_family": args.controller_family,
            "passed": bool(len(reasons) == 0),
            "reasons": reasons,
            "models": {
                "gat_path": str(gat_path),
                "gat_exists": gat_path.exists(),
                "surrogate_path": str(surrogate_path),
                "surrogate_exists": surrogate_path.exists(),
            },
            "recommended_next_action": "run generic clean closed-loop" if not reasons else "train/check GAT and graph surrogate first",
        }
        out_json = out_dir / "wr_preflight_gate.json"
        out_md = out_dir / "wr_preflight_gate.md"
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = [
            f"# WR Preflight Gate ({args.mode}, {args.controller_family})",
            f"- Passed: `{report['passed']}`",
            f"- GAT exists: `{gat_path.exists()}`",
            f"- Surrogate exists: `{surrogate_path.exists()}`",
        ]
        if reasons:
            lines.append("\n## Blocking Reasons")
            lines.extend([f"- {r}" for r in reasons])
        out_md.write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.fail_on_block and reasons:
            raise SystemExit(2)
        return

    residual_rows = int(float(overall.get("rows", 0) or 0))
    residual_events = int(float(overall.get("events", 0) or 0))
    improve_safe = float(overall.get("pfv_improve_safe_frac", 0.0) or 0.0)
    if residual_rows < int(args.min_residual_rows):
        reasons.append(f"residual rows {residual_rows} < {args.min_residual_rows}")
    if residual_events < int(args.min_events):
        reasons.append(f"residual events {residual_events} < {args.min_events}")
    if improve_safe < float(args.min_pfv_improve_safe_frac):
        reasons.append(f"PFV improve+safe fraction {improve_safe:.3f} < {args.min_pfv_improve_safe_frac:.3f}")

    tier_rows = {}
    if tier.empty:
        reasons.append("missing residual delta-tier audit table")
    else:
        for _, row in tier.iterrows():
            name = str(row.get("residual_delta_tier", "")).strip()
            if name:
                tier_rows[name] = int(float(row.get("n", 0) or 0))
        for name in ["small", "medium", "large"]:
            if tier_rows.get(name, 0) < int(args.min_tier_rows):
                reasons.append(f"{name} tier rows {tier_rows.get(name, 0)} < {args.min_tier_rows}")

    training_available = bool(best)
    if args.require_training and not training_available:
        reasons.append("missing residual action-value training report")
    if training_available:
        pfv_dir = float(best.get("PFV_direction_accuracy", 0.0) or 0.0)
        safe_precision = float(best.get("safe_precision", 0.0) or 0.0)
        peak_dir = float(best.get("peak_direction_accuracy", 0.0) or 0.0)
        if pfv_dir < float(args.min_pfv_direction):
            reasons.append(f"PFV direction accuracy {pfv_dir:.3f} < {args.min_pfv_direction:.3f}")
        if safe_precision < float(args.min_safe_precision):
            reasons.append(f"safe precision {safe_precision:.3f} < {args.min_safe_precision:.3f}")
        if peak_dir < float(args.min_peak_direction):
            reasons.append(f"peak direction accuracy {peak_dir:.3f} < {args.min_peak_direction:.3f}")

    passed = len(reasons) == 0
    report = {
        "mode": args.mode,
        "run_tag": args.run_tag,
        "controller_family": args.controller_family,
        "passed": bool(passed),
        "reasons": reasons,
        "residual": {
            "rows": residual_rows,
            "events": residual_events,
            "pfv_improve_safe_frac": improve_safe,
            "tier_rows": tier_rows,
        },
        "training_best": best,
        "group_report_rows": int(len(group)),
        "recommended_next_action": "run formal closed-loop" if passed else "fix listed data/model gaps before formal closed-loop",
    }

    out_json = out_dir / "wr_preflight_gate.json"
    out_md = out_dir / "wr_preflight_gate.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# WR Preflight Gate ({args.mode})",
        f"- Passed: `{passed}`",
        f"- Residual rows/events: `{residual_rows}` / `{residual_events}`",
        f"- PFV improve + safe fraction: `{improve_safe:.3f}`",
        f"- Tier rows: `{tier_rows}`",
    ]
    if training_available:
        lines.append(
            "- Best action-value model: "
            f"PFV_dir `{float(best.get('PFV_direction_accuracy', 0.0) or 0.0):.3f}`, "
            f"safe_precision `{float(best.get('safe_precision', 0.0) or 0.0):.3f}`, "
            f"peak_dir `{float(best.get('peak_direction_accuracy', 0.0) or 0.0):.3f}`."
        )
    if reasons:
        lines.append("\n## Blocking Reasons")
        lines.extend([f"- {r}" for r in reasons])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.fail_on_block and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
