from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.evaluation.project5_formal_gate import Project5GateThresholds, evaluate_project5_gate
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _thresholds_from_config(cfg: dict) -> Project5GateThresholds:
    gate_cfg = ((cfg.get("evaluation", {}) or {}).get("proposed_formal_gate", {}) or {})
    allowed = Project5GateThresholds.__dataclass_fields__.keys()
    values = {}
    for key in allowed:
        if key in gate_cfg:
            values[key] = gate_cfg[key]
    return Project5GateThresholds(**values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--paired-metrics", default="")
    ap.add_argument("--residual-report", default="")
    ap.add_argument("--empirical-guard", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--fail-on-block", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    eval_dir = root / "outputs" / "evaluation_project5_priority_zone"
    diag_dir = cfg_path(cfg, "outputs.diagnostics")
    paired_path = Path(args.paired_metrics) if args.paired_metrics else eval_dir / "project5_priority_paired_metrics_main.csv"
    if args.paired_metrics and not paired_path.exists():
        raise FileNotFoundError(f"Missing paired metrics: {paired_path}")
    if not args.paired_metrics and not paired_path.exists():
        paired_path = eval_dir / "project5_priority_paired_metrics.csv"
    residual_path = Path(args.residual_report) if args.residual_report else diag_dir / "residual_action_value_training_report.csv"
    guard_path = Path(args.empirical_guard) if args.empirical_guard else diag_dir / "action_template_outcomes" / "action_template_empirical_guard_table.csv"
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else eval_dir)

    report = evaluate_project5_gate(
        _read_csv(paired_path),
        _read_csv(residual_path),
        _read_csv(guard_path),
        thresholds=_thresholds_from_config(cfg),
    )
    report.update(
        {
            "paired_metrics": str(paired_path),
            "residual_report": str(residual_path),
            "empirical_guard": str(guard_path),
        }
    )
    out_json = out_dir / "project5_formal_gate.json"
    out_md = out_dir / "project5_formal_gate.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Project5 Formal Gate",
        "",
        f"- Passed: `{report['passed']}`",
        f"- Proposed policy: `{report.get('proposed_policy', 'unknown')}`",
        f"- High-risk paired events: `{report['n_high_risk_paired_events']}`",
        f"- PFV mean/median reduction: `{report['PFV_mean_reduction_pct']:.3f}` / `{report['PFV_median_reduction_pct']:.3f}` %",
        f"- TFV/peak worse fraction: `{report['TFV_worse_frac']:.3f}` / `{report['peak_worse_frac']:.3f}`",
        f"- Action-change ratio: `{report['action_change_ratio_vs_internal']}`",
        f"- Residual model PFV_dir/safe_precision/peak_dir: `{report['residual_PFV_direction_accuracy']:.3f}` / `{report['residual_safe_precision']:.3f}` / `{report['residual_peak_direction_accuracy']:.3f}`",
        f"- Allowed empirical guard event coverage: `{report['allowed_guard_event_coverage']}`",
    ]
    if report.get("baseline_comparisons"):
        lines.extend(["", "## Baseline Comparisons", ""])
        lines.append("| Baseline | Events | PFV pct events | Near-zero ref | PFV mean % | PFV median % | TFV worse frac | Peak worse frac |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in report["baseline_comparisons"]:
            lines.append(
                "| {baseline} | {events} | {pfv_events} | {near_zero} | {pfv_mean:.3f} | {pfv_median:.3f} | {tfv:.3f} | {peak:.3f} |".format(
                    baseline=row.get("baseline_policy", ""),
                    events=int(row.get("paired_events", 0) or 0),
                    pfv_events=int(row.get("PFV_percent_stat_events", 0) or 0),
                    near_zero=int(row.get("near_zero_reference_events", 0) or 0),
                    pfv_mean=float(row.get("PFV_mean_reduction_pct", 0.0) or 0.0),
                    pfv_median=float(row.get("PFV_median_reduction_pct", 0.0) or 0.0),
                    tfv=float(row.get("TFV_worse_frac", 0.0) or 0.0),
                    peak=float(row.get("peak_worse_frac", 0.0) or 0.0),
                )
            )
    if report["reasons"]:
        lines.extend(["", "## Blocking Reasons"])
        lines.extend([f"- {reason}" for reason in report["reasons"]])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.fail_on_block and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
