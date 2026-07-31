from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.evaluation.risk_stratified import RiskThresholds, build_event_table
from sewerrtc.io.priority_config import combined_priority_depth_nodes, configured_priority_nodes, priority_config_summary
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _run_root(cfg: dict, mode: str, run_tag: str) -> Path:
    root = cfg_path(cfg, "outputs.closed_loop") / mode
    if run_tag:
        root = root / run_tag
    return root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--mode", choices=["debug", "formal"], default="formal")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--baseline-policy", default="internal_rules")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = _run_root(cfg, args.mode, args.run_tag)
    baseline_path = root / "baseline_results.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline results: {baseline_path}")
    baseline = pd.read_csv(baseline_path)
    priority_nodes = configured_priority_nodes(cfg)
    depth_nodes = combined_priority_depth_nodes(cfg)
    thresholds = RiskThresholds.from_config(cfg)
    table = build_event_table(
        baseline,
        priority_nodes,
        thresholds,
        baseline_policy=args.baseline_policy,
        depth_nodes=depth_nodes,
    )

    out_dir = Path(args.out_dir) if args.out_dir else cfg_path(cfg, "project_root") / "outputs" / "evaluation"
    out_dir = ensure_dir(out_dir)
    out_csv = out_dir / "risk_stratified_event_table.csv"
    table.to_csv(out_csv, index=False)
    summary = {
        "mode": args.mode,
        "run_tag": args.run_tag,
        "baseline_policy": args.baseline_policy,
        "events": int(len(table)),
        "risk_class_counts": {str(k): int(v) for k, v in table["event_risk_class"].value_counts().to_dict().items()},
        "thresholds": thresholds.__dict__,
        "priority": priority_config_summary(cfg),
        "output": str(out_csv),
    }
    (out_dir / "risk_stratified_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
