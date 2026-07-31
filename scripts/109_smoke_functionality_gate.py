from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.evaluation.smoke_functionality_gate import evaluate_smoke_functionality
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Project6 smoke runs for real temporal and simultaneous SWMM actions.")
    parser.add_argument("--config", default="configs/wuhan_project6_36_hierarchical_residual_v7.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--fail-on-block", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else run_dir / "smoke_functionality_gate")
    control_table_path = run_dir / "control_actuator_table.csv"
    if not control_table_path.exists():
        control_table_path = cfg_path(cfg, "outputs.audit") / "actuator_table.csv"
    control_table = pd.read_csv(control_table_path)
    smoke_cfg = ((cfg.get("evaluation", {}) or {}).get("smoke_gate", {}) or {})
    candidate_cfg = (((cfg.get("controller", {}) or {}).get("temporal_joint", {}) or {}).get("candidate_search", {}) or {})
    report = evaluate_smoke_functionality(
        run_dir=run_dir,
        control_table=control_table,
        required_return_period_groups=dict(smoke_cfg.get("required_return_period_groups", {
            "light_or_t10": ["T5", "T10"],
            "medium": ["T20"],
            "severe": ["T100"],
        })),
        binary_pump_ids=list(candidate_cfg.get("binary_pump_ids", ["ADD301.2", "ADD301.3"])),
        require_action_written=bool(smoke_cfg.get("require_action_written", True)),
        require_temporal_action=bool(smoke_cfg.get("require_temporal_action", True)),
        require_simultaneous_action=bool(smoke_cfg.get("require_simultaneous_action", True)),
        forbid_fractional_binary_pumps=bool(smoke_cfg.get("forbid_fractional_binary_pumps", True)),
        forbid_storage_interlock_violations=bool(smoke_cfg.get("forbid_storage_interlock_violations", True)),
    )
    out_json = out_dir / "smoke_functionality_gate.json"
    out_csv = out_dir / "smoke_event_action_summary.csv"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(report.get("events", [])).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.fail_on_block and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
