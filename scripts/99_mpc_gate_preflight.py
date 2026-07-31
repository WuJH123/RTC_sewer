from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sewerrtc.data.three_step_research_builders import evaluate_mpc_readiness
from sewerrtc.io.project_paths import cfg_path, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Block Smoke/Formal when the temporal joint effect model or MPC contract is not ready.")
    parser.add_argument("--config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--model-report", default="outputs/models_temporal_joint_36_v3/raw_joint_36_same_state_v3_train_report.json")
    parser.add_argument("--out-json", default="outputs/research_reuse_plan/mpc_gate_preflight.json")
    parser.add_argument("--enforce", action="store_true", help="Exit non-zero if the gate is not ready.")
    parser.add_argument("--require-tier2", action="store_true", help="Exit non-zero unless the hierarchical residual gate passes.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    report_path = root / args.model_report if not Path(args.model_report).is_absolute() else Path(args.model_report)
    out_json = root / args.out_json if not Path(args.out_json).is_absolute() else Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate_mpc_readiness(config=cfg, model_report_path=report_path)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.require_tier2 and not result.get("tier2_residual_allowed", False):
        raise SystemExit(3)
    if args.enforce and not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
