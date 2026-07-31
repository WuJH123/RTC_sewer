#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.gat_holdout_trajectory import generate_and_build_holdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Project6 independent GAT holdout trajectories and sr0p15 validation cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gat-dir", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--policies", default="no_control")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tail-min", type=int, default=180)
    parser.add_argument("--control-step-sec", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gat_dir = Path(args.gat_dir)
    plan = Path(args.plan)
    if not gat_dir.is_absolute():
        gat_dir = ROOT / gat_dir
    if not plan.is_absolute():
        plan = ROOT / plan
    policies = [value.strip() for value in str(args.policies).split(",") if value.strip()]
    code, outputs = generate_and_build_holdout(
        plan_path=plan,
        gat_dir=gat_dir,
        max_events=args.max_events,
        policies=policies,
        workers=args.workers,
        tail_min=args.tail_min,
        control_step_sec=args.control_step_sec,
        max_steps=args.max_steps,
        time_stride=args.time_stride,
        resume=args.resume,
    )
    print(json.dumps({"status": "completed" if code == 0 else "blocked", "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
