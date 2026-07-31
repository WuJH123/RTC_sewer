#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.state_input_manifest import build_state_input_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an explicit state input manifest for Project6 V3 state features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-mode", required=True, choices=["project4_gat_validation", "project4_diagnostic_contaminated", "gat_independent_holdout", "project6_retrofit_baseline"])
    parser.add_argument("--trajectory-root", default="")
    parser.add_argument("--validation-manifest", default="")
    parser.add_argument("--max-samples", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    trajectory_root = Path(args.trajectory_root) if args.trajectory_root else None
    if trajectory_root is not None and not trajectory_root.is_absolute():
        trajectory_root = ROOT / trajectory_root
    code, outputs = build_state_input_manifest(
        source_mode=args.source_mode,
        out_dir=out_dir,
        trajectory_root=trajectory_root,
        validation_manifest=Path(args.validation_manifest) if args.validation_manifest else None,
        max_samples=args.max_samples,
    )
    print(json.dumps({"status": "completed" if code == 0 else "blocked", "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
