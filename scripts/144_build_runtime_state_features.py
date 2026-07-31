#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.runtime_state_features import build_runtime_state_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real seven-frame runtime state features from an explicit manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gat-lock", required=True)
    parser.add_argument("--state-input-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--state-validation-mode",
        default="full_project6_augmented_state",
        choices=["full_project6_augmented_state", "project6_full_baseline", "project4_node_only", "gat_independent_node_only"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Path(args.config)
    lock = Path(args.gat_lock)
    manifest = Path(args.state_input_manifest)
    out_dir = Path(args.out_dir)
    if not config.is_absolute():
        config = ROOT / config
    if not lock.is_absolute():
        lock = ROOT / lock
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    code, outputs = build_runtime_state_features(
        config_path=config,
        lock_path=lock,
        state_input_manifest=manifest,
        out_dir=out_dir,
        max_samples=args.max_samples,
        state_validation_mode=args.state_validation_mode,
    )
    status = "completed" if code == 0 else "blocked"
    print(json.dumps({"status": status, "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
