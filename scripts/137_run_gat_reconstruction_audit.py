#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.gat_compatibility import run_reconstruction_audit_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal Project4-cache GAT reconstruction diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        print(json.dumps({"status": "failed", "reason": "config_not_found", "config": str(config_path)}, indent=2))
        return 6
    outputs = run_reconstruction_audit_outputs(Path(args.out_dir))
    if not outputs:
        print(json.dumps({"status": "blocked", "reason": "no_reconstruction_outputs"}, indent=2))
        return 3
    print(json.dumps({"status": "completed", "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
