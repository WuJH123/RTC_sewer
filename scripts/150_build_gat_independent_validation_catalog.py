#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.gat_independent_validation import build_gat_independent_validation_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build independent sr0p15 GAT validation catalog without running GAT.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gat-dir", default=r"E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3\gat")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.config).exists():
        print(json.dumps({"status": "contract_mismatch", "reason": "config not found", "config": args.config}, indent=2))
        return 6
    report = build_gat_independent_validation_catalog(Path(args.gat_dir))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

