#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.contracts.prompt3a import OUT_ROOT
from sewerrtc.data.round0_planner import plan_round0


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan Prompt 3A Round 0 manifest without executing SWMM cases.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--event-catalog", default=str(OUT_ROOT / "event_catalog" / "event_catalog.csv"))
    parser.add_argument("--checkpoint-catalog", default=str(OUT_ROOT / "checkpoint_catalog" / "checkpoint_catalog.csv"))
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "round0"))
    args = parser.parse_args()
    code, report, _ = plan_round0(args.event_catalog, args.checkpoint_catalog, args.out_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

