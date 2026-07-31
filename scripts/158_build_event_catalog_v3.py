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
from sewerrtc.data.event_catalog import build_event_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Prompt 3A event catalog and split manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "event_catalog"))
    args = parser.parse_args()
    code, report, _ = build_event_catalog(args.out_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

