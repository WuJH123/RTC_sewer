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
from sewerrtc.data.rainfall_asset_index import build_rainfall_asset_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fixed-directory rainfall asset index for Project6 Prompt3A.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "rainfall_assets"))
    args = parser.parse_args()
    code, report, _ = build_rainfall_asset_index(args.out_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
