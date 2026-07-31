#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.contracts.native_rules import audit_native_rules
from sewerrtc.contracts.prompt3a import INP_PATH, OUT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse and audit SWMM native controls.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "native_rules"))
    args = parser.parse_args()
    code, report, _ = audit_native_rules(INP_PATH, args.out_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

