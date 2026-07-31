#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.contracts.prompt3a import static_physical_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Prompt 3A static physical fatal audit.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    code, report, _ = static_physical_audit(args.config)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

