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


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate continuous replay determinism gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "state_clone"))
    args = parser.parse_args()
    report_path = Path(args.out_dir) / "continuous_replay_determinism_report.json"
    if not report_path.exists():
        print(json.dumps({"status": "blocked", "blocking_reasons": ["continuous_replay_report_missing"]}, indent=2))
        return 3
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    status = report.get("status")
    print(json.dumps({"status": status, "report": str(report_path)}, indent=2, ensure_ascii=False))
    if status == "pass":
        return 0
    if report.get("runtime_executed"):
        return 5
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

