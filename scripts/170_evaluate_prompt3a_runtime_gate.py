#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.status.current_truth import write_runtime_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Prompt3A runtime gate from real runtime evidence.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = write_runtime_gate(ROOT)
    gate = result["gate"]
    print(json.dumps({"status": gate["status"], "gate": str(result["path"]), "blocking_reasons": gate.get("blocking_reasons", [])}, indent=2, ensure_ascii=False))
    return 0 if gate["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())

