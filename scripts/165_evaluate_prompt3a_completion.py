#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.status.current_truth import write_prompt3a_completion


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Prompt3A completion as engineering gate AND runtime gate.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = write_prompt3a_completion(ROOT)
    gate = result["gate"]
    print(json.dumps({"status": gate["status"], "gate": str(result["path"]), "runtime_blocking_reasons": gate["runtime_blocking_reasons"]}, indent=2, ensure_ascii=False))
    return 0 if gate["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())

