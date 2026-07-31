#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.prompt2_completion_gate import write_prompt2_completion_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Prompt 2 completion gate without unlocking Round 0.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    path = write_prompt2_completion_gate(out_root)
    gate = json.loads(path.read_text(encoding="utf-8-sig"))
    print(json.dumps({"status": gate.get("status"), "gate": str(path)}, indent=2))
    return 0 if gate.get("status") == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
