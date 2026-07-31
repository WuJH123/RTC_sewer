#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.prompt2_gat_readiness import evaluate_prompt2_gat_readiness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Prompt 2 GAT readiness for entering Prompt 3A.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    gate = evaluate_prompt2_gat_readiness(out_root)
    print(json.dumps({"status": gate["status"], "allowed_to_enter_prompt3a": gate["allowed_to_enter_prompt3a"]}, indent=2))
    return 0 if gate["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
