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
from sewerrtc.state.state_clone_equivalence import evaluate_state_clone_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the State Clone gate from real equivalence outputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "state_clone"))
    args = parser.parse_args()
    code, outputs = evaluate_state_clone_gate(args.out_dir)
    status = "pass" if code == 0 else "failed_gate" if code == 5 else "blocked"
    print(json.dumps({"status": status, "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
