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
from sewerrtc.state.same_state_replay import evaluate_hotstart_clone_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate verified hot-start same-state gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "state_clone"))
    args = parser.parse_args()
    del args.config
    code, outputs = evaluate_hotstart_clone_gate(Path(args.out_dir))
    print(json.dumps({"status": "pass" if code == 0 else "failed_gate" if code == 5 else "blocked", "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

