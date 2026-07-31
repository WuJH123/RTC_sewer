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
from sewerrtc.state.state_clone_equivalence import run_state_clone_equivalence


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or prepare State Clone equivalence comparison from hot-start checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "state_clone"))
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    code, outputs = run_state_clone_equivalence(
        args.out_dir,
        mode=args.mode,
        max_checkpoints=args.max_checkpoints,
        workers=args.workers,
        resume=args.resume,
    )
    status = "pass" if code == 0 else "failed_gate" if code == 5 else "blocked"
    print(json.dumps({"status": status, "mode": args.mode, "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
