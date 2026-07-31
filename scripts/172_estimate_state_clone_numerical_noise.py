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
from sewerrtc.state.state_clone_equivalence import estimate_state_clone_numerical_noise


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate State Clone numerical noise floor.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "state_clone"))
    parser.add_argument("--max-checkpoints", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    code, outputs = estimate_state_clone_numerical_noise(
        args.out_dir,
        max_checkpoints=args.max_checkpoints,
        workers=args.workers,
    )
    print(json.dumps({"status": "completed" if code == 0 else "blocked", "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
