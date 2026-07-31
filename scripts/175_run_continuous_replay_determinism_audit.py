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
from sewerrtc.state.same_state_replay import run_continuous_replay_determinism_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic full continuous replay audit for Prompt3A same-state.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "state_clone"))
    parser.add_argument("--max-checkpoints", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    del args.config, args.workers
    code, outputs = run_continuous_replay_determinism_audit(Path(args.out_dir), max_checkpoints=args.max_checkpoints)
    status = "pass" if code == 0 else "failed_gate" if code == 5 else "blocked"
    print(json.dumps({"status": status, "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

