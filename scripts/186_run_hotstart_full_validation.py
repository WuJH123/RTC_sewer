from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.hotstart_acceleration import run_hotstart_full_validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    del args.config, args.max_checkpoints, args.workers, args.resume
    code, outputs = run_hotstart_full_validation(Path(args.out_dir))
    print(json.dumps({"status": "pass" if code == 0 else "blocked" if code == 3 else "failed_gate", "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
