from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.hotstart_acceleration import benchmark_hotstart_acceleration


def _ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidate-counts", default="1,5,10,20")
    parser.add_argument("--worker-counts", default="1,2,4")
    args = parser.parse_args()
    del args.config
    code, outputs = benchmark_hotstart_acceleration(_ints(args.candidate_counts), _ints(args.worker_counts), Path(args.out_dir))
    print(json.dumps({"status": "completed", "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
