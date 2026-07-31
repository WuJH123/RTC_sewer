from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.interface_cache import audit_runoff_interface_equivalence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-events", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    del args.workers
    code, outputs = audit_runoff_interface_equivalence(args.config, args.max_events, Path(args.out_dir))
    status = "pass" if code == 0 else "blocked" if code == 3 else "failed_gate"
    print(json.dumps({"status": status, "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
