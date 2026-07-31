from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.interface_cache import candidate_prefilter_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    del args.config
    code, outputs = candidate_prefilter_audit(Path(args.out_dir))
    print(json.dumps({"status": "completed" if code == 0 else "blocked", "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
