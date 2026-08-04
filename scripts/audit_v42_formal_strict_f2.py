from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_formal_strict import audit_formal_strict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4",
    )
    args = ap.parse_args()
    payload = audit_formal_strict(args.output_root)
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if payload["strict_formal_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
