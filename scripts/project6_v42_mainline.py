from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_mainline_workflow import audit_v42_mainline


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the canonical V4.2 scientific chain: R0 -> Step1 -> Step2 -> "
            "Step3 -> closed-loop/lock/blind."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "project6_dual_reference_v4"
        / "final_v4",
    )
    args = parser.parse_args()
    audit = audit_v42_mainline(args.output_root)
    print(json.dumps(audit.as_dict(), indent=2, allow_nan=False))
    return 0 if audit.complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
