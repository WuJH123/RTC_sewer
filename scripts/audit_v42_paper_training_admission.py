from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_paper_training_admission import (
    audit_training_admission,
    write_training_admission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed admission gate for formal V4.2 paper surrogate training."
    )
    parser.add_argument("--oracle-summary", type=Path, required=True)
    parser.add_argument("--hydraulic-target-audit", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    admission = audit_training_admission(
        independent_oracle_summary=args.oracle_summary,
        hydraulic_target_audit=args.hydraulic_target_audit,
        expected_sample_count=args.expected_count,
    )
    write_training_admission(output_path=args.output, admission=admission)
    print(json.dumps(admission.as_dict(), indent=2, allow_nan=False))
    return 0 if admission.authorized else 5


if __name__ == "__main__":
    raise SystemExit(main())
