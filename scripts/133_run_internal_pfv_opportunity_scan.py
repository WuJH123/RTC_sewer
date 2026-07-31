"""Run Project6 V3 Internal PFV opportunity scan.

Fail-fast skeleton. The real implementation must use fit/design events only
and must keep online predicted PFV-active separate from formal realized PFV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.data.opportunity_scan_contract import OPPORTUNITY_SCAN_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-catalog", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    report = {
        "status": "disabled",
        "failure_reason": "internal_pfv_opportunity_scan_not_implemented",
        "formal_event_usage": "forbidden",
        "threshold_status": "uncalibrated",
        "required_fields": list(OPPORTUNITY_SCAN_FIELDS),
        "expected_outputs": [
            str(out_dir / "internal_pfv_opportunity_scan.csv"),
            str(out_dir / "pfv_opportunity_distribution.json"),
            str(out_dir / "threshold_calibration_inputs.csv"),
            str(out_dir / "round0_checkpoint_eligibility.csv"),
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
