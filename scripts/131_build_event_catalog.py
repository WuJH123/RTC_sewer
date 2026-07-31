"""Build Project6 V3 event catalog.

This script is a fail-fast skeleton. It defines the required CLI and schema,
but it must not be enabled until Project2-Project6 rainfall provenance scanning
is implemented.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.data.event_catalog_contract import EVENT_CATALOG_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    report = {
        "status": "disabled",
        "failure_reason": "event_catalog_provenance_scan_not_implemented",
        "required_fields": list(EVENT_CATALOG_FIELDS),
        "expected_outputs": [
            str(out_dir / "event_catalog.csv"),
            str(out_dir / "event_provenance_audit.json"),
            str(out_dir / "event_near_duplicate_groups.csv"),
            str(out_dir / "split_leakage_audit.csv"),
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
