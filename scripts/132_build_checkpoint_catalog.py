"""Build Project6 V3 checkpoint catalog.

Fail-fast skeleton. Requires real EventCatalog, Internal trajectories, state
contract, and cloneable hydraulic/controller memory before enabling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.data.checkpoint_catalog_contract import CHECKPOINT_CATALOG_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--event-catalog", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    report = {
        "status": "disabled",
        "failure_reason": "checkpoint_catalog_state_clone_inputs_not_implemented",
        "required_fields": list(CHECKPOINT_CATALOG_FIELDS),
        "expected_outputs": [
            str(out_dir / "checkpoint_catalog.csv"),
            str(out_dir / "checkpoint_state_hash_audit.csv"),
            str(out_dir / "checkpoint_near_duplicate_audit.csv"),
            str(out_dir / "checkpoint_split_audit.csv"),
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
