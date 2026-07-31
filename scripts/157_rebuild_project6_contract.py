#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.contracts.prompt3a import INP_PATH, OUT_ROOT, managed_facility_ids, sha256_file, utc_now, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild frozen Prompt 3A physical contract manifest.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    out = OUT_ROOT / "contracts" / "project6_prompt3a_contract_manifest.json"
    payload = {
        "status": "completed",
        "created_at": utc_now(),
        "single_network": str(INP_PATH),
        "network_sha256": sha256_file(INP_PATH),
        "managed_facility_count": len(managed_facility_ids()),
        "control_interval_min": 10,
        "prediction_horizon_min": 120,
        "prediction_horizon_steps": 12,
        "free_residual_steps": 3,
        "primary_gat": "sr0p15",
        "internal_is_pfv_benchmark": True,
        "selected_safe_fallback_is_online_benchmark": True,
        "no_control_role": "diagnostic_only",
        "round0_unlock_allowed": False,
    }
    write_json(out, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

