#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.gat_independent_validation import lock_gat_independent_validation_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock independent sr0p15 GAT validation manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gat-dir", default=r"E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3\gat")
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--acknowledge-independent-holdout", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.config).exists():
        print(json.dumps({"status": "contract_mismatch", "reason": "config not found", "config": args.config}, indent=2))
        return 6
    manifest = Path(args.validation_manifest)
    if not manifest.exists():
        print(json.dumps({"status": "blocked", "reason": "validation manifest missing", "manifest": str(manifest)}, indent=2))
        return 3
    code, lock_path = lock_gat_independent_validation_manifest(
        manifest,
        Path(args.gat_dir),
        acknowledge=args.acknowledge_independent_holdout,
    )
    status = "completed" if code == 0 else ("failed_gate" if code == 5 else "blocked")
    print(json.dumps({"status": status, "lock": str(lock_path)}, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

