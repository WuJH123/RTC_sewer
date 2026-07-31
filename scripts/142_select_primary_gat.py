#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.gat_selection import write_primary_gat_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock the user-confirmed sr0p15 primary GAT after report/hash checks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gat-dir", required=True)
    parser.add_argument("--registry-name", required=True)
    parser.add_argument("--selection-decision-path", default="docs/contracts/gat_primary_selection_decision.json")
    parser.add_argument("--acknowledge-selection", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.registry_name != "sr0p15":
        print(json.dumps({"status": "failed", "reason": "registry_name_must_be_sr0p15"}, indent=2))
        return 7
    if not args.acknowledge_selection:
        print(json.dumps({"status": "failed", "reason": "missing_acknowledgement"}, indent=2))
        return 7
    config = Path(args.config)
    decision = Path(args.selection_decision_path)
    gat_dir = Path(args.gat_dir)
    if not config.is_absolute():
        config = ROOT / config
    if not decision.is_absolute():
        decision = ROOT / decision
    if not gat_dir.is_absolute():
        gat_dir = ROOT / gat_dir
    code, payload = write_primary_gat_lock(
        config_path=config,
        decision_path=decision,
        gat_dir=gat_dir,
        script_path=Path(__file__).resolve(),
        registry_name=args.registry_name,
        acknowledgement=args.acknowledge_selection,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
