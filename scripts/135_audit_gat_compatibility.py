#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.gat_compatibility import audit_gat_registry, write_gat_compatibility_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Project4 GAT compatibility with the Project6 retrofit network.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    registry_path = Path(args.registry)
    out_dir = Path(args.out_dir)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not registry_path.is_absolute():
        registry_path = ROOT / registry_path
    out_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        print(json.dumps({"status": "failed", "reason": "config_not_found", "config": str(config_path)}, indent=2))
        return 6
    if not registry_path.exists():
        print(json.dumps({"status": "blocked", "reason": "registry_not_found", "registry": str(registry_path)}, indent=2))
        return 3

    report = audit_gat_registry(config_path=config_path, registry_path=registry_path)
    outputs = write_gat_compatibility_outputs(report, out_dir, config_path=config_path)
    payload = {
        "status": "completed_audit",
        "overall_research_status": report.overall_research_status,
        "selected_primary_gat": None,
        "selection_status": "human_selection_required",
        "compatible_strict_count": report.compatible_strict_count,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
