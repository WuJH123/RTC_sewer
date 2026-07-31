#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.gat_registry import (
    DEFAULT_PROJECT4_GAT_CANDIDATES,
    build_gat_registry,
    write_gat_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register read-only Project4 sparse-sensor GAT checkpoints for Project6 V3."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        print(json.dumps({"status": "failed", "reason": "config_not_found", "config": str(config_path)}, indent=2))
        return 6

    records = build_gat_registry(
        candidates=DEFAULT_PROJECT4_GAT_CANDIDATES,
        config_path=config_path,
        intended_use="read_only_candidate_for_project6_state_reconstruction",
    )
    outputs = write_gat_registry(records, out_dir)
    missing = [r.source_path for r in records if not r.exists]
    report = {
        "status": "registered_with_missing_sources" if missing else "registered",
        "config": str(config_path),
        "candidate_count": len(records),
        "missing_sources": missing,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "project4_read_only": True,
        "selected_primary_gat": None,
        "selection_status": "human_selection_required",
    }
    (out_dir / "gat_registration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 4 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
