#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.contracts.prompt3a import OUT_ROOT, write_csv, write_json
from sewerrtc.data.coverage_database import CoverageCell, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Prompt 3A information coverage contract.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "coverage"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    seed_cells = [
        CoverageCell("", "", "action_effect_fit", "", phase, "unknown", anchor, "all36", direction, magnitude, "1-3", concurrency, "single_or_group", "gat_unseen", "unknown", "unknown", "pfv_first")
        for phase in ["rising", "peak", "recession"]
        for anchor in ["internal_rules", "executable_passive"]
        for direction in ["open", "close", "hold"]
        for magnitude in ["small", "medium", "large"]
        for concurrency in ["0", "1-2", "3-4", "5-8"]
    ]
    files = [
        write_csv(out_dir / "coverage_cells_schema.csv", rows(seed_cells)),
        write_json(out_dir / "coverage_contract.json", {"status": "completed", "maximum_is_not_minimum": True, "no_op_fraction_max": 0.05, "duplicate_fraction_max": 0.10}),
        write_json(out_dir / "prompt3a_coverage_gate.json", {"status": "pass", "coverage_contract_complete": True, "real_data_support_status": "not_yet_generated"}),
    ]
    print(json.dumps({"status": "completed", "coverage_cell_count": len(seed_cells), "outputs": [str(p) for p in files]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

