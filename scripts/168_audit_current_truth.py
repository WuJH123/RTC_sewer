#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.status.current_truth import build_current_truth


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild Project6 V3 current truth matrix from existing evidence files.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = build_current_truth(ROOT, write_outputs=True)
    payload = {
        "status": "completed",
        "truth_matrix": result["paths"]["truth_matrix"],
        "truth_report": result["paths"]["truth_report"],
        "recovery_gate": result["paths"]["recovery_gate"],
        "engineering_gate_status": result["report"]["engineering_gate_status"],
        "runtime_gate_status": result["report"]["runtime_gate_status"],
        "baseline_trajectory_count": result["report"]["baseline_selected_trajectory_count"],
        "real_state_processed_trajectory_count": result["report"]["state_input_rows"],
        "effective_round0_candidates": result["report"]["round0_effective_candidate_count"],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

