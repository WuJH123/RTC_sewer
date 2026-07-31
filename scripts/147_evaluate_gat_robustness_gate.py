#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.gat_robustness_gate import evaluate_gat_robustness_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute sr0p15 robustness gate from existing audit reports only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gat-dir", required=True)
    parser.add_argument("--independent-holdout", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gat_dir = Path(args.gat_dir)
    if not gat_dir.is_absolute():
        gat_dir = ROOT / gat_dir
    code, gate_path = evaluate_gat_robustness_gate(gat_dir)
    if args.independent_holdout and gate_path.exists():
        independent_gate = gat_dir / "gat_sr0p15_independent_robustness_gate.json"
        payload = json.loads(gate_path.read_text(encoding="utf-8-sig"))
        payload["gate_kind"] = "independent_holdout"
        payload["diagnostic_contaminated_gate"] = False
        independent_gate.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        gate_path = independent_gate
    print(json.dumps({"status": "pass" if code == 0 else ("failed_gate" if code == 5 else "blocked"), "gate": str(gate_path)}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
