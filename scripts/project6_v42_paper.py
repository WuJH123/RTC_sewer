from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.paper_workflow_v42 import audit_paper_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Step 4 of the fail-closed V4.2 paper workflow: true-state/offline, "
            "closed loops, Policy Lock, Challenge and Formal Blind. Use "
            "scripts/project6_v42_mainline.py for the full R0->Step1->Step2->Step3->Step4 chain."
        )
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/wuhan_project6_v4_final.yaml"),
    )
    parser.add_argument("--output-root", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = Path(config.get("project", {}).get("root", PROJECT_ROOT))
    output_root = (
        Path(args.output_root)
        if args.output_root
        else root
        / str(
            config.get("project", {}).get(
                "output_root", "outputs/project6_dual_reference_v4/final_v4"
            )
        )
    )
    audit = audit_paper_workflow(output_root)
    print(json.dumps(audit.as_dict(), indent=2, allow_nan=False))
    return 0 if audit.complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
