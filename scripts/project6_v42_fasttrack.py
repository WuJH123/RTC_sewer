from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from sewerrtc.v4.v42_fasttrack import (
    audit_fasttrack_workflow,
    prepare_fasttrack_core,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "final_v4"
DEFAULT_R01 = DEFAULT_OUTPUT_ROOT / "v42_paper" / "data_reuse"
DEFAULT_FASTTRACK = DEFAULT_OUTPUT_ROOT / "v42_fasttrack" / "core_pool"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "wuhan_project6_v42_fasttrack.yaml"


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Development-only V4.2 fast-track: prove learnability/controlability before full R0."
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-core")
    prepare.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    prepare.add_argument("--r01-audit-dir", type=Path, default=DEFAULT_R01)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_FASTTRACK)
    prepare.add_argument("--max-events", type=int, default=None)
    prepare.add_argument("--cases-per-event", type=int, default=None)
    prepare.add_argument("--seed", type=int, default=None)

    gate = sub.add_parser("gate")
    gate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return p


def main() -> int:
    args = _parser().parse_args()
    cfg = _load_config(args.config)
    ft = cfg.get("fasttrack", {})
    thresholds = cfg.get("thresholds", {})

    if args.command == "prepare-core":
        evidence = prepare_fasttrack_core(
            project_root=args.project_root,
            r01_audit_dir=args.r01_audit_dir,
            output_dir=args.output_dir,
            max_events=int(args.max_events if args.max_events is not None else ft.get("max_events", 16)),
            cases_per_event=int(
                args.cases_per_event if args.cases_per_event is not None else ft.get("cases_per_event", 3)
            ),
            seed=int(args.seed if args.seed is not None else ft.get("seed", 42)),
            min_events=int(thresholds.get("core_pool", {}).get("independent_rainfall_groups", 8)),
            min_aligned_cases=int(thresholds.get("core_pool", {}).get("aligned_cases", 12)),
        )
        print(json.dumps(evidence, indent=2, allow_nan=False))
        return 0 if evidence.get("status") == "pass" else 5

    audit = audit_fasttrack_workflow(
        args.output_root,
        threshold_overrides=thresholds,
    )
    print(json.dumps(audit.as_dict(), indent=2, allow_nan=False))
    return 0 if audit.complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
