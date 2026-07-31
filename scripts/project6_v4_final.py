from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.pipeline import ALL_STAGES, build_registry, sha256_json
from sewerrtc.v4.runtime import (
    RuntimeOptions,
    atomic_write_json,
    stage_record,
    working_code_sha,
)


PAPER_CONTRACT_ID = "PROJECT6_V42_PAPER_WORKFLOW_V1"
# These legacy registry stages were designed around the V4.1/old V4.2 model
# line.  They remain reproducible development tools, but the final paper config
# must not execute them accidentally as Formal evidence.
LEGACY_FORMAL_STAGE_TOKENS = (
    "TrainV42",
    "EvaluateV42",
    "ExactClosedLoop",
    "SurrogateClosedLoop",
    "GATClosedLoop",
    "PolicyLock",
    "Challenge",
    "FormalBlind",
)


def _declared_input_hashes(config: dict, root: Path) -> dict[str, str]:
    project = config.get("project", {})
    result: dict[str, str] = {}
    for key in (
        "network",
        "contract",
        "paper_workflow_contract",
        "canonical_ids",
        "facility_semantics",
        "priority_nodes",
    ):
        raw = project.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = root / path
        if path.exists() and path.is_file():
            result[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/wuhan_project6_v4_final.yaml"),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-stages", action="store_true")
    parser.add_argument(
        "--allow-legacy-development",
        action="store_true",
        help=(
            "Allow a superseded V4.1/old-V4.2 stage for reproducibility only. "
            "It still cannot create PROJECT6_V42_PAPER_WORKFLOW_V1 Formal evidence."
        ),
    )
    return parser.parse_args()


def _legacy_formal_stage(stage: str) -> bool:
    text = str(stage)
    return any(token.casefold() in text.casefold() for token in LEGACY_FORMAL_STAGE_TOKENS)


def main() -> int:
    args = parse_args()
    if args.list_stages:
        print("\n".join(ALL_STAGES))
        print(
            "\nNOTE: final paper stages are gated by scripts/project6_v42_paper.py; "
            "legacy registry stages cannot authorize Formal V4.2 evidence.",
            file=sys.stderr,
        )
        return 0
    if not args.stage:
        print("--stage is required unless --list-stages is used", file=sys.stderr)
        return 2
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    paper_contract = str(
        config.get("paper_workflow", {}).get("contract_id", "")
    )
    if (
        paper_contract == PAPER_CONTRACT_ID
        and _legacy_formal_stage(args.stage)
        and not args.allow_legacy_development
    ):
        print(
            f"Blocked legacy stage {args.stage!r}: the active config is governed by "
            f"{PAPER_CONTRACT_ID}. Use scripts/project6_v42_paper.py and the new "
            "trajectory-first/GAT-integrated paper components. If you only need "
            "historical reproducibility, rerun with --allow-legacy-development; "
            "that output is development evidence and cannot become Formal Blind.",
            file=sys.stderr,
        )
        return 2

    project_root = Path(config.get("project", {}).get("root", PROJECT_ROOT))
    output_root = project_root / config.get("project", {}).get(
        "output_root", "outputs/project6_dual_reference_v4/final_v4"
    )
    registry = build_registry(
        project_root=project_root, output_root=output_root, config=config
    )
    options = RuntimeOptions(
        stage=args.stage,
        config=str(config_path),
        workers=max(1, min(int(args.workers), 16)),
        limit=max(0, int(args.limit)),
        resume=bool(args.resume),
        retry_failed=bool(args.retry_failed),
        dry_run=bool(args.dry_run),
    )
    started = time.time()
    result = registry.run(args.stage, options)
    record = stage_record(
        result,
        config_sha=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        code_sha=working_code_sha(project_root),
        input_sha=sha256_json(
            {
                "stage": args.stage,
                "config": config,
                "declared_inputs": _declared_input_hashes(
                    config, project_root
                ),
                "options": {
                    "workers": options.workers,
                    "limit": options.limit,
                    "resume": options.resume,
                    "retry_failed": options.retry_failed,
                    "dry_run": options.dry_run,
                    "allow_legacy_development": bool(
                        args.allow_legacy_development
                    ),
                },
            }
        ),
        started_at=started,
        run_uuid=str(uuid.uuid4()),
    )
    status_path = (
        output_root / "audits" / "stage_status" / f"{args.stage}.json"
    )
    atomic_write_json(status_path, record)
    if result.scope_complete:
        atomic_write_json(
            status_path.with_name(f"{args.stage}.completion.json"),
            {
                "stage": args.stage,
                "run_uuid": record["run_uuid"],
                "input_sha": record["input_sha"],
                "status_sha256": hashlib.sha256(
                    status_path.read_bytes()
                ).hexdigest(),
            },
        )
    print(json.dumps(record, indent=2, allow_nan=False))
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
