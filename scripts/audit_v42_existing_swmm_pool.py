"""CLI for the read-only V4.2 historical SWMM evidence-pool audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sewerrtc.v4.v42_r0_preflight import assert_r0_schema_preflight
from sewerrtc.v4.v42_r0_strict import (
    audit_existing_swmm_pool_strict,
    write_existing_pool_audit,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Discover all historical Project6 SWMM detail evidence, de-duplicate "
            "physical runs, audit target coverage and classify reuse potential. "
            "The default is the combined R0.1+R0.2 full finite audit."
        )
    )
    p.add_argument("--project-root", default=r"E:\RTC_sewer\Project6")
    p.add_argument(
        "--outputs-root",
        default=r"E:\RTC_sewer\Project6\outputs",
        help="Scan recursively; this is intentionally not limited to Train1600.",
    )
    p.add_argument(
        "--output-dir",
        default=(
            r"E:\RTC_sewer\Project6\outputs\project6_dual_reference_v4"
            r"\final_v4\v42_paper\data_reuse"
        ),
    )
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "Run discovery/metadata only. This mode cannot authorize reusable "
            "training views. Default is the combined full finite audit."
        ),
    )
    p.add_argument(
        "--full-finite-check",
        action="store_true",
        help="Backward-compatible no-op: full finite audit is already the default.",
    )
    p.add_argument(
        "--scan-cache",
        type=Path,
        default=None,
        help=(
            "Checkpoint the expensive logical-detail scan before case classification. "
            "Default: <output-dir>/_r0_logical_detail_cache.parquet."
        ),
    )
    p.add_argument(
        "--resume-scan-cache",
        action="store_true",
        help=(
            "Skip the expensive CSV audit and resume post-processing from --scan-cache. "
            "The cache is fail-closed against project/output/network/mode mismatches."
        ),
    )
    return p


def main() -> int:
    args = _parser().parse_args()
    # Catch serializer/classifier schema drift before opening any historical CSV.
    assert_r0_schema_preflight()
    print("[R0] schema preflight PASS", file=sys.stderr, flush=True)

    full_finite = not bool(args.metadata_only)
    output_dir = Path(args.output_dir)
    cache_path = (
        Path(args.scan_cache)
        if args.scan_cache is not None
        else output_dir / "_r0_logical_detail_cache.parquet"
    )
    result = audit_existing_swmm_pool_strict(
        project_root=Path(args.project_root),
        outputs_root=Path(args.outputs_root),
        full_finite_check=full_finite,
        max_workers=max(1, min(int(args.workers), 32)),
        logical_cache_path=cache_path,
        resume_from_logical_cache=bool(args.resume_scan_cache),
    )
    paths = write_existing_pool_audit(result, output_dir)
    payload = dict(result.summary)
    payload["outputs"] = {k: str(v) for k, v in paths.items()}
    payload["scan_cache"] = str(cache_path)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
