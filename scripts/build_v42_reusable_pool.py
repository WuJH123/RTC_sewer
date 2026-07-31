"""Build masked reusable V4.2 task manifests from an existing-pool audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sewerrtc.v4.v42_reusable_pool import build_reusable_paper_pool


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build target-masked reusable V4.2 manifests")
    default_root = Path(
        r"E:\RTC_sewer\Project6\outputs\project6_dual_reference_v4"
        r"\final_v4\v42_paper\data_reuse"
    )
    p.add_argument("--audit-dir", type=Path, default=default_root)
    p.add_argument("--exclude-source-domain", action="store_true")
    p.add_argument("--exclude-consumed-development", action="store_true")
    p.add_argument(
        "--allow-missing-alignment-audit",
        action="store_true",
        help=(
            "Build generic branch-level pretraining views even when the case alignment audit "
            "is absent. Counterfactual case eligibility remains false."
        ),
    )
    return p


def main() -> int:
    args = _parser().parse_args()
    root = args.audit_dir
    alignment = root / "case_alignment_audit.csv"
    if not alignment.exists() and not args.allow_missing_alignment_audit:
        raise FileNotFoundError(
            f"missing {alignment}; run scripts/audit_v42_case_alignment.py first, "
            "or explicitly use --allow-missing-alignment-audit for generic pretraining only"
        )
    result = build_reusable_paper_pool(
        physical_inventory=root / "physical_run_inventory.parquet",
        case_inventory=root / "target_coverage_by_case.csv",
        alignment_inventory=alignment if alignment.exists() else None,
        output_physical_manifest=root / "reusable_pool_manifest.parquet",
        output_case_manifest=root / "reusable_case_manifest.parquet",
        audit_output=root / "reusable_pool_summary.json",
        include_source_domain=not args.exclude_source_domain,
        include_consumed_development=not args.exclude_consumed_development,
    )
    print(
        json.dumps(
            {
                "physical_manifest": str(result.physical_manifest_path),
                "case_manifest": str(result.case_manifest_path),
                "audit": str(result.audit_path),
                "physical_rows": result.physical_row_count,
                "case_rows": result.case_row_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
