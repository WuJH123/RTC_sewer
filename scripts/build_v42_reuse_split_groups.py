"""Build cross-version rainfall grouping keys for the reusable V4.2 pool."""
from __future__ import annotations

import argparse
from pathlib import Path

from sewerrtc.v4.v42_reuse_split import build_reuse_split_groups


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(
        r"E:\RTC_sewer\Project6\outputs\project6_dual_reference_v4"
        r"\final_v4\v42_paper\data_reuse"
    )
    parser.add_argument("--audit-dir", type=Path, default=root)
    args = parser.parse_args()
    result = build_reuse_split_groups(
        reusable_physical_manifest=args.audit_dir / "reusable_pool_manifest.parquet",
        output_path=args.audit_dir / "split_group_manifest.parquet",
    )
    print(
        f"rows={len(result)} unique_groups={result['split_group_key'].nunique()} "
        f"reserved={int(result['reserved_evaluation'].sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
