from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_step1_windows import build_step1_window_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build formal 13x5-min Step-1 temporal windows from strict R0 evidence."
    )
    r0 = (
        PROJECT_ROOT
        / "outputs"
        / "project6_dual_reference_v4"
        / "final_v4"
        / "v42_paper"
        / "data_reuse"
    )
    out = (
        PROJECT_ROOT
        / "outputs"
        / "project6_dual_reference_v4"
        / "final_v4"
        / "v42_paper"
        / "step1_gat"
        / "dataset"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--r0-dir", type=Path, default=r0)
    parser.add_argument("--output-dir", type=Path, default=out)
    args = parser.parse_args()
    result = build_step1_window_manifest(
        project_root=args.project_root,
        physical_manifest=args.r0_dir / "reusable_pool_manifest.parquet",
        split_manifest=args.r0_dir / "split_group_manifest.parquet",
        output_manifest=args.output_dir / "step1_window_manifest.parquet",
        audit_output=args.output_dir / "window_audit.json",
    )
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path),
                "audit": str(result.audit_path),
                "window_count": result.window_count,
                "rainfall_group_count": result.rainfall_group_count,
            },
            indent=2,
        )
    )
    return 0 if result.window_count > 0 and result.rainfall_group_count > 1 else 5


if __name__ == "__main__":
    raise SystemExit(main())
