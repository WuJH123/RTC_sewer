from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_r0_paper_dataset import build_r0_paper_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize the formal GAT-compatible Step-2 dataset from strict Phase R0."
    )
    default = (
        PROJECT_ROOT
        / "outputs"
        / "project6_dual_reference_v4"
        / "final_v4"
        / "v42_paper"
        / "data_reuse"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--r0-dir", type=Path, default=default)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "project6_dual_reference_v4"
            / "final_v4"
            / "v42_paper"
            / "step2_surrogate"
            / "dataset"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = build_r0_paper_dataset(
        project_root=args.project_root,
        physical_manifest=args.r0_dir / "reusable_pool_manifest.parquet",
        case_manifest=args.r0_dir / "reusable_case_manifest.parquet",
        split_manifest=args.r0_dir / "split_group_manifest.parquet",
        output_manifest=args.output_dir / "trajectory_manifest.parquet",
        audit_output=args.output_dir / "dataset_audit.json",
    )
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path),
                "audit": str(result.audit_path),
                "accepted_count": result.accepted_count,
                "rejected_count": result.rejected_count,
                "sample_lineage_sha256": result.lineage_sha256,
            },
            indent=2,
        )
    )
    return 0 if result.accepted_count > 0 and result.rejected_count == 0 else 5


if __name__ == "__main__":
    raise SystemExit(main())
