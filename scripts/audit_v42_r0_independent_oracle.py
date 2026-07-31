from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_r0_independent_oracle import audit_r0_manifest_raw


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the authoritative raw Independent Oracle on the exact R0-derived Step-2 population."
    )
    default_dir = (
        PROJECT_ROOT
        / "outputs"
        / "project6_dual_reference_v4"
        / "final_v4"
        / "v42_paper"
        / "step2_surrogate"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_dir / "dataset" / "trajectory_manifest.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_dir / "independent_oracle",
    )
    args = parser.parse_args()
    audit, summary = audit_r0_manifest_raw(
        project_root=args.project_root,
        manifest_path=args.manifest,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "raw_oracle_rows.csv", index=False)
    (args.output_dir / "raw_oracle_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary.get("all_pass") is True else 5


if __name__ == "__main__":
    raise SystemExit(main())
