"""Run numeric four-reference same-state/forcing audit on the reusable pool."""
from __future__ import annotations

import argparse
from pathlib import Path

from sewerrtc.v4.v42_case_alignment_audit import audit_case_alignment


def main() -> int:
    parser = argparse.ArgumentParser()
    default_dir = Path(
        r"E:\RTC_sewer\Project6\outputs\project6_dual_reference_v4"
        r"\final_v4\v42_paper\data_reuse"
    )
    parser.add_argument("--project-root", type=Path, default=Path(r"E:\RTC_sewer\Project6"))
    parser.add_argument("--audit-dir", type=Path, default=default_dir)
    args = parser.parse_args()
    frame = audit_case_alignment(
        project_root=args.project_root,
        physical_inventory=args.audit_dir / "physical_run_inventory.parquet",
        case_inventory=args.audit_dir / "target_coverage_by_case.csv",
        output_path=args.audit_dir / "case_alignment_audit.csv",
    )
    passed = int((frame["same_state_numeric_pass"] & frame["same_forcing_pass"]).sum())
    print(f"aligned_cases={passed}/{len(frame)}")
    return 0 if len(frame) and passed > 0 else 5


if __name__ == "__main__":
    raise SystemExit(main())
