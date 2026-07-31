"""Recover historical outfall-flow sidecars only after independent validation."""
from __future__ import annotations

import argparse
from pathlib import Path

from sewerrtc.v4.v42_outfall_bulk_recovery import recover_outfall_sidecars


def main() -> int:
    parser = argparse.ArgumentParser()
    default_root = Path(
        r"E:\RTC_sewer\Project6\outputs\project6_dual_reference_v4\"
        r"final_v4\v42_paper\data_reuse"
    )
    parser.add_argument("--audit-dir", type=Path, default=default_root)
    parser.add_argument(
        "--validation-json",
        type=Path,
        required=True,
        help="Passing validation report from validate_v42_outfall_reconstruction.py",
    )
    parser.add_argument(
        "--inp",
        type=Path,
        default=Path(r"E:\RTC_sewer\Project6\data\wuhan_v8_storage_retrofit.inp"),
    )
    args = parser.parse_args()
    frame = recover_outfall_sidecars(
        physical_inventory=args.audit_dir / "physical_run_inventory.parquet",
        validation_json=args.validation_json,
        inp_path=args.inp,
        sidecar_dir=args.audit_dir / "outfall_sidecars",
        output_manifest=args.audit_dir / "recoverable_from_validated_links.csv",
    )
    recovered = int((frame["status"] == "recovered_validated").sum())
    failed = int((frame["status"] == "recovery_failed").sum())
    print(f"recovered={recovered} failed={failed} total={len(frame)}")
    return 0 if failed == 0 else 5


if __name__ == "__main__":
    raise SystemExit(main())
