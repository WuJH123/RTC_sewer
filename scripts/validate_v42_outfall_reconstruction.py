"""Validate incoming-link reconstruction against a new explicit-outfall detail."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sewerrtc.v4.v42_outfall_recovery import (
    validate_outfall_reconstruction,
    write_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", required=True, type=Path)
    parser.add_argument(
        "--inp",
        type=Path,
        default=Path(r"E:\RTC_sewer\Project6\data\wuhan_v8_storage_retrofit.inp"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--atol-m3s", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    args = parser.parse_args()
    result = validate_outfall_reconstruction(
        args.detail,
        inp_path=args.inp,
        atol_m3s=args.atol_m3s,
        rtol=args.rtol,
    )
    write_validation(args.output, result)
    print(json.dumps(result.as_dict(), indent=2, allow_nan=False))
    return 0 if result.status == "pass" else 5


if __name__ == "__main__":
    raise SystemExit(main())
