"""CLI for the read-only V4.2 historical SWMM evidence-pool audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sewerrtc.v4.v42_existing_pool_audit import (
    audit_existing_swmm_pool,
    write_existing_pool_audit,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Discover all historical Project6 SWMM detail evidence, de-duplicate "
            "physical runs, audit target coverage and classify reuse potential."
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
    p.add_argument(
        "--full-finite-check",
        action="store_true",
        help=(
            "Read all available target columns and fail finite checks on NaN/Inf. "
            "The default metadata pass is faster and never claims a finite scientific pass."
        ),
    )
    return p


def main() -> int:
    args = _parser().parse_args()
    # Since the single-pass reader already loads the full CSV into memory,
    # the finite check is essentially free.  Always enable it so that R0.1
    # and R0.2 are combined into one I/O pass.
    full_finite = True
    result = audit_existing_swmm_pool(
        project_root=Path(args.project_root),
        outputs_root=Path(args.outputs_root),
        full_finite_check=full_finite,
    )
    paths = write_existing_pool_audit(result, Path(args.output_dir))
    payload = dict(result.summary)
    payload["outputs"] = {k: str(v) for k, v in paths.items()}
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
