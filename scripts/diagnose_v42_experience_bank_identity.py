"""Diagnose legacy versus canonical action identity in the Round2 manifest."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.control.authoritative_control_metrics_v42 import action_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(
        args.manifest,
        columns=["state_key", "candidate_action_sha256", "action_candidate_readback", "source_detail_path_candidate"],
    )

    def canonical(value: object) -> str:
        return action_sha256(np.asarray(json.loads(str(value)), dtype=np.float32))

    frame["canonical_sha"] = frame["action_candidate_readback"].map(canonical)
    frame["legacy_sha"] = frame["candidate_action_sha256"].astype(str).str.strip()
    mismatch = frame["legacy_sha"].ne(frame["canonical_sha"])
    failed = frame.loc[mismatch]

    def source_family(value: object) -> str:
        match = re.search(r"(?i)(train1600|pilot_v3|v41_calibration|v41_locked|peak|round[0-9]+|aug[0-9]+)", str(value))
        return match.group(1).lower() if match else "unclassified"

    positions = failed.index.to_list()
    diagnosis = {
        "status": "LEGACY_ACTION_SHA_FORMAT_MISMATCH" if len(failed) else "NO_IDENTITY_MISMATCH",
        "manifest": str(args.manifest),
        "failure_count": int(len(failed)),
        "exception_type": "ValueError",
        "exception_message": "stored candidate_action_sha256 differs from canonical action hash",
        "row_range": {"first": int(min(positions)) if positions else None, "last": int(max(positions)) if positions else None},
        "mismatch_rows_contiguous": bool(positions and positions == list(range(min(positions), max(positions) + 1))),
        "unique_states": int(failed["state_key"].nunique()),
        "by_source_family": {str(k): int(v) for k, v in failed["source_detail_path_candidate"].map(source_family).value_counts().items()},
        "base_518_mismatch_count": int(len(failed.iloc[:518])) if len(failed) >= 518 else 0,
        "canonical_identity": {
            "source": "action_candidate_readback",
            "dtype": "float32",
            "identity": "(state_key, canonical_action_sha256)",
            "legacy_sha_used_as_authority": False,
        },
        "first_ten": [
            {
                "row": int(index),
                "state_key": str(row.state_key),
                "legacy_sha": str(row.legacy_sha),
                "canonical_sha": str(row.canonical_sha),
                "source_family": source_family(row.source_detail_path_candidate),
            }
            for index, row in failed.head(10).iterrows()
        ],
        "resolution": "canonicalize executed readback action; retain legacy SHA as metadata only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(diagnosis, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
