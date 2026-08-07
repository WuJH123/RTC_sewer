"""Materialize a compact manifest from completed targeted target-fill branches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v42_targeted_candidate_expansion import _build_expanded_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--source-manifest", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--funnel", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    plan = pd.read_csv(args.plan)
    funnel = pd.read_csv(args.funnel)
    passed = funnel[funnel.status.isin(["pass", "reused"])].to_dict("records")
    if len(passed) != len(plan):
        raise RuntimeError("target-fill funnel is incomplete")
    states = sorted(plan.state_key.astype(str).unique())
    source = pd.read_parquet(args.source_manifest, filters=[("state_key", "in", states)])
    manifest = _build_expanded_manifest(passed, source, args.project_root)
    if len(manifest) != len(passed):
        raise RuntimeError("target-fill manifest row count mismatch")
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output, index=False)
    audit = {
        "audit_id": "V42_STEP2_TARGETED_TARGET_FILL_MANIFEST_AUDIT_V1",
        "development_only": True, "formal_mainline_authorized": False,
        "rows": int(len(manifest)), "states": int(manifest.state_key.nunique()),
        "storage_complete": int(manifest.trajectory_storage_volume_candidate_available.astype(bool).sum()),
        "facility_flow_complete": int(manifest.trajectory_facility_flow_candidate_available.astype(bool).sum()),
        "output": str(output),
    }
    output.with_name("TARGETED_STEP2_TARGET_FILL_MANIFEST_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
