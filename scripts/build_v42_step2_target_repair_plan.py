"""Freeze a development-only plan for repairing missing CONTROL_CORE targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plan_v42_targeted_candidate_expansion import _prefix_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.state_audit.read_text(encoding="utf-8"))
    state_keys = [str(value) for value in audit["state_keys"]]
    source = pd.read_parquet(
        args.source_manifest,
        columns=[
            "state_key",
            "event_id",
            "rainfall_sha256",
            "checkpoint_min",
            "history_depth",
            "history_actions_readback",
            "rainfall_forecast",
        ],
    )
    rows = source[source["state_key"].astype(str).isin(state_keys)].copy()
    rows = rows.drop_duplicates("state_key", keep="first")
    if set(rows["state_key"].astype(str)) != set(state_keys):
        missing = sorted(set(state_keys) - set(rows["state_key"].astype(str)))
        raise RuntimeError(f"repair state(s) missing from source manifest: {missing}")

    plan = pd.DataFrame(
        {
            "state_key": rows["state_key"].astype(str),
            "event_id": rows["event_id"].astype(str),
            "rainfall_sha256": rows["rainfall_sha256"].astype(str),
            "checkpoint_min": rows["checkpoint_min"].astype(float),
            "prefix_state_sha256": rows.apply(_prefix_hash, axis=1),
        }
    ).sort_values("state_key", kind="stable")
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(args.output_plan, index=False)
    selected = plan.to_dict(orient="records")
    lock = {
        "lock_id": "V42_STEP2_TARGET_REPAIR_PLAN_LOCK_V1",
        "development_only": True,
        "source_manifest": str(args.source_manifest),
        "state_audit": str(args.state_audit),
        "plan_sha256": hashlib.sha256(args.output_plan.read_bytes()).hexdigest(),
        "selected_states": selected,
    }
    args.output_lock.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"states": len(selected), "plan": str(args.output_plan), "lock": str(args.output_lock)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
