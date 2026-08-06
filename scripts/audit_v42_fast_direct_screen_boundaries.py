"""Audit whether a selected best action hit an expandable Stage-A bound."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_formal_runtime import load_actuators


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--importance", type=Path, required=True)
    args = parser.parse_args()
    screen = args.screen_root
    states = pd.read_csv(screen / "FAST_DIRECT_SCREEN_FINAL_STATES.csv")
    importance = pd.read_csv(args.importance)
    manifest = pd.read_parquet(args.manifest, columns=["state_key", "action_hold_previous_readback"]).assign(state_key=lambda x: x.state_key.astype(str))
    ids = load_actuators(args.project_root).actuator_id.astype(str).tolist()
    latest: dict[tuple[str, str], dict] = {}
    for ledger_path in (screen / "stage_a/DIRECT_SCREEN_STAGE_A_LEDGER.jsonl", screen / "stage_b/DIRECT_SCREEN_STAGE_B_LEDGER.jsonl"):
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest[(str(row.get("state_key")), str(row.get("candidate_action_sha256")))] = row
    output: list[dict] = []
    for _, state in states.iterrows():
        key = str(state.state_key)
        row = latest[(key, str(state.best_action_sha256))]
        sequence = np.asarray(row.get("candidate_action") or row.get("sequence"), dtype=float)
        raw_current = manifest[manifest.state_key.eq(key)].iloc[0].action_hold_previous_readback
        current = np.asarray(json.loads(raw_current) if isinstance(raw_current, str) else raw_current, dtype=float)[0]
        selected = importance[(importance.state_key.astype(str) == key) & (~importance.binary)].sort_values(
            ["frequency", "best_tfv_gain_pct", "facility_id"], ascending=[False, False, True], kind="stable"
        ).head(6)
        hits: list[dict] = []
        for facility_id in selected.facility_id.astype(str):
            index = ids.index(facility_id)
            old_low, old_high = max(0.0, current[index] - 0.25), min(1.0, current[index] + 0.25)
            new_low, new_high = max(0.0, current[index] - 0.40), min(1.0, current[index] + 0.40)
            values = sequence[:3, index]
            if any(abs(float(value) - old_low) <= 1.0e-5 for value in values) and new_low < old_low - 1.0e-9:
                hits.append({"facility_id": facility_id, "side": "low", "old_bound": old_low, "expanded_bound": new_low})
            if any(abs(float(value) - old_high) <= 1.0e-5 for value in values) and new_high > old_high + 1.0e-9:
                hits.append({"facility_id": facility_id, "side": "high", "old_bound": old_high, "expanded_bound": new_high})
        output.append({"event_id": state.event_id, "state_key": key, "regime": state.regime, "expandable_boundary_hit": bool(hits), "hits": hits})
    result = {"audit": "FAST_DIRECT_SCREEN_BOUNDARY_AUDIT_V1", "all_best_actions_have_no_expandable_boundary": not any(x["expandable_boundary_hit"] for x in output), "states": output}
    (screen / "FAST_DIRECT_BOUNDARY_AUDIT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in row.items() if k != "hits"} for row in output]).to_csv(screen / "FAST_DIRECT_BOUNDARY_AUDIT.csv", index=False)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
