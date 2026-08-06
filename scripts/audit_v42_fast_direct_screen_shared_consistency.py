"""Verify persisted FAST-screen metrics against the shared authoritative functions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_v42_fast_direct_screen import _detail_metrics
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--rows-per-state", type=int, default=3)
    args = parser.parse_args()
    screen = args.screen_root
    all_rows = pd.read_csv(screen / "FAST_DIRECT_SCREEN_ROWS.csv")
    state_keys = all_rows.state_key.astype(str).drop_duplicates().head(int(args.states)).tolist()
    selected = all_rows[all_rows.state_key.astype(str).isin(state_keys)].groupby("state_key", sort=False).head(int(args.rows_per_state))
    reference_by_state: dict[str, str] = {}
    checkpoint_by_key: dict[tuple[str, str], float] = {}
    for ledger_path in (screen / "stage_a/DIRECT_SCREEN_STAGE_A_LEDGER.jsonl", screen / "stage_b/DIRECT_SCREEN_STAGE_B_LEDGER.jsonl"):
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item.get("source_detail_path_no_control"):
                    reference_by_state[str(item["state_key"])] = str(item["source_detail_path_no_control"])
                if item.get("checkpoint_min") is not None:
                    checkpoint_by_key[(str(item.get("state_key")), str(item.get("candidate_action_sha256")))] = float(item["checkpoint_min"])
    graph = _load_graph_topology(args.project_root)
    node_ids = [str(value) for value in graph["node_ids"]]
    priority_nodes = [node_ids[int(index)] for index in get_pfv_core_node_indices(node_ids)]
    checks = []
    for _, row in selected.iterrows():
        checkpoint = checkpoint_by_key[(str(row.state_key), str(row.candidate_action_sha256))]
        actual = _detail_metrics(Path(str(row.detail_path)), Path(reference_by_state[str(row.state_key)]), checkpoint, priority_nodes)
        errors = {
            "pfv_candidate_m3": abs(float(row.pfv_candidate_m3) - float(actual["pfv_candidate_m3"])),
            "pfv_no_control_m3": abs(float(row.pfv_no_control_m3) - float(actual["pfv_no_control_m3"])),
            "tfv_candidate_m3": abs(float(row.tfv_candidate_m3) - float(actual["tfv_candidate_m3"])),
            "pfv_budget_metric_m3": abs(float(row.pfv_budget_metric_m3) - float(actual["pfv_budget_metric_m3"])),
        }
        checks.append({"state_key": str(row.state_key), "candidate_action_sha256": str(row.candidate_action_sha256), "max_abs_error": max(errors.values()), "errors": errors, "pfv_feasible_match": bool(str(row.pfv_feasible).lower() == str(actual["pfv_feasible"]).lower())})
    passed = bool(checks) and all(float(x["max_abs_error"]) <= 1.0e-3 and x["pfv_feasible_match"] for x in checks)
    audit = {"audit_id": "FAST_DIRECT_SCREEN_SHARED_METRIC_CONSISTENCY_V1", "development_only": True, "new_swmm_started": False, "rows_tested": len(checks), "max_abs_error": max(float(x["max_abs_error"]) for x in checks) if checks else None, "status": "pass" if passed else "fail", "checks": checks}
    (screen / "FAST_DIRECT_SHARED_METRIC_CONSISTENCY.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
