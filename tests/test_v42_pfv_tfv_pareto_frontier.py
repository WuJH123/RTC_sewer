from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_v42_pfv_tfv_pareto_frontier.py"
spec = importlib.util.spec_from_file_location("audit_v42_pfv_tfv_pareto_frontier", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_pareto_rows_keep_candidate_metrics_after_state_join(tmp_path: Path) -> None:
    module._load_graph_topology = lambda _: {"node_ids": ["N0"]}
    module.get_pfv_core_node_indices = lambda _: [0]
    module.pd.qcut = lambda values, q, labels: pd.Series([labels[0]] * len(values), index=values.index)
    trajectory = json.dumps([[1.0], [1.0]])
    internal = json.dumps([[2.0], [2.0]])
    hold = json.dumps([[1.0], [1.0]])
    frame = pd.DataFrame(
        [
            {
                "state_key": "state-1", "event_id": "event-1", "rainfall_sha256": "rain-1",
                "checkpoint_min": 120, "pfv_delta": 0.0, "tfv_delta": -100.0, "peak_delta": 0.0,
                "trajectory_flood_no_control": trajectory,
                "trajectory_flood_dynamic_internal": internal,
                "trajectory_flood_candidate": trajectory,
                "action_candidate_readback": json.dumps([[0.0], [0.0]]),
                "action_hold_previous_readback": hold,
            },
            {
                "state_key": "state-1", "event_id": "event-1", "rainfall_sha256": "rain-1",
                "checkpoint_min": 120, "pfv_delta": 1000.0, "tfv_delta": -1000.0, "peak_delta": 0.0,
                "trajectory_flood_no_control": trajectory,
                "trajectory_flood_dynamic_internal": internal,
                "trajectory_flood_candidate": json.dumps([[3.0], [3.0]]),
                "action_candidate_readback": json.dumps([[0.0], [0.0]]),
                "action_hold_previous_readback": hold,
            },
        ]
    )
    manifest = tmp_path / "manifest.parquet"
    frame.to_parquet(manifest, index=False)

    rows, _ = module._load_states(manifest, tmp_path)
    annotated_rows, state_results = module._state_results(rows, relative=0.05, absolute=100.0)

    assert len(rows) == 2
    assert rows["pfv_candidate_m3"].nunique() == 2
    assert int(annotated_rows["actual_safe"].sum()) == 1
    assert int(state_results.iloc[0]["actual_safe_candidate_count"]) == 1
