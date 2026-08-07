import json

import pandas as pd

from scripts.train_v42_step2_fast import _tensorise


def _j(shape, value=0.0):
    rows = [[value] * shape[1] for _ in range(shape[0])] if len(shape) == 2 else [value] * shape[0]
    return json.dumps(rows)


length, width = 12, 2


def test_control_core_skips_present_but_unavailable_outfall_columns():
    row = {
        "history_depth": _j((13, width)),
        "history_actions_readback": _j((13, 36)),
        "rainfall_forecast": _j((12, 1)),
        "pfv_delta": 0.0,
        "tfv_delta": 0.0,
        "peak_delta": 0.0,
    }
    for branch in ("candidate", "no_control", "dynamic_internal", "hold_previous"):
        row[f"action_{branch}_readback"] = _j((length, 36))
        row[f"trajectory_depth_{branch}"] = _j((length, width))
        row[f"trajectory_flood_{branch}"] = _j((length, width))
        row[f"trajectory_outfall_flow_{branch}"] = None
        row[f"trajectory_outfall_flow_{branch}_available"] = False

    data = _tensorise(pd.DataFrame([row]))

    assert "outfall_flow_candidate" not in data
    assert "depth_candidate" in data
