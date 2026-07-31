from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "206_oracle_pareto_v4.py"
)
spec = importlib.util.spec_from_file_location("oracle_pareto_v4", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_nondominated_mask() -> None:
    values = np.array(
        [
            [1.0, 3.0, 3.0],
            [2.0, 2.0, 2.0],
            [3.0, 1.0, 3.0],
            [2.0, 3.0, 4.0],
            [1.0, 3.0, 4.0],
        ]
    )
    assert module.nondominated_mask(values).tolist() == [True, True, True, False, False]


def test_topk_deviation_limits_changed_facilities() -> None:
    anchor = np.zeros((3, 5), dtype=float)
    schedule = np.array(
        [
            [0.1, 0.5, 0.2, 0.4, 0.3],
            [0.5, 0.1, 0.4, 0.2, 0.3],
            [0.2, 0.3, 0.5, 0.1, 0.4],
        ]
    )
    out = module.topk_deviation(schedule, anchor, 2)
    assert ((np.abs(out - anchor) > 0).sum(axis=1) <= 2).all()


def test_project_schedule_binary_and_bounds() -> None:
    actuators = pd.DataFrame(
        {
            "actuator_id": ["P1", "O1"],
            "link_type": ["pump", "orifice"],
            "storage_control_type": ["", ""],
        }
    )
    raw = np.array([[0.2, -0.1], [0.8, 1.2]], dtype=float)
    anchor = np.ones_like(raw)
    cfg = {
        "controller": {
            "variable_speed_pump_ids": [],
            "per_actuator_max_delta": {"P1": 1.0, "O1": 1.0},
            "min_hold_steps_by_actuator": {"P1": 1},
            "storage_retrofit": {
                "inlet_outlet_incompatible_action_constraint": False,
            },
        }
    }
    engineering = {"engineering36": {"binary_pump_ids": ["P1"]}}
    out = module.project_schedule(
        raw,
        anchor=anchor,
        actuators=actuators,
        cfg=cfg,
        engineering_cfg=engineering,
        constraint_mode="constrained",
        max_k=2,
    )
    assert set(np.unique(out[:, 0])).issubset({0.0, 1.0})
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_classify_objectives_reachable_separately() -> None:
    frame = pd.DataFrame(
        {
            "strict_feasible": [False, False],
            "pfv_feasible": [True, False],
            "tfv_feasible": [False, True],
            "peak_feasible": [False, True],
            "constraint_mode": ["constrained", "constrained"],
        }
    )
    assert module.classify_event(frame) == "objectives_reachable_separately_not_jointly"
