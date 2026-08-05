from pathlib import Path

import numpy as np

from sewerrtc.v4.v42_formal_runtime import _node_full_depths
from sewerrtc.control.formal_baselines_f2 import equal_filling_degree_action


def test_filling_degree_capacity_uses_frozen_inp_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    inp = root / "data" / "wuhan_v8_storage_retrofit.inp"
    values = _node_full_depths(["ADD301", "ADD424"], {}, inp_path=inp)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def test_efd_changes_an_uneven_filling_state() -> None:
    result = equal_filling_degree_action(
        node_depth=np.asarray([1.0, 9.0]),
        node_full_depth=np.asarray([10.0, 10.0]),
        action_node_map=np.eye(2),
        anchor_action=np.ones(2),
        gain=0.8,
        deadband=0.02,
    )
    assert result.desired_action[0] < 1.0
    assert result.desired_action[1] == 1.0
