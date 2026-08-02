from __future__ import annotations

import numpy as np

from sewerrtc.control.formal_baselines_f2 import (
    all_close_action,
    auto_rbc_action,
    controlled_filling_degree,
    equal_filling_degree_action,
)


def test_controlled_filling_degree_uses_normalized_controlled_node_depth() -> None:
    depth = np.array([0.2, 0.8, 0.5])
    full = np.ones(3)
    amap = np.array([[1, 0, 0], [0, 1, 1]], dtype=float)
    fill = controlled_filling_degree(node_depth=depth, node_full_depth=full, action_node_map=amap)
    np.testing.assert_allclose(fill, [0.2, 0.8])


def test_efd_releases_more_filled_zone_and_keeps_binary_facility_binary() -> None:
    depth = np.array([0.2, 0.9])
    full = np.ones(2)
    amap = np.eye(2)
    anchor = np.array([0.5, 0.0])
    out = equal_filling_degree_action(
        node_depth=depth,
        node_full_depth=full,
        action_node_map=amap,
        anchor_action=anchor,
        binary_indices=[1],
        gain=1.0,
        deadband=0.0,
    )
    assert out.desired_action[1] in (0.0, 1.0)
    assert out.desired_action[1] >= anchor[1]
    assert out.desired_action[0] <= anchor[0]


def test_auto_rbc_is_fixed_and_bounded() -> None:
    depth = np.array([0.1, 0.5, 0.9])
    full = np.ones(3)
    amap = np.eye(3)
    anchor = np.full(3, 0.5)
    out = auto_rbc_action(
        node_depth=depth,
        node_full_depth=full,
        action_node_map=amap,
        anchor_action=anchor,
        binary_indices=[2],
    )
    assert out.desired_action[0] == 0.0
    assert 0.0 <= out.desired_action[1] <= 1.0
    assert out.desired_action[2] == 1.0


def test_all_close_has_exact_facility_dimension() -> None:
    out = all_close_action(36)
    assert out.desired_action.shape == (36,)
    assert np.all(out.desired_action == 0.0)
