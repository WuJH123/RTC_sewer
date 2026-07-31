from __future__ import annotations

import numpy as np
import pandas as pd

from sewerrtc.models.raw_joint_assets import build_actuator_feature_matrix, diffuse_action_node_map


def test_diffuse_action_node_map_reaches_multihop_nodes_and_normalizes() -> None:
    base = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    edge_index = np.asarray([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64)

    expanded = diffuse_action_node_map(base, edge_index, hops=3, decay=0.6)

    assert expanded.shape == base.shape
    assert expanded[0, 3] > 0.0
    np.testing.assert_allclose(expanded.sum(axis=1), 1.0, atol=1.0e-6)


def test_diffuse_action_node_map_zero_hops_preserves_base_map() -> None:
    base = np.asarray([[0.75, 0.25]], dtype=np.float32)
    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)

    expanded = diffuse_action_node_map(base, edge_index, hops=0, decay=0.6)

    np.testing.assert_allclose(expanded, base)


def test_actuator_feature_matrix_retains_hydraulic_capacity_and_head_lift() -> None:
    actuators = pd.DataFrame(
        {
            "link_type": ["orifice", "pump"],
            "near_storage": [True, False],
            "storage_control_type": ["storage_outlet", "not_storage"],
            "is_existing_rtc": [True, True],
            "is_physically_controllable": [True, True],
            "has_internal_rule": [True, False],
            "geom1": [2.0, np.nan],
            "geom2": [3.0, np.nan],
            "from_max_depth": [6.0, 7.0],
            "to_max_depth": [4.0, 3.0],
            "from_invert": [10.0, 12.0],
            "to_invert": [8.0, 18.0],
        },
        index=["O1", "P1"],
    )

    features, names = build_actuator_feature_matrix(actuators)

    assert features.shape == (2, len(names))
    assert "geom1_log_scaled" in names
    assert "invert_head_lift_scaled" in names
    assert "has_internal_rule" in names
    assert np.isfinite(features).all()
    assert features[0, names.index("geom1_log_scaled")] > features[1, names.index("geom1_log_scaled")]
    assert features[1, names.index("invert_head_lift_scaled")] > features[0, names.index("invert_head_lift_scaled")]
