import numpy as np
from types import SimpleNamespace

from sewerrtc.v4.v42_trajectory_builder import build_surrogate_action_node_map


def test_surrogate_action_map_reaches_bounded_network_neighbourhood():
    graph = {
        "n_nodes": 4,
        "n_facilities": 1,
        "node_ids": ["N0", "N1", "N2", "N3"],
        "edge_index": np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        "facility_endpoints": [{"from_node": "N0", "to_node": "N0"}],
    }
    result = build_surrogate_action_node_map(graph, radius=2)
    assert result.shape == (1, 4)
    assert np.isclose(result[0, 0], 1.0)
    assert np.isclose(result[0, 1], 0.5)
    assert np.isclose(result[0, 2], 1.0 / 3.0)
    assert result[0, 3] == 0.0


def test_surrogate_action_map_accepts_step1_graph_assets_shape():
    graph = SimpleNamespace(
        n_nodes=4,
        n_facilities=1,
        node_ids=["N0", "N1", "N2", "N3"],
        edge_index=np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        action_node_map=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    result = build_surrogate_action_node_map(graph, radius=2)
    assert np.allclose(result[0], [1.0, 0.5, 1.0 / 3.0, 0.0])
