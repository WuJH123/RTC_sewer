from __future__ import annotations

import pytest
import torch

from sewerrtc.models.temporal_sparse_gat_v42 import TemporalSparseGATReconstructorV42


def _model_and_inputs():
    torch.manual_seed(1)
    B, T, N, A, E = 2, 13, 6, 2, 10
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0, 1, 2, 3, 4]],
        dtype=torch.long,
    )
    model = TemporalSparseGATReconstructorV42(
        n_nodes=N,
        n_facilities=A,
        node_static_dim=3,
        link_static_dim=2,
        hidden_dim=16,
        heads=2,
        gat_layers=1,
        dropout=0.0,
    )
    inputs = dict(
        sparse_depth_history=torch.rand(B, T, N),
        sensor_mask_history=(torch.rand(B, T, N) > 0.5).float(),
        rainfall_history=torch.rand(B, T),
        historical_actions=torch.rand(B, T, A),
        node_static=torch.rand(N, 3),
        link_static=torch.rand(E, 2),
        edge_index=edge_index,
        action_node_map=torch.tensor(
            [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=torch.float32
        ),
    )
    return model, inputs


def test_temporal_gat_consumes_all_formal_step1_inputs():
    model, inputs = _model_and_inputs()
    out = model(**inputs)
    B, _, N = inputs["sparse_depth_history"].shape
    assert out.depth_mean.shape == (B, N)
    assert out.depth_std.shape == (B, N)
    assert out.latent_state.shape[:2] == (B, N)
    assert torch.isfinite(out.depth_mean).all()
    assert (out.depth_std > 0).all()


def test_decision_time_observed_sensors_are_preserved_exactly():
    model, inputs = _model_and_inputs()
    inputs["sensor_mask_history"][:, -1, :] = 0.0
    inputs["sensor_mask_history"][:, -1, 0] = 1.0
    inputs["sparse_depth_history"][:, -1, 0] = torch.tensor([0.4, 0.8])
    out = model(**inputs)
    assert torch.allclose(out.depth_mean[:, 0], torch.tensor([0.4, 0.8]))


def test_historical_actions_change_unobserved_reconstruction():
    model, inputs = _model_and_inputs()
    model.eval()
    inputs["sensor_mask_history"][:, -1, :] = 0.0
    with torch.no_grad():
        out1 = model(**inputs).depth_mean
        modified = dict(inputs)
        modified["historical_actions"] = inputs["historical_actions"] + 0.5
        out2 = model(**modified).depth_mean
    assert not torch.allclose(out1, out2)


def test_link_static_attributes_enter_gat_path():
    model, inputs = _model_and_inputs()
    model.eval()
    with torch.no_grad():
        out1 = model(**inputs).depth_mean
        modified = dict(inputs)
        modified["link_static"] = inputs["link_static"] + 2.0
        out2 = model(**modified).depth_mean
    assert not torch.allclose(out1, out2)


def test_temporal_gat_rejects_legacy_seven_frame_history():
    model, inputs = _model_and_inputs()
    inputs["sparse_depth_history"] = inputs["sparse_depth_history"][:, :7]
    inputs["sensor_mask_history"] = inputs["sensor_mask_history"][:, :7]
    inputs["rainfall_history"] = inputs["rainfall_history"][:, :7]
    inputs["historical_actions"] = inputs["historical_actions"][:, :7]
    with pytest.raises(ValueError, match="13"):
        model(**inputs)
