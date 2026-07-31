"""Twin-branch sensitivity, action-encoder usage and output-aliasing tests."""
from __future__ import annotations

import pytest
import torch

from sewerrtc.v4.models_v42.counterfactual_twin_dynamics import TwinGraphDynamics
from sewerrtc.v4.models_v42.actuator_action_encoder import ActuatorActionEncoder
from sewerrtc.v4.v42_trainer import TwinWithKPIHeads


@pytest.fixture
def small_model():
    m = TwinGraphDynamics(
        n_nodes=16,
        n_facilities=4,
        n_static_features=3,
        hidden_dim=32,
        gat_heads=4,
        n_gat_layers=2,
        horizon=4,
        history_frames=3,
    )
    with torch.no_grad():
        for layer in m.depth_head:
            if isinstance(layer, torch.nn.Linear):
                layer.bias.fill_(1.0)
    return m


@pytest.fixture
def small_inputs():
    torch.manual_seed(0)
    B, H, N, A, T = 3, 4, 16, 4, 3
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    return dict(
        state_history=torch.randn(B, T, N),
        rainfall=torch.randn(B, H),
        action_candidate=torch.rand(B, H, A),
        action_reference=torch.rand(B, H, A),
        action_dynamic_internal=torch.rand(B, H, A),
        action_hold_previous=torch.rand(B, H, A),
        edge_index=edge_index,
        node_static=torch.randn(N, 3).abs() + 0.1,
        action_node_map=(torch.rand(A, N) > 0.5).float(),
    )


def test_twin_branch_different_actions_different_outputs(small_model, small_inputs):
    out = small_model(**small_inputs)
    assert out["y_candidate"].shape == out["y_reference"].shape
    assert not torch.allclose(out["y_candidate"], out["y_reference"], atol=1e-6)
    assert out["delta"].std().item() > 0.0
    assert "y_dynamic_internal" in out and "y_hold_previous" in out
    assert "delta_di" in out


def test_twin_branch_same_actions_zero_delta(small_model, small_inputs):
    small_inputs["action_candidate"] = small_inputs["action_reference"].clone()
    out = small_model(**small_inputs)
    assert torch.allclose(out["y_candidate"], out["y_reference"], atol=1e-6)
    assert out["delta"].abs().max().item() < 1e-6


def test_twin_branch_action_sha_enters_output(small_model, small_inputs):
    out1 = small_model(**small_inputs)
    ac2 = small_inputs["action_candidate"].clone()
    ac2[:, 0, 0] = 1.0 - ac2[:, 0, 0]
    out2 = small_model(**dict(small_inputs, action_candidate=ac2))
    assert not torch.allclose(out1["y_candidate"], out2["y_candidate"], atol=1e-6)


def test_action_encoder_different_inputs_different_outputs():
    enc = ActuatorActionEncoder(n_facilities=4, hidden_dim=8, horizon=3)
    a = torch.zeros(2, 3, 4)
    b = torch.ones(2, 3, 4)
    node_map = (torch.rand(4, 8) > 0.5).float()
    out_a = enc(a, node_map)
    out_b = enc(b, node_map)
    assert out_a.shape == out_b.shape == (2, 3, 8, 8)
    assert not torch.allclose(out_a, out_b, atol=1e-6)


def test_action_encoder_zero_input_shape_is_stable():
    enc = ActuatorActionEncoder(n_facilities=4, hidden_dim=8, horizon=3)
    a = torch.zeros(2, 3, 4)
    node_map = (torch.rand(4, 8) > 0.5).float()
    assert enc(a, node_map).shape == (2, 3, 8, 8)


def test_actuator_injection_reaches_gru(small_model, small_inputs):
    out_full = small_model(**small_inputs)
    zero_map = dict(
        small_inputs,
        action_node_map=torch.zeros_like(small_inputs["action_node_map"]),
    )
    out_zero = small_model(**zero_map)
    assert not torch.allclose(out_full["y_candidate"], out_zero["y_candidate"], atol=1e-6)


def test_output_keys_are_distinct_tensors(small_model, small_inputs):
    out = small_model(**small_inputs)
    assert out["y_candidate"].data_ptr() != out["y_reference"].data_ptr()
    assert out["y_candidate"].data_ptr() != out["delta"].data_ptr()
    assert out["y_reference"].data_ptr() != out["delta"].data_ptr()
    assert out["y_dynamic_internal"].data_ptr() != out["y_candidate"].data_ptr()


def _kpi_model_inputs(B=8, H=4, N=16, A=4, T=3):
    torch.manual_seed(0)
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    base = TwinGraphDynamics(
        n_nodes=N,
        n_facilities=A,
        n_static_features=3,
        hidden_dim=8,
        gat_heads=2,
        n_gat_layers=1,
        horizon=H,
        history_frames=T,
    )
    model = TwinWithKPIHeads(base, hidden_dim=8)
    inputs = dict(
        state_history=torch.randn(B, T, N),
        rainfall=torch.randn(B, H),
        action_candidate=torch.rand(B, H, A),
        action_reference=torch.zeros(B, H, A),
        action_dynamic_internal=torch.rand(B, H, A),
        action_hold_previous=torch.rand(B, H, A),
        edge_index=edge_index,
        node_static=torch.randn(N, 3).abs() + 0.1,
        action_node_map=(torch.rand(A, N) > 0.5).float(),
    )
    return model, inputs


def test_kpi_head_not_constant_across_samples():
    model, inputs = _kpi_model_inputs()
    model.eval()
    with torch.no_grad():
        pred = model(**inputs)
    assert pred["pfv_delta"].std().item() > 1e-3
    assert pred["tfv_delta"].std().item() > 1e-4
    assert pred["peak_delta"].std().item() > 1e-4


def test_pfv_kpi_head_is_constant_when_candidate_equals_nc_for_identical_context():
    B, H, N, A, T = 4, 4, 16, 4, 3
    model, inputs = _kpi_model_inputs(B, H, N, A, T)
    model.eval()
    # Make all context and the Candidate/NC action identical across samples.
    for key in ("state_history", "rainfall", "action_reference"):
        inputs[key][1:] = inputs[key][0]
    inputs["action_candidate"] = inputs["action_reference"].clone()
    with torch.no_grad():
        pred = model(**inputs)
    assert pred["pfv_delta"].std().item() < 1e-5
