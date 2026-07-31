"""Action sensitivity tests under the formal four-reference V4.2 contract."""
from __future__ import annotations

import torch
from torch import nn

from sewerrtc.v4.models_v42.counterfactual_twin_dynamics import TwinGraphDynamics
from sewerrtc.v4.v42_trainer import TwinWithKPIHeads


def _make_model_and_inputs():
    torch.manual_seed(0)
    B, H, N, A, T = 4, 6, 16, 3, 3
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    base = TwinGraphDynamics(
        n_nodes=N,
        n_facilities=A,
        n_static_features=3,
        hidden_dim=16,
        gat_heads=2,
        n_gat_layers=1,
        horizon=H,
        history_frames=T,
    )
    model = TwinWithKPIHeads(base, hidden_dim=16)
    with torch.no_grad():
        for layer in model.base.depth_head:
            if isinstance(layer, nn.Linear):
                layer.bias.fill_(1.0)
    inputs = dict(
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
    return model, inputs


def test_action_shuffle_changes_trajectory():
    model, inputs = _make_model_and_inputs()
    model.eval()
    with torch.no_grad():
        y_orig = model(**inputs)["y_candidate"].clone()
    shuffled = dict(inputs)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    shuffled["action_candidate"] = inputs["action_candidate"][:, perm, :]
    with torch.no_grad():
        y_shuf = model(**shuffled)["y_candidate"]
    assert not torch.allclose(y_orig, y_shuf, atol=1e-5), (
        "Candidate trajectory unchanged after action shuffle"
    )


def test_action_shuffle_changes_kpi():
    model, inputs = _make_model_and_inputs()
    model.eval()
    with torch.no_grad():
        original = model(**inputs)
        original_kpis = {k: original[k].clone() for k in ("pfv_delta", "tfv_delta", "peak_delta")}
    shuffled = dict(inputs)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    shuffled["action_candidate"] = inputs["action_candidate"][:, perm, :]
    with torch.no_grad():
        changed = model(**shuffled)
    assert any(
        not torch.allclose(original_kpis[k], changed[k], atol=1e-5)
        for k in original_kpis
    ), "All KPI heads are action-blind"


def test_zero_action_produces_different_output_from_nonzero():
    model, inputs = _make_model_and_inputs()
    model.eval()
    with torch.no_grad():
        y_normal = model(**inputs)["y_candidate"].clone()
    zeroed = dict(inputs)
    zeroed["action_candidate"] = torch.zeros_like(inputs["action_candidate"])
    with torch.no_grad():
        y_zero = model(**zeroed)["y_candidate"]
    assert not torch.allclose(y_normal, y_zero, atol=1e-5)


def test_action_gradient_reaches_encoder():
    model, inputs = _make_model_and_inputs()
    model.train()
    out = model(**inputs)
    loss = (
        out["y_candidate"].pow(2).mean()
        + out["pfv_delta"].pow(2).mean()
        + out["tfv_delta"].pow(2).mean()
        + out["peak_delta"].pow(2).mean()
    )
    loss.backward()
    grads = [
        p.grad.abs().sum().item()
        for p in model.base.action_encoder.parameters()
        if p.grad is not None
    ]
    assert grads and sum(grads) > 0.0


def test_different_actions_per_batch_produce_different_outputs():
    model, inputs = _make_model_and_inputs()
    model.eval()
    modified = dict(inputs)
    modified["state_history"] = inputs["state_history"].clone()
    modified["state_history"][1] = inputs["state_history"][0]
    modified["rainfall"] = inputs["rainfall"].clone()
    modified["rainfall"][1] = inputs["rainfall"][0]
    # Make NC/DI/Hold identical across the pair too, leaving Candidate action
    # as the only causal difference.
    for key in ("action_reference", "action_dynamic_internal", "action_hold_previous"):
        modified[key] = inputs[key].clone()
        modified[key][1] = inputs[key][0]
    with torch.no_grad():
        out = model(**modified)
    assert not torch.allclose(out["y_candidate"][0], out["y_candidate"][1], atol=1e-5)
