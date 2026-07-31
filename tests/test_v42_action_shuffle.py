"""Action shuffle sensitivity tests.

Spec §17 items:
    test_v42_action_shuffle.py

These tests verify that:
  * shuffling the action sequence (breaking temporal correspondence)
    produces different model outputs;
  * the model does not ignore the action input entirely;
  * KPI predictions change when actions are permuted;
  * the action encoder is actually used (not dead code).
"""
from __future__ import annotations

import torch
from torch import nn

from sewerrtc.v4.models_v42.counterfactual_twin_dynamics import TwinGraphDynamics
from sewerrtc.v4.v42_trainer import TwinWithKPIHeads


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model_and_inputs():
    torch.manual_seed(0)
    B, H, N, A, T = 4, 6, 16, 3, 3
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    base = TwinGraphDynamics(
        n_nodes=N, n_facilities=A, n_static_features=3,
        hidden_dim=16, gat_heads=2, n_gat_layers=1, horizon=H, history_frames=T,
    )
    model = TwinWithKPIHeads(base, hidden_dim=16)
    # Bias depth head so untrained model produces non-zero outputs
    with torch.no_grad():
        for layer in model.base.depth_head:
            if isinstance(layer, nn.Linear):
                layer.bias.fill_(1.0)
    inputs = dict(
        state_history=torch.randn(B, T, N),
        rainfall=torch.randn(B, H),
        action_candidate=torch.rand(B, H, A),
        action_reference=torch.rand(B, H, A),
        edge_index=edge_index,
        node_static=torch.randn(N, 3).abs() + 0.1,
        action_node_map=(torch.rand(A, N) > 0.5).float(),
    )
    return model, inputs


# ---------------------------------------------------------------------------
# test_v42_action_shuffle
# ---------------------------------------------------------------------------

def test_action_shuffle_changes_trajectory():
    """Shuffling the action sequence across time steps must change the
    predicted trajectory — the model must be temporally sensitive to actions."""
    model, inputs = _make_model_and_inputs()
    model.eval()
    with torch.no_grad():
        out_orig = model(**inputs)
        y_orig = out_orig["y_candidate"].clone()

    # Shuffle actions along the horizon dimension
    shuffled = dict(inputs)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])  # non-identity permutation of H=6
    shuffled["action_candidate"] = inputs["action_candidate"][:, perm, :]
    shuffled["action_reference"] = inputs["action_reference"][:, perm, :]

    with torch.no_grad():
        out_shuf = model(**shuffled)
        y_shuf = out_shuf["y_candidate"]

    assert not torch.allclose(y_orig, y_shuf, atol=1e-5), (
        "Trajectory unchanged after action shuffle — model ignores action timing"
    )


def test_action_shuffle_changes_kpi():
    """Shuffling actions must change KPI predictions (pfv_delta, tfv_delta)."""
    model, inputs = _make_model_and_inputs()
    model.eval()
    with torch.no_grad():
        out_orig = model(**inputs)
        pfv_orig = out_orig["pfv_delta"].clone()

    shuffled = dict(inputs)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    shuffled["action_candidate"] = inputs["action_candidate"][:, perm, :]
    shuffled["action_reference"] = inputs["action_reference"][:, perm, :]

    with torch.no_grad():
        out_shuf = model(**shuffled)
        pfv_shuf = out_shuf["pfv_delta"]

    assert not torch.allclose(pfv_orig, pfv_shuf, atol=1e-5), (
        "KPI prediction unchanged after action shuffle — KPI head is "
        "action-blind"
    )


def test_zero_action_produces_different_output_from_nonzero():
    """Setting all actions to zero must produce different output from
    non-zero actions (the action encoder is not bypassed)."""
    model, inputs = _make_model_and_inputs()
    model.eval()
    with torch.no_grad():
        out_normal = model(**inputs)
        y_normal = out_normal["y_candidate"].clone()

    zeroed = dict(inputs)
    zeroed["action_candidate"] = torch.zeros_like(inputs["action_candidate"])
    zeroed["action_reference"] = torch.zeros_like(inputs["action_reference"])

    with torch.no_grad():
        out_zero = model(**zeroed)
        y_zero = out_zero["y_candidate"]

    assert not torch.allclose(y_normal, y_zero, atol=1e-5), (
        "Zero actions produce same output as non-zero actions — "
        "action encoder may be disconnected"
    )


def test_action_gradient_reaches_encoder():
    """Loss must propagate gradient back to the action encoder parameters."""
    model, inputs = _make_model_and_inputs()
    model.train()
    out = model(**inputs)
    loss = out["y_candidate"].pow(2).mean() + out["pfv_delta"].pow(2).mean()
    loss.backward()

    # Check action encoder gradients
    act_enc_grads = []
    for name, param in model.base.action_encoder.named_parameters():
        if param.grad is not None:
            act_enc_grads.append(param.grad.abs().sum().item())

    assert len(act_enc_grads) > 0, "No action encoder parameter has a gradient"
    assert sum(act_enc_grads) > 0.0, (
        "Action encoder gradients are all zero — action path is disconnected"
    )


def test_different_actions_per_batch_produce_different_outputs():
    """Samples with different action inputs must produce different outputs."""
    model, inputs = _make_model_and_inputs()
    model.eval()

    # Make sample 0 and sample 1 have identical state/rainfall but different actions
    modified = dict(inputs)
    modified["state_history"] = inputs["state_history"].clone()
    modified["state_history"][1] = inputs["state_history"][0]  # same state
    modified["rainfall"] = inputs["rainfall"].clone()
    modified["rainfall"][1] = inputs["rainfall"][0]  # same rainfall
    # But keep different actions (already random)

    with torch.no_grad():
        out = model(**modified)
        y0 = out["y_candidate"][0]
        y1 = out["y_candidate"][1]

    assert not torch.allclose(y0, y1, atol=1e-5), (
        "Same state + same rainfall + different actions → same output. "
        "Model ignores action input."
    )
