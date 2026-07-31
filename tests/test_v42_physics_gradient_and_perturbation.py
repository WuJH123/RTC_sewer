"""Physics loss perturbation + gradient connectivity.

Spec §17 items:
    test_v42_physics_gradient.py
    test_v42_physics_perturbation.py

These tests verify:
  * every physics loss propagates gradient to the trajectory prediction;
  * artificially violating a physics constraint produces a *larger* loss
    than the unviolated prediction (perturbation test);
  * when no violation is present the loss may be zero but must still be
    *connected* (i.e. gradient is non-zero once a violation is introduced).
"""
from __future__ import annotations

import torch

from sewerrtc.v4.models_v42.counterfactual_twin_dynamics import TwinGraphDynamics
from sewerrtc.v4.models_v42.physics_losses import PhysicsLosses


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model_and_inputs():
    torch.manual_seed(0)
    B, H, N, A, T = 2, 4, 16, 3, 3
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    model = TwinGraphDynamics(
        n_nodes=N, n_facilities=A, n_static_features=3,
        hidden_dim=16, gat_heads=2, n_gat_layers=1, horizon=H, history_frames=T,
    )
    # Bias depth head so untrained model produces non-zero depths.
    with torch.no_grad():
        for layer in model.depth_head:
            if isinstance(layer, torch.nn.Linear):
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
    node_max_depth = torch.full((N,), 5.0)
    return model, inputs, node_max_depth


# ---------------------------------------------------------------------------
# test_v42_physics_gradient
# ---------------------------------------------------------------------------

def test_physics_losses_all_return_tensor():
    model, inputs, node_max_depth = _make_model_and_inputs()
    pred = model(**inputs)
    loss = PhysicsLosses(n_nodes=model.n_nodes, node_max_depth=node_max_depth)
    out = loss(pred, edge_index=inputs["edge_index"])
    expected = {
        "mass_balance", "storage_continuity", "non_negative",
        "capacity_bounds", "flooding_consistency", "shared_init_state",
        "kpi_trajectory_consistency", "peak_consistency",
    }
    assert expected.issubset(out.keys()), f"missing keys: {expected - set(out)}"
    for k, v in out.items():
        assert torch.isfinite(v), f"{k} is not finite"


def test_physics_gradient_reaches_prediction():
    """Every physics loss must have a non-zero gradient w.r.t. y_candidate."""
    torch.manual_seed(0)
    B, H, N = 2, 4, 8
    # Build a synthetic prediction with explicit gradient tracking.  We do
    # not need the full model here — the goal is to verify that each loss
    # term is *connected* to the trajectory prediction, not that the model
    # produces a specific output.  We use a *negative* node_max_depth so
    # the capacity_bounds loss is active (random.randn depths are > -1,
    # so ReLU(depth - (-1)) is always positive and has non-zero gradient).
    y_cand = torch.randn(B, H, N, requires_grad=True)
    y_ref = torch.randn(B, H, N, requires_grad=True)
    pred = dict(
        y_candidate=y_cand,
        y_reference=y_ref,
        delta=y_cand - y_ref,
        pfv_delta=(y_cand - y_ref).sum(dim=(1, 2)),
        tfv_delta=(y_cand - y_ref).mean(dim=(1, 2)),
        peak_flood_rate=y_cand.amax(dim=(1, 2)),
        pfv_rate_seq=y_cand.mean(dim=2),
        tfv_rate_seq=y_cand.mean(dim=2),
    )
    node_max_depth = torch.full((N,), -1.0)  # force capacity violation
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    loss = PhysicsLosses(n_nodes=N, node_max_depth=node_max_depth)
    out = loss(pred, edge_index=edge_index)
    for k, v in out.items():
        if y_cand.grad is not None:
            y_cand.grad.zero_()
        v.backward(retain_graph=True)
        grad = y_cand.grad
        assert grad is not None, f"{k} did not produce a gradient"
        assert grad.abs().sum().item() > 0.0, (
            f"{k} gradient w.r.t. y_candidate is identically zero — loss is "
            "disconnected from the trajectory prediction"
        )


# ---------------------------------------------------------------------------
# test_v42_physics_perturbation
# ---------------------------------------------------------------------------

def test_capacity_bounds_perturbation_increases_loss():
    """Pushing depth above max_depth must increase capacity_bounds loss."""
    model, inputs, node_max_depth = _make_model_and_inputs()
    pred = model(**inputs)
    loss = PhysicsLosses(n_nodes=model.n_nodes, node_max_depth=node_max_depth)
    base = loss(pred, edge_index=inputs["edge_index"])["capacity_bounds"].item()
    # Perturb: push all depths well above max_depth.  Build a fresh pred
    # dict (no deepcopy — intermediate tensors are non-leaf and cannot be
    # deepcopied).
    perturbed = dict(pred)
    perturbed["y_candidate"] = pred["y_candidate"].detach() + 100.0
    perturbed["y_reference"] = pred["y_reference"].detach() + 100.0
    perturbed["delta"] = perturbed["y_candidate"] - perturbed["y_reference"]
    after = loss(perturbed, edge_index=inputs["edge_index"])["capacity_bounds"].item()
    assert after > base, (
        f"capacity_bounds did not increase under capacity violation: {base} → {after}"
    )


def test_non_negative_perturbation_increases_loss():
    """Pushing depth below zero must increase non_negative loss."""
    model, inputs, node_max_depth = _make_model_and_inputs()
    pred = model(**inputs)
    loss = PhysicsLosses(n_nodes=model.n_nodes, node_max_depth=node_max_depth)
    base = loss(pred, edge_index=inputs["edge_index"])["non_negative"].item()
    perturbed = dict(pred)
    perturbed["y_candidate"] = pred["y_candidate"].detach() - 100.0
    perturbed["y_reference"] = pred["y_reference"].detach() - 100.0
    perturbed["delta"] = perturbed["y_candidate"] - perturbed["y_reference"]
    after = loss(perturbed, edge_index=inputs["edge_index"])["non_negative"].item()
    assert after > base, (
        f"non_negative did not increase under negative-depth violation: {base} → {after}"
    )


def test_shared_init_state_perturbation_increases_loss():
    """Making candidate and reference start from different states must increase loss."""
    model, inputs, node_max_depth = _make_model_and_inputs()
    pred = model(**inputs)
    loss = PhysicsLosses(n_nodes=model.n_nodes, node_max_depth=node_max_depth)
    base = loss(pred, edge_index=inputs["edge_index"])["shared_init_state"].item()
    perturbed = dict(pred)
    y_cand = pred["y_candidate"].detach().clone()
    y_cand[:, 0, :] += 50.0
    perturbed["y_candidate"] = y_cand
    after = loss(perturbed, edge_index=inputs["edge_index"])["shared_init_state"].item()
    assert after > base, (
        f"shared_init_state did not increase under init mismatch: {base} → {after}"
    )
