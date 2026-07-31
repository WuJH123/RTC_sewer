"""Physics-loss tests for the physically defensible V4.2 contract.

Unsupported pseudo-physics (depth-difference mass balance, t+10 branch equality,
depth-proxy flooding/KPI volume) must remain disabled.  Supported depth-domain
constraints must still be differentiable when active.
"""
from __future__ import annotations

import pytest
import torch

from sewerrtc.v4.models_v42.physics_losses import PhysicsLosses


def _pred(y_cand, y_ref):
    return {
        "y_candidate": y_cand,
        "y_reference": y_ref,
        "delta": y_cand - y_ref,
    }


def test_physics_losses_all_return_tensor():
    y_c = torch.rand(2, 4, 8, requires_grad=True)
    y_r = torch.rand(2, 4, 8, requires_grad=True)
    out = PhysicsLosses(8, node_max_depth=torch.ones(8) * 5.0)(_pred(y_c, y_r))
    expected = {
        "mass_balance",
        "storage_continuity",
        "non_negative",
        "capacity_bounds",
        "flooding_consistency",
        "shared_init_state",
        "kpi_trajectory_consistency",
        "peak_consistency",
    }
    assert expected.issubset(out)
    assert all(torch.isfinite(v) for v in out.values())


def test_supported_depth_losses_reach_prediction_when_active():
    B, H, N = 2, 4, 8
    # Negative and temporally curved values make non-negative and smoothness
    # losses active.  A physical max depth of 0.25 activates capacity too.
    y_c = torch.tensor(
        [[[ -1.0 + 0.7 * t + 0.2 * (t ** 2) for _ in range(N)] for t in range(H)]] * B,
        requires_grad=True,
    )
    y_r = torch.zeros(B, H, N, requires_grad=True)
    losses = PhysicsLosses(N, node_max_depth=torch.full((N,), 0.25))(_pred(y_c, y_r))
    for key in ("storage_continuity", "non_negative", "capacity_bounds"):
        if y_c.grad is not None:
            y_c.grad.zero_()
        losses[key].backward(retain_graph=True)
        assert y_c.grad is not None
        assert y_c.grad.abs().sum().item() > 0.0, f"{key} is disconnected"


def test_unsupported_pseudo_physics_are_graph_connected_zero():
    y_c = torch.rand(2, 4, 8, requires_grad=True)
    y_r = torch.rand(2, 4, 8, requires_grad=True)
    out = PhysicsLosses(8)(_pred(y_c, y_r))
    for key in (
        "mass_balance",
        "flooding_consistency",
        "shared_init_state",
        "kpi_trajectory_consistency",
        "peak_consistency",
    ):
        assert out[key].item() == pytest.approx(0.0)
        assert out[key].requires_grad


def test_capacity_bounds_perturbation_increases_loss():
    B, H, N = 2, 4, 8
    y_c = torch.ones(B, H, N)
    y_r = torch.ones(B, H, N)
    loss = PhysicsLosses(N, node_max_depth=torch.full((N,), 5.0))
    base = loss(_pred(y_c, y_r))["capacity_bounds"].item()
    after = loss(_pred(y_c + 100.0, y_r + 100.0))["capacity_bounds"].item()
    assert after > base


def test_non_negative_perturbation_increases_loss():
    B, H, N = 2, 4, 8
    y_c = torch.ones(B, H, N)
    y_r = torch.ones(B, H, N)
    loss = PhysicsLosses(N, node_max_depth=torch.full((N,), 5.0))
    base = loss(_pred(y_c, y_r))["non_negative"].item()
    after = loss(_pred(y_c - 100.0, y_r - 100.0))["non_negative"].item()
    assert after > base


def test_future_tplus10_divergence_is_not_penalized_as_shared_initial_state():
    """Branches share the checkpoint, not their first post-action t+10 state."""
    B, H, N = 2, 4, 8
    y_c = torch.ones(B, H, N)
    y_r = torch.ones(B, H, N)
    loss = PhysicsLosses(N)
    base = loss(_pred(y_c, y_r))["shared_init_state"].item()
    perturbed = y_c.clone()
    perturbed[:, 0, :] += 50.0
    after = loss(_pred(perturbed, y_r))["shared_init_state"].item()
    assert base == pytest.approx(0.0)
    assert after == pytest.approx(0.0)
