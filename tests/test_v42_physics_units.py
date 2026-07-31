"""Physics-unit consistency tests for V4.2.

PFV/TFV are volumes (m³) and Peak is a rate (m³/s).  Node depth exceedance is
not accepted as a PFV/TFV proxy.  Optional consistency penalties are only
activated when explicit quantities with matching physical units are supplied.
"""
from __future__ import annotations

import pytest
import torch

from sewerrtc.v4.models_v42.physics_losses import PhysicsLosses
from sewerrtc.v4.models_v42.trajectory_losses import TrajectoryLosses


def _base_pred(B=2, H=6, N=4):
    y_c = torch.rand(B, H, N)
    y_r = torch.rand(B, H, N)
    return {
        "y_candidate": y_c,
        "y_reference": y_r,
        "delta": y_c - y_r,
    }


def test_depth_proxy_does_not_activate_kpi_consistency():
    pred = _base_pred()
    pred.update(
        {
            "pfv_delta": torch.tensor([100.0, -50.0]),
            "tfv_delta": torch.tensor([1000.0, -500.0]),
            # Old pseudo-fields must not be interpreted as volume quantities.
            "pfv_rate_seq": pred["y_candidate"].mean(dim=2),
            "tfv_rate_seq": pred["y_candidate"].mean(dim=2),
        }
    )
    out = PhysicsLosses(n_nodes=4)(pred)
    assert out["kpi_trajectory_consistency"].item() == pytest.approx(0.0)


def test_explicit_m3_kpi_consistency_uses_matching_units():
    pred = _base_pred()
    pred.update(
        {
            "pfv_delta": torch.tensor([10.0, -3.0]),
            "tfv_delta": torch.tensor([100.0, -30.0]),
            "pfv_from_trajectory_m3": torch.tensor([10.0, -3.0]),
            "tfv_from_trajectory_m3": torch.tensor([100.0, -30.0]),
        }
    )
    out = PhysicsLosses(n_nodes=4)(pred)
    assert out["kpi_trajectory_consistency"].item() < 1e-7


def test_explicit_peak_rate_sequence_consistency():
    B, H, N = 2, 6, 4
    seq = torch.tensor(
        [[1.0, 3.0, 2.0, 5.0, 4.0, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
    )
    pred = _base_pred(B, H, N)
    pred["peak_rate_sequence_m3s"] = seq
    pred["peak_delta"] = torch.tensor([5.0, 0.6])
    out = PhysicsLosses(n_nodes=N)(pred)
    assert out["peak_consistency"].item() < 1e-7


def test_trajectory_loss_uses_peak_delta_key():
    loss = TrajectoryLosses(
        pfv_dead_zone=0.0, tfv_dead_zone=0.0, peak_dead_zone=0.0
    )
    B, H, N = 2, 4, 8
    zeros = torch.zeros(B, H, N)
    pred = {
        "y_candidate": zeros,
        "y_reference": zeros,
        "y_dynamic_internal": zeros,
        "y_hold_previous": zeros,
        "delta": zeros,
        "delta_di": zeros,
        "pfv_delta": torch.tensor([1.0, -1.0]),
        "tfv_delta": torch.tensor([0.5, -0.5]),
        "peak_delta": torch.tensor([0.1, -0.1]),
    }
    target = {
        "depth_candidate": zeros,
        "depth_reference": zeros,
        "depth_dynamic_internal": zeros,
        "depth_hold_previous": zeros,
        "pfv_delta": torch.tensor([1.0, -1.0]),
        "tfv_delta": torch.tensor([0.5, -0.5]),
        "peak_delta": torch.tensor([0.1, -0.1]),
    }
    out = loss(pred, target)
    for key in ("pfv_kpi", "tfv_kpi", "peak_kpi"):
        assert out[key].item() < 1e-7


def test_mass_balance_requires_explicit_physical_residual():
    pred = _base_pred()
    base = PhysicsLosses(n_nodes=4, dt_sec=600.0)(pred)["mass_balance"]
    assert base.item() == pytest.approx(0.0)

    # A caller may provide a true mass-balance residual in m³ derived from
    # predicted flow/storage/inflow quantities.  The loss then uses it directly
    # rather than inventing a flow from node-depth differences.
    pred["mass_balance_residual_m3"] = torch.tensor([[2.0, -3.0], [1.0, -4.0]])
    active = PhysicsLosses(n_nodes=4, dt_sec=300.0)(pred)["mass_balance"]
    assert active.item() == pytest.approx(2.5)
