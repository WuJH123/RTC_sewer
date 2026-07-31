"""Physics units consistency tests.

Spec §17 items:
    test_v42_physics_units.py

These tests verify that:
  * PFV/TFV/Peak predictions and targets use consistent units;
  * trajectory-derived KPI integrals match hand-computed values;
  * the physics loss kpi_trajectory_consistency is commensurable
    (both sides in the same unit — depth metres);
  * peak is derived from the sequence max, not an independent scale.
"""
from __future__ import annotations

import torch

from sewerrtc.v4.models_v42.physics_losses import PhysicsLosses
from sewerrtc.v4.models_v42.trajectory_losses import TrajectoryLosses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hand_compute_pfv(depth: torch.Tensor, max_depth: torch.Tensor) -> torch.Tensor:
    """PFV = sum over nodes of relu(depth - max_depth), summed over time,
    mean over batch — all in depth-metre units (m)."""
    overflow = torch.relu(depth - max_depth[None, None, :])  # [B, H, N]
    return overflow.sum(dim=2).mean(dim=1)  # [B]  m


def _hand_compute_tfv(depth: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """TFV proxy = mean system depth per step, then mean over time — m."""
    sys_depth = depth.sum(dim=2) / n_nodes  # [B, H]
    return sys_depth.mean(dim=1)  # [B]  m


# ---------------------------------------------------------------------------
# test_v42_physics_units
# ---------------------------------------------------------------------------

def test_kpi_trajectory_consistency_uses_same_units():
    """The trajectory-derived PFV delta and the predicted pfv_delta must
    be in the same unit (depth metres).  We verify by constructing a
    prediction where pred_pfv = traj_pfv_c - traj_pfv_r exactly, and
    checking the loss is zero."""
    torch.manual_seed(42)
    B, H, N = 3, 6, 10
    max_depth = torch.full((N,), 2.0)
    y_cand = torch.rand(B, H, N) * 4.0  # some above capacity
    y_ref = torch.rand(B, H, N) * 4.0

    # Hand-compute trajectory PFV delta
    pfv_c = _hand_compute_pfv(y_cand, max_depth)
    pfv_r = _hand_compute_pfv(y_ref, max_depth)
    pfv_delta_gt = pfv_c - pfv_r  # [B]

    tfv_c = _hand_compute_tfv(y_cand, N)
    tfv_r = _hand_compute_tfv(y_ref, N)
    tfv_delta_gt = tfv_c - tfv_r

    pred = dict(
        y_candidate=y_cand,
        y_reference=y_ref,
        delta=y_cand - y_ref,
        pfv_delta=pfv_delta_gt,  # perfect prediction
        tfv_delta=tfv_delta_gt,
        peak_flood_rate=y_cand.amax(dim=(1, 2)),
        tfv_rate_seq=y_cand.mean(dim=2),
    )
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])

    loss = PhysicsLosses(n_nodes=N, node_max_depth=max_depth)
    out = loss(pred, edge_index=edge_index)

    # kpi_trajectory_consistency should be ~0 when prediction matches
    kpi_loss = out["kpi_trajectory_consistency"].item()
    assert kpi_loss < 1e-5, (
        f"kpi_trajectory_consistency = {kpi_loss:.4e} even when pfv/tfv "
        "predictions exactly match trajectory-derived values — unit mismatch?"
    )


def test_pfv_increases_when_depth_exceeds_capacity():
    """PFV must be 0 when all depths < max_depth, and positive otherwise."""
    torch.manual_seed(0)
    B, H, N = 2, 4, 8
    max_depth = torch.full((N,), 5.0)

    # Case 1: all depths below capacity
    y_safe = torch.rand(B, H, N) * 3.0  # max 3.0 < 5.0
    pfv_safe = _hand_compute_pfv(y_safe, max_depth)
    assert (pfv_safe == 0).all(), "PFV should be 0 when depth < max_depth"

    # Case 2: some depths above capacity
    y_unsafe = torch.rand(B, H, N) * 3.0 + 4.0  # range [4, 7], some > 5
    pfv_unsafe = _hand_compute_pfv(y_unsafe, max_depth)
    assert (pfv_unsafe > 0).all(), "PFV should be > 0 when depth > max_depth"


def test_peak_consistency_reads_sequence_max():
    """peak_consistency loss must be |max(sequence) - direct_peak|."""
    B, H, N = 2, 6, 4
    tfv_seq = torch.tensor([[1.0, 3.0, 2.0, 5.0, 4.0, 0.5],
                            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])  # [B, H]
    peak_direct = torch.tensor([5.0, 0.6])  # matches seq max

    pred = dict(
        y_candidate=torch.randn(B, H, N),
        y_reference=torch.randn(B, H, N),
        delta=torch.randn(B, H, N),
        pfv_delta=torch.zeros(B),
        tfv_delta=torch.zeros(B),
        peak_flood_rate=peak_direct,
        tfv_rate_seq=tfv_seq,
    )
    loss = PhysicsLosses(n_nodes=N)
    out = loss(pred)
    assert out["peak_consistency"].item() < 1e-6, (
        "peak_consistency should be ~0 when direct_peak = max(sequence)"
    )


def test_trajectory_loss_kpi_keys_match_model_output():
    """TrajectoryLosses must read pred['pfv_delta'], pred['tfv_delta'],
    pred['peak_flood_rate'] — verify with hand-computed errors."""
    loss = TrajectoryLosses(pfv_dead_zone=0.0, tfv_dead_zone=0.0,
                            peak_dead_zone=0.0)
    B = 2
    pred = dict(
        y_candidate=torch.zeros(B, 4, 8),
        y_reference=torch.zeros(B, 4, 8),
        delta=torch.zeros(B, 4, 8),
        pfv_delta=torch.tensor([1.0, -1.0]),
        tfv_delta=torch.tensor([0.5, -0.5]),
        peak_flood_rate=torch.tensor([0.1, -0.1]),
    )
    target = dict(
        depth_candidate=torch.zeros(B, 4, 8),
        depth_reference=torch.zeros(B, 4, 8),
        pfv_delta=torch.tensor([1.0, -1.0]),
        tfv_delta=torch.tensor([0.5, -0.5]),
        peak_delta=torch.tensor([0.1, -0.1]),
    )
    out = loss(pred, target)
    # With dead_zone=0 and perfect predictions, all KPI losses should be 0
    for k in ("pfv_kpi", "tfv_kpi", "peak_kpi"):
        assert out[k].item() < 1e-6, f"{k} should be ~0 for perfect prediction"


def test_physics_loss_dt_sec_scales_mass_balance():
    """Mass balance uses dt_sec to convert flow to volume.  Changing dt_sec
    must change the mass_balance loss value."""
    torch.manual_seed(0)
    B, H, N = 2, 4, 8
    y_cand = torch.randn(B, H, N).abs()
    y_ref = torch.randn(B, H, N).abs()
    pred = dict(
        y_candidate=y_cand, y_reference=y_ref, delta=y_cand - y_ref,
        pfv_delta=torch.zeros(B), tfv_delta=torch.zeros(B),
        peak_flood_rate=torch.zeros(B), tfv_rate_seq=torch.zeros(B, H),
    )
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])

    loss_600 = PhysicsLosses(n_nodes=N, dt_sec=600.0)
    loss_300 = PhysicsLosses(n_nodes=N, dt_sec=300.0)
    out_600 = loss_600(pred, edge_index=edge_index)["mass_balance"]
    out_300 = loss_300(pred, edge_index=edge_index)["mass_balance"]
    # They should differ (dt_sec affects the volume calculation)
    assert not torch.allclose(out_600, out_300), (
        "mass_balance should change with dt_sec"
    )
