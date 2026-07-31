"""P0/P1 target keys, peak schema, ranking direction, ranking pairs.

Spec §17 items:
    test_v42_target_keys.py
    test_v42_peak_schema.py
    test_v42_ranking_direction.py
    test_v42_ranking_pairs.py

These tests verify:
  * trajectory/KPI/physics losses read the correct target keys;
  * peak uses the same key in model output, loss, and metric;
  * ranking loss has the correct direction (better → lower pfv_delta);
  * ranking loss has non-zero gradient on both sides of the optimum;
  * valid_pairs=0 is handled (loss=0 but caller must fail-closed).
"""
from __future__ import annotations

import torch

from sewerrtc.v4.models_v42.trajectory_losses import TrajectoryLosses
from sewerrtc.v4.models_v42.ranking_losses import RankingLosses


# ---------------------------------------------------------------------------
# test_v42_target_keys
# ---------------------------------------------------------------------------

def test_trajectory_loss_reads_pfv_delta_target():
    loss = TrajectoryLosses()
    B, H, N = 4, 12, 8
    pred = dict(
        y_candidate=torch.randn(B, H, N),
        y_reference=torch.randn(B, H, N),
        delta=torch.randn(B, H, N),
        pfv_delta=torch.randn(B),
        tfv_delta=torch.randn(B),
        peak_flood_rate=torch.randn(B),
    )
    target = dict(
        depth_candidate=torch.randn(B, H, N),
        depth_reference=torch.randn(B, H, N),
        pfv_delta=torch.randn(B),
        tfv_delta=torch.randn(B),
        peak_delta=torch.randn(B),
    )
    out = loss(pred, target)
    # All three KPI losses must be present and finite.
    for k in ("pfv_kpi", "tfv_kpi", "peak_kpi"):
        assert k in out, f"{k} missing from trajectory loss output"
        assert torch.isfinite(out[k]), f"{k} is not finite"


def test_trajectory_loss_zero_when_pred_equals_target():
    loss = TrajectoryLosses()
    B, H, N = 2, 4, 4
    target = dict(
        depth_candidate=torch.randn(B, H, N),
        depth_reference=torch.randn(B, H, N),
        pfv_delta=torch.tensor([0.5, -0.5]),
        tfv_delta=torch.tensor([0.2, -0.2]),
        peak_delta=torch.tensor([0.1, -0.1]),
    )
    pred = dict(
        y_candidate=target["depth_candidate"],
        y_reference=target["depth_reference"],
        delta=target["depth_candidate"] - target["depth_reference"],
        pfv_delta=target["pfv_delta"],
        tfv_delta=target["tfv_delta"],
        peak_flood_rate=target["peak_delta"],
    )
    out = loss(pred, target)
    assert out["pfv_kpi"].item() < 1e-6
    assert out["tfv_kpi"].item() < 1e-6
    assert out["peak_kpi"].item() < 1e-6


# ---------------------------------------------------------------------------
# test_v42_peak_schema
# ---------------------------------------------------------------------------

def test_peak_schema_model_output_matches_loss_input():
    """Model outputs 'peak_flood_rate'; loss must read it under that key."""
    from sewerrtc.v4.models_v42.counterfactual_twin_dynamics import TwinGraphDynamics
    from sewerrtc.v4.v42_trainer import TwinWithKPIHeads

    torch.manual_seed(0)
    B, H, N, A, T = 2, 4, 8, 3, 2
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    base = TwinGraphDynamics(
        n_nodes=N, n_facilities=A, n_static_features=3,
        hidden_dim=4, gat_heads=1, n_gat_layers=1, horizon=H, history_frames=T,
    )
    model = TwinWithKPIHeads(base, hidden_dim=4)
    inputs = dict(
        state_history=torch.randn(B, T, N),
        rainfall=torch.randn(B, H),
        action_candidate=torch.rand(B, H, A),
        action_reference=torch.rand(B, H, A),
        edge_index=edge_index,
        node_static=torch.randn(N, 3).abs() + 0.1,
        action_node_map=(torch.rand(A, N) > 0.5).float(),
    )
    pred = model(**inputs)
    # The loss reads pred["peak_flood_rate"] — assert it exists and is a
    # 1-D tensor with one value per sample.
    assert "peak_flood_rate" in pred
    assert pred["peak_flood_rate"].shape == (B,)


# ---------------------------------------------------------------------------
# test_v42_ranking_direction
# ---------------------------------------------------------------------------

def test_ranking_direction_better_sample_lower_loss():
    """If A is truly better than B (lower pfv_delta), model predicting
    pfv_delta_A < pfv_delta_B must produce a *lower* ranking loss than
    the wrong ordering."""
    loss = RankingLosses(margin=0.1)
    # Ground truth: sample 0 is better (pfv_delta=-1) than sample 1 (pfv_delta=+1).
    target = dict(pfv_delta=torch.tensor([-1.0, 1.0]))
    # Correct prediction: sample 0 also has lower pfv_delta.
    pred_correct = dict(
        pfv_delta=torch.tensor([-0.5, 0.5]),
        tfv_delta=torch.zeros(2),
        peak_flood_rate=torch.zeros(2),
    )
    # Wrong prediction: sample 0 has *higher* pfv_delta than sample 1.
    pred_wrong = dict(
        pfv_delta=torch.tensor([0.5, -0.5]),
        tfv_delta=torch.zeros(2),
        peak_flood_rate=torch.zeros(2),
    )
    out_correct = loss(pred_correct, target)["pairwise_ranking"]
    out_wrong = loss(pred_wrong, target)["pairwise_ranking"]
    assert out_wrong.item() > out_correct.item(), (
        f"ranking loss has wrong direction: correct={out_correct.item():.4f}, "
        f"wrong={out_wrong.item():.4f}"
    )


def test_ranking_softplus_has_gradient_on_correct_side():
    """Softplus ranking loss must have non-zero gradient even when ranking is correct."""
    loss = RankingLosses(margin=0.1)
    target = dict(pfv_delta=torch.tensor([-1.0, 1.0]))
    pred = dict(
        pfv_delta=torch.tensor([-0.5, 0.5]),  # correct ordering
        tfv_delta=torch.zeros(2, requires_grad=False),
        peak_flood_rate=torch.zeros(2, requires_grad=False),
    )
    pred["pfv_delta"].requires_grad_(True)
    out = loss(pred, target)["pairwise_ranking"]
    out.backward()
    grad = pred["pfv_delta"].grad
    assert grad is not None
    # With softplus, gradient must be non-zero on the "correct" side.
    assert grad.abs().max().item() > 1e-6, (
        "ranking loss has zero gradient on correct side — training will stall"
    )


# ---------------------------------------------------------------------------
# test_v42_ranking_pairs
# ---------------------------------------------------------------------------

def test_ranking_zero_valid_pairs_returns_zero_loss():
    """When all gt deltas are within the dead zone, loss must be 0."""
    loss = RankingLosses(margin=0.1)
    target = dict(pfv_delta=torch.tensor([0.0, 0.01, -0.01]))  # all within dead zone
    pred = dict(
        pfv_delta=torch.tensor([0.5, -0.5, 0.0]),
        tfv_delta=torch.zeros(3),
        peak_flood_rate=torch.zeros(3),
    )
    out = loss(pred, target)["pairwise_ranking"]
    assert out.item() == 0.0


def test_ranking_loss_has_nonzero_gradient():
    """End-to-end: ranking loss must propagate gradient to pred['pfv_delta']."""
    loss = RankingLosses(margin=0.1)
    target = dict(pfv_delta=torch.tensor([-1.0, 1.0, 0.0]))
    pred = dict(
        pfv_delta=torch.tensor([0.0, 0.0, 0.0], requires_grad=True),
        tfv_delta=torch.zeros(3),
        peak_flood_rate=torch.zeros(3),
    )
    out = loss(pred, target)
    out["pairwise_ranking"].backward()
    assert pred["pfv_delta"].grad is not None
    assert pred["pfv_delta"].grad.abs().sum().item() > 0.0
