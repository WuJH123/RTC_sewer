"""Target/schema and PFV-first lexicographic ranking regression tests."""
from __future__ import annotations

import torch

from sewerrtc.v4.models_v42.trajectory_losses import TrajectoryLosses
from sewerrtc.v4.models_v42.ranking_losses import RankingLosses


def _full_branch_target(B: int, H: int, N: int):
    return dict(
        depth_candidate=torch.randn(B, H, N),
        depth_reference=torch.randn(B, H, N),
        depth_dynamic_internal=torch.randn(B, H, N),
        depth_hold_previous=torch.randn(B, H, N),
        pfv_delta=torch.randn(B),
        tfv_delta=torch.randn(B),
        peak_delta=torch.randn(B),
    )


def test_trajectory_loss_reads_canonical_delta_targets():
    loss = TrajectoryLosses()
    B, H, N = 4, 12, 8
    target = _full_branch_target(B, H, N)
    pred = dict(
        y_candidate=torch.randn(B, H, N),
        y_reference=torch.randn(B, H, N),
        y_dynamic_internal=torch.randn(B, H, N),
        y_hold_previous=torch.randn(B, H, N),
        delta=torch.randn(B, H, N),
        delta_di=torch.randn(B, H, N),
        pfv_delta=torch.randn(B),
        tfv_delta=torch.randn(B),
        peak_delta=torch.randn(B),
    )
    out = loss(pred, target)
    for key in ("pfv_kpi", "tfv_kpi", "peak_kpi"):
        assert key in out and torch.isfinite(out[key])


def test_trajectory_loss_zero_when_pred_equals_target():
    loss = TrajectoryLosses(pfv_dead_zone=0.0, tfv_dead_zone=0.0, peak_dead_zone=0.0)
    B, H, N = 2, 4, 4
    target = _full_branch_target(B, H, N)
    pred = dict(
        y_candidate=target["depth_candidate"].clone(),
        y_reference=target["depth_reference"].clone(),
        y_dynamic_internal=target["depth_dynamic_internal"].clone(),
        y_hold_previous=target["depth_hold_previous"].clone(),
        delta=target["depth_candidate"] - target["depth_reference"],
        delta_di=target["depth_candidate"] - target["depth_dynamic_internal"],
        pfv_delta=target["pfv_delta"].clone(),
        tfv_delta=target["tfv_delta"].clone(),
        peak_delta=target["peak_delta"].clone(),
    )
    out = loss(pred, target)
    assert out["depth_trajectory"].item() < 1e-7
    assert out["delta_trajectory"].item() < 1e-7
    assert out["pfv_kpi"].item() < 1e-7
    assert out["tfv_kpi"].item() < 1e-7
    assert out["peak_kpi"].item() < 1e-7


def test_peak_schema_model_output_matches_loss_input():
    from sewerrtc.v4.models_v42.counterfactual_twin_dynamics import TwinGraphDynamics
    from sewerrtc.v4.v42_trainer import TwinWithKPIHeads

    torch.manual_seed(0)
    B, H, N, A, T = 2, 4, 8, 3, 2
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    base = TwinGraphDynamics(
        n_nodes=N,
        n_facilities=A,
        n_static_features=3,
        hidden_dim=4,
        gat_heads=1,
        n_gat_layers=1,
        horizon=H,
        history_frames=T,
    )
    model = TwinWithKPIHeads(base, hidden_dim=4)
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
    pred = model(**inputs)
    assert "peak_delta" in pred
    assert "peak_flood_rate" not in pred
    assert pred["peak_delta"].shape == (B,)


def _ranking_target(pfv_safe, peak_safe, tfv_true, state_groups=None):
    n = len(pfv_safe)
    if state_groups is None:
        state_groups = [0] * n
    return {
        "state_group_index": torch.tensor(state_groups, dtype=torch.long),
        "pfv_safe_label": torch.tensor(pfv_safe, dtype=torch.float32),
        "peak_noninferior_label": torch.tensor(peak_safe, dtype=torch.float32),
        "tfv_delta": torch.tensor(tfv_true, dtype=torch.float32),
        "pfv_boundary_norm": torch.zeros(n),
        "peak_boundary_norm": torch.zeros(n),
    }


def test_ranking_direction_pfv_safe_beats_unsafe():
    loss = RankingLosses(margin=0.1)
    target = _ranking_target([1, 0], [1, 1], [-1.0, -100.0])
    correct = {
        "pfv_delta": torch.tensor([-1.0, 1.0]),
        "peak_delta": torch.zeros(2),
        "tfv_delta": torch.tensor([10.0, -100.0]),
    }
    wrong = {
        "pfv_delta": torch.tensor([1.0, -1.0]),
        "peak_delta": torch.zeros(2),
        "tfv_delta": torch.tensor([-100.0, 10.0]),
    }
    assert loss(correct, target)["pairwise_ranking"] < loss(wrong, target)["pairwise_ranking"]


def test_ranking_peak_safety_dominates_tfv_inside_pfv_safe_set():
    loss = RankingLosses(margin=0.0)
    target = _ranking_target([1, 1], [1, 0], [100.0, -1000.0])
    correct = {
        "pfv_delta": torch.tensor([-1.0, -1.0]),
        "peak_delta": torch.tensor([-1.0, 1.0]),
        "tfv_delta": torch.tensor([100.0, -1000.0]),
    }
    wrong = {
        "pfv_delta": torch.tensor([-1.0, -1.0]),
        "peak_delta": torch.tensor([1.0, -1.0]),
        "tfv_delta": torch.tensor([-1000.0, 100.0]),
    }
    assert loss(correct, target)["pairwise_ranking"] < loss(wrong, target)["pairwise_ranking"]


def test_ranking_tfv_only_inside_joint_safe_set_and_has_gradient():
    loss = RankingLosses(margin=0.0)
    target = _ranking_target([1, 1], [1, 1], [-5.0, 2.0])
    pfv = torch.tensor([-0.2, -0.2], requires_grad=True)
    peak = torch.tensor([-0.1, -0.1], requires_grad=True)
    tfv = torch.tensor([-1.0, 1.0], requires_grad=True)
    out = loss({"pfv_delta": pfv, "peak_delta": peak, "tfv_delta": tfv}, target)
    out["pairwise_ranking"].backward()
    assert out["valid_pair_count"].item() == 1
    assert tfv.grad is not None and tfv.grad.abs().sum().item() > 0.0


def test_ranking_never_forms_cross_state_pairs():
    loss = RankingLosses(margin=0.0)
    target = _ranking_target([1, 0], [1, 0], [-5.0, 2.0], state_groups=[0, 1])
    pred = {
        "pfv_delta": torch.tensor([-1.0, 1.0], requires_grad=True),
        "peak_delta": torch.tensor([-1.0, 1.0], requires_grad=True),
        "tfv_delta": torch.tensor([-1.0, 1.0], requires_grad=True),
    }
    out = loss(pred, target)
    assert out["valid_pair_count"].item() == 0
    assert out["pairwise_ranking"].item() == 0.0
