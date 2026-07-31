from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn


def test_action_columns_are_reordered_by_engineering_id():
    from sewerrtc.v4.v42_trajectory_builder import _resolve_columns_by_ids

    df = pd.DataFrame(
        {
            "a:FacilityB": [20.0, 21.0],
            "a:FacilityA": [10.0, 11.0],
            "a:irrelevant": [99.0, 99.0],
        }
    )
    cols, arr = _resolve_columns_by_ids(df, "a:", ["FacilityA", "FacilityB"])
    assert cols == ["a:FacilityA", "a:FacilityB"]
    np.testing.assert_allclose(arr, [[10.0, 20.0], [11.0, 21.0]])


def test_node_depth_and_flood_columns_are_reordered_by_graph_id():
    from sewerrtc.v4.v42_trajectory_builder import _resolve_columns_by_ids

    df = pd.DataFrame(
        {
            "h:N2": [2.0],
            "h:N1": [1.0],
            "flood:N2": [20.0],
            "flood:N1": [10.0],
        }
    )
    _, depth = _resolve_columns_by_ids(df, "h:", ["N1", "N2"])
    _, flood = _resolve_columns_by_ids(df, "flood:", ["N1", "N2"])
    np.testing.assert_allclose(depth, [[1.0, 2.0]])
    np.testing.assert_allclose(flood, [[10.0, 20.0]])


def test_missing_reference_fails_closed():
    from sewerrtc.v4.v42_trajectory_builder import _require_reference_branches

    with pytest.raises(ValueError, match="missing required reference"):
        _require_reference_branches(
            {"no_control": {}, "hold_previous": {}}, case_id="case-x"
        )


def test_make_batch_indexes_all_optional_sample_tensors():
    from sewerrtc.v4.v42_trainer import _make_batch

    n = 5
    base = torch.arange(n, dtype=torch.float32)
    data = {
        "state_history": base[:, None, None].expand(n, 13, 1).clone(),
        "rainfall": base[:, None].expand(n, 12).clone(),
        "action_candidate": base[:, None, None].expand(n, 12, 36).clone(),
        "action_reference": (base + 10)[:, None, None].expand(n, 12, 36).clone(),
        "action_dynamic_internal": (base + 20)[:, None, None].expand(n, 12, 36).clone(),
        "action_hold_previous": (base + 30)[:, None, None].expand(n, 12, 36).clone(),
        "depth_candidate": base[:, None, None].expand(n, 12, 1).clone(),
        "depth_reference": (base + 10)[:, None, None].expand(n, 12, 1).clone(),
        "depth_dynamic_internal": (base + 20)[:, None, None].expand(n, 12, 1).clone(),
        "depth_hold_previous": (base + 30)[:, None, None].expand(n, 12, 1).clone(),
        "pfv_delta": base.clone(),
        "tfv_delta": base.clone(),
        "peak_delta": base.clone(),
        "pfv_safe_label": torch.ones(n),
        "tfv_improved_label": torch.ones(n),
        "peak_noninferior_label": torch.ones(n),
        "state_group_index": torch.arange(n),
    }
    batch = _make_batch(data, np.array([4, 1]))
    assert batch["action_dynamic_internal"].shape[0] == 2
    assert batch["action_hold_previous"].shape[0] == 2
    assert batch["action_dynamic_internal"][0, 0, 0].item() == 24.0
    assert batch["action_dynamic_internal"][1, 0, 0].item() == 21.0
    assert batch["depth_hold_previous"][0, 0, 0].item() == 34.0


def test_four_branch_forward_passes_di_and_hold():
    from sewerrtc.v4.v42_trainer import _forward_model

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.received = None

        def forward(self, **kwargs):
            self.received = kwargs
            x = kwargs["state_history"][:, :1, :]
            return {"y_candidate": x}

    model = Dummy()
    batch = {
        "state_history": torch.zeros(2, 13, 3),
        "rainfall": torch.zeros(2, 12),
        "action_candidate": torch.zeros(2, 12, 36),
        "action_reference": torch.zeros(2, 12, 36),
        "action_dynamic_internal": torch.ones(2, 12, 36) * 0.5,
        "action_hold_previous": torch.ones(2, 12, 36) * 0.25,
        "edge_index": torch.zeros(2, 0, dtype=torch.long),
        "node_static": torch.zeros(3, 7),
        "action_node_map": torch.zeros(36, 3),
    }
    _forward_model(model, batch)
    assert model.received is not None
    assert torch.allclose(model.received["action_dynamic_internal"], batch["action_dynamic_internal"])
    assert torch.allclose(model.received["action_hold_previous"], batch["action_hold_previous"])


def test_kpi_normalization_is_train_fold_local():
    from sewerrtc.v4.v42_trainer import compute_kpi_normalization_stats

    data = {
        "pfv_delta": torch.tensor([0.0, 1.0, 1000.0]),
        "tfv_delta": torch.tensor([0.0, 2.0, 2000.0]),
        "peak_delta": torch.tensor([0.0, 3.0, 3000.0]),
    }
    stats = compute_kpi_normalization_stats(data, train_idx=np.array([0, 1]))
    assert stats["pfv_delta"] == pytest.approx((0.5, 0.5))
    assert stats["tfv_delta"] == pytest.approx((1.0, 1.0))
    assert stats["peak_delta"] == pytest.approx((1.5, 1.5))


def test_graph_history_encoder_uses_early_frames(monkeypatch):
    import sewerrtc.v4.models_v42.graph_state_encoder as mod

    class FakeGAT(nn.Module):
        def __init__(self, in_channels, out_channels, heads=1, **kwargs):
            super().__init__()
            self.out_dim = out_channels * heads
            self.proj = nn.Linear(in_channels, self.out_dim, bias=False)

        def forward(self, x, edge_index):
            return self.proj(x)

    monkeypatch.setattr(mod, "GATConv", FakeGAT)
    torch.manual_seed(7)
    enc = mod.GraphStateEncoder(
        n_nodes=2,
        n_static_features=1,
        hidden_dim=8,
        gat_heads=2,
        n_gat_layers=1,
        dropout=0.0,
    )
    enc.eval()
    a = torch.zeros(1, 13, 2)
    b = a.clone()
    b[:, 0, :] = 1.0  # early history changes; checkpoint frame remains identical
    ha = enc.encode_history(a)
    hb = enc.encode_history(b)
    assert not torch.allclose(ha, hb)


def test_pseudo_physics_terms_are_disabled():
    from sewerrtc.v4.models_v42.physics_losses import PhysicsLosses

    y_c = torch.rand(2, 12, 3, requires_grad=True)
    y_r = torch.rand(2, 12, 3, requires_grad=True)
    losses = PhysicsLosses(3, node_max_depth=torch.ones(3) * 5.0)(
        {"y_candidate": y_c, "y_reference": y_r, "delta": y_c - y_r}
    )
    for key in (
        "mass_balance",
        "flooding_consistency",
        "shared_init_state",
        "kpi_trajectory_consistency",
        "peak_consistency",
    ):
        assert losses[key].item() == pytest.approx(0.0)


def _ranking_loss(pred_values):
    from sewerrtc.v4.models_v42.ranking_losses import RankingLosses

    pred = {k: torch.tensor(v, dtype=torch.float32, requires_grad=True) for k, v in pred_values.items()}
    target = {
        "state_group_index": torch.tensor([0, 0, 0]),
        "pfv_safe_label": torch.tensor([1.0, 0.0, 1.0]),
        "peak_noninferior_label": torch.tensor([1.0, 0.0, 0.0]),
        "pfv_delta": torch.tensor([-1.0, 2.0, -0.5]),
        "peak_delta": torch.tensor([-1.0, 2.0, 1.0]),
        "tfv_delta": torch.tensor([-5.0, 4.0, -1.0]),
        "pfv_boundary_norm": torch.zeros(3),
        "peak_boundary_norm": torch.zeros(3),
    }
    losses = RankingLosses()(pred, target)
    return losses


def test_lexicographic_ranking_rewards_pfv_then_peak_then_tfv():
    good = _ranking_loss(
        {
            "pfv_delta": [-1.0, 2.0, -0.5],
            "peak_delta": [-1.0, 2.0, 1.0],
            "tfv_delta": [-5.0, 4.0, -1.0],
        }
    )
    bad = _ranking_loss(
        {
            "pfv_delta": [2.0, -1.0, 1.0],
            "peak_delta": [2.0, -1.0, -2.0],
            "tfv_delta": [4.0, -5.0, -10.0],
        }
    )
    assert good["valid_pair_count"].item() > 0
    assert good["pairwise_ranking"].item() < bad["pairwise_ranking"].item()


def test_ranking_never_pairs_different_states():
    from sewerrtc.v4.models_v42.ranking_losses import RankingLosses

    pred = {
        "pfv_delta": torch.tensor([-1.0, 2.0], requires_grad=True),
        "peak_delta": torch.tensor([-1.0, 2.0], requires_grad=True),
        "tfv_delta": torch.tensor([-1.0, 2.0], requires_grad=True),
    }
    target = {
        "state_group_index": torch.tensor([0, 1]),
        "pfv_safe_label": torch.tensor([1.0, 0.0]),
        "peak_noninferior_label": torch.tensor([1.0, 0.0]),
        "tfv_delta": torch.tensor([-1.0, 2.0]),
        "pfv_boundary_norm": torch.zeros(2),
        "peak_boundary_norm": torch.zeros(2),
    }
    losses = RankingLosses()(pred, target)
    assert losses["valid_pair_count"].item() == 0
    assert losses["pairwise_ranking"].item() == pytest.approx(0.0)


def test_peak_delta_key_is_used_by_trajectory_loss():
    from sewerrtc.v4.models_v42.trajectory_losses import TrajectoryLosses

    pred = {
        "y_candidate": torch.zeros(2, 12, 3),
        "y_reference": torch.zeros(2, 12, 3),
        "y_dynamic_internal": torch.zeros(2, 12, 3),
        "y_hold_previous": torch.zeros(2, 12, 3),
        "delta": torch.zeros(2, 12, 3),
        "delta_di": torch.zeros(2, 12, 3),
        "pfv_delta": torch.zeros(2),
        "tfv_delta": torch.zeros(2),
        "peak_delta": torch.tensor([1.0, 2.0]),
    }
    target = {
        "depth_candidate": torch.zeros(2, 12, 3),
        "depth_reference": torch.zeros(2, 12, 3),
        "depth_dynamic_internal": torch.zeros(2, 12, 3),
        "depth_hold_previous": torch.zeros(2, 12, 3),
        "pfv_delta": torch.zeros(2),
        "tfv_delta": torch.zeros(2),
        "peak_delta": torch.tensor([1.0, 2.0]),
    }
    losses = TrajectoryLosses()(pred, target)
    assert losses["peak_kpi"].item() == pytest.approx(0.0)


def test_independent_oracle_uses_max_each_branch_then_difference():
    from sewerrtc.v4.v42_independent_oracle import recompute_row

    # Candidate total rates: [10, 5], DI totals: [0, 9].
    # Correct peak delta = 10 - 9 = 1, whereas max(C-DI) = 10.
    row = pd.Series(
        {
            "checkpoint_min": 0.0,
            "future_elapsed_min": json.dumps([10.0, 20.0]),
            "trajectory_flood_candidate": json.dumps([[10.0, 0.0], [5.0, 0.0]]),
            "trajectory_flood_no_control": json.dumps([[8.0, 0.0], [4.0, 0.0]]),
            "trajectory_flood_dynamic_internal": json.dumps([[0.0, 0.0], [9.0, 0.0]]),
        }
    )
    out = recompute_row(row, [0])
    assert out["recomputed_peak_delta_m3s"] == pytest.approx(1.0)


def test_pipeline_contract_history_is_13_frames():
    contract = json.loads(
        Path("docs/contracts/PROJECT6_V4_FINAL_PIPELINE_CONTRACT.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["history_min"] == 60
    assert contract["state_record_step_sec"] == 300
    assert contract["history_frames"] == 13
    assert contract["missing_reference_policy"] == "fail_closed_no_substitution"
    assert contract["normalization_policy"] == "fit_train_fold_only"
