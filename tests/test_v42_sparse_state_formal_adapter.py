from __future__ import annotations

import torch

from sewerrtc.models.temporal_sparse_gat_v42 import TemporalSparseGATReconstructorV42
from sewerrtc.models.gat_reconstructor import SparseGATReconstructor
from sewerrtc.state.v42_sparse_state import build_sparse_state_estimate


def _common(B=2, T=13, N=6, A=2):
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0, 1, 2, 3, 4]],
        dtype=torch.long,
    )
    return dict(
        sparse_depth_history=torch.rand(B, T, N),
        sensor_mask=(torch.rand(B, T, N) > 0.5).float(),
        rainfall_history=torch.rand(B, T),
        historical_actions=torch.rand(B, T, A),
        node_static=torch.rand(N, 3),
        edge_index=edge_index,
        node_invert_m=torch.arange(N, dtype=torch.float32),
        node_max_depth_m=torch.ones(N) * 3.0,
        storage_node_mask=torch.tensor([False, True, False, False, True, False]),
    )


def test_formal_temporal_adapter_uses_full_contract_and_ood():
    torch.manual_seed(0)
    B, T, N, A, E = 2, 13, 6, 2, 10
    model = TemporalSparseGATReconstructorV42(
        n_nodes=N,
        n_facilities=A,
        node_static_dim=3,
        link_static_dim=2,
        hidden_dim=16,
        heads=2,
        gat_layers=1,
        dropout=0.0,
    )
    inputs = _common(B, T, N, A)
    inputs.update(
        link_static=torch.rand(E, 2),
        action_node_map=torch.tensor(
            [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=torch.float32
        ),
        ood_score=torch.tensor([0.1, 0.2]),
        mc_samples=1,
    )
    out = build_sparse_state_estimate(model, **inputs)
    assert out.metadata["reconstructor_contract"] == "formal_temporal_v42"
    assert out.metadata["formal_step1_input_contract_satisfied"] is True
    assert out.metadata["formal_online_state_eligible"] is True
    assert out.ood_available is True
    assert (out.node_uncertainty > 0).all()


def test_legacy_single_snapshot_gat_cannot_claim_formal_step1():
    torch.manual_seed(0)
    B, T, N, A = 2, 13, 6, 2
    model = SparseGATReconstructor(
        n_nodes=N,
        static_dim=3,
        hidden_dim=16,
        heads=2,
        dropout=0.0,
    )
    inputs = _common(B, T, N, A)
    inputs.update(ood_score=torch.tensor([0.1, 0.2]), mc_samples=1)
    out = build_sparse_state_estimate(model, **inputs)
    assert out.metadata["reconstructor_contract"] == "legacy_single_snapshot"
    assert out.metadata["formal_step1_input_contract_satisfied"] is False
    assert out.metadata["formal_online_state_eligible"] is False
