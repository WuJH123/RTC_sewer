from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from sewerrtc.control.pfvfirst_mpc_v42 import (
    EngineeringStatus,
    FrozenFallback,
    MPCandidate,
    MPCWeights,
    SafetyMargins,
    decide_pfvfirst_mpc,
)
from sewerrtc.state.state_contract import TEMPORAL_FRAME_OFFSETS_MIN
from sewerrtc.state.v42_sparse_state import build_sparse_state_estimate
from sewerrtc.v4.models_v42.hydraulic_multi_reference import (
    MultiReferenceHydraulicSurrogate,
)
from sewerrtc.v4.paper_workflow_v42 import (
    CONTRACT_ID,
    MODEL_LINE,
    PAPER_STAGE_ORDER,
    audit_paper_workflow,
    write_stage_evidence,
)


def test_state_contract_is_13_frames_at_5min():
    assert TEMPORAL_FRAME_OFFSETS_MIN == list(range(-60, 1, 5))
    assert len(TEMPORAL_FRAME_OFFSETS_MIN) == 13


class _MockFrozenGAT(nn.Module):
    def __init__(self, n_nodes: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, sparse_depth, sensor_mask, rain, node_static, edge_index):
        del rain, node_static, edge_index
        # deterministic signal at observed nodes + stochastic dropout for MC
        base = sparse_depth * sensor_mask + (1.0 - sensor_mask) * sparse_depth.mean(dim=1, keepdim=True)
        return torch.relu(self.dropout(base + 0.1))


def test_sparse_state_adapter_uses_physical_head_and_actions():
    torch.manual_seed(0)
    B, T, N, A = 2, 13, 5, 3
    gat = _MockFrozenGAT(N)
    depth_hist = torch.rand(B, T, N)
    mask = torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32)
    rain_hist = torch.rand(B, T)
    actions = torch.rand(B, T, A)
    invert = torch.arange(N, dtype=torch.float32) + 10.0
    max_depth = torch.ones(N) * 2.0
    storage_mask = torch.tensor([False, True, False, False, True])
    out = build_sparse_state_estimate(
        gat,
        sparse_depth_history=depth_hist,
        sensor_mask=mask,
        rainfall_history=rain_hist,
        historical_actions=actions,
        node_static=torch.rand(N, 4),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        node_invert_m=invert,
        node_max_depth_m=max_depth,
        storage_node_mask=storage_mask,
        mc_samples=4,
    )
    assert out.node_depth.shape == (B, N)
    assert out.node_uncertainty.shape == (B, N)
    assert torch.allclose(out.node_head, out.node_depth + invert[None, :])
    assert torch.allclose(out.node_filling_degree, out.node_depth / 2.0)
    assert torch.allclose(out.facility_current_setting, actions[:, -1])
    assert out.storage_depth.shape == (B, 2)
    assert out.storage_volume is None
    assert not out.storage_volume_available.any()
    assert out.metadata["role"] == "state_estimation_only_not_policy"


def test_sparse_state_rejects_legacy_7frame_input():
    gat = _MockFrozenGAT(4)
    with pytest.raises(ValueError, match="13"):
        build_sparse_state_estimate(
            gat,
            sparse_depth_history=torch.rand(1, 7, 4),
            sensor_mask=torch.ones(4),
            rainfall_history=torch.rand(1, 7),
            historical_actions=torch.rand(1, 7, 2),
            node_static=torch.rand(4, 3),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            node_invert_m=torch.zeros(4),
            node_max_depth_m=torch.ones(4),
            mc_samples=1,
        )


def _tiny_surrogate():
    torch.manual_seed(0)
    return MultiReferenceHydraulicSurrogate(
        n_nodes=6,
        n_facilities=2,
        state_feature_dim=1,
        static_feature_dim=3,
        hidden_dim=8,
        gat_heads=2,
        gat_layers=1,
        horizon=3,
        dt_sec=600.0,
        dropout=0.0,
    )


def test_multireference_surrogate_rolls_four_branches_and_derives_kpis():
    model = _tiny_surrogate()
    B, T, N, A, H = 2, 13, 6, 2, 3
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0, 1, 2, 3, 4]],
        dtype=torch.long,
    )
    action_map = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=torch.float32
    )
    out = model(
        state_history=torch.rand(B, T, N),
        historical_actions=torch.rand(B, T, A),
        rainfall_forecast=torch.rand(B, H),
        action_candidate=torch.rand(B, H, A),
        action_no_control=torch.ones(B, H, A),
        action_dynamic_internal=torch.rand(B, H, A),
        action_hold_previous=torch.rand(B, H, A),
        edge_index=edge_index,
        node_static=torch.rand(N, 3),
        action_node_map=action_map,
        priority_node_indices=torch.tensor([1, 4]),
        storage_node_indices=torch.tensor([2]),
        outfall_node_indices=torch.tensor([5]),
    )
    assert set(out["branches"]) == {
        "candidate",
        "no_control",
        "dynamic_internal",
        "hold_previous",
    }
    for branch in out["branches"].values():
        assert branch["node_depth"].shape == (B, H, N)
        assert branch["node_flooding_rate"].shape == (B, H, N)
        assert branch["system_flooding_rate"].shape == (B, H)
        assert branch["storage_volume"].shape == (B, H, 1)
        assert branch["facility_flow"].shape == (B, H, A)
        assert branch["outfall_flow"].shape == (B, H, 1)

    cand = out["branches"]["candidate"]["node_flooding_rate"]
    nc = out["branches"]["no_control"]["node_flooding_rate"]
    di = out["branches"]["dynamic_internal"]["node_flooding_rate"]
    expected_pfv = (
        cand[:, :, [1, 4]].sum(dim=(1, 2))
        - nc[:, :, [1, 4]].sum(dim=(1, 2))
    ) * 600.0
    expected_tfv = (cand.sum(dim=(1, 2)) - di.sum(dim=(1, 2))) * 600.0
    expected_peak = cand.sum(dim=2).max(dim=1).values - di.sum(dim=2).max(dim=1).values
    assert torch.allclose(out["pfv_delta"], expected_pfv)
    assert torch.allclose(out["tfv_delta"], expected_tfv)
    assert torch.allclose(out["peak_delta"], expected_peak)
    assert out["metadata"]["role"] == "hydraulic_surrogate_not_policy"
    assert out["metadata"]["kpis_derived_from_flooding_rate_trajectory"] is True


def _engineering(ok: bool = True) -> EngineeringStatus:
    return EngineeringStatus(ok, ok, ok, ok, ok)


def _candidate(
    cid: str,
    *,
    pfv: float = -1.0,
    peak: float = -0.1,
    tfv: float = -10.0,
    action_cost: float = 0.0,
    terminal_cost: float = 0.0,
    uncertainty_cost: float = 0.0,
    engineering: EngineeringStatus | None = None,
) -> MPCandidate:
    return MPCandidate(
        candidate_id=cid,
        action_sequence=np.zeros((12, 3), dtype=float),
        pfv_delta_ucb_m3=pfv,
        peak_delta_ucb_m3s=peak,
        tfv_delta_di_m3=tfv,
        action_cost=action_cost,
        terminal_cost=terminal_cost,
        uncertainty_cost=uncertainty_cost,
        changed_facilities=1,
        engineering=engineering or _engineering(True),
        uncertainty_pass=True,
        ood_pass=True,
        executable=True,
    )


def _fallback() -> FrozenFallback:
    return FrozenFallback(
        fallback_id="frozen_safe_fallback",
        action_sequence=np.ones((12, 3), dtype=float),
        contract_hash="abc123",
        legal=True,
    )


def test_mpc_safety_cannot_be_compensated_by_large_tfv_gain():
    unsafe = _candidate("unsafe", pfv=1.0, peak=-1.0, tfv=-1_000_000.0)
    safe = _candidate("safe", pfv=-0.1, peak=-0.01, tfv=10.0)
    decision = decide_pfvfirst_mpc(
        candidates=[unsafe, safe],
        fallback=_fallback(),
        margins=SafetyMargins(0.0, 0.0, 8),
        weights=MPCWeights(0.1, 0.1, 0.1),
        expected_fallback_contract_hash="abc123",
    )
    assert decision.selected_id == "safe"
    assert decision.used_fallback is False
    unsafe_audit = next(a for a in decision.audits if a.candidate_id == "unsafe")
    assert "pfv_safety_violation_vs_no_control" in unsafe_audit.rejection_reasons


def test_mpc_minimizes_tfv_only_inside_safe_set():
    a = _candidate("a", tfv=-20.0, action_cost=10.0)
    b = _candidate("b", tfv=-5.0, action_cost=0.0)
    decision = decide_pfvfirst_mpc(
        candidates=[a, b],
        fallback=_fallback(),
        weights=MPCWeights(action=0.1, terminal=0.0, uncertainty=0.0),
    )
    # a objective = -19, b = -5
    assert decision.selected_id == "a"
    assert decision.metadata["tfv_is_hard_safety_constraint"] is False


def test_mpc_empty_safe_set_uses_frozen_fallback():
    decision = decide_pfvfirst_mpc(
        candidates=[_candidate("bad", peak=0.1)],
        fallback=_fallback(),
        expected_fallback_contract_hash="abc123",
    )
    assert decision.used_fallback is True
    assert decision.selected_id == "frozen_safe_fallback"
    assert decision.reason == "safe_set_empty"


def _stage_payload(stage: str) -> dict:
    base = {
        "status": "pass",
        "development_evidence_substituted": False,
        "legacy_locked_evidence_substituted": False,
    }
    if stage == "true_state_offline_validation":
        base.update(
            state_source="true_state",
            four_reference_surrogate=True,
            trajectory_first_kpi_derivation=True,
        )
    elif stage == "exact_swmm_closed_loop":
        base.update(authoritative_engine="SWMM", online_future_hydraulic_truth_used=False)
    elif stage == "surrogate_closed_loop":
        base.update(surrogate_role="hydraulic_surrogate_not_policy", pfvfirst_mpc_v42=True)
    elif stage == "gat_integrated_closed_loop":
        base.update(state_source="gat_sparse_reconstruction", gat_uncertainty_used=True, ood_gate_used=True)
    elif stage == "policy_lock":
        base.update(
            policy_sha256="p",
            model_sha256="m",
            fallback_contract_sha256="f",
            post_lock_parameter_updates_allowed=False,
        )
    elif stage == "challenge":
        base.update(policy_locked_before_reveal=True, used_for_retraining=False)
    elif stage == "formal_blind":
        base.update(
            event_count=24,
            policy_locked_before_reveal=True,
            new_rainfall_sha_only=True,
            post_reveal_exclusion_used=False,
            used_for_retraining=False,
        )
    return base


def test_paper_workflow_rejects_legacy_evidence_and_enforces_order(tmp_path: Path):
    legacy = tmp_path / "v42_paper/true_state_offline/evidence.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "contract_id": "V41",
                "model_line": "v41_compact",
                "stage": "true_state_offline_validation",
                **_stage_payload("true_state_offline_validation"),
            }
        ),
        encoding="utf-8",
    )
    audit = audit_paper_workflow(tmp_path)
    assert audit.complete is False
    assert audit.next_stage == "true_state_offline_validation"
    assert "wrong_or_legacy_workflow_contract" in audit.stage_audits[0].reasons

    legacy.unlink()
    for stage in PAPER_STAGE_ORDER:
        write_stage_evidence(stage=stage, output_root=tmp_path, payload=_stage_payload(stage))
    final = audit_paper_workflow(tmp_path)
    assert final.complete is True
    assert final.passed_through == "formal_blind"


def test_paper_contract_ids_are_explicit():
    assert CONTRACT_ID == "PROJECT6_V42_PAPER_WORKFLOW_V1"
    assert MODEL_LINE == "v42_trajectory_first_multi_reference"
