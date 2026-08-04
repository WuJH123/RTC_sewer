from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.materialize_v42_step2_target_contract import _exact_future, _extract
from sewerrtc.control.pfvfirst_mpc_v42 import (
    EngineeringStatus,
    FrozenFallback,
    MPCandidate,
    MPCWeights,
    SafetyMargins,
    audit_candidate_safety,
    decide_pfvfirst_mpc,
    performance_objective,
)
from sewerrtc.v4.v42_reference_semantics import branch_equivalence, no_control_all_open


def _candidate(**updates) -> MPCandidate:
    base = dict(
        candidate_id="c",
        action_sequence=np.ones((12, 36), dtype=float),
        pfv_delta_ucb_m3=104.0,
        peak_delta_ucb_m3s=0.2,
        tfv_delta_di_m3=-300.0,
        action_cost=1.0,
        terminal_cost=0.0,
        uncertainty_cost=0.0,
        changed_facilities=2,
        engineering=EngineeringStatus(True, True, True, True, True),
        uncertainty_pass=True,
        ood_pass=True,
        executable=True,
        pfv_no_control_m3=100.0,
        priority_depth_ucb_m=(0.9, 1.0),
        priority_depth_limit_m=(1.0, 1.1),
    )
    base.update(updates)
    return MPCandidate(**base)


def test_pfv_budget_is_100_plus_five_percent_of_no_control() -> None:
    margins = SafetyMargins(require_priority_depth=True)
    assert margins.pfv_allowance_m3(100.0) == 105.0
    assert audit_candidate_safety(_candidate(pfv_delta_ucb_m3=105.0), margins=margins).safe
    audit = audit_candidate_safety(_candidate(pfv_delta_ucb_m3=105.0001), margins=margins)
    assert not audit.safe
    assert "pfv_noninferiority_budget_exceeded_vs_no_control" in audit.rejection_reasons


def test_priority_depth_is_hard_but_peak_is_performance_only() -> None:
    margins = SafetyMargins(require_priority_depth=True)
    depth_bad = audit_candidate_safety(
        _candidate(priority_depth_ucb_m=(1.01, 1.0)), margins=margins
    )
    assert not depth_bad.safe
    assert "priority_depth_safety_violation" in depth_bad.rejection_reasons
    peak_high = audit_candidate_safety(_candidate(peak_delta_ucb_m3s=99.0), margins=margins)
    assert peak_high.safe


def test_positive_peak_excess_is_penalized_without_becoming_hard_gate() -> None:
    no_peak = performance_objective(
        _candidate(peak_delta_ucb_m3s=-1.0), weights=MPCWeights(peak=600.0)
    )
    positive_peak = performance_objective(
        _candidate(peak_delta_ucb_m3s=1.0), weights=MPCWeights(peak=600.0)
    )
    assert positive_peak - no_peak == 600.0


def test_selector_minimizes_tfv_plus_peak_inside_safe_set() -> None:
    a = _candidate(candidate_id="a", tfv_delta_di_m3=-300.0, peak_delta_ucb_m3s=0.0)
    b = _candidate(candidate_id="b", tfv_delta_di_m3=-400.0, peak_delta_ucb_m3s=1.0)
    fallback = FrozenFallback("hold", np.ones((12, 36)), "hash")
    decision = decide_pfvfirst_mpc(
        candidates=[a, b],
        fallback=fallback,
        margins=SafetyMargins(require_priority_depth=True),
        weights=MPCWeights(peak=600.0, action=0.0, terminal=0.0, uncertainty=0.0),
    )
    assert decision.selected_id == "a"
    assert not decision.used_fallback


def test_reference_equivalence_and_all_open_no_control() -> None:
    all_open = np.ones((12, 36), dtype=float)
    hold = np.zeros((12, 36), dtype=float)
    actions = {
        "candidate": hold.copy(),
        "no_control": all_open,
        "dynamic_internal": hold.copy(),
        "hold_previous": hold.copy(),
    }
    assert no_control_all_open(all_open)
    result = branch_equivalence(actions)
    assert result["no_control_all_open_verified"]
    assert result["dynamic_internal_equals_hold_action"]
    assert "candidate==hold_previous" in result["action_equivalent_pairs"]
    assert result["unique_action_branch_count"] == 2


def test_same_action_does_not_imply_same_hydraulics() -> None:
    action = np.zeros((12, 36), dtype=float)
    actions = {name: action.copy() for name in ("candidate", "no_control", "dynamic_internal", "hold_previous")}
    depths = {name: np.zeros((12, 2)) for name in actions}
    floods = {name: np.zeros((12, 2)) for name in actions}
    depths["dynamic_internal"] = np.ones((12, 2))
    result = branch_equivalence(actions, depths=depths, floods=floods)
    assert "dynamic_internal==hold_previous" in result["action_equivalent_pairs"]
    assert "dynamic_internal==hold_previous" not in result["hydraulic_equivalent_pairs"]


def test_target_materializer_exact_h120_and_no_imputation() -> None:
    elapsed = np.arange(0.0, 241.0, 10.0)
    frame = pd.DataFrame(
        {
            "elapsed_min": elapsed,
            "storage_volume:S1": elapsed + 1.0,
            "flow:F1": elapsed + 2.0,
        }
    )
    future = _exact_future(frame, 120.0)
    assert future["elapsed_min"].tolist() == list(np.arange(130.0, 241.0, 10.0))
    storage, fraction = _extract(future, ["storage_volume:S1"], required=True)
    assert storage is not None and storage.shape == (12, 1)
    assert fraction == 1.0
    optional, fraction = _extract(future, ["outfall_flow:O1"], required=False)
    assert optional is None and fraction == 0.0
