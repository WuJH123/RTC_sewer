from __future__ import annotations

import numpy as np

from sewerrtc.control.pfvfirst_mpc_v42 import (
    EngineeringStatus,
    FrozenFallback,
    MPCandidate,
    MPCWeights,
    SafetyMargins,
    decide_pfvfirst_mpc,
)


def _candidate(
    candidate_id: str,
    *,
    pfv: float = 0.0,
    peak: float = 0.0,
    tfv: float = -1.0,
    changed: int = 1,
    engineering: EngineeringStatus | None = None,
    uncertainty: bool = True,
    ood: bool = True,
    executable: bool = True,
    no_control_pfv: float = 0.0,
    priority_depth=(0.5, 0.6),
    priority_limit=(1.0, 1.0),
) -> MPCandidate:
    return MPCandidate(
        candidate_id=candidate_id,
        action_sequence=np.full((12, 36), 0.5, dtype=float),
        pfv_delta_ucb_m3=pfv,
        peak_delta_ucb_m3s=peak,
        tfv_delta_di_m3=tfv,
        action_cost=0.0,
        terminal_cost=0.0,
        uncertainty_cost=0.0,
        changed_facilities=changed,
        engineering=engineering or EngineeringStatus(True, True, True, True, True),
        uncertainty_pass=uncertainty,
        ood_pass=ood,
        executable=executable,
        pfv_no_control_m3=no_control_pfv,
        priority_depth_ucb_m=tuple(priority_depth),
        priority_depth_limit_m=tuple(priority_limit),
    )


def _fallback() -> FrozenFallback:
    return FrozenFallback(
        fallback_id="hold",
        action_sequence=np.zeros((12, 36), dtype=float),
        contract_hash="frozen-hash",
    )


def test_pfv_budget_allows_100_plus_five_percent() -> None:
    decision = decide_pfvfirst_mpc(
        candidates=[_candidate("safe", pfv=109.0, no_control_pfv=200.0)],
        fallback=_fallback(),
        margins=SafetyMargins(require_priority_depth=True),
    )
    assert not decision.used_fallback
    assert decision.selected_id == "safe"

    rejected = decide_pfvfirst_mpc(
        candidates=[_candidate("unsafe", pfv=110.01, no_control_pfv=200.0)],
        fallback=_fallback(),
        margins=SafetyMargins(require_priority_depth=True),
    )
    assert rejected.used_fallback
    assert any(
        "pfv_noninferiority_budget_exceeded_vs_no_control" in audit.rejection_reasons
        for audit in rejected.audits
    )


def test_priority_depth_is_hard_constraint() -> None:
    decision = decide_pfvfirst_mpc(
        candidates=[
            _candidate(
                "too_deep",
                priority_depth=(1.01, 0.5),
                priority_limit=(1.0, 1.0),
            )
        ],
        fallback=_fallback(),
        margins=SafetyMargins(require_priority_depth=True),
    )
    assert decision.used_fallback
    assert any(
        "priority_depth_safety_violation" in audit.rejection_reasons
        for audit in decision.audits
    )


def test_peak_is_penalty_not_hard_gate() -> None:
    # B has lower TFV but a large peak excess. With a 600 s peak penalty A wins;
    # both remain admissible because Peak is no longer a hard gate.
    a = _candidate("a", tfv=-100.0, peak=0.0)
    b = _candidate("b", tfv=-500.0, peak=1.0)
    decision = decide_pfvfirst_mpc(
        candidates=[a, b],
        fallback=_fallback(),
        margins=SafetyMargins(require_priority_depth=True),
        weights=MPCWeights(peak=600.0, action=0.0, terminal=0.0, uncertainty=0.0),
    )
    assert not decision.used_fallback
    assert decision.selected_id == "a"
    assert all(
        "peak_safety_violation_vs_dynamic_internal" not in audit.rejection_reasons
        for audit in decision.audits
    )


def test_engineering_k_uncertainty_ood_and_executable_remain_hard() -> None:
    bad_engineering = EngineeringStatus(False, True, True, True, True)
    candidates = [
        _candidate("k", changed=9),
        _candidate("bounds", engineering=bad_engineering),
        _candidate("unc", uncertainty=False),
        _candidate("ood", ood=False),
        _candidate("exec", executable=False),
    ]
    decision = decide_pfvfirst_mpc(
        candidates=candidates,
        fallback=_fallback(),
        margins=SafetyMargins(require_priority_depth=True),
    )
    assert decision.used_fallback
    reasons = {reason for audit in decision.audits for reason in audit.rejection_reasons}
    assert "K_exceeded" in reasons
    assert "engineering_bounds_violation" in reasons
    assert "uncertainty_gate_failed" in reasons
    assert "ood_gate_failed" in reasons
    assert "candidate_not_executable" in reasons


def test_selects_minimum_tfv_objective_within_safe_set() -> None:
    decision = decide_pfvfirst_mpc(
        candidates=[
            _candidate("worse", tfv=-10.0),
            _candidate("better", tfv=-100.0),
        ],
        fallback=_fallback(),
        margins=SafetyMargins(require_priority_depth=True),
        weights=MPCWeights(peak=0.0, action=0.0, terminal=0.0, uncertainty=0.0),
    )
    assert decision.selected_id == "better"
    assert decision.reason == "minimum_tfv_objective_within_pfv_budget_and_depth_safe_set"


def test_empty_safe_set_executes_frozen_fallback() -> None:
    decision = decide_pfvfirst_mpc(
        candidates=[_candidate("unsafe", pfv=101.0, no_control_pfv=0.0)],
        fallback=_fallback(),
        margins=SafetyMargins(require_priority_depth=True),
    )
    assert decision.used_fallback
    assert decision.selected_id == "hold"
    assert decision.reason == "safe_set_empty"


def test_fallback_hash_can_be_pinned() -> None:
    decision = decide_pfvfirst_mpc(
        candidates=[],
        fallback=_fallback(),
        expected_fallback_contract_hash="frozen-hash",
    )
    assert decision.used_fallback
