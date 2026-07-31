from __future__ import annotations

import numpy as np
import pytest

from sewerrtc.control.pfvfirst_mpc_v42 import MPCDecision
from sewerrtc.control.pfvfirst_mpc_v42_authoritative import (
    FORMAL_FACILITY_COUNT,
    FORMAL_HORIZON_STEPS,
    ProjectionGuardEvidence,
    audit_executed_decision_readback,
    build_authoritative_mpc_candidate,
)


def _guard() -> ProjectionGuardEvidence:
    return ProjectionGuardEvidence(
        checks={
            "bounds": True,
            "rate": True,
            "ramp": True,
            "dwell": True,
            "interlock": True,
        },
        contract_sha256="guard-sha",
    )


def test_changed_facilities_are_derived_not_caller_supplied():
    seq = np.zeros((FORMAL_HORIZON_STEPS, FORMAL_FACILITY_COUNT), dtype=float)
    seq[0, [1, 3]] = 1.0
    candidate = build_authoritative_mpc_candidate(
        candidate_id="c",
        projected_action_sequence=seq,
        anchor_action=np.zeros(FORMAL_FACILITY_COUNT),
        guard_evidence=_guard(),
        pfv_delta_ucb_m3=-1.0,
        peak_delta_ucb_m3s=-0.1,
        tfv_delta_di_m3=-5.0,
        action_cost=0.0,
        terminal_cost=0.0,
        uncertainty_cost=0.0,
        uncertainty_pass=True,
        ood_pass=True,
        executable=True,
    )
    assert candidate.changed_facilities == 2
    assert candidate.engineering.passed
    assert candidate.metadata["changed_facilities_authority"].startswith("derived")
    assert candidate.action_sequence.shape == (12, 36)


def test_formal_candidate_rejects_non_h12_or_non_engineering36_shape():
    common = dict(
        candidate_id="c",
        anchor_action=np.zeros(FORMAL_FACILITY_COUNT),
        guard_evidence=_guard(),
        pfv_delta_ucb_m3=-1.0,
        peak_delta_ucb_m3s=-0.1,
        tfv_delta_di_m3=-5.0,
        action_cost=0.0,
        terminal_cost=0.0,
        uncertainty_cost=0.0,
        uncertainty_pass=True,
        ood_pass=True,
        executable=True,
    )
    with pytest.raises(ValueError):
        build_authoritative_mpc_candidate(
            projected_action_sequence=np.zeros((11, FORMAL_FACILITY_COUNT)),
            **common,
        )
    with pytest.raises(ValueError):
        build_authoritative_mpc_candidate(
            projected_action_sequence=np.zeros((FORMAL_HORIZON_STEPS, 35)),
            **common,
        )


def test_readback_audit_detects_write_or_K_mismatch():
    execute = np.zeros(FORMAL_FACILITY_COUNT)
    execute[0] = 1.0
    selected = np.zeros((FORMAL_HORIZON_STEPS, FORMAL_FACILITY_COUNT))
    selected[0] = execute
    decision = MPCDecision(
        selected_id="c",
        execute_action=execute,
        selected_sequence=selected,
        used_fallback=False,
        reason="test",
        objective=0.0,
        audits=(),
        metadata={},
    )
    ok = audit_executed_decision_readback(
        decision=decision,
        anchor_action=np.zeros(FORMAL_FACILITY_COUNT),
        written_action=execute.copy(),
        readback_action=execute.copy(),
        max_changed_facilities=1,
    )
    assert ok.passed

    bad_readback = execute.copy()
    bad_readback[1] = 1.0
    bad = audit_executed_decision_readback(
        decision=decision,
        anchor_action=np.zeros(FORMAL_FACILITY_COUNT),
        written_action=execute.copy(),
        readback_action=bad_readback,
        max_changed_facilities=1,
    )
    assert not bad.passed
    assert "written_action_differs_from_readback" in bad.reasons
    assert "readback_action_K_exceeded" in bad.reasons
