from __future__ import annotations

import numpy as np

from sewerrtc.control.pfvfirst_mpc_v42 import MPCDecision
from sewerrtc.control.pfvfirst_mpc_v42_authoritative import (
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
    seq = np.zeros((12, 4), dtype=float)
    seq[0, [1, 3]] = 1.0
    candidate = build_authoritative_mpc_candidate(
        candidate_id="c",
        projected_action_sequence=seq,
        anchor_action=np.zeros(4),
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


def test_readback_audit_detects_write_or_K_mismatch():
    decision = MPCDecision(
        selected_id="c",
        execute_action=np.asarray([1.0, 0.0, 0.0]),
        selected_sequence=np.zeros((12, 3)),
        used_fallback=False,
        reason="test",
        objective=0.0,
        audits=(),
        metadata={},
    )
    ok = audit_executed_decision_readback(
        decision=decision,
        anchor_action=np.zeros(3),
        written_action=np.asarray([1.0, 0.0, 0.0]),
        readback_action=np.asarray([1.0, 0.0, 0.0]),
        max_changed_facilities=1,
    )
    assert ok.passed
    bad = audit_executed_decision_readback(
        decision=decision,
        anchor_action=np.zeros(3),
        written_action=np.asarray([1.0, 0.0, 0.0]),
        readback_action=np.asarray([1.0, 1.0, 0.0]),
        max_changed_facilities=1,
    )
    assert not bad.passed
    assert "written_action_differs_from_readback" in bad.reasons
    assert "readback_action_K_exceeded" in bad.reasons
