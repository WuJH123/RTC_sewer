import pytest

from sewerrtc.v4.pipeline_v4_closed_loop import (
    CLOSED_LOOP_ABLATIONS,
    predictive_gate_authorizes_closed_loop,
)
from sewerrtc.v4.pipeline import ALL_STAGES, PREREQUISITES


def test_closed_loop_requires_the_v41_predictive_gate_pass() -> None:
    assert predictive_gate_authorizes_closed_loop(
        {"status": "pass", "authorizes_closed_loop": True}
    )
    assert not predictive_gate_authorizes_closed_loop(
        {"status": "pass", "authorizes_closed_loop": False}
    )
    assert not predictive_gate_authorizes_closed_loop(
        {"status": "scientific_fail", "authorizes_closed_loop": False}
    )


def test_closed_loop_ablation_matrix_is_complete_and_disjoint() -> None:
    assert CLOSED_LOOP_ABLATIONS == {
        "A": ("true_state", "exact_evaluation"),
        "B": ("true_state", "v4_surrogate"),
        "C": ("gat_state", "exact_evaluation"),
        "D": ("gat_state", "v4_surrogate"),
    }


def test_exact_closed_loop_is_released_by_v41_not_the_old_locked_stage() -> None:
    assert "AuditGATClosedLoopReadiness" in ALL_STAGES
    assert PREREQUISITES["PlanExactClosedLoop"] == (
        "AuditV4PredictiveGeneralizationGateV1",
        "AuditGATClosedLoopReadiness",
    )
