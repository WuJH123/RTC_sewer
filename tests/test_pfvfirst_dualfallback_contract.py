import numpy as np

from sewerrtc.control.pfvfirst_dualfallback import (
    CandidatePrediction,
    FallbackPrediction,
    decide_dualfallback,
    select_safe_fallback,
)


def _seq(value):
    return np.full((12, 36), value, dtype=float)


def test_select_safe_fallback_independent_of_candidate():
    passive = FallbackPrediction("passive", _seq(0.2), 10.0, 5.0, 2.0, 1.0, 1.0, True)
    internal = FallbackPrediction("internal", _seq(0.5), 1.0, 7.0, 3.0, 1.0, 1.0, True)
    assert select_safe_fallback([internal, passive]).fallback_id == "passive"


def test_candidate_must_improve_internal_and_fallback_pfv():
    fallback = FallbackPrediction("passive", _seq(0.2), 10.0, 5.0, 2.0, 1.0, 1.0, True)
    bad = CandidatePrediction("bad", _seq(0.3), 20.0, 0.0, -1.0, -1.0, True, True)
    decision = decide_dualfallback(fallbacks=[fallback], candidates=[bad], minimum_internal_improvement=10.0)
    assert decision.selected_candidate_id == "passive"


def test_execute_only_first_step_of_accepted_candidate():
    fallback = FallbackPrediction("passive", _seq(0.2), 10.0, 5.0, 2.0, 1.0, 1.0, True)
    candidate = CandidatePrediction("learned", _seq(0.7), 20.0, 5.0, -1.0, -1.0, True, True)
    decision = decide_dualfallback(fallbacks=[fallback], candidates=[candidate], minimum_internal_improvement=10.0)
    assert decision.selected_candidate_id == "learned"
    assert decision.execute_action.shape == (36,)
    assert np.allclose(decision.execute_action, 0.7)
