from __future__ import annotations

from sewerrtc.data.candidate_prefilter import prefilter_candidate


def test_add350_bounds_pending_blocks_residual_override() -> None:
    keep, reason = prefilter_candidate({"add350_residual_override": True, "override_count": 1})

    assert keep is False
    assert reason == "variable_speed_bounds_unverified"


def test_k_above_8_is_prefiltered() -> None:
    keep, reason = prefilter_candidate({"override_count": 9})

    assert keep is False
    assert reason == "K_exceeded"
