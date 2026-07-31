from __future__ import annotations

from sewerrtc.data.candidate_prefilter import prefilter_candidate


def test_binary_pump_rejects_intermediate_requested_values() -> None:
    keep, reason = prefilter_candidate({"binary_pump_values": {"ADD301.3": 0.2}, "override_count": 1})

    assert keep is False
    assert reason == "binary_pump_intermediate_value"


def test_binary_pump_accepts_zero_and_one() -> None:
    keep, reason = prefilter_candidate({"binary_pump_values": {"ADD301.2": 0, "ADD301.3": 1}, "override_count": 2})

    assert keep is True
    assert reason == ""
