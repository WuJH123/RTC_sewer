from __future__ import annotations

from sewerrtc.state.hotstart_acceleration import amortized_speedup


def test_amortized_speedup_reports_candidate_counts() -> None:
    result = amortized_speedup(prefix_sec=100.0, hotstart_load_sec=5.0, suffix_sec=20.0, replay_sec=120.0, candidate_counts=[1, 5])

    assert result["1_candidates_per_checkpoint_speedup"] > 0
    assert result["5_candidates_per_checkpoint_speedup"] > result["1_candidates_per_checkpoint_speedup"]
