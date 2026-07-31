from __future__ import annotations

import pandas as pd

from sewerrtc.state.hotstart_acceleration import detect_forcing_replay_from_start, first_divergence


def test_forcing_from_start_replay_is_rejected() -> None:
    reference = pd.DataFrame({"elapsed_min": [65.0, 70.0], "rainfall_mm_h": [3.0, 0.0]})
    hotstart = pd.DataFrame({"elapsed_min": [65.0, 70.0], "rainfall_mm_h": [0.0, 3.0]})

    result = detect_forcing_replay_from_start(reference, hotstart)

    assert result["status"] == "failed_gate"
    assert result["reason"] == "first_future_forcing_value_mismatch"


def test_first_divergence_reports_object_and_first_step() -> None:
    reference = pd.DataFrame({"datetime": ["t1", "t2"], "h:A": [1.0, 1.0], "h:B": [2.0, 2.0]})
    hotstart = pd.DataFrame({"datetime": ["t1", "t2"], "h:A": [1.1, 1.0], "h:B": [2.0, 2.0]})

    result = first_divergence(reference, hotstart, "node_depth", tolerance=1.0e-6)

    assert result["first_divergence_object_id"] == "A"
    assert result["first_divergence_timestamp"] == "t1"
    assert result["divergence_from_first_step"] is True
