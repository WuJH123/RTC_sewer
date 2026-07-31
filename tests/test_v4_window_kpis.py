from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.simulation.kpi_metrics import compute_window_kpis


def test_compute_window_kpis_converts_rates_to_volumes_and_keeps_peak_rate() -> None:
    detail = pd.DataFrame(
        {
            "elapsed_min": [0.0, 5.0, 10.0],
            "flood:priority": [1.0, 2.0, 0.0],
            "flood:other": [2.0, 0.0, 1.0],
        }
    )

    result = compute_window_kpis(
        detail,
        priority_nodes=["priority"],
        start_min=0.0,
        horizon_min=10.0,
        dt_sec=300,
    )

    assert result["PFV"] == pytest.approx(600.0)
    assert result["TFV"] == pytest.approx(900.0)
    assert result["peak_TFV_rate"] == pytest.approx(2.0)
    assert result["steps"] == 2


def test_compute_window_kpis_uses_strict_post_checkpoint_closed_horizon() -> None:
    detail = pd.DataFrame(
        {
            "elapsed_min": [0.0, 5.0, 10.0],
            "flood:priority": [1.0, 1.0, 100.0],
        }
    )

    result = compute_window_kpis(
        detail,
        priority_nodes=["priority"],
        start_min=0.0,
        horizon_min=10.0,
        dt_sec=300,
    )

    assert result["PFV"] == pytest.approx(30300.0)
    assert result["peak_TFV_rate"] == pytest.approx(100.0)


def test_compute_window_kpis_rejects_nonpositive_dt() -> None:
    detail = pd.DataFrame({"elapsed_min": [0.0], "flood:a": [1.0]})

    with pytest.raises(ValueError, match="dt_sec"):
        compute_window_kpis(
            detail,
            priority_nodes=[],
            start_min=0.0,
            horizon_min=120.0,
            dt_sec=0,
        )
