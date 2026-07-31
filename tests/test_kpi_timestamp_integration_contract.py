from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sewerrtc.evaluation.kpi_contract import compute_kpis, integrate_volume_from_rates


def test_integrates_with_real_timestamp_deltas_not_fixed_control_interval() -> None:
    ts = pd.to_datetime(
        [
            "2026-01-01 00:00:00",
            "2026-01-01 00:05:00",
            "2026-01-01 00:15:00",
            "2026-01-01 00:30:00",
        ]
    )
    rates = [1.0, 1.0, 1.0]
    assert integrate_volume_from_rates(rates, timestamps=ts) == pytest.approx(1800.0)


def test_formal_kpi_rejects_same_length_sample_timestamps() -> None:
    ts = pd.to_datetime(
        [
            "2026-01-01 00:00:00",
            "2026-01-01 00:10:00",
            "2026-01-01 00:20:00",
        ]
    )
    rates = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="n\\+1 boundaries"):
        integrate_volume_from_rates(rates, timestamps=ts)


def test_compute_kpis_uses_fixed_priority_and_full_node_universe() -> None:
    ts = pd.to_datetime(
        ["2026-01-01 00:00:00", "2026-01-01 00:10:00", "2026-01-01 00:20:00"]
    )
    priority_rates = np.asarray([[0.2], [0.2]])
    all_rates = np.asarray([[0.2, 0.1, 0.0], [0.2, 0.0, 0.0]])
    out = compute_kpis(
        all_flooding_rates=all_rates,
        priority_flooding_rates=priority_rates,
        timestamps=ts,
    )
    assert out["PFV"] == pytest.approx(240.0)
    assert out["TFV"] == pytest.approx(300.0)
    assert out["peak_TFV_rate"] == pytest.approx(0.3)
