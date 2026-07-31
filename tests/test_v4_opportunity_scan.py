from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.control.v4_opportunity import scan_control_opportunities


def test_scan_ranks_flowing_checkpoint_above_flat_checkpoint() -> None:
    detail = pd.DataFrame(
        {
            "elapsed_min": [0.0, 10.0],
            "flow:a": [0.0, 2.0],
            "a:a": [0.0, 1.0],
            "flood:p": [0.0, 0.5],
            "storage_volume:s": [0.0, 3.0],
            "rainfall_mm_h": [0.0, 1.0],
            "future_truth_PFV": [999.0, 999.0],
        }
    )

    result = scan_control_opportunities(detail, ["a"])

    assert result.loc[1, "opportunity_score"] > result.loc[0, "opportunity_score"]
    assert result.loc[1, "opportunity_class"] == "responsive"
    assert "future_truth_PFV" not in result.columns


def test_scan_marks_zero_signal_checkpoint_flat() -> None:
    detail = pd.DataFrame(
        {
            "elapsed_min": [0.0],
            "flow:a": [0.0],
            "a:a": [0.0],
            "flood:p": [0.0],
            "storage_volume:s": [0.0],
            "rainfall_mm_h": [0.0],
        }
    )

    result = scan_control_opportunities(detail, ["a"])

    assert result.loc[0, "opportunity_class"] == "flat"


def test_opportunity_score_does_not_read_future_hydraulic_rows() -> None:
    first = {
        "elapsed_min": 0.0,
        "flow:a": 0.2,
        "a:a": 0.0,
        "flood:p": 0.0,
        "storage_volume:s": 10.0,
        "rainfall_mm_h": 1.0,
    }
    future_normal = dict(first)
    future_normal.update({"elapsed_min": 10.0})
    baseline = scan_control_opportunities(
        pd.DataFrame([first, future_normal]), ["a"]
    )
    future_hydraulic_extreme = dict(future_normal)
    future_hydraulic_extreme.update(
        {
            "flow:a": 1000.0,
            "flood:p": 1000.0,
            "storage_volume:s": 1.0e9,
        }
    )
    extended = scan_control_opportunities(
        pd.DataFrame([first, future_hydraulic_extreme]), ["a"]
    )

    assert extended.loc[0, "opportunity_score"] == baseline.loc[0, "opportunity_score"]


def test_v2_uses_only_future_rainfall_as_forecast_driver() -> None:
    rows = []
    for index in range(25):
        rows.append(
            {
                "elapsed_min": float(index * 5),
                "flow:a": 2.5,
                "a:a": 0.0,
                "flood:p": 0.0,
                "storage_volume:s": 80_000.0,
                "h:u": 22.0,
                "h:d": 0.0,
                "rainfall_mm_h": 30.0 if index >= 2 else 0.0,
            }
        )
    with_forecast = scan_control_opportunities(pd.DataFrame(rows), ["a"])
    no_forecast = scan_control_opportunities(
        pd.DataFrame([{**row, "rainfall_mm_h": 0.0} for row in rows]), ["a"]
    )

    assert with_forecast.loc[0, "forecast_rain_depth_120min_mm"] > 0.0
    assert with_forecast.loc[0, "forecast_rain_peak_120min_mm_h"] == 30.0
    assert (
        with_forecast.loc[0, "opportunity_score"]
        > no_forecast.loc[0, "opportunity_score"]
    )
    assert "future_hydraulic_truth" not in with_forecast.columns


def test_v3_uses_facility_head_difference_not_global_topographic_spread() -> None:
    detail = pd.DataFrame(
        {
            "elapsed_min": [0.0],
            "flow:a": [0.0],
            "a:a": [0.0],
            "h:up": [10.0],
            "h:down": [9.8],
            "h:remote_high": [100.0],
            "h:remote_low": [0.0],
            "flood:p": [0.0],
            "storage_volume:s": [0.0],
            "rainfall_mm_h": [0.0],
        }
    )

    result = scan_control_opportunities(
        detail, ["a"], facility_nodes={"a": ("up", "down")}
    )

    assert result.loc[0, "facility_head_difference_signal"] == pytest.approx(0.2)
    assert "head_spread_signal" not in result.columns


def test_v3_keeps_real_flow_opportunity_without_rain_or_flood() -> None:
    detail = pd.DataFrame(
        {
            "elapsed_min": [0.0],
            "flow:a": [2.0],
            "a:a": [0.0],
            "h:up": [10.0],
            "h:down": [9.0],
            "flood:p": [0.0],
            "storage_volume:s": [100.0],
            "rainfall_mm_h": [0.0],
        }
    )

    result = scan_control_opportunities(
        detail, ["a"], facility_nodes={"a": ("up", "down")}
    )

    assert result.loc[0, "opportunity_score"] > 0.0
    assert result.loc[0, "hydraulic_driver"] > 0.0
