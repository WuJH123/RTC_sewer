from __future__ import annotations

import pandas as pd

from sewerrtc.control.v4_action_authority import classify_action_authority


def _frame(action: float, flow: float, flood: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "elapsed_min": [60.0, 65.0, 70.0],
            "requested_setting:a": [action] * 3,
            "a:a": [action] * 3,
            "flow:a": [flow] * 3,
            "storage_volume:s": [1.0] * 3,
            "flood:p": [flood] * 3,
        }
    )


def test_classifies_requested_difference_that_collapses_after_readback_as_a() -> None:
    reference = _frame(action=0.0, flow=1.0)
    candidate = _frame(action=0.0, flow=1.0)
    candidate["requested_setting:a"] = 1.0

    report = classify_action_authority(reference, candidate, ["a"])

    assert report.authority_class == "A_requested_diff_actual_equal"
    assert not report.command_realized


def test_classifies_realized_action_on_zero_flow_facility_as_b() -> None:
    reference = _frame(action=0.0, flow=0.0)
    candidate = _frame(action=1.0, flow=0.0)

    report = classify_action_authority(reference, candidate, ["a"])

    assert report.authority_class == "B_realized_no_hydraulic_opportunity"
    assert report.command_realized
    assert not report.locally_responsive


def test_classifies_local_response_with_unchanged_kpi_as_c() -> None:
    reference = _frame(action=0.0, flow=1.0)
    candidate = _frame(action=1.0, flow=2.0)

    report = classify_action_authority(reference, candidate, ["a"])

    assert report.authority_class == "C_local_response_kpi_flat"
    assert report.locally_responsive
    assert not report.kpi_responsive


def test_classifies_realized_action_with_flat_hydraulics_as_d() -> None:
    reference = _frame(action=0.0, flow=1.0)
    candidate = _frame(action=1.0, flow=1.0)

    report = classify_action_authority(reference, candidate, ["a"])

    assert report.authority_class == "D_realized_hydraulically_flat"
    assert report.command_realized
    assert not report.locally_responsive


def test_classifies_kpi_response_as_e() -> None:
    reference = _frame(action=0.0, flow=1.0, flood=0.0)
    candidate = _frame(action=1.0, flow=2.0, flood=0.5)

    report = classify_action_authority(reference, candidate, ["a"])

    assert report.authority_class == "E_kpi_responsive"
    assert report.kpi_responsive
