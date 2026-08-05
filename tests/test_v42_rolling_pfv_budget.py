from __future__ import annotations

import numpy as np
import pytest

from sewerrtc.control.rolling_pfv_budget_v42 import RollingPfvBudgetState


def test_rolling_budget_includes_realised_prefix_and_does_not_reset() -> None:
    state = RollingPfvBudgetState(
        relative_margin_fraction=0.05,
        absolute_margin_m3=100.0,
    )
    first = state.update(
        candidate_interval_pfv_m3=80.0,
        no_control_interval_pfv_m3=40.0,
    )
    # Prefix metric = 80 - 1.05 * 40 = 38 m3, leaving 62 m3.
    assert first.realised_prefix_budget_metric_m3 == pytest.approx(38.0)
    assert first.remaining_future_allowance_m3 == pytest.approx(62.0)
    assert first.admits(np.asarray([61.9, 62.1])).tolist() == [True, False]

    second = first.update(
        candidate_interval_pfv_m3=20.0,
        no_control_interval_pfv_m3=10.0,
    )
    assert second.update_count == 2
    assert second.realised_prefix_budget_metric_m3 == pytest.approx(47.5)
    assert second.remaining_future_allowance_m3 == pytest.approx(52.5)
    assert second.admits(np.asarray([52.4, 52.6])).tolist() == [True, False]


def test_rolling_budget_audit_explicitly_reports_no_reinitialisation() -> None:
    payload = RollingPfvBudgetState().audit_payload()
    assert payload["allowance_reinitialised_each_decision"] is False
    assert payload["realised_prefix_included"] is True
    assert payload["contract"] == "event_level_candidate_le_(1+delta)_no_control_plus_B"


def test_rolling_budget_rejects_negative_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError):
        RollingPfvBudgetState(relative_margin_fraction=-0.01)
    with pytest.raises(ValueError):
        RollingPfvBudgetState().update(
            candidate_interval_pfv_m3=-1.0,
            no_control_interval_pfv_m3=0.0,
        )
    with pytest.raises(ValueError):
        RollingPfvBudgetState().admits(np.asarray([np.nan]))
