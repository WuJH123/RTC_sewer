from __future__ import annotations

from sewerrtc.control.event_pfv_budget import EventPfvBudget


def test_event_pfv_budget_uses_two_percent_or_200_m3_floor():
    small = EventPfvBudget(predicted_event_no_control_pfv=5000.0)
    large = EventPfvBudget(predicted_event_no_control_pfv=50000.0)

    assert small.initial_budget_m3 == 200.0
    assert large.initial_budget_m3 == 1000.0


def test_event_pfv_budget_is_cumulative_not_reinitialized_per_step():
    budget = EventPfvBudget(predicted_event_no_control_pfv=50000.0)

    budget.debit_observed(300.0, label="first_hour")
    budget.set_inflight_conservative_cost(200.0, label="open_storage")

    assert budget.remaining_budget_m3 == 500.0
    assert budget.candidate_allowed(500.0)
    assert not budget.candidate_allowed(500.1)
