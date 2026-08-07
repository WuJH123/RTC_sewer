from __future__ import annotations

import numpy as np
import pandas as pd

from sewerrtc.control.pfv_tfv_tradeoff_v42 import (
    PfvContract,
    contract_scan,
    pareto_exchange_rates,
    select_knee_points,
    state_pareto_frontier,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"state_key": "s1", "candidate_action_sha256": "a", "pfv_candidate_m3": 100.0, "pfv_no_control_m3": 100.0, "tfv_candidate_m3": 1000.0, "tfv_internal_m3": 1000.0},
            {"state_key": "s1", "candidate_action_sha256": "b", "pfv_candidate_m3": 105.0, "pfv_no_control_m3": 100.0, "tfv_candidate_m3": 850.0, "tfv_internal_m3": 1000.0},
            {"state_key": "s1", "candidate_action_sha256": "c", "pfv_candidate_m3": 120.0, "pfv_no_control_m3": 100.0, "tfv_candidate_m3": 700.0, "tfv_internal_m3": 1000.0},
            {"state_key": "s1", "candidate_action_sha256": "d", "pfv_candidate_m3": 125.0, "pfv_no_control_m3": 100.0, "tfv_candidate_m3": 760.0, "tfv_internal_m3": 1000.0},
            {"state_key": "s2", "candidate_action_sha256": "e", "pfv_candidate_m3": 50.0, "pfv_no_control_m3": 50.0, "tfv_candidate_m3": 500.0, "tfv_internal_m3": 500.0},
        ]
    )


def test_contract_admission_matches_relative_plus_absolute_budget() -> None:
    contract = PfvContract(relative_margin_fraction=0.05, absolute_margin_m3=10.0)
    admitted = contract.admits(np.asarray([105.0, 116.0]), np.asarray([100.0, 100.0]))
    assert admitted.tolist() == [True, False]


def test_state_pareto_frontier_removes_dominated_action() -> None:
    frontier = state_pareto_frontier(_frame())
    s1 = frontier[frontier.state_key.eq("s1")]
    assert set(s1.candidate_action_sha256) == {"a", "b", "c"}
    assert "d" not in set(s1.candidate_action_sha256)


def test_exchange_rate_is_positive_for_more_pfv_more_tfv_benefit() -> None:
    frontier = state_pareto_frontier(_frame())
    exchange = pareto_exchange_rates(frontier)
    s1 = exchange[exchange.state_key.eq("s1")].dropna(subset=["marginal_tfv_benefit_per_pfv_m3"])
    assert (s1.marginal_tfv_benefit_per_pfv_m3 > 0).all()


def test_contract_scan_reports_all_state_zero_if_unavailable() -> None:
    _, aggregate = contract_scan(
        _frame(),
        relative_margins=(0.0,),
        absolute_margins_m3=(0.0,),
    )
    row = aggregate.iloc[0]
    assert row.total_states == 2
    assert row.admitted_states == 2
    assert np.isfinite(row.all_state_zero_if_unavailable_mean_pct)


def test_knee_selection_returns_one_row_per_state() -> None:
    knees = select_knee_points(state_pareto_frontier(_frame()))
    assert set(knees.state_key) == {"s1", "s2"}
