from __future__ import annotations

import pandas as pd

from scripts.plan_v42_true_state_direct_oracle import _select_direct_states


def _states() -> pd.DataFrame:
    rows = []
    values = {
        "LOW_LOAD": [35.0, 12.0, 7.0, -2.0, 1.0, 22.0],
        "MODERATE_LOAD": [-8.0, -1.0, 2.0, 8.0, 12.0, 4.0],
        "NEAR_CAPACITY": [-3.0, 9.0, 4.0, 1.0],
        "SEVERE_OVERLOAD": [-2.0, 6.0, 3.0, 1.0],
    }
    for regime, reductions in values.items():
        for index, reduction in enumerate(reductions):
            rows.append(
                {
                    "state_key": f"{regime}-{index}",
                    "event_id": f"event-{regime}-{index}",
                    "rainfall_sha256": f"rain-{regime}-{index}",
                    "load_regime": regime,
                    "oracle_tfv_reduction_pct": reduction,
                    "candidate_count": 10 + index,
                    "actual_safe_candidate_count": index,
                    "actual_safe_tfv_improving_count": max(index - 1, 0),
                    "oracle_candidate_action_sha256": f"action-{regime}-{index}",
                }
            )
    return pd.DataFrame(rows)


def test_direct_oracle_plan_has_required_regime_counts_and_is_deterministic() -> None:
    first = _select_direct_states(_states())
    second = _select_direct_states(_states())

    assert first["state_key"].tolist() == second["state_key"].tolist()
    assert first["state_key"].is_unique
    assert first.groupby("load_regime").size().to_dict() == {
        "LOW_LOAD": 3,
        "MODERATE_LOAD": 5,
        "NEAR_CAPACITY": 2,
        "SEVERE_OVERLOAD": 2,
    }


def test_direct_oracle_plan_selection_reasons_are_explicit() -> None:
    plan = _select_direct_states(_states())
    reasons = set(plan["selection_reason"].astype(str))
    reason_rows = plan["selection_reason"].astype(str).tolist()

    assert "low_positive_ge20" in reasons
    assert sum(reason.startswith("moderate_nonpositive") for reason in reason_rows) == 2
    assert sum(reason.startswith("moderate_low_positive") for reason in reason_rows) == 2
    assert "moderate_best_available" in reasons
    assert "near_capacity_best" in reasons
    assert "near_capacity_gap" in reasons
    assert "severe_overload_best" in reasons
    assert "severe_overload_gap" in reasons
