import pandas as pd

from scripts.plan_v42_fast_direct_screen import select_fast_states


def test_fast_plan_has_frozen_regime_counts() -> None:
    rows = []
    for regime, count in {
        "LOW_LOAD": 4,
        "MODERATE_LOAD": 6,
        "NEAR_CAPACITY": 3,
        "SEVERE_OVERLOAD": 4,
    }.items():
        for index in range(count):
            rows.append(
                {
                    "state_key": f"{regime}-{index}",
                    "event_id": f"event-{regime}-{index}",
                    "rainfall_sha256": f"rain-{regime}-{index}",
                    "load_regime": regime,
                    "oracle_tfv_reduction_pct": {
                        "LOW_LOAD": [25.0, 8.0, 1.0, -1.0],
                        "MODERATE_LOAD": [-2.0, 2.0, 12.0, 3.0, 1.0, 0.0],
                        "NEAR_CAPACITY": [5.0, 8.0, 6.0],
                        "SEVERE_OVERLOAD": [7.0, 1.0, 3.0, 4.0],
                    }[regime][index],
                    "candidate_count": 10,
                    "actual_safe_candidate_count": 2,
                }
            )
    selected = select_fast_states(pd.DataFrame(rows))
    assert selected.groupby("load_regime").size().to_dict() == {
        "LOW_LOAD": 2,
        "MODERATE_LOAD": 3,
        "NEAR_CAPACITY": 1,
        "SEVERE_OVERLOAD": 2,
    }
    assert len(selected) == 8
    assert selected["state_key"].is_unique
