import numpy as np
import pandas as pd

from sewerrtc.v4.labels import (
    add_ranking_labels,
    enforce_full_label_eligibility,
    select_h120_window,
)


def test_h120_is_strictly_after_checkpoint_and_includes_horizon_endpoint() -> None:
    detail = pd.DataFrame({"elapsed_min": [0, 5, 120, 125], "value": [1, 2, 3, 4]})
    window = select_h120_window(detail, checkpoint_min=0)

    assert window["elapsed_min"].tolist() == [5, 120]


def test_full_labels_are_nan_when_full_event_is_not_eligible() -> None:
    frame = pd.DataFrame(
        {
            "full_event_eligible": [False, True],
            "delta_pfv_full": [1.0, 2.0],
            "delta_tfv_full": [3.0, 4.0],
            "delta_peak_full": [5.0, 6.0],
        }
    )
    result = enforce_full_label_eligibility(frame)

    assert result.loc[0, ["delta_pfv_full", "delta_tfv_full", "delta_peak_full"]].isna().all()
    assert result.loc[1, "delta_pfv_full"] == 2.0


def test_feasible_ranking_and_exact_regret_are_checkpoint_local() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["e"] * 3,
            "checkpoint_id": ["c"] * 3,
            "joint_noninferior": [True, True, False],
            "delta_tfv_h120_vs_dynamic_internal": [-10.0, -5.0, -100.0],
        }
    )
    result = add_ranking_labels(frame)

    assert result["feasible_rank"].tolist()[:2] == [1.0, 2.0]
    assert result["regret_to_exact_best"].tolist()[:2] == [0.0, 5.0]
    assert np.isnan(result.loc[2, "feasible_rank"])
