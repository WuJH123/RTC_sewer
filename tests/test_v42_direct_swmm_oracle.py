from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sewerrtc.control.authoritative_control_metrics_v42 import (
    action_sha256,
    detail_horizon_metrics,
    h3_prefix_is_valid,
    pfv_budget_metric,
    pfv_feasible_first_score,
    realised_prefix_budget_metric,
    trajectory_metrics,
)


def test_h3_bounds_binary_and_h4_h12_hold() -> None:
    current = np.asarray([0.0, 1.0, 0.5], dtype=float)
    sequence = np.tile(current, (12, 1))
    sequence[:3, 0] = 0.0
    sequence[:3, 1] = 1.0
    sequence[:3, 2] = 0.75

    assert h3_prefix_is_valid(
        sequence,
        current,
        binary_indices=(0, 1),
    )
    sequence[3, 2] = 0.7
    assert not h3_prefix_is_valid(sequence, current, binary_indices=(0, 1))


def test_h3_rejects_nonfinite_out_of_bounds_and_binary_values() -> None:
    current = np.asarray([0.0, 0.5], dtype=float)
    sequence = np.tile(current, (12, 1))
    sequence[0, 0] = 0.25
    assert not h3_prefix_is_valid(sequence, current, binary_indices=(0,))
    sequence[0, 0] = 1.0
    sequence[0, 1] = 1.1
    assert not h3_prefix_is_valid(sequence, current, binary_indices=(0,))
    sequence[0, 1] = np.nan
    assert not h3_prefix_is_valid(sequence, current, binary_indices=(0,))


def test_authoritative_detail_metrics_match_array_metrics_and_prefix_semantics() -> None:
    flood = np.asarray([[0.0, 1.0], [0.5, 0.0], [1.0, 2.0], [2.0, 1.0]])
    detail = pd.DataFrame(
        {
            "elapsed_min": [0.0, 10.0, 20.0, 30.0],
            "flood:n0": flood[:, 0],
            "flood:n1": flood[:, 1],
        }
    )
    assert trajectory_metrics(flood, priority_indices=(0,), dt_sec=600) == pytest.approx(
        {"PFV": 2100.0, "TFV": 4500.0, "peak_TFV_rate": 3.0}
    )
    assert detail_horizon_metrics(detail, priority_nodes=("n0",), checkpoint_min=10.0, steps=2) == pytest.approx(
        {"PFV": 900.0, "TFV": 2100.0, "peak_TFV_rate": 3.0}
    )
    assert realised_prefix_budget_metric(
        detail.iloc[:2], detail.iloc[:2], priority_nodes=("n0",), relative_margin=0.05
    ) == pytest.approx(-15.0)


def test_pfv_first_score_never_prefers_unsafe_low_tfv() -> None:
    safe = pfv_feasible_first_score(tfv=100.0, budget_metric=99.0, absolute_margin=100.0, reference_tfv=100.0)
    unsafe = pfv_feasible_first_score(tfv=-1.0e12, budget_metric=101.0, absolute_margin=100.0, reference_tfv=100.0)
    assert safe < unsafe
    assert pfv_budget_metric(150.0, 100.0, relative_margin=0.05) == pytest.approx(45.0)


def test_action_cache_sha_is_deterministic() -> None:
    action = np.zeros((12, 3), dtype=np.float32)
    assert action_sha256(action) == action_sha256(action.copy())
    changed = action.copy()
    changed[0, 0] = 0.1
    assert action_sha256(action) != action_sha256(changed)
