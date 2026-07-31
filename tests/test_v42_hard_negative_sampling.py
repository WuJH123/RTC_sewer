"""V4.2 hard-negative sampling — type coverage and weight boost."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_sampling import (
    HARD_NEGATIVE_MULTIPLIER,
    HARD_NEGATIVE_TYPES,
    V42GroupedTrainingSampler,
)


def _hn_sampler():
    """Sampler with known hard-negative types."""
    events = [f"evt_{i % 5}" for i in range(30)]
    hn_types = [""] * 30
    hn_types[0] = "pfv_unsafe_tfv_improved"
    hn_types[1] = "peak_degraded_tfv_improved"
    hn_types[2] = "joint_hard_negative"
    hn_types[3] = "pfv_boundary"
    hn_types[4] = "peak_boundary"
    hn_types[5] = "previous_false_safe"
    return V42GroupedTrainingSampler(
        event_groups=pd.Series(events),
        hard_negative_types=pd.Series(hn_types),
        random_state=42,
    )


class TestHardNegativeSampling:
    def test_hard_negative_types_recognised(self):
        expected = {
            "pfv_unsafe_tfv_improved", "peak_degraded_tfv_improved",
            "joint_hard_negative", "pfv_boundary", "peak_boundary",
            "previous_false_safe",
        }
        assert expected == set(HARD_NEGATIVE_TYPES)

    def test_weight_boost_for_hard_negatives(self):
        sampler = _hn_sampler()
        train_idx = np.arange(30)
        weights = sampler._hard_negative_weights(train_idx)
        # Hard-negative samples (indices 0-5) should have 2× weight
        for i in range(6):
            assert weights[i] == HARD_NEGATIVE_MULTIPLIER
        # Non-hard-negative samples should have 1× weight
        for i in range(6, 30):
            assert weights[i] == 1.0

    def test_hard_negative_coverage_in_summary(self):
        sampler = _hn_sampler()
        train_idx = np.arange(30)
        summary = sampler.summary(train_idx)
        hn_counts = summary["hard_negative_counts"]
        assert hn_counts["pfv_unsafe_tfv_improved"] >= 1
        assert hn_counts["joint_hard_negative"] >= 1

    def test_no_hard_negative_returns_unit_weights(self):
        events = ["evt_0"] * 10
        sampler = V42GroupedTrainingSampler(
            event_groups=pd.Series(events),
            hard_negative_types=None,
            random_state=42,
        )
        weights = sampler._hard_negative_weights(np.arange(10))
        np.testing.assert_array_equal(weights, np.ones(10))
