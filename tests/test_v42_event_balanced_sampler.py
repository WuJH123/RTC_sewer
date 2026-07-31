"""V4.2 event-balanced sampler — per-event weight equality, large-event cap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_sampling import V42GroupedTrainingSampler


def _imbalanced_sampler():
    """Event A has 30 samples, event B has 10."""
    events = ["evt_A"] * 30 + ["evt_B"] * 10
    return V42GroupedTrainingSampler(
        event_groups=pd.Series(events),
        random_state=42,
    )


class TestEventBalancedSampler:
    def test_per_event_weight_approximately_equal(self):
        sampler = _imbalanced_sampler()
        train_idx = np.arange(40)
        weights = sampler.get_sample_weights(train_idx)
        # Sum of weights per event should be similar
        groups = pd.Series(["evt_A"] * 30 + ["evt_B"] * 10)
        w_a = weights[groups.iloc[train_idx] == "evt_A"].sum()
        w_b = weights[groups.iloc[train_idx] == "evt_B"].sum()
        # After normalization, both should be close to len(train_idx)/n_events = 20
        ratio = w_a / w_b if w_b > 0 else float("inf")
        assert 0.5 < ratio < 2.0

    def test_large_event_not_overrepresented(self):
        sampler = _imbalanced_sampler()
        train_idx = np.arange(40)
        weights = sampler.get_sample_weights(train_idx)
        # Per-sample weight for evt_A (30 samples) should be lower than evt_B (10)
        w_a_single = weights[0]  # evt_A sample
        w_b_single = weights[30]  # evt_B sample
        assert w_a_single < w_b_single

    def test_weights_sum_to_n_samples(self):
        sampler = _imbalanced_sampler()
        train_idx = np.arange(40)
        weights = sampler.get_sample_weights(train_idx)
        np.testing.assert_almost_equal(weights.sum(), len(train_idx), decimal=5)
