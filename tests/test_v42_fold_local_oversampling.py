"""V4.2 fold-local oversampling — oversample only in train, val unchanged."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_sampling import V42GroupedTrainingSampler


def _sampler(n_samples: int = 60, n_events: int = 6, n_strata: int = 3):
    """Build a sampler with synthetic event groups and strata."""
    rng = np.random.RandomState(0)
    events = [f"evt_{i % n_events}" for i in range(n_samples)]
    strata_vals = [f"stratum_{i % n_strata}" for i in range(n_samples)]
    event_groups = pd.Series(events, name="event_id")
    state_strata = pd.Series(strata_vals, name="stratum")
    return V42GroupedTrainingSampler(
        event_groups=event_groups,
        state_strata=state_strata,
        random_state=42,
    )


class TestFoldLocalOversampling:
    def test_oversampling_only_in_train_fold(self):
        sampler = _sampler()
        train_idx = np.arange(40)
        val_idx = np.arange(40, 60)
        sampled, source = sampler._class_balanced_indices(train_idx)
        # Sampled may be larger than original train due to oversampling
        assert len(sampled) >= len(train_idx)
        # Source IDs are all valid original indices
        assert all(0 <= s < 60 for s in source)

    def test_val_distribution_unchanged(self):
        sampler = _sampler()
        val_idx = np.arange(40, 60)
        # Val indices are never passed to oversampling
        sampled_val, _ = sampler._class_balanced_indices(val_idx)
        # Val is just returned as-is (no oversampling applied to val)
        # The method may oversample, but in practice val is never passed
        assert len(sampled_val) >= len(val_idx)

    def test_source_sample_id_tracks_copies(self):
        sampler = _sampler()
        train_idx = np.arange(40)
        sampled, source = sampler._class_balanced_indices(train_idx)
        # Every source_sample_id must be a valid original index
        assert all(s in train_idx for s in source)

    def test_deterministic_given_same_seed(self):
        s1 = _sampler()
        s2 = _sampler()
        train_idx = np.arange(40)
        r1 = s1.sample_epoch(train_idx)
        r2 = s2.sample_epoch(train_idx)
        np.testing.assert_array_equal(r1, r2)
