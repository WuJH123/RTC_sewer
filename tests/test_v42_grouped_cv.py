"""V4.2 grouped CV — fold assignment, group integrity, min events per fold."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_cv import (
    DEFAULT_INNER_SPLITS,
    DEFAULT_OUTER_SPLITS,
    MIN_EVENTS_PER_FOLD,
    V42CVPlan,
    compute_inner_folds,
    plan_v42_nested_grouped_cv,
)


def _events(n: int = 20, n_unique_shas: int = 10) -> tuple[list[str], list[str]]:
    """Return (event_ids, rainfall_shas) with *n* events sharing *n_unique_shas* SHAs."""
    event_ids = [f"evt_{i}" for i in range(n)]
    shas = [f"sha_{i % n_unique_shas}" for i in range(n)]
    return event_ids, shas


class TestPlanNestedGroupedCV:
    def test_every_event_assigned(self):
        eids, shas = _events(20)
        plan = plan_v42_nested_grouped_cv(eids, shas, n_outer=5)
        assert set(plan.outer_fold_assignment.keys()) == set(eids)

    def test_each_event_in_exactly_one_fold(self):
        eids, shas = _events(20)
        plan = plan_v42_nested_grouped_cv(eids, shas, n_outer=5)
        fold_vals = list(plan.outer_fold_assignment.values())
        assert all(0 <= v < 5 for v in fold_vals)

    def test_same_sha_not_split_across_folds(self):
        eids, shas = _events(20, n_unique_shas=10)
        plan = plan_v42_nested_grouped_cv(eids, shas, n_outer=5)
        # Build SHA → fold mapping
        sha_folds: dict[str, set[int]] = {}
        for eid, fold in plan.outer_fold_assignment.items():
            sha = dict(zip(eids, shas))[eid]
            sha_folds.setdefault(sha, set()).add(fold)
        for sha, folds in sha_folds.items():
            assert len(folds) == 1, f"SHA {sha} spans folds {folds}"

    def test_frozen_seeds_deterministic(self):
        eids, shas = _events(20)
        p1 = plan_v42_nested_grouped_cv(eids, shas, seed=42)
        p2 = plan_v42_nested_grouped_cv(eids, shas, seed=42)
        assert p1.frozen_seeds == p2.frozen_seeds
        assert p1.outer_fold_assignment == p2.outer_fold_assignment

    def test_min_events_per_fold(self):
        eids, shas = _events(20)
        plan = plan_v42_nested_grouped_cv(eids, shas, n_outer=5)
        fold_counts = pd.Series(list(plan.outer_fold_assignment.values())).value_counts()
        assert (fold_counts >= MIN_EVENTS_PER_FOLD).all()


class TestInnerFolds:
    def test_inner_folds_within_outer_train(self):
        eids, shas = _events(20)
        plan = plan_v42_nested_grouped_cv(eids, shas, n_outer=5, n_inner=3)
        event_df = pd.DataFrame({"event_id": eids, "rainfall_sha256": shas})
        inner = compute_inner_folds(event_df, plan, outer_fold=0)
        # Inner folds only contain events NOT in outer fold 0
        outer_test_events = {
            eid for eid, f in plan.outer_fold_assignment.items() if f == 0
        }
        inner_events = set(inner["event_id"])
        assert inner_events.isdisjoint(outer_test_events)

    def test_inner_fold_count(self):
        eids, shas = _events(20)
        plan = plan_v42_nested_grouped_cv(eids, shas, n_outer=5, n_inner=3)
        event_df = pd.DataFrame({"event_id": eids, "rainfall_sha256": shas})
        inner = compute_inner_folds(event_df, plan, outer_fold=0)
        if not inner.empty:
            assert set(inner["inner_fold"].unique()).issubset({0, 1, 2})
