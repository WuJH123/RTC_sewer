"""V4.2 nested CV no-leakage — events/SHA never cross outer folds."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_cv import plan_v42_nested_grouped_cv, compute_inner_folds


def _events(n: int = 25, n_shas: int = 10) -> tuple[list[str], list[str]]:
    eids = [f"evt_{i}" for i in range(n)]
    shas = [f"sha_{i % n_shas}" for i in range(n)]
    return eids, shas


class TestNoLeakage:
    def test_events_not_cross_outer_fold(self):
        eids, shas = _events(25)
        plan = plan_v42_nested_grouped_cv(eids, shas, n_outer=5)
        # Each event appears in exactly one fold
        fold_sets: dict[int, set[str]] = {i: set() for i in range(5)}
        for eid, fold in plan.outer_fold_assignment.items():
            fold_sets[fold].add(eid)
        # No overlap between folds
        for i in range(5):
            for j in range(i + 1, 5):
                assert fold_sets[i].isdisjoint(fold_sets[j])

    def test_rainfall_sha_not_cross_fold(self):
        eids, shas = _events(25, n_shas=10)
        plan = plan_v42_nested_grouped_cv(eids, shas, n_outer=5)
        sha_to_fold: dict[str, int] = {}
        for eid, fold in plan.outer_fold_assignment.items():
            sha = dict(zip(eids, shas))[eid]
            if sha in sha_to_fold:
                assert sha_to_fold[sha] == fold, f"SHA {sha} in multiple folds"
            else:
                sha_to_fold[sha] = fold

    def test_inner_folds_within_outer_train_only(self):
        eids, shas = _events(25)
        plan = plan_v42_nested_grouped_cv(eids, shas, n_outer=5, n_inner=3)
        event_df = pd.DataFrame({"event_id": eids, "rainfall_sha256": shas})
        for outer_fold in range(5):
            inner = compute_inner_folds(event_df, plan, outer_fold=outer_fold)
            outer_test = {
                eid for eid, f in plan.outer_fold_assignment.items() if f == outer_fold
            }
            inner_events = set(inner["event_id"])
            assert inner_events.isdisjoint(outer_test), (
                f"Inner fold of outer={outer_fold} contains outer test events"
            )
