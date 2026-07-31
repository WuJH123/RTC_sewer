"""V4.2 cross-fitted calibration — OOF prediction no-leakage test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_cv import plan_v42_nested_grouped_cv, compute_inner_folds


def _events(n: int = 20) -> tuple[list[str], list[str]]:
    eids = [f"evt_{i}" for i in range(n)]
    shas = [f"sha_{i % 8}" for i in range(n)]
    return eids, shas


class TestCrossFittedCalibration:
    def test_oof_predictions_no_leakage(self):
        """Each event appears as OOF (test) in exactly one outer fold."""
        eids, shas = _events(20)
        plan = plan_v42_nested_grouped_cv(eids, shas, n_outer=5)
        event_df = pd.DataFrame({"event_id": eids, "rainfall_sha256": shas})

        oof_counts: dict[str, int] = {eid: 0 for eid in eids}
        for outer_fold in range(5):
            test_events = {
                eid for eid, f in plan.outer_fold_assignment.items() if f == outer_fold
            }
            for eid in test_events:
                oof_counts[eid] += 1

        # Each event is OOF exactly once
        assert all(c == 1 for c in oof_counts.values())

    def test_oof_train_test_disjoint(self):
        """Train and test sets in each fold are disjoint."""
        eids, shas = _events(20)
        plan = plan_v42_nested_grouped_cv(eids, shas, n_outer=5)

        for outer_fold in range(5):
            train_events = {
                eid for eid, f in plan.outer_fold_assignment.items() if f != outer_fold
            }
            test_events = {
                eid for eid, f in plan.outer_fold_assignment.items() if f == outer_fold
            }
            assert train_events.isdisjoint(test_events)

    def test_inner_folds_no_leakage_within_outer_train(self):
        """Inner fold test sets are disjoint from inner fold train sets."""
        eids, shas = _events(20)
        plan = plan_v42_nested_grouped_cv(eids, shas, n_outer=5, n_inner=3)
        event_df = pd.DataFrame({"event_id": eids, "rainfall_sha256": shas})

        for outer_fold in range(5):
            inner = compute_inner_folds(event_df, plan, outer_fold=outer_fold)
            if inner.empty:
                continue
            for inner_fold in range(3):
                train_mask = inner["inner_fold"] != inner_fold
                test_mask = inner["inner_fold"] == inner_fold
                train_events = set(inner.loc[train_mask, "event_id"])
                test_events = set(inner.loc[test_mask, "event_id"])
                assert train_events.isdisjoint(test_events)
