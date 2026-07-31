"""State stratification: online features only, scorer fit on pilot_train."""
from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.train1600_v3 import (
    V3_STRATA,
    apply_state_feasibility_scorer,
    fit_state_feasibility_scorer,
)
from train_v3_helpers import SCORER, make_plan_chain, make_standard_catalog


def _p3_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    p3_map = pd.DataFrame(
        {
            "event_id": [f"p{i}" for i in range(12)],
            "checkpoint_id": [f"p{i}_cp0" for i in range(12)],
            "split": ["pilot_train"] * 9 + ["pilot_validation"] * 3,
            "state_feasibility_class": (
                ["joint_feasible_robust"] * 4
                + ["no_joint_found_under_budget"] * 4
                + ["joint_boundary_found"]
                + ["joint_feasible_robust"] * 3
            ),
        }
    )
    p3_catalog = pd.DataFrame(
        {
            "event_id": p3_map["event_id"],
            "checkpoint_id": p3_map["checkpoint_id"],
            "opportunity_score": [
                10.0, 9.0, 10.0, 9.5,
                4.0, 5.0, 4.5, 5.5,
                7.0,
                10.0, 10.0, 10.0,
            ],
        }
    )
    return p3_map, p3_catalog


def test_scorer_fits_on_pilot_train_only_with_online_feature() -> None:
    p3_map, p3_catalog = _p3_fixture()

    scorer = fit_state_feasibility_scorer(p3_map, p3_catalog)

    assert scorer["online_features_only"] is True
    assert scorer["feature"] == "opportunity_score"
    assert scorer["training_split"] == "pilot_train"
    # Only the 9 pilot_train states inform the thresholds; the 3 validation
    # states (all high-score robust) are excluded.
    assert scorer["training_states"] == 9
    assert scorer["t_high"] > scorer["t_low"]
    assert scorer["strata"] == list(V3_STRATA)


def test_scorer_refuses_to_fit_without_pilot_train_states() -> None:
    p3_map, p3_catalog = _p3_fixture()
    p3_map["split"] = "pilot_validation"

    with pytest.raises(ValueError, match="pilot_train"):
        fit_state_feasibility_scorer(p3_map, p3_catalog)


def test_stratification_never_reads_new_state_exact_labels() -> None:
    catalog = make_standard_catalog(4)
    # Even if an exact label column leaks into the catalog, the scorer only
    # reads the online opportunity_score; a contradictory exact label must
    # not change the assigned stratum.
    poisoned = catalog.copy()
    poisoned["state_feasibility_class"] = "no_joint_found_under_budget"

    clean = apply_state_feasibility_scorer(SCORER, catalog)
    dirty = apply_state_feasibility_scorer(SCORER, poisoned)

    assert clean["predicted_stratum"].tolist() == (
        dirty["predicted_stratum"].tolist()
    )


def test_strata_thresholds_and_low_opportunity_preserved() -> None:
    catalog = make_standard_catalog(2)
    stratified = apply_state_feasibility_scorer(SCORER, catalog)

    by_cp = stratified.set_index("checkpoint_id")["predicted_stratum"]
    for event in ("ev000", "ev001"):
        # score 10 -> high, 9/8 -> boundary, 7 -> fallback-likely.
        assert by_cp[f"{event}_cp0"] == "predicted_high_feasibility"
        assert by_cp[f"{event}_cp1"] == "predicted_boundary"
        assert by_cp[f"{event}_cp2"] == "predicted_boundary"
        assert by_cp[f"{event}_cp3"] == "predicted_fallback_likely"
        # The low-opportunity state always keeps its role regardless of score.
        assert by_cp[f"{event}_cp4"] == "low_opportunity"


def test_train_catalog_keeps_one_low_opportunity_state_per_event() -> None:
    chain = make_plan_chain()
    stratified = chain["stratified"]

    low = stratified[stratified["predicted_stratum"] == "low_opportunity"]
    assert low.groupby("event_id").size().eq(1).all()
    assert low["event_id"].nunique() == stratified["event_id"].nunique()
    # All three responsive strata are populated across the train catalog.
    train = stratified[stratified["split"] == "train"]
    assert set(V3_STRATA) <= set(train["predicted_stratum"].unique())
