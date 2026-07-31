"""Final 1600 dataset gate: accounting closed, splits exact, no leakage."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.train1600_v3 import audit_train1600_dataset_v3
from train_v3_helpers import make_plan_chain


@pytest.fixture(scope="module")
def chain() -> dict:
    return make_plan_chain()


def _make_samples(chain: dict) -> pd.DataFrame:
    # 320 states x 5 primary accepted candidates = 1600 accepted samples.
    samples = chain["role_plan"][
        chain["role_plan"]["plan_tier"] == "primary"
    ].copy()
    samples = samples.reset_index(drop=True)
    samples["actual_schedule_sha256"] = [
        f"actual_{i:04d}" for i in range(len(samples))
    ]
    rng = np.random.default_rng(7)
    samples["delta_pfv_h120_vs_no_control"] = rng.normal(size=len(samples))
    samples["delta_tfv_h120_vs_dynamic_internal"] = rng.normal(
        size=len(samples)
    )
    samples["delta_peak_h120_vs_dynamic_internal"] = rng.normal(
        size=len(samples)
    )
    samples["same_state_verified"] = True
    samples["readback_verified"] = True
    # Sampled-only state-feasibility label contract (spec section 1): every
    # Train1600 state is sampled-only, no P3 Exact search was performed.
    samples["state_feasibility_label_source"] = "sampled_only"
    samples["state_feasibility_label_validity"] = "sampled_only"
    samples["exact_search_performed"] = False
    samples["candidate_search_budget"] = 5
    joint = samples.groupby(["event_id", "checkpoint_id"]).cumcount().eq(0)
    samples["joint_found_in_sampled_set"] = (
        joint.groupby(
            [samples["event_id"], samples["checkpoint_id"]]
        ).transform("any")
    )
    return samples


def test_final_audit_passes_on_a_closed_1600_table(chain: dict) -> None:
    samples = _make_samples(chain)

    audit = audit_train1600_dataset_v3(
        samples,
        chain["train_catalog"],
        chain["selection"],
        hard_columns=("same_state_verified", "readback_verified"),
    )

    assert audit["status"] == "pass"
    checks = audit["checks"]
    assert checks["accepted_total_1600"]
    assert checks["events_64"]
    assert checks["state_groups_320"]
    assert checks["split_48_8_8"]
    assert checks["train_samples_1200"]
    assert checks["calibration_samples_200"]
    assert checks["locked_validation_samples_200"]
    assert checks["reserve_not_in_main_table"]
    assert checks["per_state_exactly_5"]
    assert checks["per_state_actual_unique"]
    assert checks["no_rainfall_sha_leakage"]
    assert checks["hard_authenticity_100pct"]
    assert checks["continuous_labels_non_degenerate"]
    assert checks["hard_negatives_present"]
    assert checks["uncertainty_coverage_gap_present"]
    assert audit["split_sample_counts"] == {
        "train": 1200,
        "calibration": 200,
        "locked_validation": 200,
    }
    # Generation never requires Calibration/Locked to contain 5 feasible
    # states; that belongs to the deferred Model Safety Gate + accrual.
    assert audit["informational"][
        "calibration_locked_not_required_to_contain_5_feasible_states"
    ] is True
    json.dumps(audit, allow_nan=False)


def test_rainfall_sha_leakage_across_splits_blocks(chain: dict) -> None:
    samples = _make_samples(chain)
    train_event = chain["selection"]["train"][0]
    calibration_event = chain["selection"]["calibration"][0]
    leaked_sha = samples.loc[
        samples["event_id"] == calibration_event, "rainfall_sha256"
    ].iloc[0]
    samples.loc[
        samples["event_id"] == train_event, "rainfall_sha256"
    ] = leaked_sha

    audit = audit_train1600_dataset_v3(
        samples, chain["train_catalog"], chain["selection"]
    )

    assert audit["status"] == "blocked"
    assert not audit["checks"]["no_rainfall_sha_leakage"]


def test_actual_duplicates_break_per_state_uniqueness(chain: dict) -> None:
    samples = _make_samples(chain)
    # Two candidates of one state collapse to the same actual schedule.
    state = samples.iloc[0][["event_id", "checkpoint_id"]]
    mask = (samples["event_id"] == state["event_id"]) & (
        samples["checkpoint_id"] == state["checkpoint_id"]
    )
    index = samples[mask].index[:2]
    samples.loc[index, "actual_schedule_sha256"] = "duplicated_actual"

    audit = audit_train1600_dataset_v3(
        samples, chain["train_catalog"], chain["selection"]
    )

    assert audit["status"] == "blocked"
    assert not audit["checks"]["per_state_actual_unique"]


def test_reserve_rows_never_enter_the_main_table(chain: dict) -> None:
    samples = _make_samples(chain)
    reserve_event = chain["selection"]["reserve"][0]
    contaminated = pd.concat(
        [samples, samples.iloc[:5].assign(event_id=reserve_event)],
        ignore_index=True,
    )

    audit = audit_train1600_dataset_v3(
        contaminated, chain["train_catalog"], chain["selection"]
    )

    assert audit["status"] == "blocked"
    assert not audit["checks"]["reserve_not_in_main_table"]
    assert not audit["checks"]["accepted_total_1600"]


def test_short_state_or_wrong_total_blocks(chain: dict) -> None:
    samples = _make_samples(chain)
    short = samples.iloc[:-1]  # one state loses its 5th accepted sample

    audit = audit_train1600_dataset_v3(
        short, chain["train_catalog"], chain["selection"]
    )

    assert audit["status"] == "blocked"
    assert not audit["checks"]["accepted_total_1600"]
    assert not audit["checks"]["per_state_exactly_5"]


def test_degenerate_continuous_labels_block(chain: dict) -> None:
    samples = _make_samples(chain)
    for column in samples.columns:
        if str(column).startswith("delta_"):
            samples[column] = 0.0

    audit = audit_train1600_dataset_v3(
        samples, chain["train_catalog"], chain["selection"]
    )

    assert audit["status"] == "blocked"
    assert not audit["checks"]["continuous_labels_non_degenerate"]


def test_sampled_only_contract_passes_on_stamped_table(chain: dict) -> None:
    samples = _make_samples(chain)

    audit = audit_train1600_dataset_v3(
        samples, chain["train_catalog"], chain["selection"]
    )

    assert audit["checks"]["sampled_only_label_contract_valid"]


def test_missing_sampled_only_fields_block(chain: dict) -> None:
    samples = _make_samples(chain).drop(columns=["joint_found_in_sampled_set"])

    audit = audit_train1600_dataset_v3(
        samples, chain["train_catalog"], chain["selection"]
    )

    assert audit["status"] == "blocked"
    assert not audit["checks"]["sampled_only_label_contract_valid"]


def test_fallback_only_label_on_sampled_state_blocks(chain: dict) -> None:
    samples = _make_samples(chain)
    # A sampled-only state must never be relabelled as a P3-Exact verdict.
    samples.loc[
        samples.index[:5], "state_feasibility_label_source"
    ] = "fallback_only_under_budget"

    audit = audit_train1600_dataset_v3(
        samples, chain["train_catalog"], chain["selection"]
    )

    assert audit["status"] == "blocked"
    assert not audit["checks"]["sampled_only_label_contract_valid"]


def test_exact_search_performed_true_blocks(chain: dict) -> None:
    samples = _make_samples(chain)
    samples["exact_search_performed"] = True

    audit = audit_train1600_dataset_v3(
        samples, chain["train_catalog"], chain["selection"]
    )

    assert audit["status"] == "blocked"
    assert not audit["checks"]["sampled_only_label_contract_valid"]
