"""Locked Validation accrual: power only, never deleting/retraining/tuning."""
from __future__ import annotations

import json
from pathlib import Path

from sewerrtc.v4.train1600_v3 import validate_accrual_plan_v3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "contracts"
    / "PROJECT6_V4_LOCKED_VALIDATION_ACCRUAL_V3.json"
)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


INITIAL = [f"lock_ev{i}" for i in range(8)]


def test_contract_is_preregistered_with_frozen_maxima() -> None:
    contract = _contract()

    assert contract["status"] == "pre_registered_frozen"
    rules = contract["rules"]
    assert rules["1_initial_events"] == 8
    assert rules["2_candidate_plans_frozen_before_labelling"] is True
    below5 = rules["4_when_exact_feasible_states_below_5"]
    assert below5["never_delete_original_events"] is True
    assert below5["never_modify_model"] is True
    assert below5["never_modify_thresholds"] is True
    assert below5["accrue_new_events_batch_size"] == [4, 8]
    assert below5["new_batch_frozen_wholesale_before_any_run"] is True
    assert rules["5_all_batches_in_final_report"] is True
    assert rules["6_frozen_maxima"] == {
        "maximum_total_locked_events": 24,
        "maximum_accrual_batches": 2,
    }
    assert "statistical power only" in rules["7_accrual_purpose"]


def test_valid_accrual_batches_pass() -> None:
    report = validate_accrual_plan_v3(
        _contract(),
        initial_event_ids=INITIAL,
        accrual_batches=[
            [f"acc_a{i}" for i in range(4)],
            [f"acc_b{i}" for i in range(8)],
        ],
    )

    assert report["status"] == "pass"
    assert all(report["checks"].values())
    assert report["total_locked_events"] == 20
    assert report["purpose"] == "statistical_power_only"


def test_accrual_never_deletes_original_events() -> None:
    report = validate_accrual_plan_v3(
        _contract(),
        initial_event_ids=INITIAL,
        accrual_batches=[[f"acc{i}" for i in range(4)]],
        deleted_events=["lock_ev0"],
    )

    assert report["status"] == "blocked"
    assert not report["checks"]["no_deleted_original_events"]


def test_accrual_never_modifies_model_or_thresholds() -> None:
    contract = _contract()

    modified_model = validate_accrual_plan_v3(
        contract,
        initial_event_ids=INITIAL,
        accrual_batches=[],
        model_modified=True,
    )
    assert modified_model["status"] == "blocked"
    assert not modified_model["checks"]["model_not_modified"]

    modified_thresholds = validate_accrual_plan_v3(
        contract,
        initial_event_ids=INITIAL,
        accrual_batches=[],
        thresholds_modified=True,
    )
    assert modified_thresholds["status"] == "blocked"
    assert not modified_thresholds["checks"]["thresholds_not_modified"]


def test_frozen_maxima_and_batch_sizes_are_enforced() -> None:
    contract = _contract()

    odd_batch = validate_accrual_plan_v3(
        contract,
        initial_event_ids=INITIAL,
        accrual_batches=[["a", "b", "c"]],  # size 3 not in {4, 8}
    )
    assert odd_batch["status"] == "blocked"
    assert not odd_batch["checks"]["batch_sizes_allowed"]

    too_many_batches = validate_accrual_plan_v3(
        contract,
        initial_event_ids=INITIAL,
        accrual_batches=[
            [f"x{i}" for i in range(4)],
            [f"y{i}" for i in range(4)],
            [f"z{i}" for i in range(4)],
        ],
    )
    assert too_many_batches["status"] == "blocked"
    assert not too_many_batches["checks"]["max_batches_respected"]

    over_cap = validate_accrual_plan_v3(
        contract,
        initial_event_ids=INITIAL,
        accrual_batches=[
            [f"x{i}" for i in range(8)],
            [f"y{i}" for i in range(8)],  # 8 + 8 + 8 = 24 <= 24 passes...
        ],
    )
    assert over_cap["status"] == "pass"
    duplicate = validate_accrual_plan_v3(
        contract,
        initial_event_ids=INITIAL,
        accrual_batches=[
            [f"x{i}" for i in range(8)],
            ["x0", *[f"y{i}" for i in range(7)]],  # reused event id
        ],
    )
    assert duplicate["status"] == "blocked"
    assert not duplicate["checks"]["no_duplicate_events"]
