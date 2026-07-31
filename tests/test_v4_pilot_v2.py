"""Synthetic-data regression for Dataset v2 merge rules and the Gate v2 verdict."""

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.pilot_v2 import (
    GATE_V2_CONTRACT_VERSION,
    REQUIRED_MODELS_V2,
    _finite_or_none,
    build_pilot_dataset_v2,
    evaluate_pilot_gate_v2,
)


def _primary(rows: int = 400) -> pd.DataFrame:
    splits = (
        ["pilot_train"] * 240
        + ["pilot_calibration"] * 60
        + ["pilot_validation"] * 50
        + ["pilot_challenge"] * 50
    )
    return pd.DataFrame(
        {
            "sample_id": [f"v1_{index:04d}" for index in range(rows)],
            "event_id": [f"E{index % 8}" for index in range(rows)],
            "checkpoint_id": [f"C{index % 40}" for index in range(rows)],
            "split": splits[:rows],
            "actual_schedule_sha256": [f"sha_v1_{index}" for index in range(rows)],
            "checkpoint_role": "candidate",
        }
    )


def _extension_row(sample_id: str, actual: str, split: str = "pilot_train") -> dict:
    return {
        "sample_id": sample_id,
        "event_id": "E0",
        "checkpoint_id": "C0",
        "split": split,
        "actual_schedule_sha256": actual,
        "checkpoint_role": "candidate",
    }


def test_dataset_v2_keeps_v1_verbatim_and_rejects_duplicates() -> None:
    extension = pd.DataFrame(
        [
            _extension_row("ext_000", "sha_ext_unique"),
            # Collides with a v1 actual schedule anywhere in primary400.
            _extension_row("ext_001", "sha_v1_7"),
            # Same state twice with the same actual schedule.
            _extension_row("ext_002", "sha_ext_repeat"),
            _extension_row("ext_003", "sha_ext_repeat"),
        ]
    )
    result = build_pilot_dataset_v2(_primary(), extension)
    samples = result["sample_manifest"]
    assert (samples["source_phase"] == "primary400").sum() == 400
    kept = samples[samples["source_phase"] == "joint_extension"]
    assert sorted(kept["sample_id"]) == ["ext_000", "ext_002"]
    assert kept["split"].eq("pilot_train").all()
    assert kept["selected_after_pilot_v1"].all()
    assert kept["source_contract_version"].eq(GATE_V2_CONTRACT_VERSION).all()
    duplicates = result["actual_duplicates"]
    reasons = dict(zip(duplicates["sample_id"], duplicates["reason"]))
    assert reasons == {
        "ext_001": "duplicate_actual_vs_primary400",
        "ext_003": "duplicate_actual_within_state_v2",
    }
    assert result["accounting"]["accounting_closed"]
    # Evaluation splits stay pure primary400.
    for column in (
        "eligible_for_calibration",
        "eligible_for_validation",
        "eligible_for_challenge",
    ):
        eligible = samples[samples[column]]
        assert eligible["source_phase"].eq("primary400").all()
    assert samples.loc[
        samples["eligible_for_training"], "split"
    ].eq("pilot_train").all()


def test_dataset_v2_rejects_wrong_primary_size_and_wrong_split() -> None:
    with pytest.raises(ValueError, match="400 rows"):
        build_pilot_dataset_v2(_primary(rows=399))
    bad_split = pd.DataFrame(
        [_extension_row("ext_bad", "sha_x", split="pilot_validation")]
    )
    with pytest.raises(ValueError, match="pilot_train"):
        build_pilot_dataset_v2(_primary(), bad_split)


def _passing_report() -> dict:
    models = {name: {} for name in REQUIRED_MODELS_V2}
    models["ridge"]["beats_zero_prediction"] = True
    models["logistic_regression"]["beats_majority_class"] = True
    models["hist_gradient_boosting"]["beats_majority_class"] = True
    return {
        "models": models,
        "split_policy": {
            "train_only_pilot_train": True,
            "eval_primary400_only": True,
        },
    }


def test_gate_v2_verdict_pass_and_never_authorizes_train1600() -> None:
    verdict = evaluate_pilot_gate_v2(
        {"status": "pass", "checks": {}, "headline": {}}, _passing_report()
    )
    assert verdict["status"] == "pass"
    assert verdict["exit_code"] == 0
    assert verdict["train1600_authorization"] == "manual_decision_required"
    assert verdict["auto_entry_into_train1600_prohibited"] is True


def test_gate_v2_verdict_fails_closed_on_any_check() -> None:
    report = _passing_report()
    report["models"]["ridge"]["beats_zero_prediction"] = False
    verdict = evaluate_pilot_gate_v2(
        {"status": "pass", "checks": {}, "headline": {}}, report
    )
    assert verdict["status"] == "scientific_fail"
    assert verdict["exit_code"] == 5
    assert not verdict["checks"]["ridge_beats_zero"]

    dataset_fail = evaluate_pilot_gate_v2(
        {"status": "scientific_fail", "checks": {}, "headline": {}},
        _passing_report(),
    )
    assert dataset_fail["exit_code"] == 5
    assert not dataset_fail["checks"]["dataset_audit_v2_pass"]


def test_finite_or_none_maps_nan_to_null_for_strict_json() -> None:
    # A constant zero-predictor yields NaN spearman; the report must stay
    # writable under allow_nan=False with NaN reported honestly as null.
    cleaned = _finite_or_none(
        {
            "spearman": float("nan"),
            "nested": [{"inf": float("inf")}, 1.5],
            "ok": 0.25,
            "n": 32,
            "flag": True,
        }
    )
    assert cleaned["spearman"] is None
    assert cleaned["nested"][0]["inf"] is None
    assert cleaned["nested"][1] == 1.5
    assert cleaned["ok"] == 0.25
    assert cleaned["n"] == 32
    assert cleaned["flag"] is True
    assert np.isfinite(cleaned["ok"])
