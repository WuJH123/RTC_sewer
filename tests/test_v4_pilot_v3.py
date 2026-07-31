"""Synthetic-data regression for Dataset v3, Baselines v3 and the Gate v3 verdict."""

import json

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.partial_audit import HARD_AUTHENTICITY_COLUMNS
from sewerrtc.v4.pilot_v3 import (
    CLASS_TO_STATE_LABEL,
    CONTRACT_STATE_COLUMNS,
    audit_pilot_dataset_v3,
    build_pilot_dataset_v3,
    evaluate_pilot_gate_v3,
    train_pilot_baselines_v3,
)

_SPLITS = (
    ["pilot_train"] * 240
    + ["pilot_calibration"] * 60
    + ["pilot_validation"] * 50
    + ["pilot_challenge"] * 50
)


def _v2(rows: int = 400) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": [f"v2_{index:04d}" for index in range(rows)],
            "event_id": [f"E{index % 8}" for index in range(rows)],
            "checkpoint_id": [f"C{index % 40}" for index in range(rows)],
            "split": _SPLITS[:rows],
            "actual_schedule_sha256": [
                f"sha_v2_{index}" for index in range(rows)
            ],
            "source_phase": "primary400",
            "checkpoint_min": 60.0,
        }
    )
    extras = pd.DataFrame(
        [
            {
                "sample_id": f"{phase}_{index}",
                "event_id": "E0",
                "checkpoint_id": "C0",
                "split": "pilot_train",
                "actual_schedule_sha256": f"sha_{phase}_{index}",
                "source_phase": phase,
                "checkpoint_min": 60.0,
            }
            for phase, count in (("joint_extension", 3), ("flat_auxiliary", 2))
            for index in range(count)
        ]
    )
    frame = pd.concat([frame, extras], ignore_index=True)
    frame["rainfall_sha256"] = [
        f"r_{event}_{split}"
        for event, split in zip(frame["event_id"], frame["split"])
    ]
    for column in HARD_AUTHENTICITY_COLUMNS:
        frame[column] = True
    frame["joint_noninferior"] = False
    frame.loc[frame.index[:6], "joint_noninferior"] = True
    frame["confirmed_flat"] = False
    # Mirrors the frozen v2 fact: 14 confirmed-flat rows with event support 1.
    flat_rows = frame.index[frame["event_id"] == "E4"][:14]
    frame.loc[flat_rows, "confirmed_flat"] = True
    return frame


def _map_frame() -> pd.DataFrame:
    # C0 (train, robust), C1 (train, no_joint), C240=calibration state C0?
    # Splits follow the v2 layout: C0/C1 are pilot_train states, C242 is a
    # calibration state, C282 a validation state (oracle revealed).
    return pd.DataFrame(
        [
            {
                "event_id": "E0",
                "checkpoint_id": "C0",
                "split": "pilot_train",
                "rainfall_phase": "mid",
                "checkpoint_min": 60.0,
                "state_feasibility_class": "joint_feasible_robust",
                "exact_joint_found": True,
                "online_joint_found": True,
                "candidate_generator_hit": True,
                "oracle_revealed": False,
                "feasibility_samples": 4,
            },
            {
                "event_id": "E1",
                "checkpoint_id": "C1",
                "split": "pilot_train",
                "rainfall_phase": "early",
                "checkpoint_min": 60.0,
                "state_feasibility_class": "no_joint_found_under_budget",
                "exact_joint_found": False,
                "online_joint_found": False,
                "candidate_generator_hit": False,
                "oracle_revealed": False,
                "feasibility_samples": 16,
            },
            {
                "event_id": "E2",
                "checkpoint_id": "C2",
                "split": "pilot_calibration",
                "rainfall_phase": "mid",
                "checkpoint_min": 60.0,
                "state_feasibility_class": "no_pfv_safe_found",
                "exact_joint_found": False,
                "online_joint_found": False,
                "candidate_generator_hit": False,
                "oracle_revealed": True,
                "feasibility_samples": 16,
            },
        ]
    )


def _feasibility() -> pd.DataFrame:
    rows = [
        # Training-eligible search rows on the two pilot_train states.
        {
            "sample_id": "feas_000",
            "event_id": "E0",
            "checkpoint_id": "C0",
            "split": "pilot_train",
            "actual_schedule_sha256": "sha_feas_0",
            "source_phase": "feasibility_p3",
            "search_result_training_eligible": True,
            "expected_replay_of": None,
        },
        {
            "sample_id": "feas_001",
            "event_id": "E1",
            "checkpoint_id": "C1",
            "split": "pilot_train",
            "actual_schedule_sha256": "sha_feas_1",
            "source_phase": "feasibility_p3",
            "search_result_training_eligible": True,
            "expected_replay_of": None,
        },
        # Positive-control replay intentionally repeats a v2 actual SHA.
        {
            "sample_id": "feas_replay",
            "event_id": "E0",
            "checkpoint_id": "C0",
            "split": "pilot_train",
            "actual_schedule_sha256": "sha_v2_0",
            "source_phase": "feasibility_p3",
            "search_result_training_eligible": True,
            "expected_replay_of": "v2_0000",
        },
        # Calibration-state search row: never training-eligible.
        {
            "sample_id": "feas_cal",
            "event_id": "E2",
            "checkpoint_id": "C2",
            "split": "pilot_calibration",
            "actual_schedule_sha256": "sha_feas_cal",
            "source_phase": "feasibility_p3",
            "search_result_training_eligible": False,
            "expected_replay_of": None,
        },
    ]
    frame = pd.DataFrame(rows)
    frame["rainfall_sha256"] = [
        f"r_{event}_{split}"
        for event, split in zip(frame["event_id"], frame["split"])
    ]
    for column in HARD_AUTHENTICITY_COLUMNS:
        frame[column] = True
    frame["joint_noninferior"] = [True, False, True, False]
    frame["confirmed_flat"] = False
    frame["checkpoint_min"] = 60.0
    return frame


def _built() -> dict:
    return build_pilot_dataset_v3(_v2(), _feasibility(), _map_frame())


def test_dataset_v3_merges_and_applies_contract_columns() -> None:
    result = _built()
    samples = result["sample_manifest"]
    assert result["accounting"]["accounting_closed"]
    assert len(samples) == 405 + 4
    for column in CONTRACT_STATE_COLUMNS:
        assert column in samples
    # State label mapping follows the frozen contract mapping.
    by_state = samples.drop_duplicates(["event_id", "checkpoint_id"]).set_index(
        ["event_id", "checkpoint_id"]
    )
    assert (
        by_state.loc[("E0", "C0"), "state_level_label"]
        == CLASS_TO_STATE_LABEL["joint_feasible_robust"]
    )
    assert (
        by_state.loc[("E1", "C1"), "state_level_label"]
        == "fallback_only_under_budget"
    )
    # Unsearched states carry no valid feasibility label.
    assert by_state.loc[("E3", "C3"), "state_feasibility_class"] == "not_searched"
    assert (
        by_state.loc[("E3", "C3"), "feasibility_label_validity"]
        == "not_searched"
    )


def test_dataset_v3_training_and_evaluation_isolation() -> None:
    samples = _built()["sample_manifest"]
    train = samples[samples["eligible_for_training"]]
    assert train["split"].eq("pilot_train").all()
    # Calibration-state search row is present but never trainable.
    cal_row = samples[samples["sample_id"] == "feas_cal"].iloc[0]
    assert not cal_row["eligible_for_training"]
    assert not cal_row["eligible_for_evaluation"]
    # Oracle-revealed states are excluded from unseen evaluation and
    # re-flagged as diagnostic-only rows.
    eval_rows = samples[samples["eligible_for_evaluation"]]
    assert not eval_rows["oracle_revealed_state"].any()
    assert eval_rows["source_phase"].eq("primary400").all()
    revealed = samples[samples["oracle_revealed_evaluation_only"]]
    assert (
        revealed["event_id"].eq("E2") & revealed["checkpoint_id"].eq("C2")
    ).all()
    assert len(revealed) > 0


def test_dataset_v3_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="400 primary400"):
        build_pilot_dataset_v3(
            _v2().iloc[:200], _feasibility(), _map_frame()
        )
    bad_phase = _feasibility()
    bad_phase["source_phase"] = "primary400"
    with pytest.raises(ValueError, match="feasibility_p3"):
        build_pilot_dataset_v3(_v2(), bad_phase, _map_frame())
    overlap = _feasibility()
    overlap.loc[0, "sample_id"] = "v2_0000"
    with pytest.raises(ValueError, match="overlap"):
        build_pilot_dataset_v3(_v2(), overlap, _map_frame())


def _audit(samples: pd.DataFrame) -> dict:
    return audit_pilot_dataset_v3(
        samples,
        expected_v2_counts={
            "primary400": 400,
            "joint_extension": 3,
            "flat_auxiliary": 2,
        },
        expected_feasibility_rows=4,
    )


def test_dataset_v3_audit_passes_and_allows_replay_duplicates() -> None:
    audit = _audit(_built()["sample_manifest"])
    assert audit["status"] == "pass", audit["checks"]
    assert audit["checks"]["nonreplay_actual_disjoint_from_v2"]
    assert audit["checks"]["actual_duplicates_0"]
    assert audit["headline"]["replay_rows"] == 1
    assert audit["headline"]["confirmed_flat_count"] == 14
    assert not audit["headline"]["flat_head_enabled"]
    # Strict JSON: the audit payload must serialise without NaN.
    json.dumps(audit, allow_nan=False)


def test_dataset_v3_audit_fails_closed_on_violations() -> None:
    # Non-replay duplicate of a v2 actual schedule in the same state.
    tampered = _built()["sample_manifest"].copy()
    index = tampered.index[tampered["sample_id"] == "feas_000"][0]
    tampered.loc[index, "actual_schedule_sha256"] = "sha_v2_0"
    audit = _audit(tampered)
    assert audit["status"] == "scientific_fail"
    assert not audit["checks"]["nonreplay_actual_disjoint_from_v2"]

    # An eval-split search row marked trainable violates isolation.
    leaky = _built()["sample_manifest"].copy()
    index = leaky.index[leaky["sample_id"] == "feas_cal"][0]
    leaky.loc[index, "eligible_for_training"] = True
    audit = _audit(leaky)
    assert audit["status"] == "scientific_fail"
    assert not audit["checks"]["no_eval_split_search_rows_trainable"]
    assert not audit["checks"]["training_only_pilot_train"]

    # Forbidden class vocabulary is rejected.
    forbidden = _built()["sample_manifest"].copy()
    forbidden.loc[
        forbidden["checkpoint_id"] == "C1", "state_feasibility_class"
    ] = "physically_infeasible"
    audit = _audit(forbidden)
    assert audit["status"] == "scientific_fail"
    assert not audit["checks"]["no_forbidden_class_terms"]

    # An oracle-revealed row flagged evaluation-eligible is rejected.
    revealed = _built()["sample_manifest"].copy()
    mask = revealed["oracle_revealed_evaluation_only"].astype(bool)
    revealed.loc[mask, "eligible_for_evaluation"] = True
    audit = _audit(revealed)
    assert audit["status"] == "scientific_fail"
    assert not audit["checks"]["evaluation_excludes_oracle_revealed"]


def _schedule(delta: float) -> str:
    matrix = np.zeros((12, 3))
    matrix[0, 0] = delta
    return json.dumps(matrix.tolist())


def _training_samples(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    splits = ["pilot_train"] * (rows // 2) + [
        "pilot_validation",
        "pilot_calibration",
        "pilot_challenge",
        "pilot_validation",
    ] * (rows // 8)
    splits = splits[:rows]
    move = rng.uniform(0.0, 1.0, rows)
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(rows)],
            "event_id": [f"E{index % 4}" for index in range(rows)],
            "checkpoint_id": [f"C{index % 8}" for index in range(rows)],
            "split": splits,
            "source_phase": "primary400",
            "projected_schedule_json": [_schedule(value) for value in move],
            "anchor_schedule_json": _schedule(0.0),
            "k_target": 4,
            "k_actual": 4,
            "checkpoint_min": 60.0,
            "candidate_family": [
                ("toward_hold" if index % 2 else "storage_headroom")
                for index in range(rows)
            ],
            "delta_pfv_h120_vs_no_control": move * 2.0,
            "delta_tfv_h120_vs_dynamic_internal": move * 10.0 - 5.0,
            "delta_peak_h120_vs_dynamic_internal": move * 0.01 - 0.005,
            "pfv_safe": move < 0.8,
            "tfv_noninferior": move < 0.5,
            "peak_noninferior": move < 0.6,
        }
    )
    frame["joint_noninferior"] = (
        frame["pfv_safe"] & frame["tfv_noninferior"] & frame["peak_noninferior"]
    )
    frame["eligible_for_training"] = frame["split"].eq("pilot_train")
    frame["eligible_for_evaluation"] = frame["split"].isin(
        ["pilot_calibration", "pilot_validation", "pilot_challenge"]
    )
    frame["oracle_revealed_evaluation_only"] = False
    return frame


def _state_manifest() -> pd.DataFrame:
    labels = (
        ["intervention_feasible_found"] * 3
        + ["fallback_only_under_budget"] * 5
        + ["boundary_or_uncertain"] * 2
    )
    return pd.DataFrame(
        {
            "event_id": [f"E{index % 4}" for index in range(10)],
            "checkpoint_id": [f"C{index}" for index in range(10)],
            "split": ["pilot_train"] * 8 + ["pilot_validation"] * 2,
            "checkpoint_min": [40.0 + 10 * index for index in range(10)],
            "rainfall_phase": ["early", "mid"] * 5,
            "state_level_label": labels,
            "feasibility_label_validity": "valid",
            "oracle_revealed_state": [False] * 8 + [True] * 2,
        }
    )


def test_train_baselines_v3_two_tiers_and_no_leakage() -> None:
    report = train_pilot_baselines_v3(
        _training_samples(), _state_manifest(), n_boot=20
    )
    policy = report["split_policy"]
    assert policy["train_only_pilot_train"]
    assert policy["eval_primary400_only"]
    assert policy["eval_excludes_oracle_revealed"]
    assert policy["no_exact_search_features"]
    # No post-run or search-derived feature ever enters the models.
    for column in report["feature_columns"]:
        assert "delta" not in column and "search" not in column
    composed = report["composed_joint"]["hist_gradient_boosting"]
    assert composed["trained"]
    assert "no standalone joint" in composed["policy"]
    assert "pilot_validation" in composed["splits"]
    # State head trains on train states only; the two non-train states are
    # oracle-revealed, so no unseen state evaluation exists.
    state = report["state_level"]
    assert state["train_states"] == 8
    assert state["unseen_state_evaluation"] is None
    assert "oracle_revealed" in state["unseen_state_evaluation_reason"]
    json.dumps(report, allow_nan=False)


def _passing_gate_inputs() -> tuple[dict, dict, dict]:
    dataset_audit = {
        "status": "pass",
        "checks": {"hard_authenticity_100pct": True},
        "headline": {
            "locked_validation_feasible_states_unrevealed": 5,
            "evaluation_feasible_states_unrevealed": 8,
            "fallback_only_event_support": 3,
        },
    }
    baseline_report = {
        "models": {
            "ridge": {"validation_rmse_improvement_vs_zero": 0.25},
            "composed_joint_hgb": {
                "validation_balanced_accuracy": 0.7,
                "validation_mcc": 0.3,
                "validation_auprc": 0.6,
                "validation_false_safe_rate": 0.1,
                "validation_positives": 10,
                "validation_n": 50,
            },
            "top5_feasible_recall_validation": 0.9,
        }
    }
    feasibility_audit = {
        "replay_success_rate": 1.0,
        "recall_report": {
            "exact_joint_feasible_states": 9,
            "online_generator_joint_states": 9,
            "candidate_generator_state_recall": 1.0,
            "event_support": 4,
        },
        "p3_gate": {"unresolved_states": 0},
    }
    return dataset_audit, baseline_report, feasibility_audit


def test_gate_v3_pass_never_auto_enters_train1600() -> None:
    verdict = evaluate_pilot_gate_v3(*_passing_gate_inputs())
    assert verdict["status"] == "pass"
    assert verdict["exit_code"] == 0
    assert verdict["train1600_authorized"] is True
    assert verdict["train1600_authorization"] == "authorized_manual_start_only"
    assert verdict["auto_entry_into_train1600_prohibited"] is True
    assert verdict["flat_fraction_core_blocking_gate"] is False


def test_gate_v3_underpowered_validation_blocks_train1600() -> None:
    dataset_audit, baseline_report, feasibility_audit = _passing_gate_inputs()
    dataset_audit["headline"][
        "locked_validation_feasible_states_unrevealed"
    ] = 2
    verdict = evaluate_pilot_gate_v3(
        dataset_audit, baseline_report, feasibility_audit
    )
    assert verdict["status"] == "underpowered_validation"
    assert verdict["exit_code"] == 5
    assert verdict["scientific_pass"] is False
    assert verdict["train1600_authorized"] is False
    assert verdict["underpowered_validation"]["triggered"]
    assert "never reuse oracle-searched states" in verdict[
        "underpowered_validation"
    ]["consequence"]


def test_gate_v3_fails_closed_on_any_check() -> None:
    dataset_audit, baseline_report, feasibility_audit = _passing_gate_inputs()
    baseline_report["models"]["composed_joint_hgb"][
        "validation_false_safe_rate"
    ] = 0.5
    verdict = evaluate_pilot_gate_v3(
        dataset_audit, baseline_report, feasibility_audit
    )
    assert verdict["status"] == "scientific_fail"
    assert verdict["exit_code"] == 5
    assert not verdict["checks"]["false_safe_at_most_0p20"]
    assert verdict["train1600_authorized"] is False

    # Missing metrics (None) must fail closed, never pass silently.
    dataset_audit, baseline_report, feasibility_audit = _passing_gate_inputs()
    baseline_report["models"]["ridge"][
        "validation_rmse_improvement_vs_zero"
    ] = None
    verdict = evaluate_pilot_gate_v3(
        dataset_audit, baseline_report, feasibility_audit
    )
    assert verdict["status"] == "scientific_fail"
    assert not verdict["checks"]["ridge_rmse_improvement_at_least_10pct"]
