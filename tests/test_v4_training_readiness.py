"""Section 2/3: training readiness verdict and model-training authorization.

Verifies that the frozen-evidence audit correctly (a) returns a
``conditional_pass`` when the only issues are pending obligations, (b) hard
fails on real learnability defects (split leakage, single-sided core labels),
and (c) authorizes training under ``conditional_pass`` while keeping the Model
Safety Gate deferred.
"""
from __future__ import annotations

import json

from sewerrtc.v4.train1600_v4 import (
    audit_train1600_learnability_v4,
    evaluate_model_training_authorization_v4,
)
from v4_readiness_helpers import (
    FROZEN_DEAD_ZONES,
    FROZEN_MARGINS,
    make_learnability_payload,
    make_manifest,
)


def _audit(df):
    return audit_train1600_learnability_v4(
        df, margins=FROZEN_MARGINS, dead_zones=FROZEN_DEAD_ZONES
    )


def test_default_manifest_is_conditional_pass_not_hard_fail() -> None:
    audit = _audit(make_manifest())
    verdict = audit["verdict"]

    assert verdict["training_readiness"] == "conditional_pass"
    assert verdict["hard_failures"] == []
    # K=1/K=2 online-only mismatch is the mandated conditional reason.
    assert "online_k1_k2_not_in_training_domain" in verdict["conditional_reasons"]
    assert verdict["training_permitted"] is True
    assert verdict["feature_leakage_count"] == 0
    assert verdict["split_leakage_free"] is True
    json.dumps(audit, allow_nan=False)


def test_split_rainfall_leakage_is_a_hard_failure() -> None:
    df = make_manifest()
    leaked = df.loc[df["split"] == "calibration", "rainfall_sha256"].iloc[0]
    df.loc[df["split"] == "train", "rainfall_sha256"] = leaked

    verdict = _audit(df)["verdict"]

    assert verdict["training_readiness"] == "fail"
    assert "split_rainfall_sha_leakage" in verdict["hard_failures"]
    assert verdict["training_permitted"] is False


def test_single_sided_train_core_label_is_a_hard_failure() -> None:
    df = make_manifest()
    df.loc[df["split"] == "train", "pfv_safe"] = True  # collapse to one side

    verdict = _audit(df)["verdict"]

    assert verdict["training_readiness"] == "fail"
    assert "train_core_label_single_sided" in verdict["hard_failures"]


def test_degenerate_continuous_label_is_a_hard_failure() -> None:
    df = make_manifest()
    df["delta_pfv_h120_vs_no_control"] = 0.0

    verdict = _audit(df)["verdict"]

    assert verdict["training_readiness"] == "fail"
    assert "continuous_label_degenerate" in verdict["hard_failures"]


def test_authorization_passes_under_conditional_pass() -> None:
    audit = _audit(make_manifest())
    result = evaluate_model_training_authorization_v4(
        dataset_audit={"status": "pass"},
        learnability=make_learnability_payload(audit),
        freeze={"immutable": True},
        unresolved_files=(),
    )

    assert result["status"] == "pass"
    assert result["model_training_authorized"] is True
    assert result["training_readiness"] == "conditional_pass"
    assert all(result["conditions"].values())
    # Conditional obligations recorded, never silently dropped.
    assert result["pending_obligations"]["k1_k2_supplement_required"] is True
    assert result["pending_obligations"]["disable_online_k1_k2"] is True
    # Model Safety Gate must stay deferred; no downstream authorized.
    assert result["model_safety_gate_status"] == "deferred"
    assert "policy_lock" in result["does_not_authorize"]
    json.dumps(result, allow_nan=False)


def test_authorization_blocks_when_readiness_fails() -> None:
    df = make_manifest()
    df.loc[df["split"] == "train", "peak_noninferior"] = True  # single-sided
    audit = _audit(df)
    result = evaluate_model_training_authorization_v4(
        dataset_audit={"status": "pass"},
        learnability=make_learnability_payload(audit),
        freeze={"immutable": True},
        unresolved_files=(),
    )

    assert result["status"] == "blocked"
    assert result["model_training_authorized"] is False
    assert result["conditions"]["train_core_labels_two_sided"] is False


def test_authorization_blocks_on_unresolved_frozen_files() -> None:
    audit = _audit(make_manifest())
    result = evaluate_model_training_authorization_v4(
        dataset_audit={"status": "pass"},
        learnability=make_learnability_payload(audit),
        freeze={"immutable": True},
        unresolved_files=("dataset/train1600_v3_sample_manifest.csv",),
    )

    assert result["model_training_authorized"] is False
    assert result["conditions"]["unresolved_files_zero"] is False


def test_authorization_blocks_when_dataset_gate_not_pass() -> None:
    audit = _audit(make_manifest())
    result = evaluate_model_training_authorization_v4(
        dataset_audit={"status": "blocked"},
        learnability=make_learnability_payload(audit),
        freeze={"immutable": True},
        unresolved_files=(),
    )

    assert result["model_training_authorized"] is False
    assert result["conditions"]["dataset_gate_pass"] is False


def test_authorization_blocks_when_freeze_not_immutable() -> None:
    audit = _audit(make_manifest())
    result = evaluate_model_training_authorization_v4(
        dataset_audit={"status": "pass"},
        learnability=make_learnability_payload(audit),
        freeze={"immutable": False},
        unresolved_files=(),
    )

    assert result["model_training_authorized"] is False
    assert result["conditions"]["freeze_immutable"] is False
