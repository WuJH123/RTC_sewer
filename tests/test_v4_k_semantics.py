"""Section 2.2: K semantics against the online candidate generator.

K is computed against the actual executed schedule (fallback anchor). The
frozen training domain contains no K=1/K=2, but the online generator emits
K in {1,2,4,6,8}; the audit must flag that the online domain exceeds the
training domain, require a Train-only K=1/K=2 supplement, and disable online
K=1/K=2 until backfilled -- never claim coverage it lacks.
"""
from __future__ import annotations

import json

from sewerrtc.v4.train1600_v4 import (
    ONLINE_CANDIDATE_K_VALUES,
    k_semantics_audit,
)
from v4_readiness_helpers import make_manifest


def test_online_k1_k2_absent_from_training_triggers_supplement() -> None:
    df = make_manifest(k_values=(4, 6, 8))

    records, summary = k_semantics_audit(
        df, online_k_values=ONLINE_CANDIDATE_K_VALUES
    )

    assert summary["training_contains_k1_or_k2"] is False
    assert summary["online_contains_k1_or_k2"] is True
    assert summary["online_domain_exceeds_training_domain"] is True
    assert summary["k1_k2_supplement_required"] is True
    assert summary["disable_online_k1_k2_until_backfilled"] is True
    assert summary["k_computed_against"] == "actual_executed_fallback_anchor"
    # Supplement is Train-only and must never touch Calibration/Locked.
    plan = summary["supplement_plan"]
    assert plan["scope"] == "train_events_only"
    assert plan["must_not_touch_calibration_or_locked"] is True
    assert plan["target_k_values"] == [1, 2]
    json.dumps({"records": records, "summary": summary}, allow_nan=False)


def test_k_records_flag_online_and_training_membership() -> None:
    df = make_manifest(k_values=(4, 6, 8))

    records, _ = k_semantics_audit(
        df, online_k_values=ONLINE_CANDIDATE_K_VALUES
    )
    by_k = {row["k"]: row for row in records}

    # K=1/K=2 are online-only: in generator, absent from training domain.
    for k in (1, 2):
        assert by_k[k]["in_online_generator"] is True
        assert by_k[k]["in_training_domain"] is False
        assert by_k[k]["actual_samples"] == 0
    # K=4 is present in the training domain.
    assert by_k[4]["in_training_domain"] is True
    assert by_k[4]["actual_samples"] > 0


def test_training_domain_with_k1_k2_needs_no_supplement() -> None:
    df = make_manifest(k_values=(1, 2, 4, 6, 8))

    _, summary = k_semantics_audit(
        df, online_k_values=ONLINE_CANDIDATE_K_VALUES
    )

    assert summary["training_contains_k1_or_k2"] is True
    assert summary["online_domain_exceeds_training_domain"] is False
    assert summary["k1_k2_supplement_required"] is False
    assert summary["disable_online_k1_k2_until_backfilled"] is False
    assert summary["supplement_plan"] is None
