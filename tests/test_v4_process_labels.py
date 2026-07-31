"""Section 2.6: process (temporal residual) label completeness.

All seven process residual columns must be present and populated as 12-step
future series (120 min horizon / 10 min step). Peak timing must be derivable
in place from the existing per-step curves without a new SWMM run, and the
original frozen manifest must stay read-only.
"""
from __future__ import annotations

import json

from sewerrtc.v4.dataset import TEMPORAL_RESIDUALS
from sewerrtc.v4.train1600_v4 import (
    HORIZON_STEPS,
    audit_train1600_learnability_v4,
    residual_schema_audit,
)
from v4_readiness_helpers import (
    FROZEN_DEAD_ZONES,
    FROZEN_MARGINS,
    make_manifest,
)


def test_complete_residuals_pass_the_schema_audit() -> None:
    audit = residual_schema_audit(make_manifest())

    assert audit["all_process_residuals_complete"] is True
    assert audit["original_manifest_read_only"] is True
    assert audit["enriched_manifest_required"] is False
    for column in TEMPORAL_RESIDUALS:
        entry = audit["columns"][column]
        assert entry["present"] is True
        assert entry["all_horizon_steps"] is True
        assert entry["step_lengths"] == [HORIZON_STEPS]
    json.dumps(audit, allow_nan=False)


def test_peak_timing_is_derivable_without_a_new_swmm_run() -> None:
    audit = residual_schema_audit(make_manifest())
    peak = audit["peak_timing"]

    assert peak["derivable_from_existing_residual_series"] is True
    assert peak["requires_new_swmm"] is False
    assert peak["materialized_column"] is False


def test_truncated_residual_series_is_incomplete() -> None:
    df = make_manifest()
    df["priority_depth_residual"] = json.dumps([0.0] * 6)  # 6 != 12 steps

    audit = residual_schema_audit(df)

    assert audit["all_process_residuals_complete"] is False
    assert audit["enriched_manifest_required"] is True
    assert audit["columns"]["priority_depth_residual"]["all_horizon_steps"] is False


def test_missing_residual_column_is_incomplete() -> None:
    df = make_manifest().drop(columns=["storage_volume_residual"])

    audit = residual_schema_audit(df)

    assert audit["all_process_residuals_complete"] is False
    assert audit["columns"]["storage_volume_residual"]["present"] is False


def test_incomplete_residuals_hard_fail_the_readiness_verdict() -> None:
    df = make_manifest()
    df["tfv_rate_residual"] = json.dumps([0.0] * 3)

    verdict = audit_train1600_learnability_v4(
        df, margins=FROZEN_MARGINS, dead_zones=FROZEN_DEAD_ZONES
    )["verdict"]

    assert verdict["training_readiness"] == "fail"
    assert "process_residual_schema_incomplete" in verdict["hard_failures"]
