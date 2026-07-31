"""Section 2.5: input feature leakage contract.

The model input schema must never expose outcome labels, delta KPIs, temporal
residuals, ranking / regret labels, or any realised future SWMM value. Rainfall
forecast is the only permitted future signal.
"""
from __future__ import annotations

import json

from sewerrtc.v4.dataset import CONTINUOUS_LABELS, TEMPORAL_RESIDUALS
from sewerrtc.v4.train1600_v4 import (
    PROHIBITED_INPUT_COLUMNS,
    feature_leakage_audit,
)


def test_input_schema_has_zero_leakage() -> None:
    audit = feature_leakage_audit()

    assert audit["leakage_count"] == 0
    assert audit["leakage_fields"] == []
    assert audit["no_future_swmm_state_in_inputs"] is True
    assert audit["allowed_future_rainfall_forecast"] is True
    json.dumps(audit, allow_nan=False)


def test_prohibited_columns_cover_labels_deltas_and_residuals() -> None:
    prohibited = set(PROHIBITED_INPUT_COLUMNS)

    for label in ("pfv_safe", "tfv_improved", "peak_noninferior"):
        assert label in prohibited
    for column in CONTINUOUS_LABELS:
        assert column in prohibited
    for column in TEMPORAL_RESIDUALS:
        assert column in prohibited
    # Regret / ranking outcomes are future information and must be excluded.
    assert "regret_to_exact_best" in prohibited
    assert "feasible_rank" in prohibited


def test_allowed_inputs_exclude_every_prohibited_column() -> None:
    audit = feature_leakage_audit()
    allowed = set(audit["allowed_input_fields"])

    assert allowed.isdisjoint(set(PROHIBITED_INPUT_COLUMNS))
    # Identity + provenance style inputs are allowed (paths / SHAs only).
    assert "rainfall_sha256" in allowed or "network_sha256" in allowed


def test_branch_evidence_exposes_paths_not_realised_values() -> None:
    audit = feature_leakage_audit()
    allowed = set(audit["allowed_input_fields"])

    # No realised branch KPI / depth / flow / delta values may be inputs.
    assert not any(
        field.endswith(("_kpi", "_depth", "_flow", "_delta"))
        for field in allowed
    )
