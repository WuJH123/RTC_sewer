"""Event catalog schema for Project6 V3."""

from __future__ import annotations

from typing import Mapping


EVENT_CATALOG_FIELDS = (
    "event_id",
    "canonical_event_id",
    "storm_family_id",
    "source_project",
    "rainfall_source_path",
    "rainfall_sha256",
    "time_series_signature",
    "start_time",
    "duration_min",
    "total_depth",
    "peak_intensity",
    "peak_time",
    "peak_count",
    "single_or_multi_peak",
    "antecedent_condition",
    "intended_split",
    "seen_by_GAT",
    "seen_by_effect_model",
    "seen_by_calibration",
    "seen_by_human_failure_analysis",
    "formal_eligibility",
    "near_duplicate_group",
    "provenance_status",
)


def validate_event_catalog_row(row: Mapping[str, object]) -> list[str]:
    return [f"missing:{field}" for field in EVENT_CATALOG_FIELDS if field not in row]
