"""Internal PFV opportunity scan schema for Project6 V3."""

from __future__ import annotations

from typing import Mapping


OPPORTUNITY_SCAN_FIELDS = (
    "event_id",
    "checkpoint_id",
    "internal_predicted_pfv",
    "passive_predicted_pfv",
    "internal_passive_pfv_opportunity",
    "uncertainty",
    "numerical_noise_floor",
    "pfv_active_threshold_status",
    "opportunity_threshold_status",
    "eligible_for_round0",
    "exclusion_reason",
)


def validate_opportunity_scan_row(row: Mapping[str, object]) -> list[str]:
    return [f"missing:{field}" for field in OPPORTUNITY_SCAN_FIELDS if field not in row]
