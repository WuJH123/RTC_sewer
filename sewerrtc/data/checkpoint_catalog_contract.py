"""Checkpoint catalog schema for Project6 V3."""

from __future__ import annotations

from typing import Mapping


CHECKPOINT_CATALOG_FIELDS = (
    "checkpoint_id",
    "event_id",
    "storm_family_id",
    "event_time",
    "phase",
    "state_source",
    "state_clone_source",
    "node_state_hash",
    "link_state_hash",
    "storage_state_hash",
    "controller_memory_hash",
    "rainfall_history_hash",
    "forecast_issue_id",
    "internal_current_action_signature",
    "passive_current_action_signature",
    "priority_risk_summary",
    "sentinel_risk_summary",
    "storage_remaining_capacity",
    "downstream_headroom",
    "pump_dwell_state",
    "state_cluster_id",
    "split",
    "eligible_for_effect_training",
    "exclusion_reason",
)


def validate_checkpoint_catalog_row(row: Mapping[str, object]) -> list[str]:
    return [f"missing:{field}" for field in CHECKPOINT_CATALOG_FIELDS if field not in row]
