from __future__ import annotations

from sewerrtc.data.checkpoint_catalog_contract import CHECKPOINT_CATALOG_FIELDS, validate_checkpoint_catalog_row
from sewerrtc.data.event_catalog_contract import EVENT_CATALOG_FIELDS, validate_event_catalog_row
from sewerrtc.data.opportunity_scan_contract import OPPORTUNITY_SCAN_FIELDS, validate_opportunity_scan_row


def test_event_catalog_schema_contains_formal_leakage_fields() -> None:
    assert "seen_by_GAT" in EVENT_CATALOG_FIELDS
    assert "seen_by_effect_model" in EVENT_CATALOG_FIELDS
    assert "formal_eligibility" in EVENT_CATALOG_FIELDS
    assert "near_duplicate_group" in EVENT_CATALOG_FIELDS
    assert validate_event_catalog_row({})[0].startswith("missing:")


def test_checkpoint_catalog_requires_clone_and_controller_memory() -> None:
    assert "state_clone_source" in CHECKPOINT_CATALOG_FIELDS
    assert "controller_memory_hash" in CHECKPOINT_CATALOG_FIELDS
    assert "eligible_for_effect_training" in CHECKPOINT_CATALOG_FIELDS
    assert validate_checkpoint_catalog_row({})[0].startswith("missing:")


def test_opportunity_scan_separates_threshold_status_and_eligibility() -> None:
    assert "pfv_active_threshold_status" in OPPORTUNITY_SCAN_FIELDS
    assert "opportunity_threshold_status" in OPPORTUNITY_SCAN_FIELDS
    assert "eligible_for_round0" in OPPORTUNITY_SCAN_FIELDS
    assert validate_opportunity_scan_row({})[0].startswith("missing:")
