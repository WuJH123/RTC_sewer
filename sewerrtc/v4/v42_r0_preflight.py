"""Zero-I/O schema preflight for the expensive V4.2 Phase-R0 scan.

This check deliberately runs before the first historical CSV is opened.  It
constructs the same dataclass objects used by the real audit, serializes them
through ``PhysicalRunRecord.as_dict()``, and verifies that the formal strict
classifier consumes exactly that schema.  A future serializer/consumer rename
therefore fails in milliseconds instead of after a multi-hour scan.
"""
from __future__ import annotations

import pandas as pd

from . import v42_existing_pool_audit as base
from .v42_r0_strict import (
    AVAILABILITY_COLUMNS,
    _classify_case_schema_safe,
    _enrich_physical_frame,
)


ROLES = ("candidate", "no_control", "dynamic_internal", "hold_previous")


def _synthetic_record(role: str, idx: int) -> base.PhysicalRunRecord:
    target = base.TargetAvailability(
        node_depth=True,
        hydraulic_head=True,
        node_flooding_rate=True,
        storage_volume=True,
        managed_facility_flow=True,
        outfall_flow=True,
        readback_setting=True,
        rainfall=True,
        history_complete=True,
        horizon_complete=True,
        finite_checked=True,
        finite_pass=True,
        depth_semantics="h_is_depth_head_is_hydraulic_head",
        outfall_reconstruction_candidate=False,
    )
    return base.PhysicalRunRecord(
        source_root="preflight",
        source_experiment="preflight",
        run_dir="preflight",
        completion_path=None,
        detail_path=f"preflight_{role}.csv",
        detail_sha256=f"detail-{idx}",
        detail_size_bytes=1,
        case_id="PRECHECK__60__c0",
        event_id="PRECHECK",
        rainfall_sha256="rain",
        checkpoint_min=60.0,
        branch_role=role,
        network_sha256="network",
        active_network_sha_match=True,
        domain_id="target_no_dwf",
        source_role="development",
        action_readback_sha256=f"action-{idx}",
        physical_identity_sha256=f"physical-{idx}",
        completion_status="pass",
        prefix_hash_match=True,
        checkpoint_hash_match=True,
        target=target,
        missing_target_groups=(),
        audit_reasons=(),
        window_anchor_count=1,
    )


def assert_r0_schema_preflight() -> None:
    """Raise immediately if serializer and formal classifier contracts diverge."""
    rows = [_synthetic_record(role, idx).as_dict() for idx, role in enumerate(ROLES)]
    frame = pd.DataFrame(rows)
    expected = set(AVAILABILITY_COLUMNS.values())
    missing = sorted(expected - set(frame.columns))
    if missing:
        raise RuntimeError(f"R0 serializer preflight missing availability columns: {missing}")
    # There must be one canonical representation, not a shadow alias that can
    # let a stale consumer pass tests while production uses another name.
    accidental_aliases = sorted(
        name
        for name in AVAILABILITY_COLUMNS
        if name in frame.columns and name != AVAILABILITY_COLUMNS[name]
    )
    if accidental_aliases:
        raise RuntimeError(
            f"R0 serializer preflight found forbidden unprefixed aliases: {accidental_aliases}"
        )
    checked = _enrich_physical_frame(frame)
    case = _classify_case_schema_safe(checked)
    if not case.four_reference_complete or case.branch_count != 4:
        raise RuntimeError("R0 classifier preflight failed four-reference contract")
    if case.classification is not base.ReuseClassification.FULL_REUSE:
        raise RuntimeError(
            f"R0 classifier preflight expected FULL_REUSE, got {case.classification.value}"
        )
