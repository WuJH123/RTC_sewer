from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from sewerrtc.v4.v42_case_alignment_audit import _select_role_rows
from sewerrtc.v4.v42_existing_pool_audit import (
    PhysicalRunRecord,
    ReuseClassification,
    TargetAvailability,
)
from sewerrtc.v4.v42_r0_preflight import assert_r0_schema_preflight
from sewerrtc.v4.v42_r0_strict import (
    AVAILABILITY_COLUMNS,
    _classify_case_schema_safe,
    _enrich_physical_frame,
    _load_scan_cache,
    _postprocess_physical_frame,
    _write_scan_cache,
)


FOUR_ROLES = ("candidate", "no_control", "dynamic_internal", "hold_previous")


def _target(*, outfall: bool = True, semantics: str = "h_is_depth_head_is_hydraulic_head") -> TargetAvailability:
    return TargetAvailability(
        node_depth=True,
        hydraulic_head=True,
        node_flooding_rate=True,
        storage_volume=True,
        managed_facility_flow=True,
        outfall_flow=outfall,
        readback_setting=True,
        rainfall=True,
        history_complete=True,
        horizon_complete=True,
        finite_checked=True,
        finite_pass=True,
        depth_semantics=semantics,
        outfall_reconstruction_candidate=not outfall,
    )


def _record(
    role: str,
    *,
    pid: str | None = None,
    source_experiment: str = "expA",
    source_role: str = "development",
    target: TargetAvailability | None = None,
) -> dict:
    target_value = target or _target()
    item = PhysicalRunRecord(
        source_root="outputs",
        source_experiment=source_experiment,
        run_dir=f"outputs/{source_experiment}/case",
        completion_path=f"outputs/{source_experiment}/case/completion.json",
        detail_path=f"outputs/{source_experiment}/case/{role}_detail.csv",
        detail_sha256=f"detail-{pid or role}",
        detail_size_bytes=123,
        case_id="E1__300__c1",
        event_id="E1",
        rainfall_sha256="rain-sha",
        checkpoint_min=300.0,
        branch_role=role,
        network_sha256="net-sha",
        active_network_sha_match=True,
        domain_id="target_no_dwf",
        source_role=source_role,
        action_readback_sha256=f"action-{pid or role}",
        physical_identity_sha256=pid or f"pid-{source_experiment}-{role}",
        completion_status="pass",
        prefix_hash_match=True,
        checkpoint_hash_match=True,
        target=target_value,
        missing_target_groups=() if target_value.outfall_flow else ("outfall_flow",),
        audit_reasons=(),
        window_anchor_count=1,
    )
    return item.as_dict()


def _group(*, source_experiment: str = "expA") -> pd.DataFrame:
    return _enrich_physical_frame(
        pd.DataFrame(
            [_record(role, source_experiment=source_experiment) for role in FOUR_ROLES]
        )
    )


def test_zero_io_preflight_exercises_real_serializer_and_classifier():
    assert_r0_schema_preflight()


def test_physical_record_serialization_and_classifier_share_prefixed_schema():
    group = _group()
    assert AVAILABILITY_COLUMNS["depth_semantics"] in group.columns
    # The formal classifier must not depend on the old unprefixed accidental name.
    assert "depth_semantics" not in group.columns
    record = _classify_case_schema_safe(group)
    assert record.classification is ReuseClassification.FULL_REUSE
    assert record.four_reference_complete is True
    assert record.branch_count == 4


def test_missing_prefixed_target_field_fails_before_case_classification():
    frame = pd.DataFrame([_record("candidate")]).drop(
        columns=[AVAILABILITY_COLUMNS["depth_semantics"]]
    )
    with pytest.raises(KeyError, match="R0 physical serialization missing columns"):
        _enrich_physical_frame(frame)


def test_duplicate_branch_lineage_is_order_invariant_and_counts_unique_roles():
    rows = [_record(role) for role in FOUR_ROLES]
    rows.append(
        _record(
            "candidate",
            pid="pid-candidate-partial",
            target=_target(outfall=False),
        )
    )
    forward = _enrich_physical_frame(pd.DataFrame(rows))
    reverse = forward.iloc[::-1].reset_index(drop=True)
    a = _classify_case_schema_safe(forward)
    b = _classify_case_schema_safe(reverse)
    assert a.classification is ReuseClassification.FULL_REUSE
    assert b.classification is ReuseClassification.FULL_REUSE
    assert a.case_uid == b.case_uid
    assert a.branch_count == b.branch_count == 4


def test_case_uid_includes_source_experiment_grouping_dimension():
    a = _classify_case_schema_safe(_group(source_experiment="expA"))
    b = _classify_case_schema_safe(_group(source_experiment="expB"))
    assert a.case_uid != b.case_uid


def test_postprocess_keeps_same_case_from_two_sources_as_distinct_case_uids():
    frame = pd.DataFrame(
        [
            *[_record(role, source_experiment="expA") for role in FOUR_ROLES],
            *[_record(role, source_experiment="expB") for role in FOUR_ROLES],
        ]
    )
    result = _postprocess_physical_frame(frame, full_finite_check=True)
    assert len(result.cases) == 2
    assert result.cases["case_uid"].nunique() == 2


def test_persisted_false_string_is_not_cast_to_true():
    frame = pd.DataFrame([_record("candidate")])
    frame.loc[0, AVAILABILITY_COLUMNS["finite_pass"]] = "False"
    checked = _enrich_physical_frame(frame)
    assert bool(checked.loc[0, AVAILABILITY_COLUMNS["finite_pass"]]) is False


def test_alignment_duplicate_role_selection_prefers_audited_complete_row():
    good = SimpleNamespace(
        branch_role="candidate",
        physical_identity_sha256="p-good",
        detail_path="good.csv",
        available_finite_pass=True,
        formal_all_target_complete=True,
        core_trajectory_complete=True,
        available_history_complete=True,
        available_horizon_complete=True,
        available_node_depth=True,
        available_storage_volume=True,
        available_managed_facility_flow=True,
        available_readback_setting=True,
        available_rainfall=True,
    )
    bad = SimpleNamespace(
        branch_role="candidate",
        physical_identity_sha256="p-bad",
        detail_path="bad.csv",
        available_finite_pass=False,
        formal_all_target_complete=False,
        core_trajectory_complete=False,
    )
    selected = _select_role_rows([bad, good])
    assert selected["candidate"].physical_identity_sha256 == "p-good"
    assert _select_role_rows([good, bad])["candidate"].physical_identity_sha256 == "p-good"


def test_scan_cache_roundtrip_is_schema_and_network_guarded(tmp_path: Path):
    project = tmp_path / "project"
    outputs = tmp_path / "outputs"
    network = project / "data" / "wuhan_v8_storage_retrofit.inp"
    network.parent.mkdir(parents=True)
    outputs.mkdir()
    network.write_text("[TITLE]\nfixture\n", encoding="utf-8")

    cache = tmp_path / "logical_cache.csv"
    frame = _group()
    _write_scan_cache(
        frame,
        cache_path=cache,
        project_root=project,
        outputs_root=outputs,
        full_finite_check=True,
    )
    loaded = _load_scan_cache(
        cache_path=cache,
        project_root=project,
        outputs_root=outputs,
        full_finite_check=True,
    )
    assert len(loaded) == len(frame)
    assert AVAILABILITY_COLUMNS["depth_semantics"] in loaded.columns

    with pytest.raises(RuntimeError, match="finite-audit mode mismatch"):
        _load_scan_cache(
            cache_path=cache,
            project_root=project,
            outputs_root=outputs,
            full_finite_check=False,
        )
