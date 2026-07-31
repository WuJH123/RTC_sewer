from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sewerrtc.state.gat_registry import DEFAULT_PROJECT4_GAT_CANDIDATES
from sewerrtc.state.local_flow_features import FlowFeature, FlowSource, unavailable_flow, validate_flow_feature
from sewerrtc.state.state_contract import FACILITY_STATE_FIELDS, NODE_STATE_FIELDS, PUMP_STATE_FIELDS, TEMPORAL_FRAME_OFFSETS_MIN
from sewerrtc.state.temporal_state_buffer import assert_no_future_observation, build_temporal_frame_schedule


ROOT = Path(__file__).resolve().parents[1]
FACILITY_SEMANTICS = ROOT / "data" / "project6_v3_facility_semantics_36.csv"
FACILITY_CONTRACT = ROOT / "docs" / "contracts" / "facility_semantics_contract.json"


def test_project4_gat_candidates_distinguish_same_filename_by_path() -> None:
    paths = [candidate[2] for candidate in DEFAULT_PROJECT4_GAT_CANDIDATES]
    names = [Path(path).name for path in paths]
    assert len(paths) == 5
    assert len(set(paths)) == 5
    assert len(set(names)) == 1


def test_gat_compatibility_status_vocabulary_is_layered() -> None:
    from sewerrtc.state import gat_compatibility as gc

    assert gc.COMPATIBLE_STRICT == "compatible_strict"
    assert gc.COMPATIBLE_SHARED_BASE == "compatible_shared_base_graph_only"
    assert gc.METADATA_INCOMPLETE == "metadata_incomplete"
    assert gc.INCOMPATIBLE == "incompatible"
    assert gc.LOAD_FAILED == "load_failed"


def test_temporal_state_uses_seven_causal_frames() -> None:
    assert TEMPORAL_FRAME_OFFSETS_MIN == [0, -10, -20, -30, -40, -50, -60]
    decision = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
    source_times = [decision - timedelta(minutes=i) for i in [0, 10, 20, 30, 40, 50, 60]]
    frames = build_temporal_frame_schedule(decision, source_times, max_age_min=10)
    assert len(frames) == 7
    assert all(frame.source_time <= frame.decision_time for frame in frames if frame.source_time is not None)
    assert_no_future_observation(frames)


def test_future_observation_is_rejected() -> None:
    decision = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
    frames = build_temporal_frame_schedule(decision, [decision + timedelta(minutes=1)], max_age_min=10)
    assert all(frame.source_time is None for frame in frames)


def test_missing_flow_is_not_encoded_as_zero() -> None:
    feature = unavailable_flow("cc006.1")
    validate_flow_feature(feature)
    with pytest.raises(ValueError):
        validate_flow_feature(
            FlowFeature(
                facility_id="cc006.1",
                link_id=None,
                source=FlowSource.UNAVAILABLE,
                value=0.0,
                availability_mask=False,
                uncertainty=None,
            )
        )


def test_state_contract_contains_required_node_and_facility_fields() -> None:
    for field in ["reconstructed_depth", "flooding_rate", "observation_mask", "uncertainty"]:
        assert field in NODE_STATE_FIELDS
    for field in ["actual_current_setting", "native_target_setting", "override_ttl", "local_flow"]:
        assert field in FACILITY_STATE_FIELDS
    for field in ["speed_setting_actual", "binary_actual", "dwell_remaining"]:
        assert field in PUMP_STATE_FIELDS


def test_add350_variable_speed_semantics_are_not_binary() -> None:
    text = FACILITY_SEMANTICS.read_text(encoding="utf-8")
    add350 = [line for line in text.splitlines() if line.startswith("add350.1,")][0]
    assert ",variable_speed,continuous," in add350
    assert "blocked_until_bounds_verified" in add350
    assert "user_confirmed" in add350
    assert "binary_toggle" not in add350


def test_add301_pumps_are_binary_only() -> None:
    text = FACILITY_SEMANTICS.read_text(encoding="utf-8")
    for pump_id in ["ADD301.2", "ADD301.3"]:
        row = [line for line in text.splitlines() if line.startswith(f"{pump_id},")][0]
        assert ",binary,binary," in row
        assert '"[0,1]"' in row
        assert ",false," in row
