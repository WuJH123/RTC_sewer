from __future__ import annotations

from sewerrtc.state.state_clone_contract import REQUIRED_CLONE_FIELDS


def test_state_clone_contract_includes_controller_memory_and_fallback() -> None:
    for field in ["controller_memory", "override_ttl", "fallback_mode", "continuation_policy"]:
        assert field in REQUIRED_CLONE_FIELDS


def test_state_clone_contract_includes_pump_specific_state() -> None:
    for field in ["add350_1_actual_speed", "ADD301_2_binary_state", "ADD301_3_binary_state", "pump_on_off_duration"]:
        assert field in REQUIRED_CLONE_FIELDS
