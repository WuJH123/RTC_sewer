from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "sewerrtc" / "state" / "runtime_state_features.py"
CONTROLLER = ROOT / "sewerrtc" / "simulation" / "controller_state.py"


def test_full_state_requires_controller_memory_fields() -> None:
    text = CONTROLLER.read_text(encoding="utf-8")
    for token in ["override_ttl", "fallback_mode", "add350_speed_actual", "binary_pump_states", "continuation_policy_id"]:
        assert token in text


def test_runtime_state_keeps_missing_flow_not_zero_contract() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    assert "missing_flow" in text
    assert "zero" in text

