from __future__ import annotations

from sewerrtc.state.hotstart_acceleration import certification_status, evaluate_hotstart_gate_rows


def test_smoke_cannot_substitute_for_full() -> None:
    rows = [
        {"checkpoint_id": "a", "certification_status": "pass"},
        {"checkpoint_id": "b", "certification_status": "pass"},
        {"checkpoint_id": "c", "certification_status": "pass"},
    ]

    gate = evaluate_hotstart_gate_rows(rows, expected_count=18)

    assert gate["status"] == "partial"
    assert gate["hotstart_acceleration_allowed"] == "per_checkpoint_only"


def test_18_of_18_required_for_current_full_gate() -> None:
    rows = [{"checkpoint_id": f"cp{i}", "certification_status": "pass"} for i in range(18)]

    gate = evaluate_hotstart_gate_rows(rows, expected_count=18)

    assert gate["status"] == "pass"
    assert gate["certified_checkpoint_count"] == 18


def test_initial_fingerprint_mismatch_invalidates_checkpoint() -> None:
    status = certification_status(
        {
            "compatibility_signature_pass": "true",
            "object_order_pass": "true",
            "checkpoint_phase_pass": "true",
            "forcing_pass": "true",
            "controller_memory_pass": "true",
            "initial_state_fingerprint_pass": "false",
        }
    )

    assert status == "failed_gate"
