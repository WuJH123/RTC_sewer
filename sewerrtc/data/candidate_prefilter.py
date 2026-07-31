from __future__ import annotations

from typing import Any


def prefilter_candidate(candidate: dict[str, Any]) -> tuple[bool, str]:
    if candidate.get("checkpoint_fingerprint_status") == "failed_gate":
        return False, "same_state_fingerprint_failed"
    if candidate.get("noop") is True:
        return False, "no_op"
    if candidate.get("duplicate") is True:
        return False, "duplicate"
    binary_values = candidate.get("binary_pump_values") or {}
    for pump_id in ("ADD301.2", "ADD301.3"):
        if pump_id in binary_values:
            try:
                value = float(binary_values[pump_id])
            except Exception:
                return False, "binary_pump_intermediate_value"
            if value not in (0.0, 1.0):
                return False, "binary_pump_intermediate_value"
    if candidate.get("binary_legality") == "fail":
        return False, "binary_legality"
    if candidate.get("add350_residual_override") is True:
        return False, "variable_speed_bounds_unverified"
    if int(candidate.get("override_count", 0) or 0) > 8:
        return False, "K_exceeded"
    return True, ""
