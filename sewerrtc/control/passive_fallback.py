from __future__ import annotations

from typing import Any

from sewerrtc.control.fallback_contract import FallbackAction


FORBIDDEN_PASSIVE_MODES = {"all_36_zero", "all_36_open", "instant_reset_all"}


def executable_passive_fallback(facility_state: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a minimum-intervention passive fallback plan.

    The default action is to hold current/native settings. Only rows explicitly
    marked with a safety constraint are changed by the caller's facility state.
    """
    actions: list[FallbackAction] = []
    changed = 0
    for row in facility_state:
        fid = str(row.get("facility_id", ""))
        actual = row.get("actual_current_setting", row.get("native_target_setting", "hold"))
        native = row.get("native_target_setting", actual)
        safety_target = row.get("minimum_passive_target", "")
        if safety_target not in {"", None} and safety_target != actual:
            target = safety_target
            changed += 1
            reason = "minimum_required_safety_intervention"
        else:
            target = actual if actual not in {"", None} else native
            reason = "hold_current_or_native"
        actions.append(FallbackAction(fid, target, target, target, reason))
    settings = [a.expected_actual_setting for a in actions]
    all_zero = bool(settings) and all(str(v) in {"0", "0.0"} for v in settings)
    all_open = bool(settings) and all(str(v) in {"1", "1.0"} for v in settings)
    legal = not all_zero and not all_open and changed <= max(1, len(actions) // 3)
    return {
        "fallback_id": "executable_passive",
        "executable": legal,
        "selected_facility_count": changed,
        "actions": actions,
        "switch_count": changed,
        "total_action_magnitude": changed,
        "transition_feasibility": "pass" if legal else "blocked",
        "rejection_reason": "" if legal else "passive_would_be_global_reset_or_non_minimal",
    }

