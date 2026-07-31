from __future__ import annotations

from typing import Any


def internal_rules_fallback(controller_memory: dict[str, Any]) -> dict[str, Any]:
    cleared = dict(controller_memory)
    cleared["learned_override_active"] = False
    cleared["override_ttl"] = 0
    cleared["candidate_target"] = None
    cleared["released_to_native"] = True
    stale = bool(controller_memory.get("learned_override_active")) and not cleared["released_to_native"]
    return {
        "fallback_id": "internal_rules",
        "executable": not stale,
        "release_override": True,
        "clear_expired_ttl": True,
        "clear_candidate_target": True,
        "wait_for_native_rule_evaluation": True,
        "native_target_check_required": True,
        "actual_setting_check_required": True,
        "stale_learned_action_present": stale,
        "controller_memory_after_release": cleared,
    }

