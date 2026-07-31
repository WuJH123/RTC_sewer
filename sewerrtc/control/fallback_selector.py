from __future__ import annotations

from typing import Any

from sewerrtc.control.fallback_contract import selection_input_hash


ORDER = ("tfv_ucb", "peak_ucb", "sentinel_storage_risk", "pfv_ucb", "switch_count", "action_magnitude", "native_transition_cost")


def select_safe_fallback(passive: dict[str, Any], internal: dict[str, Any], metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    candidates = []
    if passive.get("executable"):
        candidates.append("executable_passive")
    if internal.get("executable"):
        candidates.append("internal_rules")
    if not candidates:
        return {"selected_fallback_id": None, "selection_reason": "no_executable_fallback", "selection_input_hash": selection_input_hash({"passive": passive, "internal": internal, "metrics": metrics})}
    if len(candidates) == 1:
        selected = candidates[0]
    else:
        selected = sorted(candidates, key=lambda cid: tuple(metrics.get(cid, {}).get(k, float("inf")) for k in ORDER))[0]
    return {
        "selected_fallback_id": selected,
        "selection_reason": "lexicographic_safety_order",
        "selection_order": list(ORDER),
        "selection_input_hash": selection_input_hash({"passive": passive, "internal": internal, "metrics": metrics}),
    }

