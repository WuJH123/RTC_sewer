from __future__ import annotations

from sewerrtc.control.fallback_selector import select_safe_fallback
from sewerrtc.control.internal_fallback import internal_rules_fallback
from sewerrtc.control.passive_fallback import executable_passive_fallback

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"


def test_passive_fallback_is_not_all_zero_or_all_open() -> None:
    state = [{"facility_id": f"F{i}", "actual_current_setting": 0.5, "native_target_setting": 0.5} for i in range(36)]
    plan = executable_passive_fallback(state)
    assert plan["executable"] is True
    assert plan["selected_facility_count"] == 0


def test_internal_fallback_releases_override_and_clears_candidate() -> None:
    result = internal_rules_fallback({"learned_override_active": True, "override_ttl": 3, "candidate_target": 0.2})
    memory = result["controller_memory_after_release"]
    assert memory["released_to_native"] is True
    assert memory["override_ttl"] == 0
    assert memory["candidate_target"] is None


def test_fallback_selection_is_independent_and_lexicographic() -> None:
    passive = {"executable": True}
    internal = {"executable": True}
    selected = select_safe_fallback(
        passive,
        internal,
        {
            "executable_passive": {"tfv_ucb": 1, "peak_ucb": 0, "sentinel_storage_risk": 0, "pfv_ucb": 0, "switch_count": 1, "action_magnitude": 1, "native_transition_cost": 0},
            "internal_rules": {"tfv_ucb": 2, "peak_ucb": 0, "sentinel_storage_risk": 0, "pfv_ucb": 0, "switch_count": 0, "action_magnitude": 0, "native_transition_cost": 0},
        },
    )
    assert selected["selected_fallback_id"] == "executable_passive"
    assert "selection_input_hash" in selected


def test_fallback_selection_hash_accepts_action_objects() -> None:
    state = [{"facility_id": f"F{i}", "actual_current_setting": 0.5, "native_target_setting": 0.5} for i in range(36)]
    passive = executable_passive_fallback(state)
    internal = internal_rules_fallback({"learned_override_active": True, "override_ttl": 1, "candidate_target": 0.2})
    selected = select_safe_fallback(
        passive,
        internal,
        {
            "executable_passive": {"tfv_ucb": 0, "peak_ucb": 0, "sentinel_storage_risk": 0, "pfv_ucb": 0, "switch_count": 0, "action_magnitude": 0, "native_transition_cost": 0},
            "internal_rules": {"tfv_ucb": 1, "peak_ucb": 0, "sentinel_storage_risk": 0, "pfv_ucb": 0, "switch_count": 0, "action_magnitude": 0, "native_transition_cost": 0},
        },
    )
    assert selected["selected_fallback_id"] == "executable_passive"
    assert len(selected["selection_input_hash"]) == 64


def test_reference_roles_and_fallback_execution_outputs_are_isolated() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "scripts\\166_audit_reference_roles.py" in text
    assert "reference_roles\\reference_roles_contract.json" in text
    assert "reference_roles\\no_control_reference_contract.json" in text
    assert "fallbacks\\passive_fallback_contract.json" in text
    assert "fallbacks\\fallback_execution_audit_report.json" in text
    reference_block = text.split('function Run-AuditReferencesFallbacks', 1)[1].split('function Run-RebuildContract', 1)[0]
    assert "fallbacks\\passive_fallback_contract.json" not in reference_block
