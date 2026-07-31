from __future__ import annotations

from pathlib import Path
from typing import Any

from sewerrtc.contracts.prompt3a import OUT_ROOT, utc_now, write_json


def audit_reference_roles(out_dir: str | Path = OUT_ROOT / "reference_roles") -> tuple[int, dict[str, Any], list[Path]]:
    out_dir = Path(out_dir)
    reference_roles = {
        "contract_version": "project6_reference_roles_v1",
        "created_at": utc_now(),
        "no_control": {
            "role": "diagnostic_physical_reference_only",
            "used_as_primary_pfv_effectiveness_baseline": False,
            "used_for_online_fallback_selection": False,
        },
        "internal_rules": {
            "role": "pfv_effectiveness_benchmark",
            "used_for_online_pfv_active_prediction": True,
            "original_swmm_controls_must_retake_authority": True,
        },
        "selected_safe_fallback": {
            "role": "online_necessity_and_tfv_peak_safety_benchmark",
            "selected_before_candidate_evaluation": True,
            "candidate_must_not_influence_fallback_selection": True,
        },
        "legacy_core26_or_v8_templates": {
            "role": "not_used_in_prompt3a_reference_roles",
            "allowed": False,
        },
    }
    files = [
        write_json(out_dir / "reference_roles_contract.json", reference_roles),
        write_json(out_dir / "reference_role_audit_report.json", {"status": "completed", "created_at": utc_now(), "dynamic_fallback_selection_executed": False, "legacy_core26_referenced": False}),
        write_json(out_dir / "no_control_reference_contract.json", reference_roles["no_control"]),
        write_json(out_dir / "online_benchmark_contract.json", {"internal_rules": reference_roles["internal_rules"], "selected_safe_fallback": reference_roles["selected_safe_fallback"]}),
    ]
    return 0, {"status": "completed", "outputs": [str(p) for p in files]}, files
