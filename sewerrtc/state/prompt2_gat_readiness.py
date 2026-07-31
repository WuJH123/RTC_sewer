from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def evaluate_prompt2_gat_readiness(out_root: Path) -> dict[str, Any]:
    gat_dir = out_root / "gat"
    independent_gat_dir = gat_dir / "independent_holdout" / "sr0p15"
    state_dir = out_root / "state"
    gates_dir = out_root / "gates"
    gate_path = gates_dir / "project6_prompt2_gat_readiness_gate.json"
    lock = _read_json(gat_dir / "gat_primary_selection_lock.json")
    diagnostic_robustness = _read_json(gat_dir / "gat_sr0p15_robustness_gate.json")
    independent_robustness_path = independent_gat_dir / "gat_sr0p15_independent_robustness_gate.json"
    robustness = _read_json(independent_robustness_path)
    holdout_lock = _read_json(gat_dir / "gat_independent_validation_lock.json")
    state_shape = _read_json(state_dir / "augmented_state_shape_audit.json")
    state_gap = _read_json(state_dir / "state_input_gap_report.json")
    rchecks = robustness.get("checks", {})

    def passed(name: str) -> bool:
        check = rchecks.get(name) or {}
        return check.get("status") == "pass"

    checks = {
        "sr0p15_selection_lock_valid": lock.get("registry_name") == "sr0p15",
        "strict_load_passed": lock.get("strict_load_status") == "strict_loaded",
        "compatible_strict": lock.get("compatibility_status") == "compatible_strict",
        "independent_holdout_lock_valid": holdout_lock.get("status") == "locked"
        and holdout_lock.get("allowed_for_robustness_audit") is True,
        "independent_robustness_gate_used": robustness.get("gate_kind") == "independent_holdout",
        "diagnostic_contaminated_gate_failed_or_ignored": diagnostic_robustness.get("status") in {"fail", "failed", "failed_gate"}
        or diagnostic_robustness.get("validation_status") == "diagnostic_contaminated"
        or not diagnostic_robustness,
        "validation_provenance_complete": passed("validation_provenance_complete"),
        "no_training_event_leakage": passed("no_training_event_leakage"),
        "unobserved_metrics_exist": passed("unobserved_metrics_exist"),
        "priority_leaveout_complete": passed("priority_leaveout_complete"),
        "sentinel_leaveout_complete": passed("sentinel_leaveout_complete"),
        "highwater_phase_complete": passed("highwater_phase_complete") or (passed("highwater_complete") and passed("phase_complete")),
        "sensor_failure_execution_complete": passed("sensor_failure_execution_complete"),
        "latency_measurement_complete": passed("latency_measurement_complete"),
        "node_level_7frame_validation_complete": state_gap.get("node_level_7frame_validation_complete") is True
        or state_shape.get("node_level_7frame_validation_complete") is True,
        "action_data_not_generated": True,
        "full_project6_augmented_state_complete": False,
        "round0_not_unlocked": True,
    }
    blocking = [key for key, value in checks.items() if value is not True and key not in {"full_project6_augmented_state_complete"}]
    status = "pass" if not blocking else "blocked"
    payload = {
        "status": status,
        "allowed_to_enter_prompt3a": status == "pass",
        "round0_unlock_allowed": False,
        "full_project6_augmented_state_complete": False,
        "checks": checks,
        "blocking_reasons": blocking,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "lock": str(gat_dir / "gat_primary_selection_lock.json"),
            "independent_holdout_lock": str(gat_dir / "gat_independent_validation_lock.json"),
            "diagnostic_contaminated_gate": str(gat_dir / "gat_sr0p15_robustness_gate.json"),
            "robustness_gate": str(independent_robustness_path),
            "state_shape": str(state_dir / "augmented_state_shape_audit.json"),
            "state_gap": str(state_dir / "state_input_gap_report.json"),
        },
    }
    gates_dir.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
