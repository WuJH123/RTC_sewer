from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def evaluate_prompt2_completion(out_root: Path) -> dict[str, Any]:
    gat_dir = out_root / "gat"
    state_dir = out_root / "state"
    gates_dir = out_root / "gates"
    lock_path = gat_dir / "gat_primary_selection_lock.json"
    robustness_gate_path = gat_dir / "gat_sr0p15_robustness_gate.json"
    state_shape_path = state_dir / "augmented_state_shape_audit.json"
    state_gap_path = state_dir / "state_input_gap_report.json"

    checks = {
        "sr0p15_selection_decision_recorded": (Path("docs/contracts/gat_primary_selection_decision.json")).exists(),
        "sr0p15_selection_lock_exists": lock_path.exists(),
        "checkpoint_hash_consistent": False,
        "strict_load_and_graph_compatibility_passed": False,
        "validation_no_leakage_or_independent": False,
        "unobserved_node_diagnostics_completed": False,
        "priority_leaveout_completed": False,
        "sentinel_leaveout_completed": False,
        "highwater_phase_completed": False,
        "sensor_failure_completed": False,
        "repeatability_latency_completed": False,
        "sr0p15_robustness_gate_passed": False,
        "runtime_state_shape_causality_missingness_passed": False,
        "state_clone_allowed_to_remain_blocked": True,
        "action_data_not_generated": True,
        "round0_not_unlocked": True,
    }
    lock = _read_json(lock_path)
    if lock:
        checks["checkpoint_hash_consistent"] = lock.get("checkpoint_sha256") == "11f40e6a36016202139e604f04c7d888b5ec3805511c46172ad968a7c20d0e20"
        checks["strict_load_and_graph_compatibility_passed"] = (
            lock.get("strict_load_status") == "strict_loaded"
            and lock.get("compatibility_status") == "compatible_strict"
            and lock.get("registry_name") == "sr0p15"
        )
        checks["round0_not_unlocked"] = lock.get("round0_unlock_allowed") is False
    robustness = _read_json(robustness_gate_path)
    if robustness:
        rchecks = robustness.get("checks", {})
        checks["validation_no_leakage_or_independent"] = bool(rchecks.get("no_training_event_leakage"))
        checks["unobserved_node_diagnostics_completed"] = bool(rchecks.get("unobserved_metrics_exist"))
        checks["priority_leaveout_completed"] = bool(rchecks.get("priority_leaveout_complete"))
        checks["sentinel_leaveout_completed"] = bool(rchecks.get("sentinel_leaveout_complete"))
        checks["highwater_phase_completed"] = bool(rchecks.get("highwater_complete")) and bool(rchecks.get("phase_complete"))
        checks["sensor_failure_completed"] = bool(rchecks.get("sensor_failure_complete"))
        checks["repeatability_latency_completed"] = bool(rchecks.get("repeatability_complete")) and bool(rchecks.get("latency_complete"))
        checks["sr0p15_robustness_gate_passed"] = robustness.get("status") == "pass"
    state_shape = _read_json(state_shape_path)
    state_gap = _read_json(state_gap_path)
    checks["runtime_state_shape_causality_missingness_passed"] = (
        state_shape.get("status") == "completed"
        and state_gap.get("status") == "completed"
        and state_gap.get("runtime_state_features_generated") is True
    )

    if not lock_path.exists():
        status = "blocked_pending_manual_selection_lock"
    elif robustness.get("status") != "pass":
        status = "blocked_pending_sr0p15_robustness"
    elif not checks["runtime_state_shape_causality_missingness_passed"]:
        status = "blocked_pending_runtime_state_validation"
    elif all(checks.values()):
        status = "pass"
    else:
        status = "incomplete"
    return {
        "status": status,
        "checks": checks,
        "round0_unlock_allowed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "lock_path": str(lock_path),
            "robustness_gate_path": str(robustness_gate_path),
            "state_shape_path": str(state_shape_path),
            "state_gap_path": str(state_gap_path),
        },
    }


def write_prompt2_completion_gate(out_root: Path) -> Path:
    gate = evaluate_prompt2_completion(out_root)
    path = out_root / "gates" / "project6_prompt2_completion_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return path
