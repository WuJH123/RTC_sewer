from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_TAG = "project6_pfvfirst_dualfallback_10min_v3"

ALLOWED_STAGE_STATUSES = [
    "not_implemented",
    "implemented_not_run",
    "contract_ready",
    "structural_only",
    "runtime_partial",
    "runtime_pass",
    "scientific_gate_failed",
    "blocked",
    "stale",
    "pass",
]

MATRIX_COLUMNS = [
    "stage",
    "implementation_status",
    "runtime_status",
    "scientific_status",
    "input_paths",
    "output_paths",
    "evidence_path",
    "evidence_sha256",
    "blocking_reason",
    "next_allowed_stage",
]


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {"_read_error": "json_decode_failed"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _root_from_config(config_path: Path | str) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path.resolve().parents[1]
    return Path.cwd().resolve()


def _out_root(root: Path) -> Path:
    return root / "outputs" / RUN_TAG


def _join_paths(paths: list[Path]) -> str:
    return ";".join(str(p) for p in paths)


def _count_manifest_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    processable = 0
    completed = 0
    skipped_existing = 0
    for row in rows:
        status = row.get("status", "")
        detail = row.get("detail_file", "").strip()
        if status == "completed":
            completed += 1
            processable += 1
        elif status == "skipped_existing" and detail:
            skipped_existing += 1
            processable += 1
    return {
        "total_rows": len(rows),
        "completed_rows": completed,
        "skipped_existing_with_detail_rows": skipped_existing,
        "processable_rows": processable,
    }


def _npz_feature_audit(rows: list[dict[str, str]], shape_report: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "npz_read_status": "not_attempted",
        "state_feature_tensor_name_match": None,
        "facility_feature_tensor_name_match": None,
        "node_contract_field_count_match": None,
        "facility_contract_field_count_match": None,
        "state_npz_count": 0,
        "facility_npz_count": 0,
        "errors": [],
    }
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on user environment
        result["npz_read_status"] = "blocked_numpy_unavailable"
        result["errors"].append(str(exc))
        return result

    state_ok = True
    facility_ok = True
    state_count = 0
    facility_count = 0
    for row in rows:
        state_path = Path(row.get("state_history_path", ""))
        facility_path = Path(row.get("facility_history_path", ""))
        if state_path.exists() and state_path.is_file():
            state_count += 1
            try:
                with np.load(state_path, allow_pickle=True) as data:
                    tensor = data["state_history"]
                    feature_names = list(data["feature_names"]) if "feature_names" in data.files else []
                    state_ok = state_ok and tensor.shape[-1] == len(feature_names)
            except Exception as exc:
                state_ok = False
                result["errors"].append(f"{state_path}:{exc}")
        if facility_path.exists() and facility_path.is_file():
            facility_count += 1
            try:
                with np.load(facility_path, allow_pickle=True) as data:
                    tensor = data["facility_history"]
                    feature_names = list(data["feature_names"]) if "feature_names" in data.files else []
                    facility_ok = facility_ok and tensor.shape[-1] == len(feature_names)
            except Exception as exc:
                facility_ok = False
                result["errors"].append(f"{facility_path}:{exc}")
    node_fields = shape_report.get("node_state_fields") or []
    facility_fields = shape_report.get("facility_state_fields") or []
    state_shapes = shape_report.get("state_shapes_seen") or shape_report.get("state_shapes") or []
    facility_shapes = shape_report.get("facility_shapes_seen") or shape_report.get("facility_shapes") or []
    storage_shapes = shape_report.get("storage_shapes_seen") or shape_report.get("storage_shapes") or []
    storage_fields = shape_report.get("storage_state_fields") or []
    actual_node_features = shape_report.get("actual_node_feature_names") or []
    actual_facility_features = shape_report.get("actual_facility_feature_names") or []
    actual_storage_features = shape_report.get("actual_storage_feature_names") or []
    node_contract_match = (
        bool(state_shapes)
        and all(int(shape[-1]) == len(node_fields) for shape in state_shapes)
        and (not actual_node_features or list(actual_node_features) == list(node_fields))
    )
    facility_contract_match = (
        bool(facility_shapes)
        and all(int(shape[-1]) == len(facility_fields) for shape in facility_shapes)
        and (not actual_facility_features or list(actual_facility_features) == list(facility_fields))
    )
    storage_contract_match = (
        not storage_shapes
        or (
            bool(storage_fields)
            and all(int(shape[-1]) == len(storage_fields) for shape in storage_shapes)
            and (not actual_storage_features or list(actual_storage_features) == list(storage_fields))
        )
    )
    result.update(
        {
            "npz_read_status": "completed",
            "state_feature_tensor_name_match": state_ok if state_count else False,
            "facility_feature_tensor_name_match": facility_ok if facility_count else False,
            "node_contract_field_count_match": node_contract_match,
            "facility_contract_field_count_match": facility_contract_match,
            "storage_contract_field_count_match": storage_contract_match,
            "state_npz_count": state_count,
            "facility_npz_count": facility_count,
        }
    )
    return result


def _checkpoint_realness(rows: list[dict[str, str]]) -> dict[str, Any]:
    clone_hash_fields = [
        "state_clone_hash",
        "node_state_hash",
        "link_state_hash",
        "storage_state_hash",
        "controller_memory_hash",
    ]
    counts = {field: 0 for field in clone_hash_fields}
    for row in rows:
        for field in clone_hash_fields:
            if row.get(field, "").strip():
                counts[field] += 1
    return {
        "checkpoint_rows": len(rows),
        "hash_counts": counts,
        "all_rows_have_clone_hash": bool(rows) and counts["state_clone_hash"] == len(rows),
        "all_rows_have_controller_memory_hash": bool(rows) and counts["controller_memory_hash"] == len(rows),
    }


def _binary_pump_audit(manifest_rows: list[dict[str, str]]) -> dict[str, Any]:
    binary_ids = ["ADD301.2", "ADD301.3"]
    invalid: list[dict[str, Any]] = []
    observed = 0
    for row in manifest_rows:
        detail = Path(row.get("detail_file", ""))
        if not detail.exists() or not detail.is_file():
            continue
        for detail_row in read_csv(detail):
            for pump_id in binary_ids:
                for prefix in ("setting:", "a:"):
                    value = detail_row.get(f"{prefix}{pump_id}")
                    if value in (None, ""):
                        continue
                    observed += 1
                    try:
                        number = float(value)
                    except ValueError:
                        invalid.append({"trajectory_id": row.get("trajectory_id", ""), "pump_id": pump_id, "value": value})
                        continue
                    if number not in (0.0, 1.0):
                        invalid.append({"trajectory_id": row.get("trajectory_id", ""), "pump_id": pump_id, "value": value})
    return {
        "observed_binary_pump_values": observed,
        "invalid_binary_pump_value_count": len(invalid),
        "invalid_examples": invalid[:20],
        "binary_pump_semantics_pass": observed > 0 and not invalid,
    }


def _marker_audit(root: Path, out: Path, forbidden_stage_reasons: dict[str, str]) -> dict[str, Any]:
    marker_dir = out / "completion_markers"
    config_candidates = [
        root / "configs" / "wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml",
    ]
    current_config_hash = next((sha256_file(path) for path in config_candidates if path.exists()), None)
    rows: list[dict[str, Any]] = []
    invalid_count = 0
    stale_count = 0
    forbidden_count = 0
    if not marker_dir.exists():
        return {
            "marker_count": 0,
            "invalid_marker_count": 0,
            "stale_marker_count": 0,
            "forbidden_marker_count": 0,
            "markers": [],
        }
    for marker in sorted(marker_dir.glob("*_COMPLETED.json")):
        record = read_json(marker)
        stage = record.get("stage") or marker.name.replace("_COMPLETED.json", "")
        outputs = record.get("outputs") or []
        missing_outputs: list[str] = []
        missing_hashes = 0
        for item in outputs:
            path = Path(str(item.get("path", "")))
            if not path.exists():
                missing_outputs.append(str(path))
            if not item.get("sha256"):
                missing_hashes += 1
        config_hash = record.get("config_hash") or record.get("config_scope_hash")
        stale = bool(config_hash and current_config_hash and config_hash != current_config_hash)
        forbidden_reason = forbidden_stage_reasons.get(stage, "")
        invalid = bool(missing_outputs or missing_hashes or not outputs or forbidden_reason)
        if invalid:
            invalid_count += 1
        if stale:
            stale_count += 1
        if forbidden_reason:
            forbidden_count += 1
        rows.append(
            {
                "stage": stage,
                "marker_path": str(marker),
                "marker_sha256": sha256_file(marker),
                "output_count": len(outputs),
                "missing_output_count": len(missing_outputs),
                "missing_output_hash_count": missing_hashes,
                "config_hash_stale": stale,
                "forbidden_marker_reason": forbidden_reason,
                "marker_valid": not invalid and not stale,
            }
        )
    return {
        "marker_count": len(rows),
        "invalid_marker_count": invalid_count,
        "stale_marker_count": stale_count,
        "forbidden_marker_count": forbidden_count,
        "markers": rows,
    }


def build_current_truth(root: Path, *, write_outputs: bool = True) -> dict[str, Any]:
    out = _out_root(root)
    status_dir = out / "status"
    gates_dir = out / "gates"

    prompt2_gate = read_json(gates_dir / "project6_prompt2_gat_readiness_gate.json")
    prompt3a_entry = read_json(gates_dir / "prompt3a_entry_gate.json")
    prompt3a_completion = read_json(gates_dir / "project6_prompt3a_completion_gate.json")
    sr0p15_lock = read_json(out / "gat" / "gat_primary_selection_lock.json")
    holdout_lock = read_json(out / "gat" / "gat_independent_validation_lock.json")
    independent_gate = read_json(out / "gat" / "independent_holdout" / "sr0p15" / "gat_sr0p15_independent_robustness_gate.json")

    baseline_report_path = out / "baseline_trajectories" / "baseline_trajectory_generation_report.json"
    baseline_manifest_path = out / "baseline_trajectories" / "baseline_trajectory_manifest.csv"
    baseline_report = read_json(baseline_report_path)
    baseline_rows = read_csv(baseline_manifest_path)
    baseline_recovery_path = out / "baseline_trajectories" / "baseline_recovery_audit.csv"
    baseline_recovery_rows = read_csv(baseline_recovery_path)
    baseline_counts = _count_manifest_rows(baseline_rows)
    baseline_unique_events = sorted({row.get("event_id", "") for row in baseline_rows if row.get("event_id", "")})
    baseline_unique_policies = sorted({row.get("policy_id", "") for row in baseline_rows if row.get("policy_id", "")})

    state_input_path = out / "state_inputs" / "state_input_manifest_v1.csv"
    state_gap_path = out / "state_inputs" / "state_trajectory_gap_report.json"
    state_rows = read_csv(state_input_path)
    state_gap = read_json(state_gap_path)
    causality_rows = read_csv(out / "state" / "augmented_state_causality_audit.csv")
    shape_audit_path = out / "state" / "augmented_state_shape_audit.json"
    shape_audit = read_json(shape_audit_path)
    feature_audit = _npz_feature_audit(state_rows, shape_audit)

    checkpoint_path = out / "checkpoint_catalog" / "checkpoint_catalog.csv"
    checkpoint_rows = read_csv(checkpoint_path)
    checkpoint_audit = _checkpoint_realness(checkpoint_rows)

    state_clone_contract_path = out / "state" / "state_clone_contract.json"
    state_clone_contract = read_json(state_clone_contract_path)
    state_clone_gate_path = out / "state_clone" / "state_clone_gate.json"
    state_clone_gate = read_json(state_clone_gate_path)
    same_state_branch_gate_path = out / "state_clone" / "same_state_branch_gate.json"
    same_state_branch_gate = read_json(same_state_branch_gate_path)
    round0_plan_path = out / "round0" / "round0_plan_report.json"
    round0_plan = read_json(round0_plan_path)
    dryrun_report_path = out / "round0" / "round0_dryrun_report.json"
    dryrun_report = read_json(dryrun_report_path)

    binary_pumps = _binary_pump_audit(baseline_rows)

    baseline_selected = int(baseline_report.get("selected_trajectory_count") or 0)
    state_input_count = len(state_rows)
    processable_count = baseline_counts["processable_rows"]
    all_trajectories_in_state = processable_count > 0 and state_input_count == processable_count
    shape_feature_match = (
        bool(shape_audit.get("node_feature_count_matches_tensor"))
        and bool(shape_audit.get("facility_feature_count_matches_tensor"))
        and bool(shape_audit.get("storage_feature_count_matches_tensor", True))
    )
    state_schema_match = (
        bool(feature_audit["node_contract_field_count_match"])
        and bool(feature_audit["facility_contract_field_count_match"])
        and bool(feature_audit.get("storage_contract_field_count_match", True))
        and shape_feature_match
    )
    real_hotstart = checkpoint_audit["all_rows_have_clone_hash"]
    real_controller_memory = checkpoint_audit["all_rows_have_controller_memory_hash"]
    state_clone_pass = (
        same_state_branch_gate.get("status") == "pass"
        and same_state_branch_gate.get("formal_same_state_unlock_allowed") is True
    ) or (
        state_clone_gate.get("status") == "pass"
        and state_clone_gate.get("hotstart_equivalence_status") == "pass"
        and state_clone_gate.get("controller_memory_restore_status") == "pass"
    )
    hydraulic_dryrun_pass = (
        dryrun_report.get("status") == "runtime_pass"
        and dryrun_report.get("same_state_hotstart_execution_status") == "pass"
    )
    round0_effective = int(round0_plan.get("effective_candidate_count") or 0)
    round0_target = round0_plan.get("target_candidate_range") or [1500, 2000]
    round0_target_met = isinstance(round0_target, list) and len(round0_target) == 2 and int(round0_target[0]) <= round0_effective <= int(round0_target[1])
    recovery_contract_complete = (
        processable_count > 0
        and len(baseline_recovery_rows) == processable_count
        and all(str(row.get("recovery_status", "")) in {"recovered", "censored", "not_recovered"} for row in baseline_recovery_rows)
        and all(str(row.get("actual_tail_min", "")) != "" for row in baseline_recovery_rows)
    )
    no_future_manifest = bool(state_rows) and all(str(row.get("contains_future_data", "")).lower() in {"false", "0", "no"} for row in state_rows)
    no_future_causality = bool(causality_rows) and all(str(row.get("valid_before_decision", "")).lower() in {"true", "1", "yes"} for row in causality_rows)
    truth_leakage_zero = no_future_manifest and no_future_causality
    forbidden_markers: dict[str, str] = {}
    if not state_clone_pass:
        forbidden_markers["StateCloneTest"] = "hotstart_equivalence_not_run"
    if not hydraulic_dryrun_pass:
        forbidden_markers["DryRunRound0"] = "hydraulic_dryrun_not_run"
    marker_audit = _marker_audit(root, out, forbidden_markers)

    engineering_checks = {
        "prompt2_pass": prompt2_gate.get("status") == "pass",
        "prompt3a_entry_pass": prompt3a_entry.get("status") == "pass",
        "sr0p15_lock_exists": sr0p15_lock.get("registry_name") == "sr0p15",
        "independent_holdout_lock_exists": holdout_lock.get("status") == "locked",
        "independent_gat_gate_pass": independent_gate.get("status") == "pass",
        "baseline_small_plan_generated": baseline_report.get("status") == "completed" and baseline_selected == 6,
        "state_schema_exists": shape_audit_path.exists(),
        "coverage_schema_exists": (out / "coverage" / "coverage_cells_schema.csv").exists(),
        "candidate_preview_exists": round0_plan_path.exists() and round0_effective > 0,
        "structural_dryrun_report_exists": dryrun_report_path.exists(),
    }
    engineering_pass = all(engineering_checks.values())

    runtime_checks = {
        "all_input_trajectories_enter_state_pipeline": all_trajectories_in_state,
        "actual_features_match_schema": state_schema_match,
        "tail_recovery_verified_or_censored": recovery_contract_complete,
        "real_hotstart_files_present": real_hotstart,
        "real_controller_memory_present": real_controller_memory,
        "state_clone_equivalence_pass": state_clone_pass,
        "hydraulic_candidate_dryrun_pass": hydraulic_dryrun_pass,
        "truth_leakage_zero": truth_leakage_zero,
        "engineering_violations_zero": bool(binary_pumps["binary_pump_semantics_pass"]),
        "formal_round0_candidate_target_met": round0_target_met,
    }
    runtime_pass = all(runtime_checks.values())

    matrix: list[dict[str, Any]] = []

    def add_row(
        stage: str,
        implementation_status: str,
        runtime_status: str,
        scientific_status: str,
        inputs: list[Path],
        outputs: list[Path],
        evidence: Path,
        blocking_reason: str,
        next_allowed_stage: str,
    ) -> None:
        for value in (implementation_status, runtime_status, scientific_status):
            if value not in ALLOWED_STAGE_STATUSES:
                raise ValueError(f"invalid stage status {value!r} for {stage}")
        matrix.append(
            {
                "stage": stage,
                "implementation_status": implementation_status,
                "runtime_status": runtime_status,
                "scientific_status": scientific_status,
                "input_paths": _join_paths(inputs),
                "output_paths": _join_paths(outputs),
                "evidence_path": str(evidence),
                "evidence_sha256": sha256_file(evidence) or "",
                "blocking_reason": blocking_reason,
                "next_allowed_stage": next_allowed_stage,
            }
        )

    add_row(
        "Prompt2GATReadiness",
        "runtime_pass" if prompt2_gate.get("status") == "pass" else "blocked",
        "runtime_pass" if prompt2_gate.get("status") == "pass" else "blocked",
        "pass" if prompt2_gate.get("status") == "pass" else "blocked",
        [],
        [gates_dir / "project6_prompt2_gat_readiness_gate.json"],
        gates_dir / "project6_prompt2_gat_readiness_gate.json",
        "" if prompt2_gate.get("status") == "pass" else "prompt2 gate is not pass",
        "Prompt3AEngineeringGate",
    )
    add_row(
        "BaselineTrajectories",
        "runtime_pass" if baseline_report.get("status") == "completed" else "blocked",
        "runtime_pass" if baseline_report.get("completed_trajectory_count") == baseline_selected == 6 else "runtime_partial",
        "structural_only",
        [],
        [baseline_report_path, baseline_manifest_path],
        baseline_report_path,
        "" if baseline_report.get("completed_trajectory_count") == baseline_selected == 6 else "baseline selected/completed count mismatch",
        "BuildStateInputManifest",
    )
    add_row(
        "StateInputManifest",
        "runtime_pass" if state_input_path.exists() else "implemented_not_run",
        "runtime_pass" if all_trajectories_in_state else "runtime_partial",
        "structural_only",
        [baseline_manifest_path],
        [state_input_path, state_gap_path],
        state_gap_path if state_gap_path.exists() else state_input_path,
        "" if all_trajectories_in_state else f"state input rows {state_input_count} do not match processable baseline trajectories {processable_count}",
        "BuildStateFeatures",
    )
    add_row(
        "StateFeatures",
        "runtime_pass" if shape_audit_path.exists() else "implemented_not_run",
        "runtime_pass" if state_schema_match else "runtime_partial",
        "structural_only",
        [state_input_path],
        [shape_audit_path],
        shape_audit_path,
        "" if state_schema_match else "actual tensor feature dimensions do not match the frozen state/facility schema",
        "StateCloneTest",
    )
    add_row(
        "CheckpointCatalog",
        "runtime_pass" if checkpoint_path.exists() else "implemented_not_run",
        "runtime_pass" if real_hotstart and real_controller_memory else "runtime_partial",
        "structural_only",
        [baseline_manifest_path],
        [checkpoint_path],
        checkpoint_path,
        "" if real_hotstart and real_controller_memory else "checkpoint catalog lacks real state clone and controller-memory hashes",
        "StateCloneTest",
    )
    add_row(
        "StateCloneTest",
        "implemented_not_run",
        "blocked",
        "blocked",
        [state_input_path],
        [state_clone_contract_path],
        state_clone_contract_path,
        "hotstart equivalence is not_run; no real SWMM clone equivalence evidence exists",
        "EvaluatePrompt3ARuntimeGate",
    )
    add_row(
        "Round0DryRun",
        "structural_only" if dryrun_report_path.exists() else "implemented_not_run",
        "blocked",
        "blocked",
        [round0_plan_path],
        [dryrun_report_path],
        dryrun_report_path,
        "round0 dry-run is structural only; same-state hotstart execution is not_run",
        "EvaluatePrompt3ARuntimeGate",
    )
    add_row(
        "Round0Plan",
        "runtime_pass" if round0_plan_path.exists() else "implemented_not_run",
        "runtime_partial",
        "blocked",
        [checkpoint_path],
        [round0_plan_path],
        round0_plan_path,
        f"effective candidates {round0_effective} do not meet target range {round0_target}",
        "GenerateRound0OnlyAfterRuntimeGate",
    )
    add_row(
        "Prompt3AEngineeringGate",
        "contract_ready",
        "runtime_pass" if engineering_pass else "blocked",
        "pass" if engineering_pass else "blocked",
        [],
        [gates_dir / "project6_prompt3a_engineering_gate.json"],
        gates_dir / "prompt3a_entry_gate.json",
        "" if engineering_pass else "one or more engineering checks failed",
        "EvaluatePrompt3ARuntimeGate",
    )
    runtime_blocking = [key for key, ok in runtime_checks.items() if not ok]
    add_row(
        "Prompt3ARuntimeGate",
        "contract_ready",
        "runtime_pass" if runtime_pass else "blocked",
        "pass" if runtime_pass else "blocked",
        [],
        [gates_dir / "project6_prompt3a_runtime_gate.json"],
        shape_audit_path if shape_audit_path.exists() else state_gap_path,
        "" if runtime_pass else ",".join(runtime_blocking),
        "Prompt3ACompletion" if runtime_pass else "RecoveryPlan",
    )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "allowed_stage_statuses": ALLOWED_STAGE_STATUSES,
        "prompt2_status": prompt2_gate.get("status"),
        "sr0p15_primary_lock_valid": sr0p15_lock.get("registry_name") == "sr0p15",
        "independent_holdout_lock_valid": holdout_lock.get("status") == "locked",
        "independent_robustness_status": independent_gate.get("status"),
        "baseline_selected_trajectory_count": baseline_selected,
        "baseline_completed_trajectory_count": baseline_report.get("completed_trajectory_count"),
        "baseline_unique_event_count": len(baseline_unique_events),
        "baseline_unique_events": baseline_unique_events,
        "baseline_unique_policy_count": len(baseline_unique_policies),
        "baseline_unique_policies": baseline_unique_policies,
        "baseline_manifest_counts": baseline_counts,
        "baseline_recovery_audit_rows": len(baseline_recovery_rows),
        "recovery_contract_complete": recovery_contract_complete,
        "state_input_rows": state_input_count,
        "state_input_gap_status": state_gap.get("status"),
        "feature_audit": feature_audit,
        "checkpoint_audit": checkpoint_audit,
        "state_clone_status": state_clone_contract.get("status"),
        "state_clone_gate_status": state_clone_gate.get("status"),
        "same_state_branch_gate_status": same_state_branch_gate.get("status"),
        "selected_same_state_method": same_state_branch_gate.get("selected_same_state_method", "none"),
        "state_clone_hotstart_equivalence": state_clone_gate.get("hotstart_equivalence_status", state_clone_contract.get("hotstart_equivalence_status", "not_run")),
        "same_state_branch_gate": same_state_branch_gate,
        "round0_dryrun_status": dryrun_report.get("status"),
        "hydraulic_dryrun_real_pass": hydraulic_dryrun_pass,
        "round0_effective_candidate_count": round0_effective,
        "round0_target_candidate_range": round0_target,
        "binary_pump_audit": binary_pumps,
        "completion_marker_audit": marker_audit,
        "add350_variable_speed_contract_status": "confirmed_variable_speed_bounds_pending",
        "engineering_checks": engineering_checks,
        "runtime_checks": runtime_checks,
        "engineering_gate_status": "pass" if engineering_pass else "blocked",
        "runtime_gate_status": "pass" if runtime_pass else "blocked",
    }

    recovery_gate = {
        "created_at": report["created_at"],
        "status": "blocked" if not runtime_pass else "pass",
        "prompt2": report["prompt2_status"],
        "prompt3a_engineering_gate": report["engineering_gate_status"],
        "prompt3a_runtime_gate": report["runtime_gate_status"],
        "baseline_trajectory_count": baseline_selected,
        "real_state_processed_trajectory_count": state_input_count,
        "hotstart_equivalence": report["state_clone_hotstart_equivalence"],
        "selected_same_state_method": report["selected_same_state_method"],
        "hydraulic_dryrun": "pass" if hydraulic_dryrun_pass else dryrun_report.get("same_state_hotstart_execution_status", "not_run"),
        "effective_round0_candidates": round0_effective,
        "invalid_completion_marker_count": marker_audit["invalid_marker_count"],
        "stale_completion_marker_count": marker_audit["stale_marker_count"],
        "blocking_reasons": runtime_blocking,
    }

    if write_outputs:
        write_csv(status_dir / "project6_current_truth_matrix.csv", matrix, MATRIX_COLUMNS)
        write_json(status_dir / "project6_current_truth_report.json", report)
        write_json(gates_dir / "project6_recovery_gate.json", recovery_gate)
    return {
        "matrix": matrix,
        "report": report,
        "recovery_gate": recovery_gate,
        "paths": {
            "truth_matrix": str(status_dir / "project6_current_truth_matrix.csv"),
            "truth_report": str(status_dir / "project6_current_truth_report.json"),
            "recovery_gate": str(gates_dir / "project6_recovery_gate.json"),
        },
    }


def write_engineering_gate(root: Path) -> dict[str, Any]:
    result = build_current_truth(root, write_outputs=True)
    report = result["report"]
    gate = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": report["engineering_gate_status"],
        "engineering_checks": report["engineering_checks"],
        "runtime_gate_status": report["runtime_gate_status"],
        "runtime_gate_required_before_science_pass": True,
        "round0_unlock_allowed": False,
    }
    path = _out_root(root) / "gates" / "project6_prompt3a_engineering_gate.json"
    write_json(path, gate)
    return {"gate": gate, "path": path, **result}


def write_runtime_gate(root: Path) -> dict[str, Any]:
    result = build_current_truth(root, write_outputs=True)
    report = result["report"]
    checks = report["runtime_checks"]
    gate = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": report["runtime_gate_status"],
        "runtime_checks": checks,
        "blocking_reasons": [key for key, ok in checks.items() if not ok],
        "engineering_gate_status": report["engineering_gate_status"],
        "round0_unlock_allowed": False,
        "completion_marker_allowed": report["runtime_gate_status"] == "pass",
    }
    path = _out_root(root) / "gates" / "project6_prompt3a_runtime_gate.json"
    write_json(path, gate)
    return {"gate": gate, "path": path, **result}


def write_prompt3a_completion(root: Path) -> dict[str, Any]:
    engineering = write_engineering_gate(root)
    runtime = write_runtime_gate(root)
    status = "pass" if engineering["gate"]["status"] == "pass" and runtime["gate"]["status"] == "pass" else "blocked"
    gate = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "engineering_gate": str(engineering["path"]),
        "runtime_gate": str(runtime["path"]),
        "engineering_gate_status": engineering["gate"]["status"],
        "runtime_gate_status": runtime["gate"]["status"],
        "runtime_blocking_reasons": runtime["gate"].get("blocking_reasons", []),
        "scientific_pass_allowed": status == "pass",
        "round0_unlock_allowed": False,
        "completion_marker_allowed": status == "pass",
    }
    path = _out_root(root) / "gates" / "project6_prompt3a_completion_gate.json"
    write_json(path, gate)
    return {"gate": gate, "path": path, "engineering": engineering, "runtime": runtime}
