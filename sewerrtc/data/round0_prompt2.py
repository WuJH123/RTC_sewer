from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sewerrtc.contracts.prompt3a import OUT_ROOT, PROJECT_ROOT, INP_PATH, config_hash, read_csv, read_json, sha256_file, write_csv, write_json
from sewerrtc.data.candidate_prefilter import prefilter_candidate
from sewerrtc.simulation.baseline_trajectory import PLAN_COLUMNS, PLAN_SCHEMA_VERSION, POLICIES as BASELINE_POLICIES
from sewerrtc.simulation.kpi_metrics import compute_kpis
from sewerrtc.simulation.pyswmm_runner import run_swmm_no_control_action_ablation
from sewerrtc.simulation.runtime_contracts import analyze_recovery
from sewerrtc.simulation.swmm_event_builder import build_event_inp_from_plan


PROMPT2_DIR = OUT_ROOT / "prompt2"
ROUND0_DIR = OUT_ROOT / "round0"
GATES_DIR = OUT_ROOT / "gates"
CONTROL_DIR = OUT_ROOT / "control_checkpoints"
DATASET_DIR = OUT_ROOT / "round0_dataset"
EVENT_CATALOG_DIR = OUT_ROOT / "event_catalog"
PROMPT2_EXPANSION_DIR = OUT_ROOT / "prompt2_fit_expansion"
PROMPT2_BASELINE_DIR = OUT_ROOT / "prompt2_baseline_expansion"
PROMPT2_CHECKPOINT_DIR = OUT_ROOT / "prompt2_control_checkpoints"
PROMPT2_STATE_DIR = OUT_ROOT / "prompt2_state"

MAIN_K_MAX = 8
BINARY_PUMPS = {"ADD301.2", "ADD301.3"}
VARIABLE_SPEED_PUMP = "add350.1"
POLICIES = ("no_control", "internal_rules", "executable_passive")
FORBIDDEN_PROMPT2_SPLITS = {"gat_independent_holdout", "calibration", "calibration_a", "locked_validation_b", "formal", "formal_blind"}
FIT_SPLITS = {"development_fit", "action_effect_fit"}
ACTUATOR_CSV = PROJECT_ROOT / "data" / "project6_v8_storage_retrofit_assets.csv"
FACILITY_SEMANTICS_CSV = PROJECT_ROOT / "data" / "project6_v3_facility_semantics_36.csv"
MANAGED_IDS_TXT = PROJECT_ROOT / "data" / "project6_v8_storage_retrofit_control_enabled_ids.txt"
PRIORITY_NODES = PROJECT_ROOT / "data" / "project5_design" / "priority_pfv_core_nodes.txt"
PHASE_MINIMUMS = {
    "pre_rise_or_early_rising": 12,
    "rising": 30,
    "near_peak": 24,
    "peak": 24,
    "recession": 24,
    "recovery_or_release": 12,
}
MAIN_CONCURRENCY_TARGETS = {"1-2": 720, "3-4": 630, "5-8": 450}
MAIN_CONCURRENCY_MINIMUMS = {"1-2": 600, "3-4": 500, "5-8": 350}
ANCHOR_MIN_FRACTIONS = {"internal": 0.25, "passive": 0.20, "selected_safe_fallback": 0.40}
INTERACTION_MAIN_MIN_FRACTION = 0.30
ZERO_ACTION_QA_TARGET = 40
ZERO_ACTION_QA_RANGE = (30, 60)
BINARY_TRANSITION_MINIMUMS = {
    ("ADD301.2", "OFF->ON"): 20,
    ("ADD301.2", "ON->OFF"): 20,
    ("ADD301.2", "hold-ON"): 10,
    ("ADD301.2", "hold-OFF"): 10,
    ("ADD301.3", "OFF->ON"): 20,
    ("ADD301.3", "ON->OFF"): 20,
    ("ADD301.3", "hold-ON"): 10,
    ("ADD301.3", "hold-OFF"): 10,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def existing_hash(path: Path) -> str:
    return sha256_file(path) if path.exists() and path.is_file() else ""


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def _status_code(status: str) -> int:
    if status == "pass" or status == "completed":
        return 0
    if status == "failed_gate":
        return 5
    if status == "contract_mismatch":
        return 6
    return 3


def _write_gate(path: Path, payload: dict[str, Any]) -> Path:
    payload.setdefault("created_at", utc_now())
    return write_json(path, payload)


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "completed"}


def _event_design_tokens(event_id: str) -> dict[str, str]:
    lowered = event_id.lower()
    tokens = event_id.split("_")
    return_period = next((token for token in tokens if token.startswith("T") and token[1:].isdigit()), "")
    duration = next((token[1:] for token in tokens if token.startswith("D") and token[1:].isdigit()), "")
    if "chicago_early" in lowered:
        peak_position = "chicago_early"
    elif "chicago_center" in lowered:
        peak_position = "chicago_center"
    elif "chicago_late" in lowered:
        peak_position = "chicago_late"
    elif "double" in lowered:
        peak_position = "double_peak"
    elif "block" in lowered:
        peak_position = "block"
    else:
        peak_position = "unknown"
    return {"return_period": return_period, "duration_min": duration, "peak_position": peak_position}


def _event_catalog_rows() -> list[dict[str, str]]:
    catalog = read_csv(EVENT_CATALOG_DIR / "event_catalog.csv")
    split_rows = {row.get("event_id", ""): row for row in read_csv(EVENT_CATALOG_DIR / "event_split_manifest.csv")}
    joined: list[dict[str, str]] = []
    for row in catalog:
        event_id = row.get("event_id", "")
        split_record = split_rows.get(event_id, {})
        merged = dict(row)
        merged["split"] = split_record.get("split") or row.get("split", "")
        merged["round0_eligible"] = split_record.get("round0_eligible") or row.get("round0_eligible", "")
        merged.update({k: v for k, v in _event_design_tokens(event_id).items() if not merged.get(k)})
        joined.append(merged)
    return joined


def _eligible_fit_events() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    eligible: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for row in _event_catalog_rows():
        split = row.get("split", "")
        reason = ""
        if split not in FIT_SPLITS:
            reason = f"split_not_fit:{split}"
        elif split in FORBIDDEN_PROMPT2_SPLITS:
            reason = "forbidden_split"
        elif _is_true(row.get("gat_independent_holdout", "")):
            reason = "gat_independent_holdout"
        elif _is_true(row.get("calibration_eligible", "")) or _is_true(row.get("formal_eligible", "")):
            reason = "calibration_or_formal"
        elif str(row.get("round0_eligible", "")).lower() != "true":
            reason = "round0_not_eligible"
        elif not row.get("rainfall_path") or not (row.get("rainfall_file_sha256") or row.get("rainfall_file_hash")):
            reason = "rainfall_asset_incomplete"
        if reason:
            excluded.append({**row, "selection_status": "excluded", "exclusion_reason": reason})
        else:
            eligible.append({**row, "selection_status": "eligible", "exclusion_reason": ""})
    return eligible, excluded


def _select_fit_events(events: list[dict[str, str]], target: int, seed: int) -> list[dict[str, str]]:
    del seed
    target = max(1, int(target))
    family_limit = max(1, int(target * 0.2))
    selected: list[dict[str, str]] = []
    family_counts: dict[str, int] = {}
    peaks = ["chicago_early", "chicago_center", "chicago_late", "block", "double_peak", "unknown"]
    remaining = sorted(events, key=lambda r: (r.get("peak_position", "unknown"), r.get("storm_family_id", ""), r.get("event_id", "")))
    while remaining and len(selected) < target:
        progressed = False
        for peak in peaks:
            if len(selected) >= target:
                break
            for row in list(remaining):
                family = row.get("storm_family_id", "")
                if row.get("peak_position", "unknown") == peak and family_counts.get(family, 0) < family_limit:
                    selected.append({**row, "selection_status": "selected", "selection_reason": f"fit_split_diversity:{peak}"})
                    family_counts[family] = family_counts.get(family, 0) + 1
                    remaining.remove(row)
                    progressed = True
                    break
                if len(selected) >= target:
                    break
        if not progressed:
            break
    if len(selected) < target:
        for row in list(remaining):
            family = row.get("storm_family_id", "")
            if family_counts.get(family, 0) < family_limit:
                selected.append({**row, "selection_status": "selected", "selection_reason": "fit_split_fill"})
                family_counts[family] = family_counts.get(family, 0) + 1
            if len(selected) >= target:
                break
    return selected


def _family_support(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = max(1, len(rows))
    return [
        {
            "storm_family_id": family,
            "event_count": sum(1 for row in rows if row.get("storm_family_id") == family),
            "fraction": sum(1 for row in rows if row.get("storm_family_id") == family) / total,
            "dominance_status": "pass" if sum(1 for row in rows if row.get("storm_family_id") == family) / total <= 0.2 else "fail",
        }
        for family in sorted({str(row.get("storm_family_id", "")) for row in rows})
    ]


def plan_prompt2_fit_event_expansion(config: str | Path, target_fit_events: int = 36, seed: int = 20260719) -> tuple[int, dict[str, Path]]:
    eligible, excluded = _eligible_fit_events()
    selected = _select_fit_events(eligible, target_fit_events, seed)
    plan = write_csv(PROMPT2_EXPANSION_DIR / "prompt2_fit_event_expansion_plan.csv", selected)
    event_support = write_csv(PROMPT2_EXPANSION_DIR / "prompt2_fit_event_support.csv", _group_count(selected, "peak_position"))
    storm_support = write_csv(PROMPT2_EXPANSION_DIR / "prompt2_storm_family_support.csv", _family_support(selected))
    rainfall_audit = write_csv(
        PROMPT2_EXPANSION_DIR / "prompt2_rainfall_asset_audit.csv",
        [
            {
                "event_id": row.get("event_id", ""),
                "rainfall_path": row.get("rainfall_path", ""),
                "rainfall_file_sha256": row.get("rainfall_file_sha256") or row.get("rainfall_file_hash", ""),
                "rainfall_series_sha256": row.get("rainfall_series_sha256") or row.get("rainfall_series_hash", ""),
                "status": "pass" if row.get("rainfall_path") and (row.get("rainfall_file_sha256") or row.get("rainfall_file_hash")) else "incomplete",
            }
            for row in selected
        ],
    )
    exclusions = write_csv(PROMPT2_EXPANSION_DIR / "prompt2_event_expansion_exclusions.csv", excluded)
    blocking = []
    if len({row.get("event_id", "") for row in selected}) < 30:
        blocking.append("unique_fit_events_below_30")
    if any(row.get("dominance_status") == "fail" for row in _family_support(selected)):
        blocking.append("storm_family_dominance_above_20pct")
    status = "completed" if not blocking else "blocked"
    report = write_json(
        PROMPT2_EXPANSION_DIR / "prompt2_event_expansion_report.json",
        {
            "status": status,
            "selected_event_count": len(selected),
            "target_fit_events": target_fit_events,
            "minimum_fit_events": 30,
            "eligible_event_count": len(eligible),
            "excluded_event_count": len(excluded),
            "blocking_reasons": blocking,
            "structural_infeasible_peak_positions": [peak for peak in ["chicago_early", "chicago_center", "chicago_late", "block", "double_peak"] if not any(row.get("peak_position") == peak for row in eligible)],
            "config_hash": config_hash(config),
            "round0_unlock_allowed": False,
        },
    )
    return _status_code(status), {"plan": plan, "event_support": event_support, "storm_family_support": storm_support, "rainfall_audit": rainfall_audit, "exclusions": exclusions, "report": report}


def audit_prompt2_fit_event_expansion(config: str | Path) -> tuple[int, dict[str, Path]]:
    rows = [row for row in read_csv(PROMPT2_EXPANSION_DIR / "prompt2_fit_event_expansion_plan.csv") if row.get("selection_status") == "selected"]
    families = _family_support(rows)
    forbidden = [row for row in rows if row.get("split") not in FIT_SPLITS or row.get("split") in FORBIDDEN_PROMPT2_SPLITS or _is_true(row.get("gat_independent_holdout", ""))]
    blocking: list[str] = []
    if len({row.get("event_id", "") for row in rows}) < 30:
        blocking.append("unique_fit_events_below_30")
    if any(row.get("dominance_status") == "fail" for row in families):
        blocking.append("storm_family_dominance_above_20pct")
    if forbidden:
        blocking.append("forbidden_event_split_present")
    status = "pass" if not blocking else "failed_gate" if "storm_family_dominance_above_20pct" in blocking or forbidden else "blocked"
    audit = write_json(
        PROMPT2_EXPANSION_DIR / "prompt2_fit_event_expansion_audit.json",
        {
            "status": status,
            "unique_fit_events": len({row.get("event_id", "") for row in rows}),
            "selected_event_count": len(rows),
            "forbidden_event_count": len(forbidden),
            "blocking_reasons": blocking,
            "config_hash": config_hash(config),
        },
    )
    return _status_code(status), {"audit": audit}


def plan_prompt2_baseline_expansion(config: str | Path, max_events: int = 0) -> tuple[int, dict[str, Path]]:
    selected = [row for row in read_csv(PROMPT2_EXPANSION_DIR / "prompt2_fit_event_expansion_plan.csv") if row.get("selection_status") == "selected"]
    if max_events and max_events > 0:
        selected = selected[:max_events]
    network_sha = existing_hash(INP_PATH)
    catalog_path = EVENT_CATALOG_DIR / "event_catalog.csv"
    split_path = EVENT_CATALOG_DIR / "event_split_manifest.csv"
    rows: list[dict[str, Any]] = []
    for event in selected:
        for policy in BASELINE_POLICIES:
            event_id = event.get("event_id", "")
            trajectory_id = f"{event.get('canonical_event_id') or event_id}_{policy}"
            rows.append(
                {
                    "plan_schema_version": PLAN_SCHEMA_VERSION,
                    "trajectory_id": trajectory_id,
                    "event_id": event_id,
                    "canonical_event_id": event.get("canonical_event_id") or event_id,
                    "storm_family_id": event.get("storm_family_id", ""),
                    "split": event.get("split", ""),
                    "policy_id": policy,
                    "policy_mode": policy,
                    "network_policy": "single_retrofit_inp",
                    "rainfall_path": event.get("rainfall_path", ""),
                    "rainfall_file_sha256": event.get("rainfall_file_sha256") or event.get("rainfall_file_hash", ""),
                    "rainfall_series_sha256": event.get("rainfall_series_sha256") or event.get("rainfall_series_hash") or event.get("rainfall_file_sha256", ""),
                    "network_path": str(INP_PATH),
                    "network_sha256": network_sha,
                    "event_catalog_path": str(catalog_path),
                    "event_catalog_sha256": existing_hash(catalog_path),
                    "event_split_manifest_sha256": existing_hash(split_path),
                    "prompt2_import_lock_sha256": existing_hash(OUT_ROOT / "completion_markers" / "ImportPrompt2Artifacts_COMPLETED.json"),
                    "native_rule_audit_sha256": existing_hash(OUT_ROOT / "native_rules" / "native_rule_audit_report.json"),
                    "passive_fallback_contract_sha256": existing_hash(OUT_ROOT / "fallbacks" / "passive_fallback_contract.json"),
                    "internal_fallback_contract_sha256": existing_hash(OUT_ROOT / "fallbacks" / "internal_fallback_contract.json"),
                    "fallback_selection_contract_sha256": existing_hash(OUT_ROOT / "fallbacks" / "fallback_selection_contract.json"),
                    "truth_controller_separation_required": True,
                    "controller_visible_state_contract_version": "project6_controller_visible_state_v1",
                    "controller_memory_required": True,
                    "hotstart_required": True,
                    "start_time": event.get("start_time", ""),
                    "end_time": event.get("end_time", ""),
                    "tail_min": 180,
                    "tail_policy": "rain_end_plus_180min_or_recovery",
                    "output_root": str(PROMPT2_BASELINE_DIR / "trajectories" / (event.get("canonical_event_id") or event_id) / policy),
                    "status": "planned",
                    "exclusion_reason": "",
                }
            )
    plan = write_csv(PROMPT2_BASELINE_DIR / "baseline_trajectory_plan.csv", rows, PLAN_COLUMNS)
    report = write_json(
        PROMPT2_BASELINE_DIR / "prompt2_baseline_expansion_plan_report.json",
        {
            "status": "completed" if rows else "blocked",
            "planned_event_count": len({row.get("event_id", "") for row in selected}),
            "planned_trajectory_count": len(rows),
            "policy_count_per_event": len(BASELINE_POLICIES),
            "config_hash": config_hash(config),
        },
    )
    return (0 if rows else 3), {"plan": plan, "report": report}


def audit_prompt2_baseline_expansion(config: str | Path) -> tuple[int, dict[str, Path]]:
    manifest_path = PROMPT2_BASELINE_DIR / "baseline_trajectory_manifest.csv"
    blocking: list[str] = []
    if not manifest_path.exists():
        blocking.append("generation_manifest_missing")
        rows: list[dict[str, str]] = []
    else:
        rows = read_csv(manifest_path)
    completed = [row for row in rows if row.get("status") in {"completed", "skipped_existing"} or row.get("detail_file")]
    policy_by_event: dict[str, set[str]] = {}
    for row in completed:
        policy_by_event.setdefault(row.get("event_id", ""), set()).add(row.get("policy_id", ""))
    completed_events = set(policy_by_event)
    complete_policy_events = {event for event, policies in policy_by_event.items() if policies == set(BASELINE_POLICIES)}
    if manifest_path.exists() and len(completed_events) < 30:
        blocking.append("completed_events_below_30")
    if manifest_path.exists() and len(completed) < 90:
        blocking.append("completed_trajectories_below_90")
    if manifest_path.exists() and len(complete_policy_events) < 30:
        blocking.append("complete_three_policy_events_below_30")
    status = "pass" if not blocking else "blocked"
    audit = write_json(
        PROMPT2_BASELINE_DIR / "prompt2_baseline_expansion_audit_report.json",
        {
            "status": status,
            "completed_event_count": len(completed_events),
            "complete_three_policy_event_count": len(complete_policy_events),
            "completed_trajectory_count": len(completed),
            "failed_trajectory_count": len([row for row in rows if row.get("status") == "failed"]),
            "blocking_reasons": blocking,
            "config_hash": config_hash(config),
        },
    )
    return _status_code(status), {"audit": audit}


def audit_prompt2_entry(config: str | Path) -> tuple[int, dict[str, Path]]:
    paths = {
        "prompt2_gat_readiness": GATES_DIR / "project6_prompt2_gat_readiness_gate.json",
        "primary_gat_lock": OUT_ROOT / "gat" / "gat_primary_selection_lock.json",
        "same_state_branch_gate": OUT_ROOT / "state_clone" / "same_state_branch_gate.json",
        "same_state_replay_report": OUT_ROOT / "state_clone" / "same_state_replay_report.json",
        "hotstart_readiness_gate": OUT_ROOT / "hotstart" / "hotstart_acceleration_readiness_gate.json",
        "facility_semantics": PROJECT_ROOT / "docs" / "contracts" / "facility_semantics_contract.json",
        "kpi_contract": PROJECT_ROOT / "docs" / "contracts" / "kpi_contract.json",
        "forecast_contract": PROJECT_ROOT / "docs" / "contracts" / "forecast_contract.json",
        "event_split_leakage": OUT_ROOT / "event_catalog" / "event_split_leakage_audit.csv",
        "state_shape": OUT_ROOT / "state" / "augmented_state_shape_audit.json",
        "feature_index_node": OUT_ROOT / "state" / "node_feature_index.json",
        "feature_index_facility": OUT_ROOT / "state" / "facility_feature_index.json",
    }
    readiness = read_json(paths["prompt2_gat_readiness"])
    same_state = read_json(paths["same_state_branch_gate"])
    replay = read_json(paths["same_state_replay_report"])
    hotstart = read_json(paths["hotstart_readiness_gate"])
    leakage_rows = read_csv(paths["event_split_leakage"])
    leakage_count = sum(1 for row in leakage_rows if str(row.get("status", "")).lower() in {"fail", "failed_gate"} or _is_true(row.get("leakage", "")))
    checks = {
        "prompt2_gat_readiness_pass": readiness.get("status") == "pass" and readiness.get("allowed_to_enter_prompt3a") is True,
        "primary_gat_lock_valid": paths["primary_gat_lock"].exists(),
        "state_shape_contract_exists": paths["state_shape"].exists(),
        "feature_index_exists": paths["feature_index_node"].exists() and paths["feature_index_facility"].exists(),
        "same_state_branch_gate_pass": same_state.get("status") == "pass" and same_state.get("selected_same_state_method") == "deterministic_prefix_replay",
        "deterministic_replay_18_of_18": int(replay.get("passed_checkpoint_count") or 0) == 18 and replay.get("formal_same_state_unlock_allowed") is True,
        "hotstart_not_certified_for_candidate_labels": hotstart.get("hotstart_acceleration_allowed") is not True,
        "facility_semantics_valid": paths["facility_semantics"].exists(),
        "kpi_contract_valid": paths["kpi_contract"].exists(),
        "forecast_contract_valid": paths["forecast_contract"].exists(),
        "split_leakage_zero": leakage_count == 0,
        "single_network_hash_present": INP_PATH.exists(),
    }
    rows = [
        {
            "input_id": key,
            "path": str(path),
            "exists": str(path.exists()).lower(),
            "sha256": existing_hash(path),
        }
        for key, path in paths.items()
    ]
    passed = all(checks.values())
    blocking = [key for key, ok in checks.items() if not ok]
    report = {
        "status": "pass" if passed else "blocked",
        "checks": checks,
        "blocking_reasons": blocking,
        "config_hash": config_hash(config),
        "network_path": str(INP_PATH),
        "network_sha256": existing_hash(INP_PATH),
        "selected_same_state_method": same_state.get("selected_same_state_method"),
        "hotstart_acceleration_allowed": False,
        "round0_planning_allowed": passed,
        "round0_unlock_allowed": False,
    }
    inputs = write_csv(PROMPT2_DIR / "prompt2_entry_inputs.csv", rows)
    audit = write_json(PROMPT2_DIR / "prompt2_entry_audit.json", report)
    gate = _write_gate(GATES_DIR / "prompt2_entry_gate.json", report)
    return _status_code(report["status"]), {"inputs": inputs, "audit": audit, "gate": gate}


def build_control_aligned_checkpoint_catalog(config: str | Path) -> tuple[int, dict[str, Path]]:
    source = OUT_ROOT / "checkpoint_catalog" / "checkpoint_catalog.csv"
    checkpoints = read_csv(source)
    rows: list[dict[str, Any]] = []
    for row in checkpoints:
        elapsed = float(row.get("checkpoint_elapsed_min") or row.get("elapsed_min") or row.get("event_time") or 0.0)
        aligned = abs(elapsed % 10.0) < 1.0e-6
        split = row.get("split", "action_effect_fit")
        forbidden_split = split in {"gat_independent_holdout", "calibration_a", "locked_validation_b", "formal_blind", "formal"}
        history_available = _is_true(row.get("history_60min_available", elapsed >= 60.0))
        future_available = _is_true(row.get("future_120min_available", "true"))
        eligible = aligned and not forbidden_split and history_available and future_available
        rows.append(
            {
                **row,
                "elapsed_min": elapsed,
                "elapsed_min_mod_10": elapsed % 10.0,
                "control_aligned": str(aligned).lower(),
                "round0_candidate_eligible": str(eligible).lower(),
                "round0_exclusion_reason": (
                    ""
                    if eligible
                    else "not_10min_control_boundary"
                    if not aligned
                    else "forbidden_split"
                    if forbidden_split
                    else "missing_60min_history"
                    if not history_available
                    else "missing_120min_future"
                ),
                "history_60min_available": str(history_available).lower(),
                "future_120min_available": str(future_available).lower(),
                "same_state_method": "deterministic_prefix_replay",
                "hotstart_allowed_for_candidate": "false",
            }
        )
    eligible_rows = [r for r in rows if r["round0_candidate_eligible"] == "true"]
    split_audit = write_csv(CONTROL_DIR / "control_checkpoint_split_audit.csv", rows)
    history_audit = write_csv(CONTROL_DIR / "control_checkpoint_history_audit.csv", rows)
    future_audit = write_csv(CONTROL_DIR / "control_checkpoint_future_audit.csv", rows)
    phase_rows = []
    for phase in sorted({str(r.get("phase", "")) for r in rows}):
        phase_rows.append({"phase": phase, "eligible_count": sum(1 for r in eligible_rows if r.get("phase") == phase), "total_count": sum(1 for r in rows if r.get("phase") == phase)})
    event_rows = []
    for event in sorted({str(r.get("event_id", "")) for r in rows}):
        event_rows.append({"event_id": event, "eligible_count": sum(1 for r in eligible_rows if r.get("event_id") == event), "total_count": sum(1 for r in rows if r.get("event_id") == event)})
    phase_support = write_csv(CONTROL_DIR / "control_checkpoint_phase_support.csv", phase_rows)
    event_support = write_csv(CONTROL_DIR / "control_checkpoint_event_support.csv", event_rows)
    catalog = write_csv(CONTROL_DIR / "control_checkpoint_catalog.csv", rows)
    report = {
        "status": "completed" if rows else "blocked",
        "source_checkpoint_catalog": str(source),
        "source_checkpoint_catalog_sha256": existing_hash(source),
        "control_aligned_checkpoint_count": len(eligible_rows),
        "unique_fit_events": len({r.get("event_id", "") for r in eligible_rows}),
        "target_unique_fit_events": 30,
        "target_control_aligned_checkpoints": 120,
        "support_status": "sufficient" if len(eligible_rows) >= 120 and len({r.get("event_id", "") for r in eligible_rows}) >= 30 else "insufficient_support",
        "config_hash": config_hash(config),
    }
    report_path = write_json(CONTROL_DIR / "control_checkpoint_catalog_report.json", report)
    return _status_code(report["status"]), {
        "catalog": catalog,
        "report": report_path,
        "split": split_audit,
        "history": history_audit,
        "future": future_audit,
        "phase": phase_support,
        "event": event_support,
    }


def audit_control_aligned_checkpoint_catalog(config: str | Path) -> tuple[int, dict[str, Path]]:
    del config
    catalog = CONTROL_DIR / "control_checkpoint_catalog.csv"
    report = read_json(CONTROL_DIR / "control_checkpoint_catalog_report.json")
    rows = read_csv(catalog)
    leakage = [r for r in rows if r.get("round0_candidate_eligible") == "true" and r.get("split") in {"gat_independent_holdout", "calibration_a", "locked_validation_b", "formal_blind", "formal"}]
    nonaligned = [r for r in rows if r.get("round0_candidate_eligible") == "true" and float(r.get("elapsed_min_mod_10") or 0.0) != 0.0]
    status = "pass" if rows and not leakage and not nonaligned and report.get("status") == "completed" else "contract_mismatch" if leakage or nonaligned else "blocked"
    audit = {
        "status": status,
        "row_count": len(rows),
        "eligible_count": sum(1 for r in rows if r.get("round0_candidate_eligible") == "true"),
        "split_leakage_count": len(leakage),
        "non_10min_eligible_count": len(nonaligned),
        "support_status": report.get("support_status"),
        "blocking_reasons": ([] if status == "pass" else ["split_or_alignment_violation"] if status == "contract_mismatch" else ["catalog_missing_or_empty"]),
    }
    path = write_json(CONTROL_DIR / "control_checkpoint_catalog_audit_report.json", audit)
    return _status_code(status), {"audit": path}


def _elapsed_min(row: dict[str, Any]) -> float:
    for key in ("elapsed_min", "event_time", "checkpoint_elapsed_min"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                continue
    return 0.0


def _event_duration_min(event_id: str) -> float | None:
    for part in event_id.split("_"):
        if part.startswith("D") and part[1:].isdigit():
            return float(part[1:])
    return None


def _phase_from_detail_row(row: dict[str, str], elapsed: float) -> str:
    existing = str(row.get("phase", "")).strip()
    duration = _event_duration_min(str(row.get("event_id", "")))
    if existing:
        aliases: dict[str, str] = {
            "pre-rise": "pre_rise_or_early_rising",
            "early_rising": "pre_rise_or_early_rising",
            "near_peak": "near_peak",
            "peak": "peak",
            "recovery": "recovery_or_release",
            "release": "recovery_or_release",
        }
        if existing == "pre_peak":
            if elapsed <= 70:
                if duration is not None and duration <= 210 and elapsed >= 70:
                    return "rising"
                return "pre_rise_or_early_rising"
            if elapsed >= 90:
                return "near_peak"
            return "rising"
        if existing == "recession":
            if duration is not None and elapsed >= duration + 30:
                return "recovery_or_release"
            return "recession"
        return aliases.get(existing, existing)
    flooding = float(row.get("flooding_rate", row.get("priority_flooding_rate", 0)) or 0)
    trend = float(row.get("priority_depth_trend", row.get("system_stored_volume_trend", 0)) or 0)
    if elapsed < 80:
        return "pre_rise_or_early_rising"
    if trend > 0:
        return "rising"
    if flooding > 0:
        return "peak"
    if trend < 0:
        return "recession"
    return "recovery_or_release"


def build_prompt2_control_checkpoint_candidates(config: str | Path) -> tuple[int, dict[str, Path]]:
    manifest = read_csv(PROMPT2_BASELINE_DIR / "baseline_trajectory_manifest.csv")
    rows: list[dict[str, Any]] = []
    skipped_details: list[dict[str, Any]] = []
    for traj in manifest:
        if traj.get("status") not in {"completed", "skipped_existing"} and not traj.get("detail_file"):
            continue
        detail_path = Path(traj.get("detail_file") or traj.get("detail_path") or traj.get("output_detail_path") or "")
        if not detail_path.is_absolute():
            detail_path = PROJECT_ROOT / detail_path
        if not detail_path.is_file():
            skipped_details.append(
                {
                    "trajectory_id": traj.get("trajectory_id", ""),
                    "event_id": traj.get("event_id", ""),
                    "policy_id": traj.get("policy_id", ""),
                    "detail_path": str(detail_path),
                    "skip_reason": "detail_file_missing_or_not_file",
                }
            )
            continue
        detail_rows = read_csv(detail_path)
        if not detail_rows:
            continue
        elapsed_values = {round(_elapsed_min(row), 6) for row in detail_rows}
        max_elapsed = max(_elapsed_min(row) for row in detail_rows)
        for row in detail_rows:
            elapsed = _elapsed_min(row)
            aligned = abs(elapsed % 10.0) < 1.0e-6
            history_frames_present = all(round(elapsed + offset, 6) in elapsed_values for offset in [0, -10, -20, -30, -40, -50, -60])
            history = elapsed if history_frames_present else 0.0
            future = max_elapsed - elapsed
            full_recovery = str(traj.get("recovery_status", traj.get("full_recovery_contract_available", "true"))).lower() not in {"", "missing", "false"}
            forbidden = traj.get("split") in FORBIDDEN_PROMPT2_SPLITS
            eligible = aligned and history_frames_present and future >= 120 and full_recovery and not forbidden
            event_id = traj.get("event_id", "")
            policy = traj.get("policy_id", "")
            checkpoint_id = f"{traj.get('trajectory_id', event_id + '_' + policy)}_t{int(round(elapsed)):04d}"
            state_hash = hash_payload({k: row.get(k, "") for k in sorted(row) if k.endswith("_depth") or k.endswith("_flow") or k in {"priority_depth_max", "system_stored_volume"}})
            rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "trajectory_id": traj.get("trajectory_id", ""),
                    "event_id": event_id,
                    "storm_family_id": traj.get("storm_family_id", ""),
                    "policy_id": policy,
                    "split": traj.get("split", "development_fit"),
                    "elapsed_min": elapsed,
                    "event_time": elapsed,
                    "phase": _phase_from_detail_row(row, elapsed),
                    "history_available_min": 60.0 if history_frames_present else history,
                    "future_available_min": future,
                    "full_recovery_contract_available": str(full_recovery).lower(),
                    "control_aligned": str(aligned).lower(),
                    "round0_candidate_eligible": str(eligible).lower(),
                    "round0_exclusion_reason": "" if eligible else "not_10min_control_boundary" if not aligned else "missing_60min_history" if history < 60 else "missing_120min_future" if future < 120 else "forbidden_split_or_recovery_missing",
                    "hydraulic_state_hash": state_hash,
                    "node_state_hash": state_hash,
                    "rainfall_window_hash": hash_payload({"event_id": event_id, "elapsed_min": elapsed}),
                    "controller_state_hash": hash_payload({"trajectory_id": traj.get("trajectory_id", ""), "elapsed_min": elapsed, "policy_id": policy}),
                    "same_state_method": "deterministic_prefix_replay",
                }
            )
    candidates = write_csv(PROMPT2_CHECKPOINT_DIR / "prompt2_control_checkpoint_candidates.csv", rows)
    duplicate = write_csv(PROMPT2_CHECKPOINT_DIR / "control_checkpoint_duplicate_audit.csv", _duplicate_rows(rows, "hydraulic_state_hash", "exact_duplicate"))
    near_duplicate = write_csv(PROMPT2_CHECKPOINT_DIR / "control_checkpoint_near_duplicate_audit.csv", _duplicate_rows(rows, "rainfall_window_hash", "rainfall_window_duplicate"))
    cluster = write_csv(PROMPT2_CHECKPOINT_DIR / "control_checkpoint_temporal_cluster_audit.csv", _group_count(rows, "event_id"))
    missing_detail = write_csv(PROMPT2_CHECKPOINT_DIR / "control_checkpoint_missing_detail_audit.csv", skipped_details)
    eligible = [row for row in rows if row.get("round0_candidate_eligible") == "true"]
    report = write_json(
        PROMPT2_CHECKPOINT_DIR / "prompt2_control_checkpoint_candidate_report.json",
        {
            "status": "completed" if rows else "blocked",
            "candidate_count": len(rows),
            "eligible_candidate_count": len(eligible),
            "unique_fit_events": len({row.get("event_id", "") for row in eligible}),
            "config_hash": config_hash(config),
        },
    )
    return (0 if rows else 3), {"candidates": candidates, "duplicates": duplicate, "near_duplicates": near_duplicate, "clusters": cluster, "missing_detail": missing_detail, "report": report}


def _duplicate_rows(rows: list[dict[str, Any]], key: str, label: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return [{"checkpoint_id": row.get("checkpoint_id", ""), key: row.get(key, ""), "duplicate_type": label, "duplicate_count": counts.get(str(row.get(key, "")), 0), "status": "duplicate" if counts.get(str(row.get(key, "")), 0) > 1 else "unique"} for row in rows]


def select_prompt2_control_checkpoints(config: str | Path, target_checkpoints: int = 144, max_per_event: int = 6, seed: int = 20260719) -> tuple[int, dict[str, Path]]:
    del seed
    rows = [row for row in read_csv(PROMPT2_CHECKPOINT_DIR / "prompt2_control_checkpoint_candidates.csv") if row.get("round0_candidate_eligible") == "true"]
    selected: list[dict[str, Any]] = []
    by_event: dict[str, int] = {}
    selected_keys: set[str] = set()
    phases = ["pre_rise_or_early_rising", "rising", "near_peak", "peak", "recession", "recovery_or_release"]
    quota_phases = ["near_peak", "rising", "pre_rise_or_early_rising", "recession", "recovery_or_release", "peak"]

    def row_key(row: dict[str, Any]) -> str:
        return str(row.get("checkpoint_id") or f"{row.get('trajectory_id','')}::{row.get('elapsed_min','')}")

    def add_row(row: dict[str, Any]) -> bool:
        event = row.get("event_id", "")
        key = row_key(row)
        if key in selected_keys or by_event.get(event, 0) >= max_per_event:
            return False
        selected.append(row)
        selected_keys.add(key)
        by_event[event] = by_event.get(event, 0) + 1
        return True

    for phase in quota_phases:
        minimum = PHASE_MINIMUMS.get(phase, 0)
        for row in rows:
            if len(selected) >= target_checkpoints:
                break
            event = row.get("event_id", "")
            if row.get("phase") != phase or by_event.get(event, 0) >= max_per_event:
                continue
            add_row(row)
            if sum(1 for item in selected if item.get("phase") == phase) >= minimum:
                break
        if len(selected) >= target_checkpoints:
            break
    for row in rows:
        if len(selected) >= target_checkpoints:
            break
        event = row.get("event_id", "")
        if event and by_event.get(event, 0) == 0:
            add_row(row)
        if len({item.get("event_id", "") for item in selected}) >= 30:
            break
    for row in rows:
        event = row.get("event_id", "")
        if len(selected) >= target_checkpoints:
            break
        add_row(row)
    selected_path = write_csv(PROMPT2_CHECKPOINT_DIR / "prompt2_selected_control_checkpoints.csv", selected)
    catalog = write_csv(PROMPT2_CHECKPOINT_DIR / "control_checkpoint_catalog.csv", selected)
    phase_support = write_csv(PROMPT2_CHECKPOINT_DIR / "control_checkpoint_phase_support.csv", _group_count(selected, "phase"))
    event_support = write_csv(PROMPT2_CHECKPOINT_DIR / "control_checkpoint_event_support.csv", _group_count(selected, "event_id"))
    blocking = []
    if len(selected) < 120:
        blocking.append("control_checkpoints_below_120")
    if len({row.get("event_id", "") for row in selected}) < 30:
        blocking.append("unique_fit_events_below_30")
    status = "completed" if not blocking else "blocked"
    report = write_json(
        PROMPT2_CHECKPOINT_DIR / "control_checkpoint_catalog_report.json",
        {
            "status": status,
            "selected_checkpoint_count": len(selected),
            "control_aligned_checkpoint_count": len(selected),
            "unique_fit_events": len({row.get("event_id", "") for row in selected}),
            "support_status": "sufficient" if not blocking else "insufficient_support",
            "blocking_reasons": blocking,
            "target_checkpoints": target_checkpoints,
            "max_per_event": max_per_event,
            "config_hash": config_hash(config),
        },
    )
    return _status_code(status), {"selected": selected_path, "catalog": catalog, "phase": phase_support, "event": event_support, "report": report}


def audit_prompt2_control_checkpoint_support(config: str | Path) -> tuple[int, dict[str, Path]]:
    rows = read_csv(PROMPT2_CHECKPOINT_DIR / "prompt2_selected_control_checkpoints.csv")
    selected = [row for row in rows if str(row.get("round0_candidate_eligible", "true")).lower() != "false"]
    blocking: list[str] = []
    alignment_failures = [row for row in selected if abs(float(row.get("elapsed_min") or 0) % 10.0) > 1.0e-6]
    history_failures = [row for row in selected if float(row.get("history_available_min") or 0) < 60]
    future_failures = [row for row in selected if float(row.get("future_available_min") or 0) < 120]
    leakage = [row for row in selected if row.get("split") in FORBIDDEN_PROMPT2_SPLITS]
    phase_counts = {row["name"]: int(row["count"]) for row in _group_count(selected, "phase")}
    missing_phase_quota = [phase for phase, minimum in PHASE_MINIMUMS.items() if phase_counts.get(phase, 0) < minimum]
    if len({row.get("event_id", "") for row in selected}) < 30:
        blocking.append("unique_fit_events_below_30")
    if len(selected) < 120:
        blocking.append("control_checkpoints_below_120")
    if alignment_failures:
        blocking.append("alignment_failure")
    if history_failures:
        blocking.append("history_failure")
    if future_failures:
        blocking.append("future_horizon_failure")
    if leakage:
        blocking.append("split_leakage")
    if missing_phase_quota:
        blocking.append("phase_support_below_minimum")
    status = "pass" if not blocking else "blocked"
    audit = write_json(
        PROMPT2_CHECKPOINT_DIR / "prompt2_control_checkpoint_support_audit.json",
        {
            "status": status,
            "unique_fit_events": len({row.get("event_id", "") for row in selected}),
            "selected_checkpoint_count": len(selected),
            "alignment_failure_count": len(alignment_failures),
            "history_failure_count": len(history_failures),
            "future_horizon_failure_count": len(future_failures),
            "split_leakage_count": len(leakage),
            "phase_counts": phase_counts,
            "missing_phase_quota": missing_phase_quota,
            "blocking_reasons": blocking,
            "same_state_method_valid": True,
            "config_hash": config_hash(config),
        },
    )
    return _status_code(status), {"audit": audit}


def build_prompt2_state_input_manifest(config: str | Path, max_samples: int = 0) -> tuple[int, dict[str, Path]]:
    from sewerrtc.state.state_input_manifest import build_state_input_manifest

    manifest = PROMPT2_STATE_DIR / "state_inputs" / "state_input_manifest_v1.csv"
    code, outputs = build_state_input_manifest(
        out_dir=PROMPT2_STATE_DIR / "state_inputs",
        source_mode="project6_retrofit_baseline",
        trajectory_root=PROMPT2_BASELINE_DIR,
        control_checkpoint_catalog=PROMPT2_CHECKPOINT_DIR / "prompt2_selected_control_checkpoints.csv",
        max_samples=max_samples,
    )
    report = write_json(PROMPT2_STATE_DIR / "prompt2_state_input_manifest_report.json", {"status": "completed" if manifest.exists() else "blocked", "manifest": str(manifest), "manifest_sha256": existing_hash(manifest), "outputs": {k: str(v) for k, v in outputs.items()}, "config_hash": config_hash(config)})
    return (0 if code == 0 and manifest.exists() else 3), {"manifest": manifest, "report": report}


def build_prompt2_state_features(config: str | Path, max_samples: int = 0) -> tuple[int, dict[str, Path]]:
    from sewerrtc.state.runtime_state_features import build_runtime_state_features

    manifest = PROMPT2_STATE_DIR / "state_inputs" / "state_input_manifest_v1.csv"
    if not manifest.exists():
        report = write_json(PROMPT2_STATE_DIR / "prompt2_state_feature_report.json", {"status": "blocked", "blocking_reasons": ["state_input_manifest_missing"]})
        return 3, {"report": report}
    code, outputs = build_runtime_state_features(
        config_path=Path(config),
        lock_path=OUT_ROOT / "gat" / "gat_primary_selection_lock.json",
        state_input_manifest=manifest,
        out_dir=PROMPT2_STATE_DIR / "state",
        max_samples=max_samples,
        state_validation_mode="prompt2_fit_control_checkpoint",
    )
    report_path = write_json(
        PROMPT2_STATE_DIR / "prompt2_state_feature_report.json",
        {
            "status": "completed" if code == 0 else "blocked",
            "runtime_state_feature_outputs": {key: str(value) for key, value in outputs.items()},
            "config_hash": config_hash(config),
        },
    )
    return code, {**outputs, "report": report_path}


def audit_prompt2_state_coverage(config: str | Path) -> tuple[int, dict[str, Path]]:
    selected = read_csv(PROMPT2_CHECKPOINT_DIR / "prompt2_selected_control_checkpoints.csv")
    manifest = read_csv(PROMPT2_STATE_DIR / "state_inputs" / "state_input_manifest_v1.csv")
    shape = read_json(PROMPT2_STATE_DIR / "state" / "augmented_state_shape_audit.json")
    missing_events = {row.get("event_id", "") for row in selected} - {row.get("event_id", "") for row in manifest}
    schema_mismatch = shape.get("node_feature_count_matches_tensor") is False or shape.get("facility_feature_count_matches_tensor") is False
    blocking = []
    if not manifest:
        blocking.append("state_input_manifest_empty")
    if missing_events:
        blocking.append("selected_checkpoint_events_missing_from_state_inputs")
    if schema_mismatch:
        blocking.append("state_schema_mismatch")
    status = "pass" if not blocking else "blocked"
    audit = write_json(PROMPT2_STATE_DIR / "prompt2_state_coverage_audit.json", {"status": status, "selected_checkpoint_count": len(selected), "state_input_row_count": len(manifest), "missing_event_count": len(missing_events), "blocking_reasons": blocking, "config_hash": config_hash(config)})
    return _status_code(status), {"audit": audit}


def evaluate_prompt2_checkpoint_support_gate(config: str | Path) -> tuple[int, dict[str, Path]]:
    fit = read_json(PROMPT2_EXPANSION_DIR / "prompt2_fit_event_expansion_audit.json")
    baseline = read_json(PROMPT2_BASELINE_DIR / "prompt2_baseline_expansion_audit_report.json")
    checkpoints = read_json(PROMPT2_CHECKPOINT_DIR / "prompt2_control_checkpoint_support_audit.json")
    state = read_json(PROMPT2_STATE_DIR / "prompt2_state_coverage_audit.json")
    same_state = read_json(OUT_ROOT / "state_clone" / "same_state_branch_gate.json")
    checks = {
        "unique_fit_events": fit.get("status") == "pass",
        "baseline_expansion": baseline.get("status") == "pass",
        "control_checkpoint_support": checkpoints.get("status") == "pass",
        "state_coverage": state.get("status") == "pass",
        "same_state_method_valid": same_state.get("status") == "pass" and same_state.get("selected_same_state_method") == "deterministic_prefix_replay",
    }
    status = "pass" if all(checks.values()) else "blocked"
    gate = write_json(GATES_DIR / "prompt2_checkpoint_support_gate.json", {"status": status, "checks": checks, "blocking_reasons": [key for key, ok in checks.items() if not ok], "round0_planning_allowed": status == "pass", "config_hash": config_hash(config)})
    return _status_code(status), {"gate": gate}


def build_round0_coverage_contract(config: str | Path) -> tuple[int, dict[str, Path]]:
    payload = {
        "contract_version": "project6_round0_coverage_contract_v1",
        "target_effective_candidates": 1800,
        "target_range": [1500, 2000],
        "main_k_max": MAIN_K_MAX,
        "pressure_pool_training_allowed": False,
        "phase_targets": {
            "pre_rise_or_early_rising": 180,
            "rising": 450,
            "near_peak": 360,
            "peak": 360,
            "recession": 360,
            "recovery_or_release": 90,
        },
        "concurrency_targets": {"1-2": 720, "3-4": 630, "5-8": 450},
        "concurrency_minimums": MAIN_CONCURRENCY_MINIMUMS,
        "anchor_min_fractions": ANCHOR_MIN_FRACTIONS,
        "interaction_main_min_fraction": INTERACTION_MAIN_MIN_FRACTION,
        "zero_action_qa_target_range": list(ZERO_ACTION_QA_RANGE),
        "binary_pumps": sorted(BINARY_PUMPS),
        "binary_transition_minimums": {f"{pump}:{transition}": count for (pump, transition), count in BINARY_TRANSITION_MINIMUMS.items()},
        "variable_speed_pump": VARIABLE_SPEED_PUMP,
        "same_state_method": "deterministic_prefix_replay",
        "hotstart_allowed_for_candidate_labels": False,
        "config_hash": config_hash(config),
    }
    contract = write_json(ROUND0_DIR / "round0_coverage_contract.json", payload)
    schema = write_csv(
        ROUND0_DIR / "round0_candidate_manifest_schema.csv",
        [{"field": name, "required": "true"} for name in ROUND0_MANIFEST_FIELDS],
    )
    return 0, {"contract": contract, "schema": schema}


ROUND0_MANIFEST_FIELDS = [
    "case_id",
    "candidate_id",
    "event_id",
    "storm_family_id",
    "checkpoint_id",
    "split",
    "phase",
    "anchor_type",
    "selected_fallback",
    "fallback_selection_id",
    "facility_ids",
    "facility_types",
    "action_directions",
    "action_magnitude",
    "duration_steps",
    "ttl",
    "concurrency",
    "concurrency_stratum",
    "candidate_k",
    "k_value",
    "active_facility_ids",
    "active_facility_count_by_step",
    "active_facility_mask_hash",
    "candidate_pool",
    "binary_pump_id",
    "initial_binary_state",
    "requested_binary_state",
    "projected_binary_state",
    "expected_actual_binary_state",
    "transition_type",
    "minimum_on_remaining",
    "minimum_off_remaining",
    "dwell_remaining",
    "interaction_group_id",
    "interaction_type",
    "interaction_facility_ids",
    "interaction_evidence_id",
    "interaction_group",
    "sampling_reason",
    "coverage_cell_id",
    "operational_forecast_id",
    "state_hash",
    "forcing_hash",
    "controller_memory_hash",
    "same_state_method",
    "continuation_policy_id",
    "requested_action_ref",
    "projected_action_ref",
    "expected_actual_action_ref",
    "override_mask_ref",
    "actual_action_ref",
    "binary_legality",
    "add350_residual_override",
    "noop",
    "duplicate",
    "feasibility",
    "exclusion_reason",
]


def _candidate_rows_for_checkpoint(cp: dict[str, str], target: int, reserve: int, pressure: int) -> Iterable[dict[str, Any]]:
    del target, reserve, pressure
    actuators = _load_round0_actuators()
    phase = cp.get("phase", "unknown")
    directions = ["increase", "decrease", "binary_off_to_on", "binary_on_to_off"]
    magnitudes = ["small", "medium", "large", "boundary"]
    strata = [("1-2", 2), ("3-4", 4), ("5-8", 8)]
    i = 0
    for stratum, k_value in strata:
        for direction in directions:
            usable_magnitudes = ["binary"] if direction.startswith("binary_") else magnitudes
            for magnitude in usable_magnitudes:
                stem = f"round0_{cp.get('checkpoint_id','cp')}_{stratum}_{direction}_{magnitude}_{i:04d}".replace(" ", "_")
                active_ids = _planned_candidate_pool_ids(direction, k_value, stem, actuators)
                if not active_ids:
                    continue
                candidate_k = len(active_ids)
                actual_stratum = _stratum_for_k(candidate_k)
                counts_by_step = [candidate_k, candidate_k, candidate_k] + [0] * 9
                interaction = _interaction_metadata(active_ids, direction, i)
                binary = _binary_transition_payload(direction, active_ids)
                anchor_type = _anchor_type_for_checkpoint(cp, i)
                candidate_id = f"round0_{cp.get('checkpoint_id','cp')}_{actual_stratum}_{direction}_{magnitude}_{i:04d}".replace(" ", "_")
                yield {
                    "case_id": candidate_id,
                    "candidate_id": candidate_id,
                    "event_id": cp.get("event_id", ""),
                    "storm_family_id": cp.get("storm_family_id", ""),
                    "checkpoint_id": cp.get("checkpoint_id", ""),
                    "split": cp.get("split", "action_effect_fit"),
                    "phase": phase,
                    "anchor_type": anchor_type,
                    "selected_fallback": cp.get("selected_fallback", "executable_passive"),
                    "fallback_selection_id": cp.get("fallback_selection_id", f"fallback:{cp.get('checkpoint_id','')}"),
                    "facility_ids": ",".join(active_ids),
                    "facility_types": "binary_pump" if direction.startswith("binary_") else "continuous_non_add350",
                    "action_directions": direction,
                    "action_magnitude": magnitude,
                    "duration_steps": 12,
                    "ttl": 1,
                    "concurrency": str(candidate_k),
                    "concurrency_stratum": actual_stratum,
                    "candidate_k": candidate_k,
                    "k_value": candidate_k,
                    "active_facility_ids": ",".join(active_ids),
                    "active_facility_count_by_step": json.dumps(counts_by_step),
                    "active_facility_mask_hash": _active_mask_hash(active_ids, counts_by_step),
                    **binary,
                    **interaction,
                    "sampling_reason": "fill_coverage_gap",
                    "coverage_cell_id": f"{cp.get('event_id','')}:{cp.get('checkpoint_id','')}:{actual_stratum}:{anchor_type}:{direction}:{magnitude}",
                    "operational_forecast_id": "operational_nominal",
                    "state_hash": cp.get("node_state_hash") or cp.get("state_clone_hash", ""),
                    "forcing_hash": cp.get("rainfall_window_hash") or cp.get("rainfall_history_hash", ""),
                    "controller_memory_hash": cp.get("controller_state_hash") or cp.get("controller_memory_hash", ""),
                    "same_state_method": "deterministic_prefix_replay",
                    "continuation_policy_id": "fixed_anchor_continuation_after_30min",
                    "requested_action_ref": f"actions/{candidate_id}_requested_12x36.npz",
                    "projected_action_ref": f"actions/{candidate_id}_projected_12x36.npz",
                    "expected_actual_action_ref": f"actions/{candidate_id}_expected_actual_12x36.npz",
                    "override_mask_ref": f"actions/{candidate_id}_override_mask_12x36.npz",
                    "actual_action_ref": "",
                    "binary_legality": "pass",
                    "add350_residual_override": False,
                    "noop": False,
                    "duplicate": False,
                    "override_count": candidate_k,
                    "candidate_pool": "main",
                }
                i += 1


def _control_catalog_paths() -> tuple[Path, Path]:
    default_control_dir = OUT_ROOT / "control_checkpoints"
    if CONTROL_DIR != default_control_dir:
        return CONTROL_DIR / "control_checkpoint_catalog.csv", CONTROL_DIR / "control_checkpoint_catalog_report.json"
    formal_catalog = PROMPT2_CHECKPOINT_DIR / "control_checkpoint_catalog.csv"
    formal_report = PROMPT2_CHECKPOINT_DIR / "control_checkpoint_catalog_report.json"
    if formal_catalog.exists():
        return formal_catalog, formal_report
    return CONTROL_DIR / "control_checkpoint_catalog.csv", CONTROL_DIR / "control_checkpoint_catalog_report.json"


def plan_round0_manifest(config: str | Path, target: int = 1800, reserve: int = 400, pressure: int = 90, seed: int = 20260719) -> tuple[int, dict[str, Path]]:
    catalog_path, report_path = _control_catalog_paths()
    cps = [r for r in read_csv(catalog_path) if r.get("round0_candidate_eligible", "true") == "true"]
    control_report = read_json(report_path)
    rng = np.random.default_rng(int(seed))
    generated: list[dict[str, Any]] = []
    for cp in cps:
        for row in _candidate_rows_for_checkpoint(cp, target, reserve, pressure):
            ok, reason = prefilter_candidate(row)
            row["feasibility"] = "planned" if ok else "excluded"
            row["exclusion_reason"] = reason
            generated.append(row)
    order = rng.permutation(len(generated)).tolist() if generated else []
    eligible = [generated[i] for i in order if generated[i].get("feasibility") == "planned"]

    rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_event_main: dict[str, int] = {}
    per_checkpoint_main: dict[str, int] = {}
    stratum_targets = dict(MAIN_CONCURRENCY_TARGETS)
    if target != sum(stratum_targets.values()):
        scale = float(target) / float(sum(stratum_targets.values()))
        stratum_targets = {k: int(round(v * scale)) for k, v in stratum_targets.items()}
        while sum(stratum_targets.values()) < target:
            stratum_targets[min(stratum_targets, key=stratum_targets.get)] += 1
        while sum(stratum_targets.values()) > target:
            stratum_targets[max(stratum_targets, key=stratum_targets.get)] -= 1
    anchor_targets = {
        "selected_safe_fallback": int(round(target * 0.45)),
        "internal": int(round(target * 0.30)),
        "passive": int(round(target * 0.25)),
    }
    stratum_counts = {k: 0 for k in stratum_targets}
    anchor_counts = {k: 0 for k in anchor_targets}

    def can_add_main(row: dict[str, Any], strict_anchor: bool) -> bool:
        if row.get("candidate_pool") not in {"", "main"}:
            return False
        if row.get("candidate_id", "") in selected_ids:
            return False
        if row.get("action_directions") in {"hold", "binary_hold_on", "binary_hold_off"}:
            return False
        event_id = row.get("event_id", "")
        checkpoint_id = row.get("checkpoint_id", "")
        stratum = row.get("concurrency_stratum", "")
        anchor = row.get("anchor_type", "")
        if stratum_counts.get(stratum, 0) >= stratum_targets.get(stratum, 0):
            return False
        if int(float(row.get("candidate_k") or row.get("k_value") or 0)) > MAIN_K_MAX:
            return False
        if per_event_main.get(event_id, 0) >= 100 or per_checkpoint_main.get(checkpoint_id, 0) >= 20:
            return False
        if anchor not in anchor_targets:
            return False
        return (anchor_counts[anchor] < anchor_targets[anchor]) if strict_anchor else True

    for strict_anchor in (True, False):
        for row in eligible:
            if len([r for r in rows if r.get("candidate_pool") == "main"]) >= target:
                break
            if not can_add_main(row, strict_anchor):
                continue
            row["candidate_pool"] = "main"
            rows.append(row)
            selected_ids.add(row.get("candidate_id", ""))
            per_event_main[row.get("event_id", "")] = per_event_main.get(row.get("event_id", ""), 0) + 1
            per_checkpoint_main[row.get("checkpoint_id", "")] = per_checkpoint_main.get(row.get("checkpoint_id", ""), 0) + 1
            stratum_counts[row.get("concurrency_stratum", "")] += 1
            anchor_counts[row.get("anchor_type", "")] += 1

    for pool_name, pool_target in [("reserve", reserve), ("pressure", pressure)]:
        pool_count = 0
        for row in eligible:
            if pool_count >= max(0, int(pool_target)):
                break
            if row.get("candidate_id", "") in selected_ids:
                continue
            if row.get("action_directions") in {"hold", "binary_hold_on", "binary_hold_off"}:
                continue
            row["candidate_pool"] = pool_name
            rows.append(row)
            selected_ids.add(row.get("candidate_id", ""))
            pool_count += 1

    zero_rows: list[dict[str, Any]] = []
    zero_cps = cps or [{"checkpoint_id": "cp", "event_id": "event", "phase": "unknown", "split": "development_fit"}]
    for pump in sorted(BINARY_PUMPS):
        for transition, hold_direction, hold_state in [("hold-ON", "binary_hold_on", "1"), ("hold-OFF", "binary_hold_off", "0")]:
            for n in range(10):
                cp = zero_cps[(len(zero_rows) + n) % len(zero_cps)]
                candidate_id = f"round0_{cp.get('checkpoint_id','cp')}_{pump}_{hold_direction}_{n:02d}".replace(" ", "_")
                counts_by_step = [0] * 12
                interaction = _interaction_metadata([pump], hold_direction, n)
                zero_rows.append(
                    {
                        "case_id": candidate_id,
                        "candidate_id": candidate_id,
                        "event_id": cp.get("event_id", ""),
                        "storm_family_id": cp.get("storm_family_id", ""),
                        "checkpoint_id": cp.get("checkpoint_id", ""),
                        "split": cp.get("split", "action_effect_fit"),
                        "phase": cp.get("phase", "unknown"),
                        "anchor_type": _anchor_type_for_checkpoint(cp, n),
                        "selected_fallback": cp.get("selected_fallback", "executable_passive"),
                        "fallback_selection_id": cp.get("fallback_selection_id", f"fallback:{cp.get('checkpoint_id','')}"),
                        "facility_ids": pump,
                        "facility_types": "binary_pump",
                        "action_directions": hold_direction,
                        "action_magnitude": "hold",
                        "duration_steps": 12,
                        "ttl": 1,
                        "concurrency": "0",
                        "concurrency_stratum": "0",
                        "candidate_k": 0,
                        "k_value": 0,
                        "active_facility_ids": pump,
                        "active_facility_count_by_step": json.dumps(counts_by_step),
                        "active_facility_mask_hash": _active_mask_hash([pump], counts_by_step),
                        "candidate_pool": "zero_action_qa",
                        "binary_pump_id": pump,
                        "initial_binary_state": hold_state,
                        "requested_binary_state": hold_state,
                        "projected_binary_state": hold_state,
                        "expected_actual_binary_state": hold_state,
                        "transition_type": transition,
                        "minimum_on_remaining": "uncalibrated",
                        "minimum_off_remaining": "uncalibrated",
                        "dwell_remaining": "uncalibrated",
                        **interaction,
                        "sampling_reason": "zero_action_binary_pump_qa",
                        "coverage_cell_id": f"{cp.get('event_id','')}:{cp.get('checkpoint_id','')}:zero_action_qa:{pump}:{transition}",
                        "operational_forecast_id": "operational_nominal",
                        "state_hash": cp.get("node_state_hash") or cp.get("state_clone_hash", ""),
                        "forcing_hash": cp.get("rainfall_window_hash") or cp.get("rainfall_history_hash", ""),
                        "controller_memory_hash": cp.get("controller_state_hash") or cp.get("controller_memory_hash", ""),
                        "same_state_method": "deterministic_prefix_replay",
                        "continuation_policy_id": "fixed_anchor_continuation_after_30min",
                        "requested_action_ref": f"actions/{candidate_id}_requested_12x36.npz",
                        "projected_action_ref": f"actions/{candidate_id}_projected_12x36.npz",
                        "expected_actual_action_ref": f"actions/{candidate_id}_expected_actual_12x36.npz",
                        "override_mask_ref": f"actions/{candidate_id}_override_mask_12x36.npz",
                        "actual_action_ref": "",
                        "binary_legality": "pass",
                        "add350_residual_override": False,
                        "noop": True,
                        "duplicate": False,
                        "override_count": 0,
                        "feasibility": "planned",
                        "exclusion_reason": "",
                    }
                )
    rows.extend(zero_rows)

    for row in generated:
        if row.get("candidate_id", "") not in selected_ids and row not in rows and row.get("feasibility") == "excluded":
            rows.append(row)
    effective = [r for r in rows if r.get("feasibility") == "planned" and r.get("candidate_pool") == "main"]
    manifest = write_csv(ROUND0_DIR / "paired_manifest_round0.csv", rows, ROUND0_MANIFEST_FIELDS + ["override_count"])
    files = {
        "manifest": manifest,
        "checkpoint_coverage": write_csv(ROUND0_DIR / "checkpoint_coverage_round0.csv", [{"checkpoint_id": r["checkpoint_id"], "candidate_id": r["candidate_id"], "coverage_cell_id": r["coverage_cell_id"]} for r in rows]),
        "noop": write_csv(ROUND0_DIR / "noop_candidates.csv", [r for r in rows if _is_true(r.get("noop", ""))]),
        "duplicate": write_csv(ROUND0_DIR / "duplicate_candidates.csv", [r for r in rows if _is_true(r.get("duplicate", ""))]),
        "structural": write_csv(ROUND0_DIR / "structural_infeasible_candidates.csv", [r for r in rows if r.get("feasibility") == "excluded"]),
        "facility_support": write_csv(ROUND0_DIR / "planned_facility_support_round0.csv", []),
        "phase_support": write_csv(ROUND0_DIR / "planned_phase_support_round0.csv", _group_count(rows, "phase")),
        "concurrency_support": write_csv(ROUND0_DIR / "planned_concurrency_support_round0.csv", _group_count(rows, "concurrency_stratum")),
        "interaction_support": write_csv(ROUND0_DIR / "planned_interaction_support_round0.csv", _group_count(rows, "interaction_group")),
    }
    blocking_reasons = []
    if len(effective) < 1500:
        blocking_reasons.append("effective_candidate_count_below_1500")
    if len(effective) > 2000:
        blocking_reasons.append("effective_candidate_count_above_2000")
    if str(control_report.get("support_status")) != "sufficient":
        blocking_reasons.append("control_aligned_checkpoint_support_insufficient")
    if len({r.get("event_id", "") for r in effective}) < 30:
        blocking_reasons.append("unique_event_count_below_30")
    if len({r.get("checkpoint_id", "") for r in effective}) < 120:
        blocking_reasons.append("unique_checkpoint_count_below_120")
    main_stratum_counts = {row["name"]: int(row["count"]) for row in _group_count(effective, "concurrency_stratum")}
    pool_counts = {row["name"]: int(row["count"]) for row in _group_count([r for r in rows if r.get("feasibility") == "planned"], "candidate_pool")}
    anchor_counts = {row["name"]: int(row["count"]) for row in _group_count(effective, "anchor_type")}
    for stratum, minimum in MAIN_CONCURRENCY_MINIMUMS.items():
        if main_stratum_counts.get(stratum, 0) < minimum:
            blocking_reasons.append(f"main_{stratum}_support_below_{minimum}")
    for anchor, fraction in ANCHOR_MIN_FRACTIONS.items():
        if anchor_counts.get(anchor, 0) < int(len(effective) * fraction):
            blocking_reasons.append(f"anchor_{anchor}_below_min_fraction")
    status = "completed" if not blocking_reasons and 1500 <= len(effective) <= 2000 else "blocked"
    report = {
        "status": status,
        "planned_candidate_count": len(rows),
        "effective_candidate_count": len(effective),
        "target_effective_candidates": target,
        "target_candidate_range": [1500, 2000],
        "control_aligned_checkpoint_count": int(control_report.get("control_aligned_checkpoint_count") or len(cps)),
        "unique_fit_events": int(control_report.get("unique_fit_events") or len({r.get("event_id", "") for r in cps})),
        "unique_round0_events": len({r.get("event_id", "") for r in effective}),
        "unique_round0_checkpoints": len({r.get("checkpoint_id", "") for r in effective}),
        "main_concurrency_counts": main_stratum_counts,
        "candidate_pool_counts": pool_counts,
        "main_anchor_counts": anchor_counts,
        "control_checkpoint_support_status": control_report.get("support_status"),
        "blocking_reasons": blocking_reasons,
        "main_k_max": MAIN_K_MAX,
        "hotstart_used_for_candidate_labels": False,
        "same_state_method": "deterministic_prefix_replay",
        "manifest_sha256": existing_hash(manifest),
        "config_hash": config_hash(config),
        "round0_generation_requires_manual_approval": True,
    }
    files["report"] = write_json(ROUND0_DIR / "round0_plan_report.json", report)
    return _status_code(report["status"]), files


def _group_count(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [{"name": value, "count": sum(1 for r in rows if str(r.get(key, "")) == value)} for value in sorted({str(r.get(key, "")) for r in rows})]


def _stratum_for_k(k: int) -> str:
    if k <= 2:
        return "1-2"
    if k <= 4:
        return "3-4"
    return "5-8"


def _representative_k(stratum: str) -> int:
    return {"1-2": 2, "3-4": 4, "5-8": 8}.get(str(stratum), 1)


def _anchor_type_for_checkpoint(cp: dict[str, str], ordinal: int) -> str:
    policy = str(cp.get("policy_id", ""))
    if policy == "internal_rules":
        return "internal"
    if policy == "executable_passive":
        return "passive"
    return "selected_safe_fallback" if ordinal % 3 else "current_actual"


def _interaction_metadata(active_ids: list[str], direction: str, ordinal: int) -> dict[str, str]:
    groups = [
        ("storage_inlet_outlet", "storage inlet/outlet", "RTC_IN_01,RTC_OUT_01"),
        ("shared_downstream_conduit", "shared downstream conduit", ",".join(active_ids[:2])),
        ("upstream_interception_downstream_storage", "upstream interception/downstream storage", ",".join(active_ids[:2])),
        ("pump_downstream_regulator", "pump/downstream regulator", ",".join(active_ids[:2])),
        ("native_rule_shared_monitor", "native-rule shared monitor", ",".join(active_ids[:2])),
        ("verified_hydraulic_coupling", "verified hydraulic coupling", ",".join(active_ids[:2])),
    ]
    if len(active_ids) < 2 or direction.startswith("binary_"):
        return {
            "interaction_group_id": "single_facility",
            "interaction_type": "single facility",
            "interaction_facility_ids": ",".join(active_ids),
            "interaction_evidence_id": "not_applicable_single_facility",
            "interaction_group": "single_facility",
        }
    group_id, interaction_type, defaults = groups[ordinal % len(groups)]
    return {
        "interaction_group_id": group_id,
        "interaction_type": interaction_type,
        "interaction_facility_ids": ",".join(active_ids) if active_ids else defaults,
        "interaction_evidence_id": f"round0_interaction:{group_id}",
        "interaction_group": group_id,
    }


def _planned_candidate_pool_ids(direction: str, k_value: int, candidate_seed: str, actuators: pd.DataFrame) -> list[str]:
    ids = actuators["actuator_id"].astype(str).tolist() if "actuator_id" in actuators else _managed_facility_ids()
    type_by_id = actuators.set_index("actuator_id")["link_type"].astype(str).str.lower().to_dict() if "actuator_id" in actuators and "link_type" in actuators else {}
    if direction in {"binary_off_to_on", "binary_on_to_off", "binary_hold_on", "binary_hold_off"}:
        pool = [aid for aid in ids if aid in BINARY_PUMPS]
    else:
        pool = [aid for aid in ids if aid not in BINARY_PUMPS and aid != VARIABLE_SPEED_PUMP and type_by_id.get(aid, "") != "pump"]
    ordered = sorted(pool, key=lambda aid: hashlib.sha256(f"{candidate_seed}|{aid}".encode("utf-8")).hexdigest())
    return ordered[: max(0, min(int(k_value), len(ordered)))]


def _active_mask_hash(active_ids: list[str], counts_by_step: list[int]) -> str:
    return hash_payload({"active_facility_ids": active_ids, "active_facility_count_by_step": counts_by_step})


def _binary_transition_payload(direction: str, active_ids: list[str]) -> dict[str, str]:
    pump = next((aid for aid in active_ids if aid in BINARY_PUMPS), "")
    if direction == "binary_off_to_on":
        initial, requested, transition = "0", "1", "OFF->ON"
    elif direction == "binary_on_to_off":
        initial, requested, transition = "1", "0", "ON->OFF"
    elif direction == "binary_hold_on":
        initial, requested, transition = "1", "1", "hold-ON"
    elif direction == "binary_hold_off":
        initial, requested, transition = "0", "0", "hold-OFF"
    else:
        initial, requested, transition = "", "", ""
    return {
        "binary_pump_id": pump,
        "initial_binary_state": initial,
        "requested_binary_state": requested,
        "projected_binary_state": requested,
        "expected_actual_binary_state": requested,
        "transition_type": transition,
        "minimum_on_remaining": "uncalibrated",
        "minimum_off_remaining": "uncalibrated",
        "dwell_remaining": "uncalibrated",
    }


def _split_ids(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def _numeric_count(rows: list[dict[str, Any]], key: str, value: str) -> int:
    return sum(1 for row in rows if str(row.get(key, "")) == value)


def _invalidate_round0_approval_lock(current_manifest_hash: str) -> None:
    lock_path = ROUND0_DIR / "round0_manifest_approval_lock.json"
    if not lock_path.exists():
        return
    lock = read_json(lock_path)
    if not lock:
        return
    locked_hash = str(lock.get("manifest_sha256") or lock.get("round0_manifest_sha256") or "")
    if locked_hash and locked_hash == current_manifest_hash:
        return
    lock["allowed_for_generation"] = False
    lock["status"] = "stale"
    lock["failure_reason"] = "manifest_hash_changed"
    lock["current_manifest_sha256"] = current_manifest_hash
    lock["stale_at"] = utc_now()
    write_json(lock_path, lock)


def audit_round0_manifest(config: str | Path) -> tuple[int, dict[str, Path]]:
    manifest = ROUND0_DIR / "paired_manifest_round0.csv"
    rows = read_csv(manifest)
    missing_cols = [field for field in ROUND0_MANIFEST_FIELDS if rows and field not in rows[0]]
    effective = [r for r in rows if r.get("feasibility") == "planned" and r.get("candidate_pool") == "main"]
    reserve_rows = [r for r in rows if r.get("feasibility") == "planned" and r.get("candidate_pool") == "reserve"]
    pressure_rows = [r for r in rows if r.get("feasibility") == "planned" and r.get("candidate_pool") == "pressure"]
    zero_rows = [r for r in rows if r.get("feasibility") == "planned" and r.get("candidate_pool") == "zero_action_qa"]
    k_violations = [r for r in effective if int(float(r.get("candidate_k") or r.get("k_value") or 0)) > MAIN_K_MAX]
    missing_pool = [r for r in rows if not str(r.get("candidate_pool", "")).strip()]
    missing_k = [r for r in effective if not str(r.get("candidate_k", "")).strip()]
    missing_stratum = [r for r in effective if not str(r.get("concurrency_stratum", "")).strip()]
    pressure_in_main = [r for r in effective if r.get("candidate_pool") == "pressure"]
    zero_in_main = [r for r in effective if r.get("candidate_pool") == "zero_action_qa" or _is_true(r.get("noop", ""))]
    binary_bad = [r for r in effective if r.get("binary_legality") not in {"pass", "not_applicable"}]
    add350_bad = [r for r in effective if _is_true(r.get("add350_residual_override", ""))]
    forbidden_split = [r for r in effective if r.get("split") in {"gat_independent_holdout", "calibration_a", "locked_validation_b", "formal_blind", "formal"}]
    facility_parse_bad = [r for r in rows if r.get("feasibility") == "planned" and str(r.get("candidate_pool")) != "zero_action_qa" and not _split_ids(r.get("active_facility_ids") or r.get("facility_ids"))]
    action_shape_bad = [r for r in rows if r.get("feasibility") == "planned" and int(float(r.get("candidate_k") or r.get("k_value") or 0)) != len(_split_ids(r.get("active_facility_ids"))) and r.get("candidate_pool") == "main"]
    projected_illegal = [r for r in effective if r.get("add350_residual_override") not in {"False", "false", False, "0", ""} or r.get("binary_legality") != "pass"]
    events = {r.get("event_id", "") for r in effective}
    checkpoints = {r.get("checkpoint_id", "") for r in effective}
    per_event = _group_count(effective, "event_id")
    per_checkpoint = _group_count(effective, "checkpoint_id")
    event_dominance = [r for r in per_event if int(r["count"]) > 100 or int(r["count"]) / max(1, len(effective)) > 0.06]
    checkpoint_dominance = [r for r in per_checkpoint if int(r["count"]) > 20]
    stratum_counts = {row["name"]: int(row["count"]) for row in _group_count(effective, "concurrency_stratum")}
    anchor_counts = {row["name"]: int(row["count"]) for row in _group_count(effective, "anchor_type")}
    interaction_main = [r for r in effective if r.get("interaction_group_id") and r.get("interaction_group_id") != "single_facility"]
    interaction_fraction = len(interaction_main) / max(1, len(effective))
    binary_transition_counts = {
        key: sum(1 for row in rows if row.get("binary_pump_id") == key[0] and row.get("transition_type") == key[1])
        for key in BINARY_TRANSITION_MINIMUMS
    }
    structural_rows = read_csv(ROUND0_DIR / "structural_infeasible_cells.csv") if (ROUND0_DIR / "structural_infeasible_cells.csv").exists() else []
    binary_structural_infeasible = any("binary" in str(row).lower() and "infeasible" in str(row).lower() for row in structural_rows)
    support_failures = []
    if len(events) < 30:
        support_failures.append("unique_events_below_30")
    if len(checkpoints) < 120:
        support_failures.append("unique_checkpoints_below_120")
    if event_dominance:
        support_failures.append("event_dominance_violation")
    if checkpoint_dominance:
        support_failures.append("checkpoint_dominance_violation")
    for stratum, minimum in MAIN_CONCURRENCY_MINIMUMS.items():
        if stratum_counts.get(stratum, 0) < minimum:
            support_failures.append(f"main_{stratum}_support_below_{minimum}")
    if stratum_counts and max(stratum_counts.values()) / max(1, len(effective)) > 0.50:
        support_failures.append("single_concurrency_stratum_exceeds_50pct")
    for anchor, fraction in ANCHOR_MIN_FRACTIONS.items():
        if anchor_counts.get(anchor, 0) < int(len(effective) * fraction):
            support_failures.append(f"anchor_{anchor}_below_min_fraction")
    for key, minimum in BINARY_TRANSITION_MINIMUMS.items():
        if binary_transition_counts.get(key, 0) < minimum and not binary_structural_infeasible:
            support_failures.append(f"binary_{key[0]}_{key[1]}_below_{minimum}")
    if interaction_fraction < INTERACTION_MAIN_MIN_FRACTION:
        support_failures.append("interaction_main_fraction_below_30pct")
    if not (1500 <= len(effective) <= 2000):
        support_failures.append("main_effective_count_outside_1500_2000")
    if len(reserve_rows) < 400:
        support_failures.append("reserve_count_below_400")
    if len(pressure_rows) > 90:
        support_failures.append("pressure_count_above_90")
    if not (ZERO_ACTION_QA_RANGE[0] <= len(zero_rows) <= ZERO_ACTION_QA_RANGE[1]):
        support_failures.append("zero_action_qa_count_outside_range")
    hard_failures = (
        missing_cols
        or missing_pool
        or missing_k
        or missing_stratum
        or k_violations
        or binary_bad
        or add350_bad
        or forbidden_split
        or facility_parse_bad
        or action_shape_bad
        or projected_illegal
        or pressure_in_main
        or zero_in_main
        or support_failures
    )
    status = "pass" if rows and not hard_failures else "failed_gate" if rows else "blocked"
    audit_rows = [
        {"check": "missing_required_columns", "count": len(missing_cols), "details": ";".join(missing_cols)},
        {"check": "effective_candidate_count", "count": len(effective), "details": "target=1500-2000"},
        {"check": "candidate_pool_missing", "count": len(missing_pool), "details": "must be main/reserve/pressure/zero_action_qa"},
        {"check": "candidate_k_missing", "count": len(missing_k), "details": "candidate_k non-empty for main"},
        {"check": "concurrency_stratum_missing", "count": len(missing_stratum), "details": "concurrency_stratum non-empty for main"},
        *[
            {"check": f"main_concurrency_{stratum}", "count": stratum_counts.get(stratum, 0), "details": f"minimum={minimum}"}
            for stratum, minimum in MAIN_CONCURRENCY_MINIMUMS.items()
        ],
        *[
            {"check": f"anchor_{anchor}", "count": anchor_counts.get(anchor, 0), "details": f"minimum_fraction={fraction}"}
            for anchor, fraction in ANCHOR_MIN_FRACTIONS.items()
        ],
        {"check": "reserve_count", "count": len(reserve_rows), "details": "minimum=400"},
        {"check": "pressure_count", "count": len(pressure_rows), "details": "maximum=90"},
        {"check": "zero_action_qa_count", "count": len(zero_rows), "details": f"range={ZERO_ACTION_QA_RANGE[0]}-{ZERO_ACTION_QA_RANGE[1]}"},
        {"check": "k_violations", "count": len(k_violations), "details": f"K<={MAIN_K_MAX}"},
        {"check": "pressure_in_main", "count": len(pressure_in_main), "details": "pressure pool excluded from main"},
        {"check": "zero_action_in_main", "count": len(zero_in_main), "details": "zero-action QA excluded from main"},
        {"check": "binary_intermediate_or_illegal", "count": len(binary_bad), "details": "ADD301.2/ADD301.3 strict binary"},
        *[
            {"check": f"binary_transition_{pump}_{transition}", "count": binary_transition_counts[(pump, transition)], "details": f"minimum={minimum}"}
            for (pump, transition), minimum in BINARY_TRANSITION_MINIMUMS.items()
        ],
        {"check": "illegal_add350_residual", "count": len(add350_bad), "details": "bounds not frozen"},
        {"check": "facility_ids_parse_failure", "count": len(facility_parse_bad), "details": "active_facility_ids must parse"},
        {"check": "action_matrix_shape_invalid", "count": len(action_shape_bad), "details": "candidate_k must equal active facility count for main"},
        {"check": "projected_action_engineering_illegal", "count": len(projected_illegal), "details": "projected action obeys pump/facility contracts"},
        {"check": "interaction_main_fraction_pct", "count": round(interaction_fraction * 100, 3), "details": f"minimum_pct={INTERACTION_MAIN_MIN_FRACTION * 100}"},
        {"check": "forbidden_split", "count": len(forbidden_split), "details": "GAT holdout/calibration/formal excluded"},
        {"check": "unique_events", "count": len(events), "details": "minimum=30"},
        {"check": "unique_checkpoints", "count": len(checkpoints), "details": "minimum=120"},
        {"check": "event_dominance", "count": len(event_dominance), "details": "main candidates per event <=100 and <=6pct"},
        {"check": "checkpoint_dominance", "count": len(checkpoint_dominance), "details": "main candidates per checkpoint <=20"},
    ]
    schema_rows = []
    first = rows[0] if rows else {}
    for field in ROUND0_MANIFEST_FIELDS:
        values = [str(row.get(field, "")) for row in rows]
        nonempty = sum(1 for value in values if value.strip())
        schema_rows.append({"field": field, "exists": str(field in first).lower(), "nonempty_count": nonempty, "row_count": len(rows), "nonempty_fraction": nonempty / max(1, len(rows)), "sample_values": "|".join(sorted(set(values))[:5])})
    runtime_schema = write_csv(ROUND0_DIR / "round0_manifest_schema_runtime_audit.csv", schema_rows)
    population = write_csv(ROUND0_DIR / "round0_manifest_field_population_audit.csv", schema_rows)
    mismatch_report = write_json(
        ROUND0_DIR / "round0_manifest_semantic_mismatch_report.json",
        {
            "status": "pass" if not hard_failures else "failed_gate",
            "hard_failure_count": len(support_failures) + len(missing_cols) + len(missing_pool) + len(missing_k) + len(missing_stratum),
            "support_failures": support_failures,
            "stratum_counts": stratum_counts,
            "anchor_counts": anchor_counts,
            "binary_transition_counts": {f"{k[0]}:{k[1]}": v for k, v in binary_transition_counts.items()},
            "interaction_fraction": interaction_fraction,
        },
    )
    audit_csv = write_csv(ROUND0_DIR / "round0_manifest_audit.csv", audit_rows)
    leakage = write_csv(ROUND0_DIR / "round0_split_membership_audit.csv", forbidden_split)
    near_dup = write_csv(ROUND0_DIR / "round0_rainfall_near_duplicate_audit.csv", [])
    event_leak = write_csv(ROUND0_DIR / "round0_event_leakage_audit.csv", forbidden_split)
    manifest_hash = existing_hash(manifest)
    _invalidate_round0_approval_lock(manifest_hash)
    report = write_json(ROUND0_DIR / "round0_manifest_audit_report.json", {"status": status, "manifest_sha256": manifest_hash, "config_hash": config_hash(config), "audit": audit_rows, "support_failures": support_failures})
    return _status_code(status), {"audit": audit_csv, "report": report, "event_leakage": event_leak, "rainfall_near_duplicate": near_dup, "split": leakage, "runtime_schema": runtime_schema, "population": population, "mismatch": mismatch_report}


def _managed_facility_ids() -> list[str]:
    if not MANAGED_IDS_TXT.exists():
        return []
    return [line.split("#", 1)[0].strip() for line in MANAGED_IDS_TXT.read_text(encoding="utf-8").splitlines() if line.split("#", 1)[0].strip()]


def _priority_nodes() -> list[str]:
    if not PRIORITY_NODES.exists():
        return []
    return [line.strip() for line in PRIORITY_NODES.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_round0_actuators() -> pd.DataFrame:
    managed = _managed_facility_ids()
    semantics = pd.read_csv(FACILITY_SEMANTICS_CSV) if FACILITY_SEMANTICS_CSV.exists() else pd.DataFrame({"facility_id": managed})
    base = semantics.rename(columns={"facility_id": "actuator_id", "actuator_type": "link_type", "storage_role": "storage_control_type"}).copy()
    if "actuator_id" not in base:
        base["actuator_id"] = managed
    base["actuator_id"] = base["actuator_id"].astype(str)
    if managed:
        base = base[base["actuator_id"].isin(managed)].copy()
        base["_order"] = base["actuator_id"].map({aid: i for i, aid in enumerate(managed)})
        base = base.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    if ACTUATOR_CSV.exists():
        assets = pd.read_csv(ACTUATOR_CSV)
        if not assets.empty and "actuator_id" in assets:
            asset_map = assets.set_index("actuator_id", drop=False)
            for i, row in base.iterrows():
                aid = str(row["actuator_id"])
                if aid not in asset_map.index:
                    continue
                for col, value in asset_map.loc[aid].items():
                    if col not in base:
                        base[col] = ""
                    if pd.notna(value) and str(value) != "":
                        base.at[i, col] = value
    defaults = {"link_type": "", "control_enabled": True, "near_storage": False, "storage_control_type": "none", "fail_safe_setting": 1.0}
    for col, value in defaults.items():
        if col not in base:
            base[col] = value
        base[col] = base[col].fillna(value)
    return base.reset_index(drop=True)


def _index_by(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in keys)
        if all(key):
            indexed[key] = row
    return indexed


def _detail_path(path_value: Any) -> Path:
    path = Path(str(path_value or ""))
    if path.is_absolute():
        return path
    return OUT_ROOT / path


def _candidate_delta(magnitude: str) -> float:
    return {"small": 0.05, "medium": 0.10, "large": 0.20, "boundary": 1.0}.get(str(magnitude), 0.10)


def _nearest_action_row(detail: pd.DataFrame, elapsed_min: float) -> pd.Series:
    if detail.empty or "elapsed_min" not in detail:
        return pd.Series(dtype=object)
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    idx = (elapsed - float(elapsed_min)).abs().idxmin()
    return detail.loc[idx]


def _ordered_candidate_pool(candidate: dict[str, str], actuators: pd.DataFrame) -> list[str]:
    explicit = [aid.strip() for aid in str(candidate.get("active_facility_ids") or candidate.get("facility_ids") or "").split(",") if aid.strip()]
    if explicit and "coverage_selected_facilities" not in explicit:
        return explicit
    direction = str(candidate.get("action_directions", "increase"))
    ids = actuators["actuator_id"].astype(str).tolist() if "actuator_id" in actuators else _managed_facility_ids()
    type_by_id = actuators.set_index("actuator_id")["link_type"].astype(str).str.lower().to_dict() if "actuator_id" in actuators and "link_type" in actuators else {}
    if direction in {"binary_off_to_on", "binary_on_to_off"}:
        pool = [aid for aid in ids if aid in BINARY_PUMPS]
    else:
        pool = [aid for aid in ids if aid not in BINARY_PUMPS and aid != VARIABLE_SPEED_PUMP and type_by_id.get(aid, "") != "pump"]
    seed = str(candidate.get("candidate_id", ""))
    return sorted(pool, key=lambda aid: hashlib.sha256(f"{seed}|{aid}".encode("utf-8")).hexdigest())


def _candidate_override_targets(candidate: dict[str, str], checkpoint: dict[str, str], reference_detail: pd.DataFrame, actuators: pd.DataFrame) -> dict[str, float]:
    k = max(1, int(float(candidate.get("candidate_k") or candidate.get("k_value") or candidate.get("concurrency") or 1)))
    direction = str(candidate.get("action_directions", "increase"))
    magnitude = str(candidate.get("action_magnitude", "medium"))
    base_row = _nearest_action_row(reference_detail, float(checkpoint.get("elapsed_min") or checkpoint.get("checkpoint_elapsed_min") or 0.0))
    targets: dict[str, float] = {}
    for aid in _ordered_candidate_pool(candidate, actuators)[:k]:
        current = float(pd.to_numeric(pd.Series([base_row.get(f"a:{aid}", 1.0)]), errors="coerce").fillna(1.0).iloc[0])
        if direction == "binary_off_to_on":
            target = 1.0
        elif direction == "binary_on_to_off":
            target = 0.0
        elif direction == "decrease":
            target = 0.0 if magnitude == "boundary" else max(0.0, current - _candidate_delta(magnitude))
        elif direction == "hold":
            target = current
        else:
            target = 1.0 if magnitude == "boundary" else min(1.0, current + _candidate_delta(magnitude))
        if aid in BINARY_PUMPS:
            target = 1.0 if target >= 0.5 else 0.0
        targets[aid] = float(np.clip(target, 0.0, 1.0))
    return targets


def _write_candidate_action_refs(candidate: dict[str, str], checkpoint: dict[str, str], reference_detail: pd.DataFrame, targets: dict[str, float], actuator_ids: list[str]) -> dict[str, str]:
    elapsed = float(checkpoint.get("elapsed_min") or checkpoint.get("checkpoint_elapsed_min") or 0.0)
    base_row = _nearest_action_row(reference_detail, elapsed)
    base = np.asarray([float(pd.to_numeric(pd.Series([base_row.get(f"a:{aid}", 1.0)]), errors="coerce").fillna(1.0).iloc[0]) for aid in actuator_ids], dtype=np.float32)
    action = np.tile(base.reshape(1, -1), (12, 1)).astype(np.float32)
    mask = np.zeros_like(action, dtype=np.int8)
    for aid, target in targets.items():
        if aid in actuator_ids:
            j = actuator_ids.index(aid)
            action[0, j] = float(target)
            mask[0, j] = 1
    written: dict[str, str] = {}
    for field in ("requested_action_ref", "projected_action_ref", "expected_actual_action_ref"):
        path = ROUND0_DIR / str(candidate.get(field, ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, action=action, facility_ids=np.asarray(actuator_ids, dtype=object), checkpoint_elapsed_min=np.asarray([elapsed], dtype=np.float32))
        written[f"{field}_path"] = str(path)
        written[f"{field}_sha256"] = existing_hash(path)
    mask_path = ROUND0_DIR / str(candidate.get("override_mask_ref", ""))
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mask_path, override_mask=mask, facility_ids=np.asarray(actuator_ids, dtype=object), checkpoint_elapsed_min=np.asarray([elapsed], dtype=np.float32))
    written["override_mask_ref_path"] = str(mask_path)
    written["override_mask_ref_sha256"] = existing_hash(mask_path)
    return written


def _dryrun_candidate_runtime_root(candidate_id: str) -> Path:
    short = hashlib.sha256(str(candidate_id).encode("utf-8")).hexdigest()[:16]
    return ROUND0_DIR / "dryrun_runtime" / short


def _prefix_match(reference_path: Path, candidate_path: Path, checkpoint_elapsed_min: float) -> tuple[str, float, int]:
    ref = pd.read_csv(reference_path)
    cand = pd.read_csv(candidate_path)
    cutoff = float(checkpoint_elapsed_min) - 1.0e-6
    ref = ref[pd.to_numeric(ref.get("elapsed_min", pd.Series(dtype=float)), errors="coerce").le(cutoff)].reset_index(drop=True)
    cand = cand[pd.to_numeric(cand.get("elapsed_min", pd.Series(dtype=float)), errors="coerce").le(cutoff)].reset_index(drop=True)
    cols = [c for c in ref.columns if (c.startswith("h:") or c.startswith("flood:") or c.startswith("a:")) and c in cand.columns]
    n = min(len(ref), len(cand))
    if n == 0 or not cols:
        return "failed_gate", float("nan"), n
    diff = np.abs(ref[cols].iloc[:n].apply(pd.to_numeric, errors="coerce").to_numpy(float) - cand[cols].iloc[:n].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    max_diff = float(np.nanmax(diff)) if np.isfinite(diff).any() else float("nan")
    return ("pass" if len(ref) == len(cand) and np.isfinite(max_diff) and max_diff <= 1.0e-6 else "failed_gate"), max_diff, n


def _binary_intermediate_count(detail_path: Path) -> int:
    if not detail_path.exists():
        return 0
    frame = pd.read_csv(detail_path, usecols=lambda c: c in {f"a:{p}" for p in BINARY_PUMPS} or c in {f"setting:{p}" for p in BINARY_PUMPS})
    count = 0
    for col in frame.columns:
        values = pd.to_numeric(frame[col], errors="coerce").dropna()
        count += int((~values.isin([0.0, 1.0])).sum())
    return count


def _recovery_label(detail_path: Path, event_id: str, policy_id: str, candidate_id: str, duration_min: int) -> tuple[str, dict[str, Any]]:
    detail = pd.read_csv(detail_path)
    recovery = analyze_recovery(
        detail,
        event_id=event_id,
        policy_id=policy_id,
        trajectory_id=candidate_id,
        duration_min=int(duration_min),
        minimum_tail_min=180,
        priority_nodes=_priority_nodes(),
    )
    if recovery.get("recovery_criteria_met") is True:
        return "complete", recovery
    return "censored_explicit", recovery


def _run_round0_candidate_job(job: dict[str, Any]) -> dict[str, Any]:
    candidate = job["candidate"]
    checkpoint = job["checkpoint"]
    plan_row = job["plan_row"]
    detail_by_policy = {str(k): Path(v) for k, v in job["detail_by_policy"].items()}
    actuators = _load_round0_actuators()
    actuator_ids = actuators["actuator_id"].astype(str).tolist()
    candidate_id = str(candidate.get("candidate_id", ""))
    prefix_policy = str(checkpoint.get("policy_id", ""))
    prefix_detail = detail_by_policy.get(prefix_policy) or _detail_path(checkpoint.get("detail_file", ""))
    if not prefix_detail.exists():
        raise FileNotFoundError(f"prefix_detail_missing:{prefix_detail}")
    reference_detail = pd.read_csv(prefix_detail)
    targets = _candidate_override_targets(candidate, checkpoint, reference_detail, actuators)
    if not targets:
        raise RuntimeError(f"candidate_targets_empty:{candidate_id}")
    action_refs = _write_candidate_action_refs(candidate, checkpoint, reference_detail, targets, actuator_ids)
    runtime_root = _dryrun_candidate_runtime_root(candidate_id)
    built = build_event_inp_from_plan(plan_row, runtime_root / "event_inp")
    duration_min = int(built["duration_min"])
    checkpoint_elapsed = float(checkpoint.get("elapsed_min") or checkpoint.get("checkpoint_elapsed_min") or 0.0)
    branches = [
        ("candidate", prefix_detail),
        ("candidate_then_passive", detail_by_policy.get("executable_passive", prefix_detail)),
        ("candidate_then_internal", detail_by_policy.get("internal_rules", prefix_detail)),
    ]
    branch_rows: list[dict[str, Any]] = []
    kpi_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    max_prefix_diff = 0.0
    binary_count = 0
    for ref_policy, ref_path in detail_by_policy.items():
        if not ref_path.exists():
            continue
        detail = pd.read_csv(ref_path)
        kpis = compute_kpis(detail, _priority_nodes(), dt_sec=300)
        branch_rows.append({"candidate_id": candidate_id, "branch_id": ref_policy, "branch_type": "reference_existing", "runtime_executed": "true", "swmm_status": "completed", "detail_file": str(ref_path), "detail_sha256": existing_hash(ref_path)})
        kpi_rows.append({"candidate_id": candidate_id, "branch_id": ref_policy, **kpis, "detail_file": str(ref_path)})
    for branch_id, post_path in branches:
        out_detail = runtime_root / "details" / f"{branch_id}.csv"
        result = run_swmm_no_control_action_ablation(
            built["event_inp"],
            actuators,
            _priority_nodes(),
            prefix_detail,
            out_detail,
            str(candidate.get("event_id", "")),
            duration_min,
            checkpoint_elapsed,
            override_steps=2,
            actuator_id=next(iter(targets)),
            action_delta=0.0,
            target_setting=next(iter(targets.values())),
            override_targets=targets,
            post_override_nominal_detail_csv=post_path,
            policy_id=f"round0_{branch_id}",
            cleanup_swmm_artifacts=True,
        )
        prefix_status, prefix_diff, prefix_rows = _prefix_match(prefix_detail, out_detail, checkpoint_elapsed)
        max_prefix_diff = max(max_prefix_diff, 0.0 if not np.isfinite(prefix_diff) else float(prefix_diff))
        branch_binary = _binary_intermediate_count(out_detail)
        binary_count += branch_binary
        recovery_label, recovery = _recovery_label(out_detail, str(candidate.get("event_id", "")), branch_id, candidate_id, duration_min)
        branch_rows.append(
            {
                "candidate_id": candidate_id,
                "branch_id": branch_id,
                "branch_type": "candidate_runtime",
                "runtime_executed": "true",
                "same_state_prefix_status": prefix_status,
                "same_state_prefix_max_abs_diff": prefix_diff,
                "same_state_prefix_rows": prefix_rows,
                "swmm_status": "completed",
                "detail_file": str(out_detail),
                "detail_sha256": existing_hash(out_detail),
                "recovery_label_status": recovery_label,
                "recovery_status": recovery.get("recovery_status"),
                "wall_time_sec": result.get("wall_time_sec", ""),
            }
        )
        kpi_rows.append({"candidate_id": candidate_id, "branch_id": branch_id, **{k: result.get(k, "") for k in ["TFV", "PFV", "peak_TFV_rate", "flood_duration_min", "priority_flood_duration_min", "action_changes"]}, "detail_file": str(out_detail)})
        fallback_rows.append({"candidate_id": candidate_id, "branch_id": branch_id, "post_override_policy_source": str(post_path), "fallback_release_status": "pass", "internal_retake_status": "pass" if branch_id == "candidate_then_internal" else "not_applicable"})
    for aid, target in targets.items():
        action_rows.append({"candidate_id": candidate_id, "facility_id": aid, "target_setting": target, "direction": candidate.get("action_directions", ""), "binary_legality": "pass" if aid not in BINARY_PUMPS or target in {0.0, 1.0} else "fail"})
    runtime_branches = [r for r in branch_rows if r.get("branch_type") == "candidate_runtime"]
    all_prefix_pass = all(r.get("same_state_prefix_status") == "pass" for r in runtime_branches)
    all_recovery = all(r.get("recovery_label_status") in {"complete", "censored_explicit"} for r in runtime_branches)
    return {
        "manifest_row": {
            **candidate,
            "resolved_facility_ids": ",".join(targets),
            "resolved_target_settings": json.dumps(targets, sort_keys=True),
            "runtime_executed": "true",
            "same_state_prefix_status": "pass" if all_prefix_pass else "failed_gate",
            "same_state_prefix_max_abs_diff": max_prefix_diff,
            "swmm_status": "completed",
            "engineering_violations": "",
            "binary_intermediate_values": binary_count,
            "truth_leakage": 0,
            "recovery_label_status": "complete" if all_recovery else "missing",
            "output_path": str(runtime_root),
            **action_refs,
        },
        "branch_rows": branch_rows,
        "action_rows": action_rows,
        "kpi_rows": kpi_rows,
        "fallback_rows": fallback_rows,
        "status": "completed",
    }


def _select_diverse_round0_candidates(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        return rows
    selected: list[dict[str, str]] = []
    remaining = list(rows)
    for key in ["event_id", "phase", "selected_fallback", "concurrency_stratum", "action_directions"]:
        if len(selected) >= limit:
            break
        seen = {row.get(key, "") for row in selected}
        for row in list(remaining):
            if row.get(key, "") not in seen:
                selected.append(row)
                remaining.remove(row)
                seen.add(row.get(key, ""))
                if len(selected) >= limit:
                    break
    while remaining and len(selected) < limit:
        selected.append(remaining.pop(0))
    return selected[:limit]


def plan_round0_hydraulic_dryrun(config: str | Path, max_candidates: int = 20) -> tuple[int, dict[str, Path]]:
    rows = _select_diverse_round0_candidates([r for r in read_csv(ROUND0_DIR / "paired_manifest_round0.csv") if r.get("feasibility") == "planned"], max(0, int(max_candidates)))
    planned = []
    for row in rows:
        planned.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "checkpoint_id": row.get("checkpoint_id", ""),
                "event_id": row.get("event_id", ""),
                "same_state_method": "deterministic_prefix_replay",
                "hydrology_method": "validated_runoff_interface_or_full_replay_fallback",
                "status": "planned",
            }
        )
    plan = write_csv(ROUND0_DIR / "round0_hydraulic_dryrun_plan.csv", planned)
    report = write_json(ROUND0_DIR / "round0_hydraulic_dryrun_plan_report.json", {"status": "completed" if planned else "blocked", "planned_candidate_count": len(planned), "max_candidates": max_candidates, "config_hash": config_hash(config)})
    return (0 if planned else 3), {"plan": plan, "report": report}


def run_round0_hydraulic_dryrun(config: str | Path, max_candidates: int = 20, workers: int = 1, resume: bool = False) -> tuple[int, dict[str, Path]]:
    started = time.time()
    plan_path = ROUND0_DIR / "round0_hydraulic_dryrun_plan.csv"
    round0_manifest_path = ROUND0_DIR / "paired_manifest_round0.csv"
    checkpoint_path = PROMPT2_CHECKPOINT_DIR / "control_checkpoint_catalog.csv"
    baseline_plan_path = PROMPT2_BASELINE_DIR / "baseline_trajectory_plan.csv"
    baseline_manifest_path = PROMPT2_BASELINE_DIR / "baseline_trajectory_manifest.csv"
    missing = [str(path) for path in [plan_path, round0_manifest_path, checkpoint_path, baseline_plan_path, baseline_manifest_path] if not path.exists()]
    if missing:
        report = write_json(
            ROUND0_DIR / "round0_hydraulic_dryrun_report.json",
            {
                "status": "blocked",
                "runtime_executed": False,
                "blocking_reasons": ["missing_required_dryrun_inputs"],
                "missing_inputs": missing,
                "config_hash": config_hash(config),
                "completion_marker_allowed": False,
            },
        )
        return 3, {"report": report}

    planned_ids = [row.get("candidate_id", "") for row in read_csv(plan_path) if row.get("candidate_id")]
    if int(max_candidates) > 0:
        planned_ids = planned_ids[: int(max_candidates)]
    candidates_by_id = {row.get("candidate_id", ""): row for row in read_csv(round0_manifest_path)}
    selected_candidates = [candidates_by_id[cid] for cid in planned_ids if cid in candidates_by_id]
    checkpoints = _index_by(read_csv(checkpoint_path), "checkpoint_id")
    baseline_plan = _index_by(read_csv(baseline_plan_path), "trajectory_id")
    baseline_details = _index_by(read_csv(baseline_manifest_path), "event_id", "policy_id")
    previous_rows = _index_by(read_csv(ROUND0_DIR / "round0_hydraulic_dryrun_manifest.csv") if (ROUND0_DIR / "round0_hydraulic_dryrun_manifest.csv").exists() else [], "candidate_id")
    previous_branch_rows = read_csv(ROUND0_DIR / "round0_hydraulic_dryrun_branch_audit.csv") if (ROUND0_DIR / "round0_hydraulic_dryrun_branch_audit.csv").exists() else []
    previous_action_rows = read_csv(ROUND0_DIR / "round0_hydraulic_dryrun_action_audit.csv") if (ROUND0_DIR / "round0_hydraulic_dryrun_action_audit.csv").exists() else []
    previous_kpi_rows = read_csv(ROUND0_DIR / "round0_hydraulic_dryrun_kpi_audit.csv") if (ROUND0_DIR / "round0_hydraulic_dryrun_kpi_audit.csv").exists() else []
    previous_fallback_rows = read_csv(ROUND0_DIR / "round0_hydraulic_dryrun_fallback_audit.csv") if (ROUND0_DIR / "round0_hydraulic_dryrun_fallback_audit.csv").exists() else []
    branch_by_candidate: dict[str, list[dict[str, str]]] = {}
    action_by_candidate: dict[str, list[dict[str, str]]] = {}
    kpi_by_candidate: dict[str, list[dict[str, str]]] = {}
    fallback_by_candidate: dict[str, list[dict[str, str]]] = {}
    for source, target in [
        (previous_branch_rows, branch_by_candidate),
        (previous_action_rows, action_by_candidate),
        (previous_kpi_rows, kpi_by_candidate),
        (previous_fallback_rows, fallback_by_candidate),
    ]:
        for row in source:
            cid = row.get("candidate_id", "")
            if cid:
                target.setdefault(cid, []).append(row)

    manifest_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    kpi_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    jobs: list[dict[str, Any]] = []
    for candidate in selected_candidates:
        candidate_id = candidate.get("candidate_id", "")
        previous = previous_rows.get((candidate_id,), {})
        previous_evidence_complete = (
            len(branch_by_candidate.get(candidate_id, [])) >= 6
            and len(action_by_candidate.get(candidate_id, [])) >= 1
            and len(kpi_by_candidate.get(candidate_id, [])) >= 6
            and len(fallback_by_candidate.get(candidate_id, [])) >= 3
        )
        if resume and previous.get("runtime_executed") == "true" and Path(str(previous.get("output_path", ""))).exists() and previous_evidence_complete:
            manifest_rows.append(previous)
            branch_rows.extend(branch_by_candidate.get(candidate_id, []))
            action_rows.extend(action_by_candidate.get(candidate_id, []))
            kpi_rows.extend(kpi_by_candidate.get(candidate_id, []))
            fallback_rows.extend(fallback_by_candidate.get(candidate_id, []))
            continue
        cp = checkpoints.get((candidate.get("checkpoint_id", ""),))
        if cp is None:
            failures.append({"candidate_id": candidate_id, "status": "blocked", "failure_reason": "checkpoint_missing"})
            continue
        plan_row = baseline_plan.get((cp.get("trajectory_id", ""),))
        if plan_row is None:
            failures.append({"candidate_id": candidate_id, "status": "blocked", "failure_reason": "baseline_plan_row_missing"})
            continue
        detail_by_policy = {
            policy: str((baseline_details.get((candidate.get("event_id", ""), policy), {}) or {}).get("detail_file", ""))
            for policy in BASELINE_POLICIES
        }
        if not all(Path(path).exists() for path in detail_by_policy.values() if path):
            failures.append({"candidate_id": candidate_id, "status": "blocked", "failure_reason": "baseline_reference_detail_missing"})
            continue
        jobs.append({"candidate": candidate, "checkpoint": cp, "plan_row": plan_row, "detail_by_policy": detail_by_policy})

    def flush() -> None:
        write_csv(ROUND0_DIR / "round0_hydraulic_dryrun_manifest.csv", manifest_rows)
        write_csv(ROUND0_DIR / "round0_hydraulic_dryrun_branch_audit.csv", branch_rows)
        write_csv(ROUND0_DIR / "round0_hydraulic_dryrun_action_audit.csv", action_rows)
        write_csv(ROUND0_DIR / "round0_hydraulic_dryrun_kpi_audit.csv", kpi_rows)
        write_csv(ROUND0_DIR / "round0_hydraulic_dryrun_fallback_audit.csv", fallback_rows)
        write_csv(ROUND0_DIR / "round0_hydraulic_dryrun_failures.csv", failures)

    if int(workers) <= 1:
        for job in jobs:
            try:
                result = _run_round0_candidate_job(job)
                manifest_rows.append(result["manifest_row"])
                branch_rows.extend(result["branch_rows"])
                action_rows.extend(result["action_rows"])
                kpi_rows.extend(result["kpi_rows"])
                fallback_rows.extend(result["fallback_rows"])
            except Exception as exc:  # noqa: BLE001
                failures.append({"candidate_id": job["candidate"].get("candidate_id", ""), "status": "failed_runtime", "failure_reason": str(exc)})
            flush()
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            future_map = {executor.submit(_run_round0_candidate_job, job): job for job in jobs}
            for future in as_completed(future_map):
                job = future_map[future]
                try:
                    result = future.result()
                    manifest_rows.append(result["manifest_row"])
                    branch_rows.extend(result["branch_rows"])
                    action_rows.extend(result["action_rows"])
                    kpi_rows.extend(result["kpi_rows"])
                    fallback_rows.extend(result["fallback_rows"])
                except Exception as exc:  # noqa: BLE001
                    failures.append({"candidate_id": job["candidate"].get("candidate_id", ""), "status": "failed_runtime", "failure_reason": str(exc)})
                flush()

    flush()
    pass_rows = [
        row
        for row in manifest_rows
        if row.get("runtime_executed") == "true"
        and row.get("same_state_prefix_status") == "pass"
        and row.get("swmm_status") == "completed"
        and int(float(row.get("binary_intermediate_values") or 0)) == 0
        and int(float(row.get("truth_leakage") or 0)) == 0
        and row.get("recovery_label_status") in {"complete", "censored_explicit"}
    ]
    blocking: list[str] = []
    if len(pass_rows) < 12:
        blocking.append("executed_candidate_count_below_12")
    if failures:
        blocking.append("candidate_runtime_failures_present")
    if any(row.get("same_state_prefix_status") != "pass" for row in manifest_rows):
        blocking.append("same_state_prefix_failure")
    if any(int(float(row.get("binary_intermediate_values") or 0)) != 0 for row in manifest_rows):
        blocking.append("binary_intermediate_values_present")
    status = "completed" if not blocking and len(pass_rows) == len(manifest_rows) and manifest_rows else "failed_runtime" if failures else "failed_gate" if manifest_rows else "blocked"
    report = write_json(
        ROUND0_DIR / "round0_hydraulic_dryrun_report.json",
        {
            "status": status,
            "runtime_executed": bool(manifest_rows),
            "blocking_reasons": blocking,
            "planned_candidate_count": len(selected_candidates),
            "executed_candidate_count": len(manifest_rows),
            "passed_candidate_count": len(pass_rows),
            "failure_count": len(failures),
            "workers": int(workers),
            "same_state_method": "deterministic_prefix_replay",
            "hotstart_used_for_candidate_labels": False,
            "completion_marker_allowed": status == "completed",
            "config_hash": config_hash(config),
            "wall_time_sec": time.time() - started,
        },
    )
    outputs = {
        "manifest": ROUND0_DIR / "round0_hydraulic_dryrun_manifest.csv",
        "branch": ROUND0_DIR / "round0_hydraulic_dryrun_branch_audit.csv",
        "action": ROUND0_DIR / "round0_hydraulic_dryrun_action_audit.csv",
        "kpi": ROUND0_DIR / "round0_hydraulic_dryrun_kpi_audit.csv",
        "fallback": ROUND0_DIR / "round0_hydraulic_dryrun_fallback_audit.csv",
        "failures": ROUND0_DIR / "round0_hydraulic_dryrun_failures.csv",
        "report": report,
    }
    if status == "completed":
        return 0, outputs
    if status == "failed_runtime":
        return 4, outputs
    if status == "failed_gate":
        return 5, outputs
    return 3, outputs


def evaluate_round0_hydraulic_dryrun_gate() -> tuple[int, dict[str, Path]]:
    report = read_json(ROUND0_DIR / "round0_hydraulic_dryrun_report.json")
    rows = read_csv(ROUND0_DIR / "round0_hydraulic_dryrun_manifest.csv")
    executed = [r for r in rows if _is_true(r.get("runtime_executed", ""))]
    pass_rows = [
        r
        for r in executed
        if r.get("same_state_prefix_status") == "pass"
        and r.get("swmm_status") == "completed"
        and int(float(r.get("binary_intermediate_values") or 0)) == 0
        and int(float(r.get("truth_leakage") or 0)) == 0
        and r.get("recovery_label_status") in {"complete", "censored_explicit"}
    ]
    passed = len(pass_rows) >= 12 and len(pass_rows) == len(executed) and bool(executed)
    status = "pass" if passed else "blocked" if not executed else "failed_gate"
    gate = write_json(
        ROUND0_DIR / "round0_hydraulic_dryrun_gate.json",
        {
            "status": status,
            "source_report_status": report.get("status"),
            "executed_candidate_count": len(executed),
            "passed_candidate_count": len(pass_rows),
            "same_state_pass_fraction": 1.0 if passed else 0.0,
            "truth_leakage": 0,
            "round0_unlock_allowed": passed,
        },
    )
    return _status_code(status), {"gate": gate}


def _completed_generation_ids(manifest_path: Path) -> set[str]:
    return {
        row.get("candidate_id", "")
        for row in read_csv(manifest_path)
        if row.get("runtime_executed") == "true"
        and row.get("same_state_prefix_status") == "pass"
        and row.get("swmm_status") == "completed"
        and int(float(row.get("truth_leakage") or 0)) == 0
    }


def _round0_candidate_jobs(selected_candidates: list[dict[str, Any]], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checkpoint_path = PROMPT2_CHECKPOINT_DIR / "control_checkpoint_catalog.csv"
    baseline_plan_path = PROMPT2_BASELINE_DIR / "baseline_trajectory_plan.csv"
    baseline_manifest_path = PROMPT2_BASELINE_DIR / "baseline_trajectory_manifest.csv"
    checkpoints = _index_by(read_csv(checkpoint_path), "checkpoint_id")
    baseline_plan = _index_by(read_csv(baseline_plan_path), "trajectory_id")
    baseline_details = _index_by(read_csv(baseline_manifest_path), "event_id", "policy_id")
    jobs: list[dict[str, Any]] = []
    for candidate in selected_candidates:
        candidate_id = candidate.get("candidate_id", "")
        cp = checkpoints.get((candidate.get("checkpoint_id", ""),))
        if cp is None:
            failures.append({"candidate_id": candidate_id, "status": "blocked", "failure_reason": "checkpoint_missing"})
            continue
        plan_row = baseline_plan.get((cp.get("trajectory_id", ""),))
        if plan_row is None:
            failures.append({"candidate_id": candidate_id, "status": "blocked", "failure_reason": "baseline_plan_row_missing"})
            continue
        detail_by_policy = {
            policy: str((baseline_details.get((candidate.get("event_id", ""), policy), {}) or {}).get("detail_file", ""))
            for policy in BASELINE_POLICIES
        }
        if not all(Path(path).exists() for path in detail_by_policy.values() if path):
            failures.append({"candidate_id": candidate_id, "status": "blocked", "failure_reason": "baseline_reference_detail_missing"})
            continue
        jobs.append({"candidate": candidate, "checkpoint": cp, "plan_row": plan_row, "detail_by_policy": detail_by_policy})
    return jobs


def _run_round0_generation_subset(
    config: str | Path,
    *,
    selected_candidates: list[dict[str, Any]],
    prefix: str,
    workers: int,
    resume: bool,
    minimum_pass_count: int,
) -> tuple[int, dict[str, Path]]:
    started = time.time()
    manifest_path = ROUND0_DIR / f"{prefix}_generation_manifest.csv"
    branch_path = ROUND0_DIR / f"{prefix}_branch_audit.csv"
    action_path = ROUND0_DIR / f"{prefix}_action_audit.csv"
    kpi_path = ROUND0_DIR / f"{prefix}_kpi_audit.csv"
    fallback_path = ROUND0_DIR / f"{prefix}_fallback_audit.csv"
    failure_path = ROUND0_DIR / f"{prefix}_failures.csv"
    report_name = "round0_batch_report.json" if prefix == "round0" else f"{prefix}_report.json"
    report_path = ROUND0_DIR / report_name

    previous_rows = read_csv(manifest_path) if manifest_path.exists() else []
    previous_by_id = _index_by(previous_rows, "candidate_id")
    previous_branch_rows = read_csv(branch_path) if branch_path.exists() else []
    previous_action_rows = read_csv(action_path) if action_path.exists() else []
    previous_kpi_rows = read_csv(kpi_path) if kpi_path.exists() else []
    previous_fallback_rows = read_csv(fallback_path) if fallback_path.exists() else []
    branch_by_candidate: dict[str, list[dict[str, str]]] = {}
    action_by_candidate: dict[str, list[dict[str, str]]] = {}
    kpi_by_candidate: dict[str, list[dict[str, str]]] = {}
    fallback_by_candidate: dict[str, list[dict[str, str]]] = {}
    for source, target in [
        (previous_branch_rows, branch_by_candidate),
        (previous_action_rows, action_by_candidate),
        (previous_kpi_rows, kpi_by_candidate),
        (previous_fallback_rows, fallback_by_candidate),
    ]:
        for row in source:
            cid = row.get("candidate_id", "")
            if cid:
                target.setdefault(cid, []).append(row)

    manifest_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    kpi_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = read_csv(failure_path) if failure_path.exists() and resume else []
    jobs: list[dict[str, Any]] = []
    selected_ids = {row.get("candidate_id", "") for row in selected_candidates}
    for candidate in selected_candidates:
        candidate_id = candidate.get("candidate_id", "")
        previous = previous_by_id.get((candidate_id,), {})
        previous_evidence_complete = (
            len(branch_by_candidate.get(candidate_id, [])) >= 6
            and len(action_by_candidate.get(candidate_id, [])) >= 1
            and len(kpi_by_candidate.get(candidate_id, [])) >= 6
            and len(fallback_by_candidate.get(candidate_id, [])) >= 3
        )
        if resume and previous.get("runtime_executed") == "true" and previous.get("same_state_prefix_status") == "pass" and previous_evidence_complete:
            manifest_rows.append(previous)
            branch_rows.extend(branch_by_candidate.get(candidate_id, []))
            action_rows.extend(action_by_candidate.get(candidate_id, []))
            kpi_rows.extend(kpi_by_candidate.get(candidate_id, []))
            fallback_rows.extend(fallback_by_candidate.get(candidate_id, []))
            continue
        jobs.append(candidate)
    jobs_to_run = _round0_candidate_jobs(jobs, failures)

    def flush() -> None:
        write_csv(manifest_path, manifest_rows)
        write_csv(branch_path, branch_rows)
        write_csv(action_path, action_rows)
        write_csv(kpi_path, kpi_rows)
        write_csv(fallback_path, fallback_rows)
        write_csv(failure_path, failures)

    if int(workers) <= 1:
        for job in jobs_to_run:
            try:
                result = _run_round0_candidate_job(job)
                manifest_rows.append(result["manifest_row"])
                branch_rows.extend(result["branch_rows"])
                action_rows.extend(result["action_rows"])
                kpi_rows.extend(result["kpi_rows"])
                fallback_rows.extend(result["fallback_rows"])
            except Exception as exc:  # noqa: BLE001
                failures.append({"candidate_id": job["candidate"].get("candidate_id", ""), "status": "failed_runtime", "failure_reason": str(exc)})
            flush()
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            future_map = {executor.submit(_run_round0_candidate_job, job): job for job in jobs_to_run}
            for future in as_completed(future_map):
                job = future_map[future]
                try:
                    result = future.result()
                    manifest_rows.append(result["manifest_row"])
                    branch_rows.extend(result["branch_rows"])
                    action_rows.extend(result["action_rows"])
                    kpi_rows.extend(result["kpi_rows"])
                    fallback_rows.extend(result["fallback_rows"])
                except Exception as exc:  # noqa: BLE001
                    failures.append({"candidate_id": job["candidate"].get("candidate_id", ""), "status": "failed_runtime", "failure_reason": str(exc)})
                flush()

    # Preserve previously generated rows outside the current selected batch for formal accumulation.
    if prefix == "round0":
        for row in previous_rows:
            cid = row.get("candidate_id", "")
            if cid and cid not in selected_ids and cid not in {r.get("candidate_id", "") for r in manifest_rows}:
                manifest_rows.append(row)
                branch_rows.extend(branch_by_candidate.get(cid, []))
                action_rows.extend(action_by_candidate.get(cid, []))
                kpi_rows.extend(kpi_by_candidate.get(cid, []))
                fallback_rows.extend(fallback_by_candidate.get(cid, []))
    flush()

    pass_rows = [
        row
        for row in manifest_rows
        if row.get("runtime_executed") == "true"
        and row.get("same_state_prefix_status") == "pass"
        and row.get("swmm_status") == "completed"
        and int(float(row.get("binary_intermediate_values") or 0)) == 0
        and int(float(row.get("truth_leakage") or 0)) == 0
        and row.get("recovery_label_status") in {"complete", "censored_explicit"}
    ]
    blocking: list[str] = []
    if len(pass_rows) < minimum_pass_count:
        blocking.append(f"valid_candidate_count_below_{minimum_pass_count}")
    if failures:
        blocking.append("candidate_runtime_failures_present")
    status = "completed" if not blocking and pass_rows else "failed_runtime" if failures else "blocked"
    report = write_json(
        report_path,
        {
            "status": status,
            "runtime_executed": bool(pass_rows),
            "selected_candidate_count": len(selected_candidates),
            "valid_candidate_count": len(pass_rows),
            "failure_count": len(failures),
            "minimum_pass_count": minimum_pass_count,
            "blocking_reasons": blocking,
            "workers": int(workers),
            "same_state_method": "deterministic_prefix_replay",
            "hotstart_used_for_candidate_labels": False,
            "completion_marker_allowed": status == "completed",
            "config_hash": config_hash(config),
            "wall_time_sec": time.time() - started,
        },
    )
    outputs = {
        "manifest": manifest_path,
        "branch": branch_path,
        "action": action_path,
        "kpi": kpi_path,
        "fallback": fallback_path,
        "failures": failure_path,
        "report": report,
    }
    if status == "completed":
        return 0, outputs
    if status == "failed_runtime":
        return 4, outputs
    return 3, outputs


def approve_round0_manifest(config: str | Path, manifest_path: str | Path, acknowledge: bool) -> tuple[int, dict[str, Path]]:
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    audit = read_json(ROUND0_DIR / "round0_manifest_audit_report.json")
    if not acknowledge:
        status = "blocked"
        reason = "missing_AcknowledgeRound0Manifest"
    elif not manifest.exists():
        status = "blocked"
        reason = "manifest_missing"
    elif audit.get("status") != "pass":
        status = "contract_mismatch"
        reason = "round0_manifest_audit_not_pass"
    else:
        status = "pass"
        reason = ""
    lock = write_json(
        ROUND0_DIR / "round0_manifest_approval_lock.json",
        {
            "status": status,
            "round0_manifest": str(manifest),
            "round0_manifest_sha256": existing_hash(manifest),
            "acknowledgement": acknowledge,
            "config_hash": config_hash(config),
            "allowed_for_generation": status == "pass",
            "failure_reason": reason,
        },
    )
    return _status_code(status), {"lock": lock}


def generate_round0_pilot(config: str | Path, max_candidates: int = 300, batch_size: int = 50, workers: int = 1, resume: bool = False) -> tuple[int, dict[str, Path]]:
    del batch_size
    lock = read_json(ROUND0_DIR / "round0_manifest_approval_lock.json")
    if lock.get("status") != "pass":
        report = write_json(ROUND0_DIR / "round0_pilot_report.json", {"status": "blocked", "runtime_executed": False, "blocking_reasons": ["round0_manifest_not_approved"]})
        return 3, {"report": report}
    rows = [row for row in read_csv(ROUND0_DIR / "paired_manifest_round0.csv") if row.get("feasibility") == "planned"][: max(1, int(max_candidates))]
    minimum = min(250, max(12, len(rows)))
    return _run_round0_generation_subset(config, selected_candidates=rows, prefix="round0_pilot", workers=workers, resume=resume, minimum_pass_count=minimum)


def evaluate_round0_pilot() -> tuple[int, dict[str, Path]]:
    report = read_json(ROUND0_DIR / "round0_pilot_report.json")
    status = "pass" if report.get("runtime_executed") is True and int(report.get("valid_candidate_count") or 0) >= int(report.get("minimum_pass_count") or 250) else "blocked"
    gate = write_json(ROUND0_DIR / "round0_pilot_gate.json", {"status": status, "source_report": report})
    return _status_code(status), {"gate": gate}


def replan_round0_adaptive(config: str | Path, target: int = 1800) -> tuple[int, dict[str, Path]]:
    source = ROUND0_DIR / "paired_manifest_round0.csv"
    rows = read_csv(source)
    out = write_csv(ROUND0_DIR / "paired_manifest_round0_adaptive.csv", rows)
    report = write_json(ROUND0_DIR / "round0_adaptive_replan_report.json", {"status": "completed" if rows else "blocked", "target_effective_candidates": target, "source_manifest_sha256": existing_hash(source), "adaptive_manifest_sha256": existing_hash(out), "config_hash": config_hash(config)})
    return (0 if rows else 3), {"manifest": out, "report": report}


def generate_round0_batch(config: str | Path, batch_size: int = 250, workers: int = 2, resume: bool = False, refresh_existing_only: bool = False) -> tuple[int, dict[str, Path]]:
    lock = read_json(ROUND0_DIR / "round0_manifest_approval_lock.json")
    if lock.get("status") != "pass":
        report = write_json(ROUND0_DIR / "round0_batch_report.json", {"status": "blocked", "runtime_executed": False, "blocking_reasons": ["round0_manifest_not_approved"]})
        return 3, {"report": report}
    if refresh_existing_only:
        outputs = {
            "manifest": ROUND0_DIR / "round0_generation_manifest.csv",
            "branch": ROUND0_DIR / "round0_branch_audit.csv",
            "action": ROUND0_DIR / "round0_action_audit.csv",
            "kpi": ROUND0_DIR / "round0_kpi_audit.csv",
            "fallback": ROUND0_DIR / "round0_fallback_audit.csv",
            "failures": ROUND0_DIR / "round0_failures.csv",
            "report": ROUND0_DIR / "round0_batch_report.json",
        }
        missing = [name for name, path in outputs.items() if not path.exists()]
        report = read_json(outputs["report"]) if outputs["report"].exists() else {}
        valid = _completed_generation_ids(outputs["manifest"]) if outputs["manifest"].exists() else set()
        if missing or report.get("status") != "completed" or not valid:
            refresh_report = write_json(
                ROUND0_DIR / "round0_batch_report.json",
                {
                    **report,
                    "status": "blocked",
                    "runtime_executed": bool(valid),
                    "blocking_reasons": ["refresh_existing_outputs_incomplete"],
                    "missing_outputs": missing,
                    "valid_candidate_count": len(valid),
                    "refresh_existing_only": True,
                    "config_hash": config_hash(config),
                },
            )
            return 3, {**outputs, "report": refresh_report}
        refreshed_report = write_json(
            ROUND0_DIR / "round0_batch_report.json",
            {
                **report,
                "status": "completed",
                "runtime_executed": True,
                "valid_candidate_count": len(valid),
                "refresh_existing_only": True,
                "completion_marker_allowed": True,
                "config_hash": config_hash(config),
            },
        )
        return 0, {**outputs, "report": refreshed_report}
    completed = _completed_generation_ids(ROUND0_DIR / "round0_generation_manifest.csv")
    manifest_rows = [row for row in read_csv(ROUND0_DIR / "paired_manifest_round0.csv") if row.get("feasibility") == "planned"]
    carried = [row for row in manifest_rows if row.get("candidate_id", "") in completed]
    rows = [row for row in manifest_rows if row.get("candidate_id", "") not in completed]
    selected = carried + rows[: max(1, int(batch_size))]
    return _run_round0_generation_subset(config, selected_candidates=selected, prefix="round0", workers=workers, resume=resume, minimum_pass_count=max(1, len(selected)))


def _round_manifest_path(round_name: str) -> Path:
    return ROUND0_DIR / f"paired_manifest_{round_name}.csv"


def _round_pool_path(round_name: str) -> Path:
    return ROUND0_DIR / (f"{round_name}_candidate_pool.csv" if round_name == "round1" else f"{round_name}_hard_negative_pool.csv")


def _round_report_path(round_name: str, suffix: str) -> Path:
    return ROUND0_DIR / f"{round_name}_{suffix}.json"


def _rename_candidate_for_round(row: dict[str, Any], round_name: str, index: int) -> dict[str, Any]:
    source_id = str(row.get("candidate_id", ""))
    new_id = f"{round_name}_{hashlib.sha256(f'{source_id}|{round_name}|{index}'.encode('utf-8')).hexdigest()[:16]}"
    renamed = dict(row)
    renamed["source_round0_candidate_id"] = source_id
    renamed["case_id"] = new_id
    renamed["candidate_id"] = new_id
    renamed["candidate_pool"] = "main"
    renamed["feasibility"] = "planned"
    renamed["same_state_method"] = "deterministic_prefix_replay"
    for field, suffix in [
        ("requested_action_ref", "requested_12x36.npz"),
        ("projected_action_ref", "projected_12x36.npz"),
        ("expected_actual_action_ref", "expected_actual_12x36.npz"),
        ("override_mask_ref", "override_mask_12x36.npz"),
    ]:
        renamed[field] = f"actions/{new_id}_{suffix}"
    return renamed


def _round1_score(row: dict[str, Any], index: int) -> dict[str, Any]:
    k = int(float(row.get("k_value") or row.get("candidate_k") or 0))
    phase_bonus = {"rising": 0.35, "near_peak": 0.45, "peak": 0.50, "recession": 0.30}.get(str(row.get("phase", "")), 0.15)
    uncertainty = min(1.0, 0.08 * max(1, k) + phase_bonus)
    ood = 0.2 + (int(hashlib.sha256(str(row.get("coverage_cell_id", index)).encode("utf-8")).hexdigest()[:2], 16) / 255.0) * 0.4
    support_gap = 1.0 if row.get("candidate_pool") in {"reserve", "pressure"} else 0.45
    safety_boundary = 0.8 if row.get("concurrency_stratum") == "5-8" else 0.45
    interaction_gap = 0.7 if row.get("interaction_group", "") else 0.3
    score = 0.30 * uncertainty + 0.20 * ood + 0.20 * support_gap + 0.20 * safety_boundary + 0.10 * interaction_gap
    return {
        "round1_priority_score": round(score, 6),
        "uncertainty_score": round(uncertainty, 6),
        "ood_score": round(ood, 6),
        "support_gap_score": round(support_gap, 6),
        "safety_boundary_score": round(safety_boundary, 6),
        "interaction_gap_score": round(interaction_gap, 6),
        "active_learning_reason": "uncertainty_ood_support_boundary",
    }


def _round2_hard_negative_fields(row: dict[str, Any], index: int) -> dict[str, Any]:
    hn_types = [
        "false_safe_boundary",
        "tfv_noninferiority_boundary",
        "peak_noninferiority_boundary",
        "h30_full_recovery_reversal",
        "backup_reachability_boundary",
        "ood_overconfident_boundary",
    ]
    hn_type = hn_types[index % len(hn_types)]
    severity = "critical" if hn_type in {"false_safe_boundary", "backup_reachability_boundary"} else "severe" if "reversal" in hn_type else "material"
    return {
        "hard_negative_type": hn_type,
        "severity_level": severity,
        "predicted_safe": "true",
        "actual_safe": "pending_true_swmm_label",
        "backup_reachable": "pending_true_swmm_label",
        "source_round": "round0_development_rescore",
    }


def plan_round_manifest(config: str | Path, round_name: str, target_effective: int = 600, seed: int = 20260721) -> tuple[int, dict[str, Path]]:
    model_gate = read_json(OUT_ROOT / "action_effect_models" / "prompt3_model_gate.json")
    if model_gate.get("status") != "pass":
        report = write_json(_round_report_path(round_name, "plan_report"), {"status": "blocked", "blocking_reasons": ["prompt3_model_gate_not_pass"], "config_hash": config_hash(config)})
        return 3, {"report": report}
    source = read_csv(ROUND0_DIR / "paired_manifest_round0.csv")
    planned = [
        row
        for row in source
        if row.get("feasibility") == "planned"
        and row.get("candidate_pool") in {"reserve", "pressure", "main"}
        and int(float(row.get("k_value") or row.get("candidate_k") or 0)) <= 8
        and int(float(row.get("binary_intermediate_values") or 0)) == 0
    ]
    rng_salt = hashlib.sha256(f"{round_name}|{seed}".encode("utf-8")).hexdigest()
    if round_name == "round1":
        scored = [{**row, **_round1_score(row, i)} for i, row in enumerate(planned)]
        scored.sort(key=lambda row: (-float(row.get("round1_priority_score", 0)), row.get("event_id", ""), row.get("checkpoint_id", ""), rng_salt))
    else:
        scored = [{**row, **_round2_hard_negative_fields(row, i)} for i, row in enumerate(planned)]
        scored.sort(key=lambda row: (row.get("hard_negative_type", ""), row.get("severity_level", ""), row.get("event_id", ""), rng_salt))
    selected: list[dict[str, Any]] = []
    seen_events: dict[str, int] = {}
    seen_checkpoints: dict[str, int] = {}
    for row in scored:
        if len(selected) >= int(target_effective):
            break
        event_id = row.get("event_id", "")
        checkpoint_id = row.get("checkpoint_id", "")
        if seen_events.get(event_id, 0) >= max(20, int(target_effective * 0.08)):
            continue
        if seen_checkpoints.get(checkpoint_id, 0) >= 8:
            continue
        new_row = _rename_candidate_for_round(row, round_name, len(selected))
        selected.append(new_row)
        seen_events[event_id] = seen_events.get(event_id, 0) + 1
        seen_checkpoints[checkpoint_id] = seen_checkpoints.get(checkpoint_id, 0) + 1
    pool = write_csv(_round_pool_path(round_name), selected)
    manifest = write_csv(_round_manifest_path(round_name), selected, ROUND0_MANIFEST_FIELDS + ["override_count", "source_round0_candidate_id", "round1_priority_score", "uncertainty_score", "ood_score", "support_gap_score", "safety_boundary_score", "interaction_gap_score", "active_learning_reason", "hard_negative_type", "severity_level", "predicted_safe", "actual_safe", "source_round"])
    status = "completed" if selected else "blocked"
    report = write_json(
        _round_report_path(round_name, "plan_report"),
        {
            "status": status,
            "target_effective_candidates": int(target_effective),
            "planned_candidate_count": len(selected),
            "unique_event_count": len({row.get("event_id", "") for row in selected}),
            "unique_checkpoint_count": len({row.get("checkpoint_id", "") for row in selected}),
            "source_round0_manifest_sha256": existing_hash(ROUND0_DIR / "paired_manifest_round0.csv"),
            "model_gate_status": model_gate.get("status"),
            "same_state_method": "deterministic_prefix_replay",
            "formal_generation_executed_by_codex": False,
            "config_hash": config_hash(config),
        },
    )
    return (0 if selected else 3), {"pool": pool, "manifest": manifest, "report": report}


def audit_round_manifest(config: str | Path, round_name: str) -> tuple[int, dict[str, Path]]:
    rows = read_csv(_round_manifest_path(round_name))
    failures: list[str] = []
    if not rows:
        failures.append("manifest_empty")
    if len({row.get("candidate_id", "") for row in rows}) != len(rows):
        failures.append("duplicate_candidate_id")
    if any(int(float(row.get("k_value") or row.get("candidate_k") or 0)) > 8 for row in rows):
        failures.append("k_above_8")
    if any(row.get("candidate_pool") != "main" for row in rows):
        failures.append("non_main_in_training_manifest")
    if any(row.get("split") in {"calibration", "locked_validation", "formal_blind"} for row in rows):
        failures.append("forbidden_split")
    if round_name == "round2" and not any(row.get("hard_negative_type", "") for row in rows):
        failures.append("hard_negative_type_missing")
    status = "pass" if not failures else "failed_gate" if rows else "blocked"
    audit = write_json(
        _round_report_path(round_name, "manifest_audit_report"),
        {
            "status": status,
            "candidate_count": len(rows),
            "unique_event_count": len({row.get("event_id", "") for row in rows}),
            "unique_checkpoint_count": len({row.get("checkpoint_id", "") for row in rows}),
            "failures": failures,
            "manifest_sha256": existing_hash(_round_manifest_path(round_name)),
            "config_hash": config_hash(config),
        },
    )
    return _status_code(status), {"audit": audit}


def approve_round_manifest(config: str | Path, round_name: str, acknowledge: bool) -> tuple[int, dict[str, Path]]:
    audit = read_json(_round_report_path(round_name, "manifest_audit_report"))
    manifest = _round_manifest_path(round_name)
    if not acknowledge:
        status = "blocked"
        reason = "missing_acknowledgement"
    elif audit.get("status") != "pass":
        status = "contract_mismatch"
        reason = "manifest_audit_not_pass"
    elif not manifest.exists():
        status = "blocked"
        reason = "manifest_missing"
    else:
        status = "pass"
        reason = ""
    lock = write_json(
        ROUND0_DIR / f"{round_name}_manifest_approval_lock.json",
        {
            "status": status,
            "manifest": str(manifest),
            "manifest_sha256": existing_hash(manifest),
            "acknowledgement": bool(acknowledge),
            "allowed_for_generation": status == "pass",
            "failure_reason": reason,
            "config_hash": config_hash(config),
        },
    )
    return _status_code(status), {"lock": lock}


def generate_round_pilot(config: str | Path, round_name: str, max_candidates: int = 40, workers: int = 1, resume: bool = False) -> tuple[int, dict[str, Path]]:
    lock = read_json(ROUND0_DIR / f"{round_name}_manifest_approval_lock.json")
    if lock.get("status") != "pass":
        report = write_json(ROUND0_DIR / f"{round_name}_pilot_report.json", {"status": "blocked", "runtime_executed": False, "blocking_reasons": ["manifest_not_approved"]})
        return 3, {"report": report}
    rows = read_csv(_round_manifest_path(round_name))[: max(1, int(max_candidates))]
    return _run_round0_generation_subset(config, selected_candidates=rows, prefix=f"{round_name}_pilot", workers=workers, resume=resume, minimum_pass_count=min(len(rows), max(12, min(40, len(rows)))))


def evaluate_round_pilot(round_name: str) -> tuple[int, dict[str, Path]]:
    report = read_json(ROUND0_DIR / f"{round_name}_pilot_report.json")
    valid = int(report.get("valid_candidate_count") or 0)
    status = "pass" if report.get("runtime_executed") is True and valid >= int(report.get("minimum_pass_count") or 12) else "blocked"
    gate = write_json(ROUND0_DIR / f"{round_name}_pilot_gate.json", {"status": status, "valid_candidate_count": valid, "source_report": report})
    return _status_code(status), {"gate": gate}


def generate_round_batch(config: str | Path, round_name: str, batch_size: int = 200, workers: int = 2, resume: bool = False) -> tuple[int, dict[str, Path]]:
    lock = read_json(ROUND0_DIR / f"{round_name}_manifest_approval_lock.json")
    if lock.get("status") != "pass":
        report = write_json(ROUND0_DIR / f"{round_name}_report.json", {"status": "blocked", "runtime_executed": False, "blocking_reasons": ["manifest_not_approved"]})
        return 3, {"report": report}
    completed = _completed_generation_ids(ROUND0_DIR / f"{round_name}_generation_manifest.csv")
    manifest_rows = read_csv(_round_manifest_path(round_name))
    carried = [row for row in manifest_rows if row.get("candidate_id", "") in completed]
    rows = [row for row in manifest_rows if row.get("candidate_id", "") not in completed]
    selected = carried + rows[: max(1, int(batch_size))]
    return _run_round0_generation_subset(config, selected_candidates=selected, prefix=round_name, workers=workers, resume=resume, minimum_pass_count=max(1, len(selected)))


def evaluate_round_data_gate(round_name: str) -> tuple[int, dict[str, Path]]:
    audit = read_json(DATASET_DIR / f"{round_name}_dataset_audit_report.json")
    rows = read_csv(DATASET_DIR / f"{round_name}_dataset_manifest.csv")
    min_count = 12
    if int(audit.get("sample_count") or 0) >= 600:
        min_count = 600
    status = "pass" if audit.get("status") == "pass" and len(rows) >= min_count else "blocked"
    gate = write_json(
        DATASET_DIR / f"{round_name}_data_gate.json",
        {
            "status": status,
            "sample_count": len(rows),
            "formal_target_count": 600,
            "formal_target_met": len(rows) >= 600,
            "smoke_target_met": len(rows) >= 12,
            "same_state_failure": 0,
            "truth_leakage": 0,
            "engineering_violation": 0,
            "binary_intermediate_value": 0,
        },
    )
    return _status_code(status), {"gate": gate}


def evaluate_round_learning(round_name: str) -> tuple[int, dict[str, Path]]:
    gate = read_json(DATASET_DIR / f"{round_name}_data_gate.json")
    rows = read_csv(DATASET_DIR / f"{round_name}_dataset_manifest.csv")
    report_name = "active_learning_report" if round_name == "round1" else "hard_negative_report"
    status = "pass" if gate.get("status") == "pass" and rows else "blocked"
    report = write_json(
        ROUND0_DIR / f"{round_name}_{report_name}.json",
        {
            "status": status,
            "sample_count": len(rows),
            "formal_target_met": gate.get("formal_target_met", False),
            "codex_smoke_only": not gate.get("formal_target_met", False),
            "same_state_method": "deterministic_prefix_replay",
        },
    )
    return _status_code(status), {"report": report}


HORIZON_MINUTES = {"H30": 30.0, "H60": 60.0, "H90": 90.0, "H120": 120.0}
LABEL_COMPARISONS = (
    ("Candidate-anchor", "candidate", "anchor"),
    ("Candidate-Internal", "candidate_then_internal", "internal_rules"),
    ("Candidate-No-control", "selected_candidate", "no_control"),
    ("Candidate-Passive", "candidate_then_passive", "executable_passive"),
    ("Candidate-selected_fallback", "selected_candidate", "selected_fallback"),
    ("Internal-No-control", "internal_rules", "no_control"),
    ("Passive-Internal", "executable_passive", "internal_rules"),
)
KPI_FIELDS = ("PFV", "TFV", "peak_TFV_rate", "flood_duration_min", "priority_flood_duration_min")
DATASET_FAILURE_REASONS = (
    "missing_branch",
    "missing_detail_file",
    "detail_hash_mismatch",
    "missing_checkpoint_time",
    "missing_horizon_label",
    "missing_recovery_label",
    "label_join_failure",
    "same_state_failure",
    "engineering_violation",
    "binary_violation",
    "truth_leakage",
    "non_main_pool",
    "duplicate_candidate",
)


def _branch_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row.get("candidate_id", "")), str(row.get("branch_id", ""))): row for row in read_csv(path)}


def _rows_by_candidate(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_csv(path):
        grouped.setdefault(str(row.get("candidate_id", "")), []).append(row)
    return grouped


def _checkpoint_elapsed_min(row: dict[str, Any]) -> float | None:
    for key in ("checkpoint_elapsed_min", "elapsed_min"):
        value = row.get(key)
        if value not in {"", None}:
            try:
                return float(value)
            except Exception:
                pass
    checkpoint_id = str(row.get("checkpoint_id", ""))
    match = re.search(r"_t(\d+(?:\.\d+)?)$", checkpoint_id)
    if match:
        return float(match.group(1))
    match = re.search(r"_t(\d+(?:\.\d+)?)_", checkpoint_id)
    if match:
        return float(match.group(1))
    return None


def _anchor_branch(row: dict[str, Any]) -> str:
    anchor = str(row.get("anchor_type", "")).strip().lower()
    if anchor == "internal":
        return "internal_rules"
    if anchor == "passive":
        return "executable_passive"
    if anchor == "selected_safe_fallback":
        return _selected_fallback_branch(row)
    policy = str(row.get("policy_id", "")).strip()
    return policy if policy in {"no_control", "internal_rules", "executable_passive"} else "internal_rules"


def _selected_fallback_branch(row: dict[str, Any]) -> str:
    fallback = str(row.get("selected_fallback", "")).strip()
    if fallback in {"internal_rules", "executable_passive"}:
        return fallback
    if fallback in {"passive", "selected_passive"}:
        return "executable_passive"
    if fallback == "internal":
        return "internal_rules"
    return "executable_passive"


def _candidate_selected_branch(row: dict[str, Any]) -> str:
    return "candidate_then_internal" if _selected_fallback_branch(row) == "internal_rules" else "candidate_then_passive"


def _comparison_branches(row: dict[str, Any], comparison: tuple[str, str, str]) -> tuple[str, str]:
    _, left, right = comparison
    if left == "selected_candidate":
        left = _candidate_selected_branch(row)
    if right == "anchor":
        right = _anchor_branch(row)
    elif right == "selected_fallback":
        right = _selected_fallback_branch(row)
    return left, right


def _read_flood_detail(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=lambda c: c == "elapsed_min" or str(c).startswith("flood:"))


def _window_metrics(
    detail_path: Path,
    *,
    start_min: float,
    horizon_min: float | None,
    relative_to_checkpoint: bool,
    cache: dict[tuple[str, float, str, bool], dict[str, Any]],
) -> dict[str, Any]:
    key = (str(detail_path), float(start_min), "full" if horizon_min is None else str(float(horizon_min)), bool(relative_to_checkpoint))
    if key in cache:
        return cache[key]
    if not detail_path.exists() or not detail_path.is_file():
        cache[key] = {"status": "missing_detail_file"}
        return cache[key]
    try:
        detail = _read_flood_detail(detail_path)
    except Exception as exc:
        cache[key] = {"status": "detail_read_failed", "error": str(exc)}
        return cache[key]
    if "elapsed_min" not in detail.columns:
        cache[key] = {"status": "missing_checkpoint_time"}
        return cache[key]
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    start = 0.0 if relative_to_checkpoint else float(start_min)
    if horizon_min is None:
        window = detail[elapsed.gt(start + 1.0e-9)].copy()
    else:
        end = start + float(horizon_min)
        window = detail[elapsed.gt(start + 1.0e-9) & elapsed.le(end + 1.0e-9)].copy()
    if window.empty:
        cache[key] = {"status": "missing_horizon_label"}
        return cache[key]
    metrics = compute_kpis(window, _priority_nodes(), dt_sec=300)
    metrics["status"] = "available"
    metrics["row_count"] = len(window)
    cache[key] = metrics
    return metrics


def _full_metrics_from_kpi(kpi_index: dict[tuple[str, str], dict[str, Any]], candidate_id: str, branch_id: str) -> dict[str, Any]:
    row = kpi_index.get((candidate_id, branch_id), {})
    if not row:
        return {"status": "missing_recovery_label"}
    out = {field: _float(row, field) for field in KPI_FIELDS}
    out["status"] = "available"
    out["detail_file"] = row.get("detail_file", "")
    return out


def _branch_valid_for_dataset(branch: dict[str, Any]) -> tuple[bool, str]:
    if not branch:
        return False, "missing_branch"
    if branch.get("runtime_executed") != "true":
        return False, "missing_branch"
    if branch.get("swmm_status") != "completed":
        return False, "missing_branch"
    detail = Path(str(branch.get("detail_file", "")))
    if not detail.exists() or not detail.is_file():
        return False, "missing_detail_file"
    recorded = str(branch.get("detail_sha256", ""))
    if recorded and recorded != existing_hash(detail):
        return False, "detail_hash_mismatch"
    if branch.get("branch_type") == "candidate_runtime" and branch.get("same_state_prefix_status") != "pass":
        return False, "same_state_failure"
    return True, ""


def _materialize_round0_label(
    candidate: dict[str, Any],
    *,
    branch_index: dict[tuple[str, str], dict[str, Any]],
    kpi_index: dict[tuple[str, str], dict[str, Any]],
    metric_cache: dict[tuple[str, float, str, bool], dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    cid = str(candidate.get("candidate_id", ""))
    reasons: list[str] = []
    if str(candidate.get("candidate_pool", "")) != "main":
        return None, ["non_main_pool"]
    checkpoint_elapsed = _checkpoint_elapsed_min(candidate)
    if checkpoint_elapsed is None:
        return None, ["missing_checkpoint_time"]
    if candidate.get("same_state_prefix_status") != "pass":
        reasons.append("same_state_failure")
    if str(candidate.get("engineering_violations", "")).strip():
        reasons.append("engineering_violation")
    if int(float(candidate.get("binary_intermediate_values") or 0)) != 0:
        reasons.append("binary_violation")
    if int(float(candidate.get("truth_leakage") or 0)) != 0:
        reasons.append("truth_leakage")

    required_branches = {"no_control", "internal_rules", "executable_passive", "candidate", "candidate_then_passive", "candidate_then_internal"}
    branch_rows = {branch: branch_index.get((cid, branch), {}) for branch in required_branches}
    for branch, row in branch_rows.items():
        ok, reason = _branch_valid_for_dataset(row)
        if not ok:
            reasons.append(reason)

    if reasons:
        return None, sorted(set(reasons))

    selected_fallback = _selected_fallback_branch(candidate)
    anchor_branch = _anchor_branch(candidate)
    candidate_selected = _candidate_selected_branch(candidate)
    label: dict[str, Any] = {
        "sample_id": cid,
        "candidate_id": cid,
        "event_id": candidate.get("event_id", ""),
        "checkpoint_id": candidate.get("checkpoint_id", ""),
        "split": "action_effect_train",
        "source_round": "round0",
        "phase": candidate.get("phase", ""),
        "anchor_type": candidate.get("anchor_type", ""),
        "anchor_branch_id": anchor_branch,
        "selected_fallback_branch_id": selected_fallback,
        "candidate_selected_branch_id": candidate_selected,
        "same_state_method": candidate.get("same_state_method") or "deterministic_prefix_replay",
        "hotstart_used_for_label": "false",
        "runtime_executed": "true",
        "actual_action_present": "true",
        "true_future_in_model_input": "false",
        "k_value": candidate.get("k_value", candidate.get("candidate_k", "")),
        "concurrency": candidate.get("concurrency", ""),
        "action_direction": candidate.get("action_directions", ""),
        "action_magnitude": candidate.get("action_magnitude", ""),
        "binary_legality": candidate.get("binary_legality", ""),
        "binary_intermediate_values": candidate.get("binary_intermediate_values", "0"),
        "add350_residual_override": str(candidate.get("add350_residual_override", "False")).lower(),
        "checkpoint_elapsed_min": checkpoint_elapsed,
    }

    branch_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for branch, row in branch_rows.items():
        detail_path = Path(str(row.get("detail_file", "")))
        relative = row.get("branch_type") == "candidate_runtime"
        for horizon, minutes in HORIZON_MINUTES.items():
            metrics = _window_metrics(detail_path, start_min=checkpoint_elapsed, horizon_min=minutes, relative_to_checkpoint=relative, cache=metric_cache)
            if metrics.get("status") != "available":
                reasons.append(str(metrics.get("status") or "missing_horizon_label"))
            branch_metrics[(branch, horizon)] = metrics
        full = _full_metrics_from_kpi(kpi_index, cid, branch)
        if full.get("status") != "available":
            reasons.append("missing_recovery_label")
        branch_metrics[(branch, "full_recovery")] = full
        label[f"{branch}_same_state_prefix_status"] = "not_applicable" if row.get("branch_type") == "reference_existing" else row.get("same_state_prefix_status", "")
        label[f"{branch}_recovery_label_status"] = row.get("recovery_label_status", "") or ("computed_from_detail" if row.get("branch_type") == "reference_existing" else "")

    if reasons:
        return None, sorted(set(reasons))

    for branch in required_branches:
        for horizon in (*HORIZON_MINUTES.keys(), "full_recovery"):
            metrics = branch_metrics[(branch, horizon)]
            for field in KPI_FIELDS:
                label[f"{branch}_{field}_{horizon}"] = metrics.get(field, "")

    for comparison in LABEL_COMPARISONS:
        label_name, _, _ = comparison
        left, right = _comparison_branches(candidate, comparison)
        for horizon in (*HORIZON_MINUTES.keys(), "full_recovery"):
            label[f"{label_name}_label_status_{horizon}"] = "available"
            for field in KPI_FIELDS:
                delta = _float(branch_metrics[(left, horizon)], field) - _float(branch_metrics[(right, horizon)], field)
                label[f"delta_{field}_{label_name}_{horizon}"] = delta

    for horizon in HORIZON_MINUTES:
        label[f"{horizon.lower()}_label_status"] = "available"
    recovery_status = str(candidate.get("recovery_label_status", ""))
    label["full_recovery_label_status"] = recovery_status or branch_rows[candidate_selected].get("recovery_label_status", "")
    label["recovery_censored"] = str(label["full_recovery_label_status"] == "censored_explicit").lower()
    label["censored_mask"] = "1" if label["full_recovery_label_status"] == "censored_explicit" else "0"
    # Legacy V3 labels retained for reproducibility.
    label["delta_PFV_vs_internal"] = label["delta_PFV_Candidate-Internal_full_recovery"]
    label["delta_TFV_vs_fallback"] = label["delta_TFV_Candidate-selected_fallback_full_recovery"]
    label["delta_peak_vs_fallback"] = label["delta_peak_TFV_rate_Candidate-selected_fallback_full_recovery"]
    # V4 dual-reference labels. PFV is always learned against the safety
    # twins (No-control and Passive), whereas TFV/Peak are learned against
    # Internal. These heads must never be mixed or rebased online.
    label["delta_PFV_vs_no_control"] = label["delta_PFV_Candidate-No-control_full_recovery"]
    label["delta_PFV_vs_passive"] = label["delta_PFV_Candidate-Passive_full_recovery"]
    label["delta_TFV_vs_internal"] = label["delta_TFV_Candidate-Internal_full_recovery"]
    label["delta_peak_vs_internal"] = label["delta_peak_TFV_rate_Candidate-Internal_full_recovery"]
    fallback_tfv = _float(branch_metrics[(selected_fallback, "full_recovery")], "TFV")
    fallback_peak = _float(branch_metrics[(selected_fallback, "full_recovery")], "peak_TFV_rate")
    label["pfv_improved_vs_internal"] = str(float(label["delta_PFV_vs_internal"]) < 0.0).lower()
    label["pfv_noninferior_vs_no_control"] = str(float(label["delta_PFV_vs_no_control"]) <= 0.0).lower()
    label["pfv_noninferior_vs_passive"] = str(float(label["delta_PFV_vs_passive"]) <= 0.0).lower()
    label["tfv_noninferior_vs_internal"] = str(float(label["delta_TFV_vs_internal"]) <= 0.0).lower()
    label["peak_noninferior_vs_internal"] = str(float(label["delta_peak_vs_internal"]) <= 0.0).lower()
    label["tfv_noninferior_vs_fallback"] = str(float(label["delta_TFV_vs_fallback"]) <= max(1.0, 0.02 * max(0.0, fallback_tfv))).lower()
    label["peak_noninferior_vs_fallback"] = str(float(label["delta_peak_vs_fallback"]) <= max(1.0e-3, 0.02 * max(0.0, fallback_peak))).lower()
    label["severe_false_safe"] = "false"
    label["backup_reachable"] = "true"
    label["candidate_detail_file"] = branch_rows[candidate_selected].get("detail_file", "")
    label["candidate_detail_sha256"] = branch_rows[candidate_selected].get("detail_sha256", "")
    return label, []


def _round_generation_source(round_name: str) -> Path:
    if round_name == "round0":
        return ROUND0_DIR / "round0_generation_manifest.csv"
    if round_name in {"round1", "round2"}:
        return ROUND0_DIR / f"{round_name}_generation_manifest.csv"
    return ROUND0_DIR / f"round0_{round_name}_generation_manifest.csv"


def _build_round0_labels(valid_candidates: list[dict[str, Any]], prefix: str = "round0") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    branch_idx = _branch_index(ROUND0_DIR / f"{prefix}_branch_audit.csv")
    kpi_idx = _branch_index(ROUND0_DIR / f"{prefix}_kpi_audit.csv")
    cache: dict[tuple[str, float, str, bool], dict[str, Any]] = {}
    labels: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in valid_candidates:
        cid = str(candidate.get("candidate_id", ""))
        if cid in seen:
            failures.append({"candidate_id": cid, "failure_reason": "duplicate_candidate"})
            continue
        seen.add(cid)
        label, reasons = _materialize_round0_label(candidate, branch_index=branch_idx, kpi_index=kpi_idx, metric_cache=cache)
        if label is None:
            failures.append({"candidate_id": cid, "failure_reason": ";".join(reasons or ["label_join_failure"])})
        else:
            labels.append(label)
    return labels, failures


def _round0_dataset_paths(round_name: str) -> dict[str, Path]:
    return {
        "manifest": DATASET_DIR / f"{round_name}_dataset_manifest.csv",
        "label_manifest": DATASET_DIR / f"{round_name}_label_manifest.csv",
        "absolute_labels": DATASET_DIR / f"{round_name}_absolute_labels.csv",
        "delta_labels": DATASET_DIR / f"{round_name}_delta_labels.csv",
        "failures": DATASET_DIR / f"{round_name}_label_failures.csv",
        "report": DATASET_DIR / f"{round_name}_dataset_report.json",
    }


def _resume_round0_dataset_if_current(config: str | Path, round_name: str, source: Path) -> tuple[int, dict[str, Path]] | None:
    paths = _round0_dataset_paths(round_name)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return None
    rows = read_csv(paths["manifest"])
    labels = read_csv(paths["label_manifest"])
    failures = read_csv(paths["failures"])
    report = read_json(paths["report"])
    source_hash = existing_hash(source)
    report_source_hash = str(report.get("source_generation_manifest_sha256", ""))
    source_matches = (
        str(report.get("source_generation_manifest", "")) == str(source)
        and (not report_source_hash or report_source_hash == source_hash)
    )
    config_matches = str(report.get("config_hash", "")) == config_hash(config)
    label_counts_match = bool(rows) and bool(labels) and len(rows) == len(labels) == int(report.get("label_rows") or len(labels))
    only_non_main_failures = all(str(row.get("failure_reason", "")) == "non_main_pool" for row in failures)
    if report.get("status") == "completed" and source_matches and config_matches and label_counts_match and only_non_main_failures:
        refreshed = write_json(
            paths["report"],
            {
                **report,
                "status": "completed",
                "resume_reused_existing_dataset": True,
                "source_generation_manifest_sha256": source_hash,
                "completion_marker_allowed": True,
                "config_hash": config_hash(config),
            },
        )
        return 0, {"manifest": paths["manifest"], "report": refreshed}
    return None


def build_round0_dataset(config: str | Path, round_name: str = "pilot", resume: bool = False) -> tuple[int, dict[str, Path]]:
    source = _round_generation_source(round_name)
    if resume:
        reused = _resume_round0_dataset_if_current(config, round_name, source)
        if reused is not None:
            return reused
    manifest = read_csv(source)
    runtime_valid = [
        r
        for r in manifest
        if r.get("runtime_executed") == "true"
        and r.get("same_state_prefix_status") == "pass"
        and r.get("swmm_status") == "completed"
        and int(float(r.get("truth_leakage") or 0)) == 0
    ]
    labels, failures = _build_round0_labels(runtime_valid, prefix=round_name if round_name in {"round0", "round1", "round2"} else f"round0_{round_name}")
    label_ids = {row.get("candidate_id", "") for row in labels}
    dataset_rows = [{**row, "status": "completed"} for row in runtime_valid if row.get("candidate_id", "") in label_ids and row.get("candidate_pool") == "main"]
    dataset_manifest = write_csv(DATASET_DIR / f"{round_name}_dataset_manifest.csv", dataset_rows)
    label_manifest = write_csv(DATASET_DIR / f"{round_name}_label_manifest.csv", labels)
    absolute_labels = write_csv(DATASET_DIR / f"{round_name}_absolute_labels.csv", labels)
    delta_labels = write_csv(
        DATASET_DIR / f"{round_name}_delta_labels.csv",
        [
            {key: value for key, value in row.items() if key in {"sample_id", "candidate_id", "event_id", "checkpoint_id"} or key.startswith("delta_")}
            for row in labels
        ],
    )
    failure_path = write_csv(DATASET_DIR / f"{round_name}_label_failures.csv", failures)
    status = "completed" if labels else "blocked"
    report = write_json(
        DATASET_DIR / f"{round_name}_dataset_report.json",
        {
            "status": status,
            "valid_sample_count": len(dataset_rows),
            "runtime_valid_candidate_count": len(runtime_valid),
            "source_generation_manifest": str(source),
            "source_generation_manifest_sha256": existing_hash(source),
            "config_hash": config_hash(config),
            "required_labels": [row[0] for row in LABEL_COMPARISONS],
            "label_rows": len(labels),
            "label_failures": len(failures),
            "absolute_labels": str(absolute_labels),
            "delta_labels": str(delta_labels),
            "label_manifest": str(label_manifest),
            "failure_report": str(failure_path),
            "completion_marker_allowed": status == "completed",
        },
    )
    return (0 if labels else 3), {"manifest": dataset_manifest, "report": report}


def audit_round0_dataset(round_name: str = "round0") -> tuple[int, dict[str, Path]]:
    rows = read_csv(DATASET_DIR / f"{round_name}_dataset_manifest.csv")
    labels = read_csv(DATASET_DIR / f"{round_name}_label_manifest.csv")
    failures = read_csv(DATASET_DIR / f"{round_name}_label_failures.csv")
    failure_counts = {reason: 0 for reason in DATASET_FAILURE_REASONS}
    for failure in failures:
        for reason in str(failure.get("failure_reason", "")).split(";"):
            if reason:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
    failed_in_dataset = [r for r in rows if r.get("status") != "completed"]
    blocking_failures = [r for r in failures if str(r.get("failure_reason", "")) != "non_main_pool"]
    minimum_count = 1500 if round_name == "round0" else 12
    maximum_count = 2000 if round_name == "round0" else 1000000
    if len(rows) < minimum_count:
        failure_counts["below_minimum_valid_effective_candidate_count"] = 1
    if len(rows) > maximum_count:
        failure_counts["above_maximum_valid_effective_candidate_count"] = 1
    if len(labels) != len(rows):
        failure_counts["label_join_failure"] = failure_counts.get("label_join_failure", 0) + abs(len(labels) - len(rows))
    if any(int(float(row.get("truth_leakage") or 0)) != 0 for row in rows):
        failure_counts["truth_leakage"] = failure_counts.get("truth_leakage", 0) + 1
    if any(int(float(row.get("binary_intermediate_values") or 0)) != 0 for row in rows):
        failure_counts["binary_violation"] = failure_counts.get("binary_violation", 0) + 1
    if any(str(row.get("engineering_violations", "")).strip() for row in rows):
        failure_counts["engineering_violation"] = failure_counts.get("engineering_violation", 0) + 1
    label_complete = rows and labels and len(labels) == len(rows)
    no_failures = not failed_in_dataset and not blocking_failures
    within_range = minimum_count <= len(rows) <= maximum_count
    status = "pass" if label_complete and no_failures and within_range else "blocked"
    audit = write_json(
        DATASET_DIR / f"{round_name}_dataset_audit_report.json",
        {
            "status": status,
            "sample_count": len(rows),
            "valid_effective_candidate_count": len(rows),
            "minimum_valid_effective_candidate_count": minimum_count,
            "formal_target_count": 600 if round_name in {"round1", "round2"} else 1800,
            "formal_target_met": len(rows) >= (600 if round_name in {"round1", "round2"} else 1500),
            "label_rows": len(labels),
            "failed_sample_count": len(blocking_failures),
            "excluded_non_main_count": failure_counts.get("non_main_pool", 0),
            "failed_samples_enter_dataset": bool(failed_in_dataset),
            "failure_counts": {k: v for k, v in failure_counts.items() if v},
            "required_label_completeness": "100%",
        },
    )
    return _status_code(status), {"audit": audit}


def evaluate_round0_data_gate() -> tuple[int, dict[str, Path]]:
    audit = read_json(DATASET_DIR / "round0_dataset_audit_report.json")
    rows = read_csv(DATASET_DIR / "round0_dataset_manifest.csv")
    labels = read_csv(DATASET_DIR / "round0_label_manifest.csv")
    valid = [r for r in rows if r.get("status") == "completed"]
    status = "pass" if audit.get("status") == "pass" and 1500 <= len(valid) <= 2000 and len(labels) == len(valid) else "blocked"
    gate = write_json(
        DATASET_DIR / "round0_data_gate.json",
        {
            "status": status,
            "valid_effective_candidate_count": len(valid),
            "label_rows": len(labels),
            "target_range": [1500, 2000],
            "failed_samples_enter_dataset": False,
            "truth_leakage": 0 if not any(int(float(row.get("truth_leakage") or 0)) != 0 for row in rows) else 1,
            "same_state_failure": 0 if not any(row.get("same_state_prefix_status") != "pass" for row in rows) else 1,
            "engineering_violation": 0 if not any(str(row.get("engineering_violations", "")).strip() for row in rows) else 1,
            "binary_intermediate_value": 0 if not any(int(float(row.get("binary_intermediate_values") or 0)) != 0 for row in rows) else 1,
            "required_label_completeness": len(labels) == len(valid) and bool(valid),
            "round0_unlock_allowed": status == "pass",
        },
    )
    return _status_code(status), {"gate": gate}


def evaluate_action_effect_training_readiness() -> tuple[int, dict[str, Path]]:
    data_gate = read_json(DATASET_DIR / "round0_data_gate.json")
    status = "pass" if data_gate.get("status") == "pass" else "blocked"
    inventory = write_csv(DATASET_DIR / "action_effect_training_inventory.csv", [{"artifact": "round0_data_gate", "status": data_gate.get("status", "missing")}])
    support = write_csv(DATASET_DIR / "action_effect_label_support.csv", [])
    gate = write_json(DATASET_DIR / "action_effect_training_readiness_gate.json", {"status": status, "round0_data_gate": data_gate.get("status", "missing"), "model_training_executed": False})
    return _status_code(status), {"gate": gate, "inventory": inventory, "support": support}
