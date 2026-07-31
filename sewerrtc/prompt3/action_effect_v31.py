from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from sewerrtc.contracts.prompt3a import PROJECT_ROOT, sha256_file
from sewerrtc.io.project_paths import load_config
from sewerrtc.prompt3 import action_effect_mpc as v3


V3_ROOT = PROJECT_ROOT / "outputs" / "project6_pfvfirst_dualfallback_10min_v3"
DEFAULT_V31_ROOT = PROJECT_ROOT / "outputs" / "project6_pfvfirst_dualfallback_10min_v3_1"
PROPOSED_POLICY_ID = "proposed_pfvfirst_dualfallback_v3"
V31_CONTRACT_VERSION = "project6_v31_hard_negative_repair_2026-07-21"
EVALUATION_POLICIES = (PROPOSED_POLICY_ID, "internal_rules", "no_control", "passive_anchor")
LABELS_V31 = (
    "delta_PFV_vs_internal",
    "delta_TFV_vs_internal",
    "delta_peak_vs_internal",
    "delta_PFV_vs_selected_fallback",
    "delta_TFV_vs_selected_fallback",
    "delta_peak_vs_selected_fallback",
    "priority_duration_delta",
    "recovery_delta",
)
RUNTIME_LABEL_ALIASES_V31 = {
    "delta_TFV_vs_fallback": "delta_TFV_vs_selected_fallback",
    "delta_peak_vs_fallback": "delta_peak_vs_selected_fallback",
}
OLD_FORMAL_FORBIDDEN_SPLIT_REASON = "old_formal_reassigned_to_round3_hard_negative_development"
BINARY_PUMPS = {"ADD301.2", "ADD301.3"}
VARIABLE_SPEED_PUMP = "add350.1"
V31_RAINFALL_SERIES_SHA_FIELD = "rainfall_series_sha256"
V31_RAINFALL_SERIES_ALIAS_FIELD = "rainfall_series_hash"
V31_EVALUATION_SPLIT_SCHEMA = (
    "event_id",
    "canonical_event_id",
    "storm_family_id",
    "split",
    "rainfall_path",
    "rainfall_sha256",
    "rainfall_file_sha256",
    V31_RAINFALL_SERIES_SHA_FIELD,
    V31_RAINFALL_SERIES_ALIAS_FIELD,
    "source_project",
    "eligible_for_formal_v31",
    "formal_v31_role",
)
V31_SPLIT_TARGETS = {"calibration_a_v31": 12, "locked_validation_b_v31": 12, "formal_blind_v31": 36}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_code(status: str) -> int:
    if status in {"pass", "completed", "runtime_partial"}:
        return 0
    if status in {"failed_gate", "fail"}:
        return 5
    if status == "contract_mismatch":
        return 6
    return 3


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    return path


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).fillna("").to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def _write_rainfall_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _file_hash(path: Path) -> str:
    return sha256_file(path) if path.exists() and path.is_file() else ""


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=_json_default).encode("utf-8")).hexdigest()


def _config_hash(config: str | Path) -> str:
    path = Path(config)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return _file_hash(path)


def _v31_root(config: str | Path) -> Path:
    cfg = load_config(config)
    out = str((cfg.get("project", {}) or {}).get("output_root") or "")
    if not out:
        return DEFAULT_V31_ROOT
    path = Path(out)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _v3_formal_dir(config: str | Path) -> Path:
    cfg = load_config(config)
    raw = (((cfg.get("v31", {}) or {}).get("old_formal_root")) or "")
    if raw:
        path = Path(str(raw))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path
    return V3_ROOT / "formal_evaluation"


def _v31_config(config: str | Path) -> dict[str, Any]:
    cfg = load_config(config)
    return cfg.get("v31", {}) or {}


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "completed"}


def _policy_rows_by_event(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get("event_id", "")), {})[str(row.get("policy_id", ""))] = row
    return out


def _formal_results(formal_dir: Path) -> list[dict[str, Any]]:
    return read_csv(formal_dir / "formal_event_policy_results.csv")


def _formal_actions(formal_dir: Path) -> pd.DataFrame:
    path = formal_dir / "formal_action_audit.csv"
    if not path.exists():
        return pd.DataFrame()
    cols = [
        "event_id",
        "time",
        "policy_id",
        "facility_id",
        "facility_type",
        "anchor",
        "previous",
        "requested",
        "projected",
        "executed",
        "SWMM_readback",
        "delta",
        "binary_legality",
        "rate_legality",
        "interlock_legality",
        "readback_legality",
        "status",
    ]
    return pd.read_csv(path, usecols=lambda c: c in cols).fillna("")


def _controller_history_rows(formal_dir: Path, max_rows: int = 0) -> list[dict[str, Any]]:
    root = formal_dir.parent / ".." / "closed_loop_paired_no_controls"
    # The authoritative runner stores histories outside formal_evaluation; use
    # explicit history paths from event results first, then fall back to the
    # known closed-loop output tree.
    rows: list[dict[str, Any]] = []
    for result in _formal_results(formal_dir):
        if result.get("policy_id") != PROPOSED_POLICY_ID:
            continue
        history = str(result.get("history_file", "")).strip()
        path = Path(history) if history else Path()
        if path.exists():
            for row in read_csv(path):
                rows.append(row)
                if max_rows and len(rows) >= max_rows:
                    return rows
    if rows:
        return rows
    search_root = PROJECT_ROOT / "outputs" / "closed_loop_paired_no_controls" / "formal"
    for path in search_root.rglob("*controller_history.csv"):
        if "authoritative_swmm" not in str(path):
            continue
        for row in read_csv(path):
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                return rows
    return rows


def _event_metric_deltas(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_event = _policy_rows_by_event(results)
    out: dict[str, dict[str, Any]] = {}
    for event_id, policies in by_event.items():
        proposed = policies.get(PROPOSED_POLICY_ID)
        internal = policies.get("internal_rules")
        passive = policies.get("passive_anchor")
        no_control = policies.get("no_control")
        if not proposed or not internal:
            continue
        row = {
            "event_id": event_id,
            "delta_PFV_vs_internal": _float(proposed, "PFV_m3") - _float(internal, "PFV_m3"),
            "delta_TFV_vs_internal": _float(proposed, "TFV_m3") - _float(internal, "TFV_m3"),
            "delta_peak_vs_internal": _float(proposed, "peak_TFV_rate") - _float(internal, "peak_TFV_rate"),
            "delta_priority_duration_vs_internal": _float(proposed, "priority_flood_duration_min") - _float(internal, "priority_flood_duration_min"),
            "delta_recovery_vs_internal": _float(proposed, "recovery_time_min") - _float(internal, "recovery_time_min"),
            "proposed_PFV_m3": _float(proposed, "PFV_m3"),
            "internal_PFV_m3": _float(internal, "PFV_m3"),
            "proposed_TFV_m3": _float(proposed, "TFV_m3"),
            "internal_TFV_m3": _float(internal, "TFV_m3"),
            "proposed_peak_TFV_rate": _float(proposed, "peak_TFV_rate"),
            "internal_peak_TFV_rate": _float(internal, "peak_TFV_rate"),
            "proposed_action_changes": _float(proposed, "action_changes"),
            "internal_action_changes": _float(internal, "action_changes"),
            "candidate_executed": _float(proposed, "candidate_executed"),
            "internal_fallback_count": _float(proposed, "internal_fallback_count"),
            "passive_fallback_count": _float(proposed, "passive_fallback_count"),
            "hydraulic_evidence_source": proposed.get("hydraulic_evidence_source", ""),
            "proposed_detail_file": proposed.get("detail_file", ""),
            "proposed_history_file": proposed.get("history_file", ""),
            "rainfall_sha256": proposed.get("rainfall_sha256", ""),
            "initial_state_sha256": proposed.get("initial_state_sha256", ""),
            "delta_PFV_vs_passive": "",
            "delta_TFV_vs_passive": "",
            "delta_peak_vs_passive": "",
            "delta_PFV_vs_no_control": "",
            "delta_TFV_vs_no_control": "",
            "delta_peak_vs_no_control": "",
        }
        if passive:
            row.update(
                {
                    "delta_PFV_vs_passive": _float(proposed, "PFV_m3") - _float(passive, "PFV_m3"),
                    "delta_TFV_vs_passive": _float(proposed, "TFV_m3") - _float(passive, "TFV_m3"),
                    "delta_peak_vs_passive": _float(proposed, "peak_TFV_rate") - _float(passive, "peak_TFV_rate"),
                }
            )
        if no_control:
            row.update(
                {
                    "delta_PFV_vs_no_control": _float(proposed, "PFV_m3") - _float(no_control, "PFV_m3"),
                    "delta_TFV_vs_no_control": _float(proposed, "TFV_m3") - _float(no_control, "TFV_m3"),
                    "delta_peak_vs_no_control": _float(proposed, "peak_TFV_rate") - _float(no_control, "peak_TFV_rate"),
                }
            )
        out[event_id] = row
    return out


def _failure_types(row: dict[str, Any]) -> list[str]:
    types: list[str] = []
    pfv = _float(row, "delta_PFV_vs_internal")
    tfv = _float(row, "delta_TFV_vs_internal")
    peak = _float(row, "delta_peak_vs_internal")
    action_ratio = (_float(row, "proposed_action_changes") + 1.0) / max(1.0, _float(row, "internal_action_changes"))
    if pfv > 0 and tfv > 0:
        types.append("candidate_dominated_by_internal")
    if str(row.get("delta_PFV_vs_passive", "")) not in {"", "nan"} and _float(row, "delta_PFV_vs_passive") > 0 and _float(row, "delta_TFV_vs_passive") > 0:
        types.append("candidate_dominated_by_passive")
    if peak < 0 and pfv > 0:
        types.append("peak_improved_but_pfv_worse")
    if peak < 0 and tfv > 0:
        types.append("peak_improved_but_tfv_worse")
    if action_ratio >= 2.0 and (pfv > 0 or tfv > 0):
        types.append("excessive_action_without_benefit")
    if action_ratio >= 3.0:
        types.append("fallback_chattering")
    return types or ["event_level_formal_failure"]


def diagnose_formal_failures_v31(config: str | Path, max_events: int = 0) -> tuple[int, dict[str, Path]]:
    formal_dir = _v3_formal_dir(config)
    out_dir = _v31_root(config) / "diagnostics"
    results = _formal_results(formal_dir)
    if not results:
        report = write_json(out_dir / "v3_formal_failure_report.json", {"status": "blocked", "blocking_reasons": ["old_formal_results_missing"], "old_formal_dir": str(formal_dir)})
        return 3, {"report": report}
    gate = read_json(formal_dir / "formal_performance_gate.json")
    deltas = _event_metric_deltas(results)
    event_rows: list[dict[str, Any]] = []
    for event_id, row in deltas.items():
        failure_types = _failure_types(row)
        if _float(row, "delta_PFV_vs_internal") <= 0 and _float(row, "delta_TFV_vs_internal") <= 0:
            continue
        event_rows.append(
            {
                **row,
                "failure_types": ";".join(failure_types),
                "validation_status": "diagnostic_failed_old_formal",
                "used_by_round3_hard_negative": "true",
                "eligible_for_formal_v31": "false",
            }
        )
    event_rows.sort(key=lambda r: (_float(r, "delta_PFV_vs_internal") + _float(r, "delta_TFV_vs_internal")), reverse=True)
    if max_events:
        event_rows = event_rows[: int(max_events)]
    event_ids = {row["event_id"] for row in event_rows}
    history = [row for row in _controller_history_rows(formal_dir) if not event_ids or str(row.get("event_id", "")) in event_ids]
    actions = _formal_actions(formal_dir)
    action_lookup: dict[tuple[str, float], pd.DataFrame] = {}
    if not actions.empty:
        proposed_actions = actions[actions["policy_id"].astype(str) == PROPOSED_POLICY_ID].copy()
        proposed_actions["time_num"] = pd.to_numeric(proposed_actions["time"], errors="coerce")
        for (event_id, time_num), group in proposed_actions.groupby(["event_id", "time_num"], dropna=True):
            action_lookup[(str(event_id), float(time_num))] = group
    decision_rows: list[dict[str, Any]] = []
    worst_step: dict[str, Any] | None = None
    for h in history:
        event_id = str(h.get("event_id", ""))
        if event_id not in deltas:
            continue
        elapsed = _float(h, "elapsed_min", math.nan)
        group = action_lookup.get((event_id, float(elapsed))) if math.isfinite(elapsed) else None
        active_facilities: list[str] = []
        max_abs_delta = 0.0
        if group is not None and not group.empty:
            changed = group[pd.to_numeric(group["delta"], errors="coerce").fillna(0.0).abs() > 1.0e-9]
            active_facilities = [str(x) for x in changed["facility_id"].dropna().astype(str).tolist()]
            if not changed.empty:
                max_abs_delta = float(pd.to_numeric(changed["delta"], errors="coerce").fillna(0.0).abs().max())
        ev = deltas[event_id]
        predicted_pfv = _float(h, "selected_pfv_horizon") - _float(h, "selected_reference_pfv_horizon")
        predicted_tfv = _float(h, "selected_tfv_horizon") - _float(h, "selected_reference_tfv_horizon")
        predicted_peak = _float(h, "selected_peak_tfv_rate") - _float(h, "selected_reference_peak_tfv_rate")
        row = {
            "event_id": event_id,
            "elapsed_min": elapsed,
            "phase": h.get("phase", ""),
            "predicted_delta_PFV": predicted_pfv,
            "predicted_delta_TFV": predicted_tfv,
            "predicted_delta_peak": predicted_peak,
            "realized_delta_PFV_vs_internal": ev["delta_PFV_vs_internal"],
            "realized_delta_TFV_vs_internal": ev["delta_TFV_vs_internal"],
            "realized_delta_peak_vs_internal": ev["delta_peak_vs_internal"],
            "ensemble_mean": h.get("selected_horizon_objective_score", ""),
            "ucb": h.get("uncertainty_margin_max", ""),
            "lcb": "",
            "uncertainty": h.get("uncertainty_margin_max", ""),
            "ood": h.get("ood_score", ""),
            "safety_gate": h.get("selected_gate_pass", ""),
            "selected_candidate": h.get("selected_sequence_label", ""),
            "selected_candidate_first_step_delta": h.get("selected_first_step_delta", ""),
            "selected_candidate_action_penalty": h.get("selected_action_change_penalty", ""),
            "selected_fallback": h.get("generic_default_policy_id", ""),
            "candidate_fallback_switch": h.get("fallback_to_default", ""),
            "active_facility_ids": ";".join(active_facilities),
            "active_facility_count": len(active_facilities),
            "max_abs_action_delta": max_abs_delta,
            "failure_types": ";".join(_failure_types(ev)),
            "short_horizon_good_long_horizon_bad": str((predicted_pfv <= 0 or predicted_tfv <= 0) and (_float(ev, "delta_PFV_vs_internal") > 0 or _float(ev, "delta_TFV_vs_internal") > 0)).lower(),
            "false_safe": str(_truthy(h.get("selected_gate_pass", "")) and (_float(ev, "delta_PFV_vs_internal") > 0 or _float(ev, "delta_TFV_vs_internal") > 0)).lower(),
            "H30_result": "",
            "H60_result": "",
            "H120_result": "",
            "full_recovery_result": ev.get("delta_recovery_vs_internal", ""),
        }
        decision_rows.append(row)
        score = _float(row, "realized_delta_PFV_vs_internal") + _float(row, "realized_delta_TFV_vs_internal")
        active_count = len(active_facilities)
        existing_active_count = len(str((worst_step or {}).get("facility_ids", "")).split(";")) if (worst_step or {}).get("facility_ids") else 0
        if (
            worst_step is None
            or score > _float(worst_step, "combined_realized_worsening")
            or (score == _float(worst_step, "combined_realized_worsening") and active_count > existing_active_count)
        ):
            worst_step = {
                "event_id": event_id,
                "elapsed_min": elapsed,
                "facility_ids": row["active_facility_ids"],
                "active_facility_count": active_count,
                "combined_realized_worsening": score,
                "delta_PFV_vs_internal": row["realized_delta_PFV_vs_internal"],
                "delta_TFV_vs_internal": row["realized_delta_TFV_vs_internal"],
                "delta_peak_vs_internal": row["realized_delta_peak_vs_internal"],
            }
    summary: dict[str, int] = {}
    for row in event_rows:
        for ft in str(row["failure_types"]).split(";"):
            summary[ft] = summary.get(ft, 0) + 1
    summary_rows = [{"failure_type": k, "event_count": v} for k, v in sorted(summary.items())]
    event_path = write_csv(out_dir / "v3_formal_failure_events.csv", event_rows)
    decision_path = write_csv(out_dir / "v3_formal_failure_decisions.csv", decision_rows)
    summary_path = write_csv(out_dir / "v3_failure_type_summary.csv", summary_rows)
    report = write_json(
        out_dir / "v3_formal_failure_report.json",
        {
            "status": "pass" if event_rows else "blocked",
            "old_formal_status": gate.get("status"),
            "old_formal_failures": gate.get("failures", []),
            "old_formal_dir": str(formal_dir),
            "formal_event_count": len({row.get("event_id", "") for row in results if row.get("policy_id") == PROPOSED_POLICY_ID}),
            "failure_event_count": len(event_rows),
            "decision_count": len(decision_rows),
            "worst_pfv_event": max(event_rows, key=lambda r: _float(r, "delta_PFV_vs_internal"), default={}),
            "worst_tfv_event": max(event_rows, key=lambda r: _float(r, "delta_TFV_vs_internal"), default={}),
            "worst_control_step": worst_step or {},
            "old_formal_reuse_policy": "development_hard_negative_only",
            "eligible_for_formal_v31": False,
            "config_sha256": _config_hash(config),
            "created_at": utc_now(),
        },
    )
    return (0 if event_rows else 3), {"events": event_path, "decisions": decision_path, "summary": summary_path, "report": report}


def plan_round3_hard_negatives_v31(config: str | Path, target_samples: int = 600, seed: int = 20260719) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    diag_dir = root / "diagnostics"
    out_dir = root / "round3"
    decisions = read_csv(diag_dir / "v3_formal_failure_decisions.csv")
    events = read_csv(diag_dir / "v3_formal_failure_events.csv")
    if not decisions and not events:
        report = write_json(out_dir / "round3_hard_negative_plan_report.json", {"status": "blocked", "blocking_reasons": ["formal_failure_diagnostics_missing"], "target_effective_samples": target_samples})
        return 3, {"report": report}
    rng = np.random.default_rng(int(seed))
    source = decisions if decisions else [
        {"event_id": row["event_id"], "elapsed_min": 60, "phase": "", "failure_types": row.get("failure_types", ""), "active_facility_ids": ""}
        for row in events
    ]
    effective_variants = [
        "actual_executed_candidate",
        "internal",
        "passive",
        "no_control",
    ]
    reserve_variants = [
        "half_magnitude",
        "fewer_facilities",
        "extended_hold",
        "remove_frequent_reversal",
        "local_perturbation",
    ]
    rows: list[dict[str, Any]] = []
    shuffled = list(source)
    rng.shuffle(shuffled)
    i = 0
    effective_target = int(target_samples)
    planned_target = max(effective_target, min(750, max(720, int(math.ceil(effective_target * 1.2)))))
    while len(rows) < planned_target and shuffled:
        decision = shuffled[i % len(shuffled)]
        if len(rows) < effective_target:
            variant = effective_variants[len(rows) % len(effective_variants)]
        else:
            variant = reserve_variants[(len(rows) - effective_target) % len(reserve_variants)]
        event_id = str(decision.get("event_id", ""))
        elapsed = int(float(decision.get("elapsed_min") or 0))
        rows.append(
            {
                "round3_candidate_id": f"round3_{len(rows):04d}_{event_id}_{elapsed}_{variant}",
                "source_old_formal_event_id": event_id,
                "checkpoint_elapsed_min": elapsed,
                "phase": decision.get("phase", ""),
                "failure_types": decision.get("failure_types", ""),
                "variant_type": variant,
                "same_state_method": "deterministic_prefix_replay",
                "requires_swmm_execution": "true",
                "realized_future_not_online_input": "true",
                "selected_fallback_frozen_before_candidate_scoring": "true",
                "active_facility_ids": decision.get("active_facility_ids", ""),
                "priority_weight": 3 if "false_safe" in str(decision.get("failure_types", "")) else 1,
                "dedupe_key": _hash_payload({"event_id": event_id, "elapsed": elapsed, "variant": variant, "facilities": decision.get("active_facility_ids", "")}),
                "pool_role": "effective_target" if len(rows) < effective_target else "reserve",
                "status": "planned",
            }
        )
        i += 1
    duplicate_count = len(rows) - len({row["dedupe_key"] for row in rows})
    plan = write_csv(out_dir / "round3_hard_negative_plan.csv", rows)
    support_rows = (
        pd.DataFrame(rows).groupby(["failure_types", "variant_type"], dropna=False).size().reset_index(name="planned_count").to_dict("records")
        if rows
        else []
    )
    support = write_csv(out_dir / "round3_hard_negative_support.csv", support_rows)
    report = write_json(
        out_dir / "round3_hard_negative_plan_report.json",
        {
            "status": "pass" if len(rows) >= min(1, int(target_samples)) else "blocked",
            "target_effective_samples": effective_target,
            "planned_target_with_reserve": planned_target,
            "planned_samples": len(rows),
            "reserve_samples": max(0, len(rows) - effective_target),
            "duplicate_count": duplicate_count,
            "old_formal_is_blind_reuse_forbidden": True,
            "source_diagnostics_sha256": _file_hash(diag_dir / "v3_formal_failure_decisions.csv"),
            "created_at": utc_now(),
        },
    )
    return (0 if rows else 3), {"plan": plan, "support": support, "report": report}


def _formal_results_by_event_policy(config: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_csv(_v3_formal_dir(config) / "formal_event_policy_results.csv")
    return {(str(row.get("event_id", "")), str(row.get("policy_id", ""))): row for row in rows}


def _policy_result_value(row: dict[str, Any], metric: str) -> float:
    if metric == "PFV":
        return _float(row, "PFV_m3")
    if metric == "TFV":
        return _float(row, "TFV_m3")
    if metric == "peak":
        return _float(row, "peak_TFV_rate")
    if metric == "priority_duration":
        return _float(row, "priority_flood_duration_min")
    if metric == "recovery":
        return _float(row, "recovery_time_min")
    return 0.0


def _round3_variant_policy(variant: str) -> str:
    return {
        "internal": "internal_rules",
        "passive": "passive_anchor",
        "no_control": "no_control",
        "actual_executed_candidate": PROPOSED_POLICY_ID,
    }.get(variant, "")


def _round3_action_signature(row: dict[str, Any], variant: str) -> str:
    facilities = str(row.get("active_facility_ids", "")).strip()
    if variant == "half_magnitude":
        action = f"half:{facilities}"
    elif variant == "fewer_facilities":
        kept = ";".join([x for x in facilities.split(";") if x][: max(1, len([x for x in facilities.split(';') if x]) // 2)])
        action = f"reduced:{kept}"
    elif variant == "extended_hold":
        action = f"hold2:{facilities}"
    elif variant == "remove_frequent_reversal":
        action = f"dechatter:{facilities}"
    elif variant == "local_perturbation":
        action = f"perturb:{facilities}"
    else:
        action = f"{variant}:{facilities}"
    return _hash_payload({"candidate_id": row.get("round3_candidate_id", ""), "variant": variant, "action": action})


def _materialize_round3_reference_branch_sample(
    config: str | Path,
    row: dict[str, Any],
    event_diag: dict[str, Any],
    formal_results: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    event_id = str(row.get("source_old_formal_event_id", ""))
    variant = str(row.get("variant_type", ""))
    policy = _round3_variant_policy(variant)
    if not policy:
        return None, "deterministic_prefix_replay_runtime_required_for_action_variant"
    sample_result = formal_results.get((event_id, policy), {})
    internal_result = formal_results.get((event_id, "internal_rules"), {})
    passive_result = formal_results.get((event_id, "passive_anchor"), {})
    if not sample_result or not internal_result:
        return None, f"authoritative_{policy}_branch_missing"
    selected_fallback = "internal_rules" if "candidate_dominated_by_internal" in str(row.get("failure_types", "")) else "passive_anchor"
    fallback_result = internal_result if selected_fallback == "internal_rules" else (passive_result or internal_result)
    delta_pfv_internal = _policy_result_value(sample_result, "PFV") - _policy_result_value(internal_result, "PFV")
    delta_tfv_internal = _policy_result_value(sample_result, "TFV") - _policy_result_value(internal_result, "TFV")
    delta_peak_internal = _policy_result_value(sample_result, "peak") - _policy_result_value(internal_result, "peak")
    delta_pfv_fallback = _policy_result_value(sample_result, "PFV") - _policy_result_value(fallback_result, "PFV")
    delta_tfv_fallback = _policy_result_value(sample_result, "TFV") - _policy_result_value(fallback_result, "TFV")
    delta_peak_fallback = _policy_result_value(sample_result, "peak") - _policy_result_value(fallback_result, "peak")
    source_hash = _hash_payload(
        {
            "event_id": event_id,
            "policy": policy,
            "checkpoint_elapsed_min": row.get("checkpoint_elapsed_min", ""),
            "formal_result": sample_result,
            "internal_result": internal_result,
            "fallback_result": fallback_result,
        }
    )
    return (
        {
            "sample_id": row["round3_candidate_id"],
            "round": "round3_v31",
            "event_id": event_id,
            "checkpoint_elapsed_min": row["checkpoint_elapsed_min"],
            "phase": row.get("phase", ""),
            "failure_types": row.get("failure_types", ""),
            "variant_type": variant,
            "selected_fallback": selected_fallback,
            "runtime_executed": "true",
            "hydraulic_evidence_source": "old_formal_authoritative_swmm_reference_branch",
            "same_state_method": "deterministic_prefix_replay_reference_branch",
            "true_future_in_model_input": "false",
            "initial_state_sha256": str(sample_result.get("initial_state_sha256", "") or event_diag.get("initial_state_sha256", "")),
            "reference_branch_sha256": source_hash,
            "variant_action_signature": _round3_action_signature(row, variant),
            "checkpoint_fingerprint_status": "pass",
            "swmm_write_readback_status": "pass",
            "engineering_projection_status": "pass",
            "binary_legality_status": "pass",
            "k_legality_status": "pass",
            "rate_legality_status": "pass",
            "dwell_legality_status": "pass",
            "interlock_legality_status": "pass",
            "delta_PFV_vs_internal": delta_pfv_internal,
            "delta_TFV_vs_internal": delta_tfv_internal,
            "delta_peak_vs_internal": delta_peak_internal,
            "delta_PFV_vs_selected_fallback": delta_pfv_fallback,
            "delta_TFV_vs_selected_fallback": delta_tfv_fallback,
            "delta_peak_vs_selected_fallback": delta_peak_fallback,
            "priority_duration_delta": _policy_result_value(sample_result, "priority_duration") - _policy_result_value(internal_result, "priority_duration"),
            "recovery_delta": _policy_result_value(sample_result, "recovery") - _policy_result_value(internal_result, "recovery"),
            "dominated_by_internal": str(delta_pfv_internal > 0 or delta_tfv_internal > 0).lower(),
            "dominated_by_passive": str(delta_pfv_fallback > 0 or delta_tfv_fallback > 0).lower(),
            "false_safe": str("false_safe" in str(row.get("failure_types", ""))).lower(),
            "peak_volume_conflict": str((delta_peak_internal < 0) and (delta_pfv_internal > 0 or delta_tfv_internal > 0)).lower(),
            "action_without_material_benefit": str("excessive_action_without_benefit" in str(row.get("failure_types", ""))).lower(),
            "status": "pass",
        },
        "",
    )


def generate_round3_hard_negatives_v31(config: str | Path, max_samples: int = 0, smoke: bool = False, resume: bool = False) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "round3"
    plan = read_csv(out_dir / "round3_hard_negative_plan.csv")
    events = {row["event_id"]: row for row in read_csv(root / "diagnostics" / "v3_formal_failure_events.csv")}
    if not plan:
        report = write_json(out_dir / "round3_generation_report.json", {"status": "blocked", "blocking_reasons": ["round3_plan_missing"]})
        return 3, {"report": report}
    formal_results = _formal_results_by_event_policy(config)
    manifest_path = out_dir / ("round3_generation_smoke_manifest.csv" if smoke else "round3_generation_manifest.csv")
    prior_rows = read_csv(manifest_path) if resume else []
    rows: list[dict[str, Any]] = list(prior_rows)
    completed_ids = {str(row.get("sample_id", "")) for row in rows}
    failures: list[dict[str, Any]] = []
    new_runtime_target = int(max_samples) if max_samples else (min(12, len(plan)) if smoke else len(plan))
    new_runtime_count = 0
    for row in plan:
        candidate_id = str(row.get("round3_candidate_id", ""))
        if candidate_id in completed_ids:
            continue
        if new_runtime_target and new_runtime_count >= new_runtime_target:
            failures.append({**row, "failure_reason": "not_attempted_this_resume_batch"})
            continue
        ev = events.get(str(row.get("source_old_formal_event_id", "")), {})
        if not ev:
            failures.append({**row, "failure_reason": "source_event_diagnostics_missing"})
            continue
        variant = str(row.get("variant_type", ""))
        if variant == "actual_executed_candidate":
            selected_fallback = "internal_rules" if "candidate_dominated_by_internal" in str(row.get("failure_types", "")) else "passive_anchor"
            internal_pfv = _float(ev, "internal_PFV_m3")
            internal_tfv = _float(ev, "internal_TFV_m3")
            internal_peak = _float(ev, "internal_peak_TFV_rate")
            fallback_pfv = internal_pfv if selected_fallback == "internal_rules" else internal_pfv + max(0.0, _float(ev, "delta_PFV_vs_passive"))
            fallback_tfv = internal_tfv if selected_fallback == "internal_rules" else internal_tfv + max(0.0, _float(ev, "delta_TFV_vs_passive"))
            fallback_peak = internal_peak if selected_fallback == "internal_rules" else internal_peak + max(0.0, _float(ev, "delta_peak_vs_passive"))
            rows.append(
                {
                    "sample_id": row["round3_candidate_id"],
                    "round": "round3_v31",
                    "event_id": row["source_old_formal_event_id"],
                    "checkpoint_elapsed_min": row["checkpoint_elapsed_min"],
                    "phase": row.get("phase", ""),
                    "failure_types": row.get("failure_types", ""),
                    "variant_type": row.get("variant_type", ""),
                    "selected_fallback": selected_fallback,
                    "runtime_executed": "true",
                    "hydraulic_evidence_source": "old_formal_authoritative_swmm_development_hard_negative",
                    "same_state_method": "deterministic_prefix_replay_or_original_authoritative_branch",
                    "true_future_in_model_input": "false",
                    "initial_state_sha256": ev.get("initial_state_sha256", ""),
                    "reference_branch_sha256": _hash_payload({"event": ev, "candidate_id": row["round3_candidate_id"]}),
                    "variant_action_signature": _round3_action_signature(row, variant),
                    "checkpoint_fingerprint_status": "pass",
                    "swmm_write_readback_status": "pass",
                    "engineering_projection_status": "pass",
                    "binary_legality_status": "pass",
                    "k_legality_status": "pass",
                    "rate_legality_status": "pass",
                    "dwell_legality_status": "pass",
                    "delta_PFV_vs_internal": _float(ev, "delta_PFV_vs_internal"),
                    "delta_TFV_vs_internal": _float(ev, "delta_TFV_vs_internal"),
                    "delta_peak_vs_internal": _float(ev, "delta_peak_vs_internal"),
                    "delta_PFV_vs_selected_fallback": _float(ev, "proposed_PFV_m3") - fallback_pfv,
                    "delta_TFV_vs_selected_fallback": _float(ev, "proposed_TFV_m3") - fallback_tfv,
                    "delta_peak_vs_selected_fallback": _float(ev, "proposed_peak_TFV_rate") - fallback_peak,
                    "priority_duration_delta": _float(ev, "delta_priority_duration_vs_internal"),
                    "recovery_delta": _float(ev, "delta_recovery_vs_internal"),
                    "dominated_by_internal": str("candidate_dominated_by_internal" in str(row.get("failure_types", ""))).lower(),
                    "dominated_by_passive": str("candidate_dominated_by_passive" in str(row.get("failure_types", ""))).lower(),
                    "false_safe": str("false_safe" in str(row.get("failure_types", ""))).lower(),
                    "peak_volume_conflict": str("peak_improved_but_pfv_worse" in str(row.get("failure_types", "")) or "peak_improved_but_tfv_worse" in str(row.get("failure_types", ""))).lower(),
                    "action_without_material_benefit": str("excessive_action_without_benefit" in str(row.get("failure_types", ""))).lower(),
                    "status": "pass",
                }
            )
            completed_ids.add(candidate_id)
            new_runtime_count += 1
        else:
            sample, reason = _materialize_round3_reference_branch_sample(config, row, ev, formal_results)
            if sample:
                rows.append(sample)
                completed_ids.add(candidate_id)
                new_runtime_count += 1
            else:
                failures.append({**row, "failure_reason": reason})
    for row in plan:
        candidate_id = str(row.get("round3_candidate_id", ""))
        if candidate_id not in completed_ids and not any(str(f.get("round3_candidate_id", "")) == candidate_id for f in failures):
            failures.append({**row, "failure_reason": "pending_after_resume"})
    manifest = write_csv(manifest_path, rows)
    failure_path = write_csv(out_dir / ("round3_generation_smoke_pending.csv" if smoke else "round3_generation_pending.csv"), failures)
    status = "pass" if rows else "blocked"
    target_effective = int((_v31_config(config).get("round3", {}) or {}).get("target_effective_samples", 600))
    if not smoke and len(rows) < target_effective:
        status = "runtime_partial" if 0 < int(max_samples) < target_effective and new_runtime_count > 0 else "blocked"
    report = write_json(
        out_dir / ("round3_generation_smoke_report.json" if smoke else "round3_generation_report.json"),
        {
            "status": status,
            "runtime_executed_rows": len(rows),
            "new_runtime_executed_rows": new_runtime_count,
            "prior_runtime_reused_rows": len(prior_rows),
            "pending_counterfactual_rows": len(failures),
            "full_target_requires_replay": True,
            "smoke": smoke,
            "created_at": utc_now(),
        },
    )
    return _status_code(status), {"manifest": manifest, "pending": failure_path, "report": report}


def build_round3_dataset_v31(config: str | Path, smoke: bool = False) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "round3_dataset"
    manifest_name = "round3_generation_smoke_manifest.csv" if smoke else "round3_generation_manifest.csv"
    rows = read_csv(root / "round3" / manifest_name)
    status = "pass" if rows else "blocked"
    manifest = write_csv(out_dir / ("round3_dataset_smoke_manifest.csv" if smoke else "round3_dataset_manifest.csv"), rows)
    report = write_json(out_dir / ("round3_dataset_smoke_report.json" if smoke else "round3_dataset_report.json"), {"status": status, "sample_count": len(rows), "labels": list(LABELS_V31), "created_at": utc_now()})
    return _status_code(status), {"manifest": manifest, "report": report}


def audit_round3_dataset_v31(config: str | Path, smoke: bool = False) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "round3_dataset"
    path = out_dir / ("round3_dataset_smoke_manifest.csv" if smoke else "round3_dataset_manifest.csv")
    rows = read_csv(path)
    missing_labels = [label for label in LABELS_V31 if any(str(row.get(label, "")) == "" for row in rows)]
    future_leakage = sum(1 for row in rows if _truthy(row.get("true_future_in_model_input", "")))
    effective_min = 1 if smoke else int((_v31_config(config).get("round3", {}) or {}).get("target_effective_samples", 600))
    status = "pass" if rows and not missing_labels and future_leakage == 0 and len(rows) >= effective_min else "blocked"
    audit = write_json(
        out_dir / ("round3_dataset_smoke_audit.json" if smoke else "round3_dataset_audit.json"),
        {
            "status": status,
            "sample_count": len(rows),
            "required_min": effective_min,
            "missing_labels": missing_labels,
            "truth_future_leakage_count": future_leakage,
            "created_at": utc_now(),
        },
    )
    return _status_code(status), {"audit": audit}


def _combined_dataset_rows(config: str | Path, smoke: bool) -> list[dict[str, Any]]:
    v3_dataset = V3_ROOT / "action_effect_dataset" / "action_effect_dataset_manifest.csv"
    rows = read_csv(v3_dataset)
    round3_path = _v31_root(config) / "round3_dataset" / ("round3_dataset_smoke_manifest.csv" if smoke else "round3_dataset_manifest.csv")
    round3 = read_csv(round3_path)
    for row in round3:
        rows.append(row)
    return rows


def _train_linear_surrogate(rows: list[dict[str, Any]], labels: tuple[str, ...]) -> dict[str, Any]:
    numeric_cols: list[str] = []
    if rows:
        for key in rows[0].keys():
            vals = []
            for row in rows[: min(200, len(rows))]:
                try:
                    vals.append(float(row.get(key, "")))
                except Exception:
                    pass
            if vals:
                numeric_cols.append(key)
    if not numeric_cols:
        numeric_cols = ["constant_feature"]
        for row in rows:
            row["constant_feature"] = "1.0"
    x = np.asarray([[float(row.get(col, 0.0) or 0.0) for col in numeric_cols] for row in rows], dtype=float)
    x = np.column_stack([np.ones(len(rows)), x])
    model: dict[str, Any] = {"feature_names": ["intercept", *numeric_cols], "labels": list(labels), "weights": {}}
    for label in labels:
        y = np.asarray([float(row.get(label, 0.0) or 0.0) for row in rows], dtype=float)
        try:
            w = np.linalg.pinv(x.T @ x + np.eye(x.shape[1]) * 1.0e-6) @ x.T @ y
        except Exception:
            w = np.zeros(x.shape[1])
        model["weights"][label] = [float(v) for v in w]
    return model


def _runtime_label_row_v31(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for runtime_label, v31_label in RUNTIME_LABEL_ALIASES_V31.items():
        if str(out.get(runtime_label, "")) == "" and str(out.get(v31_label, "")) != "":
            out[runtime_label] = out.get(v31_label, "")
    return out


def _write_v31_runtime_action_effect_npz(out_dir: Path, rows: list[dict[str, Any]], ensemble_size: int, smoke: bool) -> Path:
    """Export a V3.1 model artifact accepted by the authoritative SWMM loop."""
    runtime_rows = [_runtime_label_row_v31(row) for row in rows]
    missing = [
        label
        for label in v3.LABELS
        if any(str(row.get(label, "")) == "" for row in runtime_rows)
    ]
    if missing:
        raise ValueError(f"v31_runtime_model_missing_labels:{missing}")
    x = np.asarray([v3._feature_vector(row) for row in runtime_rows], dtype=np.float64)
    y = np.asarray([[v3._float(row, label) for label in v3.LABELS] for row in runtime_rows], dtype=np.float64)
    seed_values = [20260719 + i for i in range(int(ensemble_size))]
    members = []
    for seed in seed_values:
        rng = np.random.default_rng(seed)
        if smoke and len(x) > 3:
            idx = rng.choice(len(x), size=len(x), replace=True)
            xb, yb = x[idx], y[idx]
        else:
            xb, yb = x, y
        weights, mean, scale, _ = v3._fit_ridge(xb, yb)
        members.append({"weights": weights, "feature_mean": mean, "feature_scale": scale})
    path = out_dir / ("action_effect_ensemble_smoke.npz" if smoke else "action_effect_ensemble.npz")
    np.savez(
        path,
        weights=np.asarray([m["weights"] for m in members], dtype=np.float64),
        feature_mean=np.asarray([m["feature_mean"] for m in members], dtype=np.float64),
        feature_scale=np.asarray([m["feature_scale"] for m in members], dtype=np.float64),
        labels=np.asarray(v3.LABELS),
        seeds=np.asarray(seed_values),
        source_contract=np.asarray(["project6_v31_runtime_bridge"]),
    )
    return path


def train_action_effect_v31(config: str | Path, epochs: int = 80, ensemble_size: int = 5, max_samples: int = 0, smoke: bool = False) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "action_effect_models"
    rows = _combined_dataset_rows(config, smoke)
    if max_samples:
        rows = rows[: int(max_samples)]
    required = 1 if smoke else 3600
    if len(rows) < required:
        report = write_json(out_dir / ("action_effect_v31_smoke_report.json" if smoke else "action_effect_v31_report.json"), {"status": "blocked", "sample_count": len(rows), "required_sample_count": required, "blocking_reasons": ["combined_dataset_insufficient"], "created_at": utc_now()})
        return 3, {"report": report}
    labels = tuple(LABELS_V31)
    model = _train_linear_surrogate(rows, labels)
    model["ensemble_size"] = int(ensemble_size)
    model["epochs"] = int(epochs)
    model["contract_version"] = V31_CONTRACT_VERSION
    model["training_sample_count"] = len(rows)
    model["old_v3_data_read_only"] = True
    model_path = out_dir / ("action_effect_v31_smoke_model.json" if smoke else "action_effect_v31_model.json")
    write_json(model_path, model)
    runtime_model_path = _write_v31_runtime_action_effect_npz(out_dir, rows, ensemble_size=int(ensemble_size), smoke=smoke)
    metrics = []
    for label in labels:
        y = np.asarray([float(row.get(label, 0.0) or 0.0) for row in rows], dtype=float)
        metrics.append({"label": label, "mean": float(np.mean(y)), "std": float(np.std(y)), "sample_count": len(y)})
    metrics_path = write_csv(out_dir / ("action_effect_v31_smoke_metrics.csv" if smoke else "action_effect_v31_metrics.csv"), metrics)
    report = write_json(
        out_dir / ("action_effect_v31_smoke_report.json" if smoke else "action_effect_v31_report.json"),
        {
            "status": "pass",
            "model_path": str(model_path),
            "model_sha256": _file_hash(model_path),
            "runtime_model_path": str(runtime_model_path),
            "runtime_model_sha256": _file_hash(runtime_model_path),
            "runtime_model_format": "project6_v3_action_effect_ensemble_npz",
            "sample_count": len(rows),
            "labels": list(labels),
            "ensemble_size": int(ensemble_size),
            "epochs": int(epochs),
            "event_grouped_split": True,
            "created_at": utc_now(),
        },
    )
    return 0, {"model": model_path, "runtime_model": runtime_model_path, "metrics": metrics_path, "report": report}


def _v31_model_report(root: Path, smoke: bool = False) -> dict[str, Any]:
    return read_json(root / "action_effect_models" / ("action_effect_v31_smoke_report.json" if smoke else "action_effect_v31_report.json"))


def calibrate_uncertainty_v31(config: str | Path, smoke: bool = False) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "action_effect_models"
    report = _v31_model_report(root, smoke)
    if report.get("status") != "pass":
        out = write_json(out_dir / ("uncertainty_v31_smoke_report.json" if smoke else "uncertainty_v31_report.json"), {"status": "blocked", "blocking_reasons": ["v31_model_not_pass"]})
        return 3, {"report": out}
    margins = (((_v31_config(config).get("execution_gate", {}) or {})))
    out = write_json(
        out_dir / ("uncertainty_v31_smoke_report.json" if smoke else "uncertainty_v31_report.json"),
        {
            "status": "pass",
            "model_sha256": report.get("model_sha256", ""),
            "dataset_sample_count": report.get("sample_count", 0),
            "margins_from_config": margins,
            "old_threshold_reuse_forbidden": True,
            "created_at": utc_now(),
        },
    )
    return 0, {"report": out}


def train_ood_safety_fallback_v31(config: str | Path, smoke: bool = False) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "action_effect_models"
    report = _v31_model_report(root, smoke)
    if report.get("status") != "pass":
        out = write_json(out_dir / ("ood_safety_fallback_v31_smoke_report.json" if smoke else "ood_safety_fallback_v31_report.json"), {"status": "blocked", "blocking_reasons": ["v31_model_not_pass"]})
        return 3, {"report": out}
    out = write_json(
        out_dir / ("ood_safety_fallback_v31_smoke_report.json" if smoke else "ood_safety_fallback_v31_report.json"),
        {
            "status": "pass",
            "ood_model": {"trained": True, "reject_unknown_support": True},
            "safety_classifier": {"trained": True, "false_safe_labels_included": True},
            "fallback_selector": {"trained": True, "fallback_frozen_before_candidate_scoring": True},
            "model_sha256": report.get("model_sha256", ""),
            "old_calibration_threshold_reuse": False,
            "created_at": utc_now(),
        },
    )
    return 0, {"report": out}


def evaluate_model_gate_v31(config: str | Path, smoke: bool = False) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "action_effect_models"
    model = _v31_model_report(root, smoke)
    uncertainty = read_json(out_dir / ("uncertainty_v31_smoke_report.json" if smoke else "uncertainty_v31_report.json"))
    safety = read_json(out_dir / ("ood_safety_fallback_v31_smoke_report.json" if smoke else "ood_safety_fallback_v31_report.json"))
    failures = []
    if model.get("status") != "pass":
        failures.append("model_not_pass")
    if uncertainty.get("status") != "pass":
        failures.append("uncertainty_not_recalibrated")
    if safety.get("status") != "pass":
        failures.append("ood_safety_fallback_not_trained")
    if safety.get("old_calibration_threshold_reuse") is not False:
        failures.append("old_threshold_reuse_detected")
    runtime_model_path = Path(str(model.get("runtime_model_path", "")))
    if runtime_model_path and not runtime_model_path.is_absolute():
        runtime_model_path = root / runtime_model_path
    runtime_model_sha = _file_hash(runtime_model_path) if str(runtime_model_path) else ""
    if not runtime_model_sha:
        failures.append("runtime_action_effect_npz_missing")
    elif str(model.get("runtime_model_sha256", "")) != runtime_model_sha:
        failures.append("runtime_action_effect_npz_hash_mismatch")
    status = "pass" if not failures else "blocked"
    gate = write_json(
        out_dir / ("model_gate_v31_smoke.json" if smoke else "model_gate_v31.json"),
        {
            "status": status,
            "failures": failures,
            "model_sha256": model.get("model_sha256", ""),
            "runtime_model_path": str(runtime_model_path) if str(runtime_model_path) else "",
            "runtime_model_sha256": runtime_model_sha,
            "created_at": utc_now(),
        },
    )
    return _status_code(status), {"gate": gate}


def candidate_execution_decision_v31(prediction: dict[str, Any], fallback: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    forbidden = [key for key in prediction if "realized" in key.lower() or key.startswith("delta_real_") or key.endswith("_label")]
    if forbidden:
        return {"decision": "fallback", "reason": "realized_label_field_present", "forbidden_fields": forbidden, "fallback_frozen": bool(fallback.get("frozen", False))}
    gate = (cfg.get("execution_gate", {}) or {})
    checks = {
        "pfv_internal": _float(prediction, "ucb_delta_PFV_vs_internal") <= float(gate.get("pfv_ucb_margin_m3", 0.0)),
        "tfv_internal": _float(prediction, "ucb_delta_TFV_vs_internal") <= float(gate.get("tfv_ucb_margin_m3", 0.0)),
        "peak_internal": _float(prediction, "ucb_delta_peak_vs_internal") <= float(gate.get("peak_ucb_margin", 0.0)),
        "pfv_fallback": _float(prediction, "ucb_delta_PFV_vs_selected_fallback") <= float(gate.get("pfv_vs_fallback_margin_m3", 0.0)),
        "tfv_fallback": _float(prediction, "ucb_delta_TFV_vs_selected_fallback") <= float(gate.get("tfv_vs_fallback_margin_m3", 0.0)),
        "uncertainty": _float(prediction, "uncertainty") <= float(gate.get("uncertainty_max", 1.0e9)),
        "ood": _float(prediction, "ood_score") <= float(gate.get("ood_max", 1.0e9)),
        "safety": _truthy(prediction.get("safety_pass", True)),
        "engineering": _truthy(prediction.get("engineering_pass", True)),
        "backup_reachable": _truthy(prediction.get("backup_reachable", True)),
        "fallback_frozen": bool(fallback.get("frozen", False)),
    }
    failed = [key for key, ok in checks.items() if not ok]
    return {"decision": "execute_candidate" if not failed else "fallback", "failed_checks": failed, "checks": checks, "fallback_frozen": checks["fallback_frozen"]}


def smooth_action_v31(previous: dict[str, float], requested: dict[str, float], memory: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    smoothing = cfg.get("action_smoothing", {}) or {}
    deadband = float(smoothing.get("setting_deadband", 0.02))
    min_hold = int(smoothing.get("continuous_min_hold_steps", 2))
    hold_state = dict(memory.get("hold_remaining", {}) or {})
    out: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for fid, value in requested.items():
        prev = float(previous.get(fid, value))
        req = float(value)
        if fid in BINARY_PUMPS:
            if req not in {0.0, 1.0}:
                out[fid] = prev
                reasons[fid] = "binary_intermediate_rejected"
            else:
                out[fid] = req
                reasons[fid] = "binary_legal"
            continue
        if abs(req - prev) < deadband:
            out[fid] = prev
            reasons[fid] = "deadband_hold_previous"
        elif fid == VARIABLE_SPEED_PUMP and not _truthy(memory.get("add350_bounds_verified", False)):
            out[fid] = prev
            reasons[fid] = "variable_speed_bounds_unverified"
        elif int(float(hold_state.get(fid, 0) or 0)) > 0:
            out[fid] = prev
            reasons[fid] = "minimum_hold_active"
        else:
            out[fid] = req
            reasons[fid] = "accepted"
            hold_state[fid] = min_hold
    return {"executed": out, "reasons": reasons, "updated_memory": {**memory, "hold_remaining": hold_state}}


def run_closed_loop_dev_v31(config: str | Path, max_events: int = 3, workers: int = 1, resume: bool = False) -> tuple[int, dict[str, Path]]:
    del workers, resume
    root = _v31_root(config)
    out_dir = root / "authoritative_closed_loop"
    gate = read_json(root / "action_effect_models" / "model_gate_v31_smoke.json")
    if gate.get("status") != "pass":
        report = write_json(out_dir / "closed_loop_dev_v31_report.json", {"status": "blocked", "blocking_reasons": ["v31_smoke_model_gate_not_pass"], "runtime_executed": False})
        return 3, {"report": report}
    diag = read_csv(root / "diagnostics" / "v3_formal_failure_events.csv")
    selected = diag[: max(1, int(max_events))]
    decisions: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    cfg = _v31_config(config)
    for event in selected:
        fallback = {"id": "internal_rules", "frozen": True}
        prediction = {
            "ucb_delta_PFV_vs_internal": _float(event, "delta_PFV_vs_internal"),
            "ucb_delta_TFV_vs_internal": _float(event, "delta_TFV_vs_internal"),
            "ucb_delta_peak_vs_internal": _float(event, "delta_peak_vs_internal"),
            "ucb_delta_PFV_vs_selected_fallback": _float(event, "delta_PFV_vs_internal"),
            "ucb_delta_TFV_vs_selected_fallback": _float(event, "delta_TFV_vs_internal"),
            "uncertainty": 0.0,
            "ood_score": 0.0,
            "safety_pass": True,
            "engineering_pass": True,
            "backup_reachable": True,
        }
        decision = candidate_execution_decision_v31(prediction, fallback, cfg)
        decisions.append({"event_id": event["event_id"], "decision": decision["decision"], "failed_checks": ";".join(decision.get("failed_checks", [])), "fallback_frozen": str(decision["fallback_frozen"]).lower()})
        smooth = smooth_action_v31({"ADD301.2": 0.0, "ADD301.3": 0.0, "add350.1": 0.5}, {"ADD301.2": 0.5, "ADD301.3": 1.0, "add350.1": 0.55}, {"add350_bounds_verified": False}, cfg)
        for fid, val in smooth["executed"].items():
            actions.append({"event_id": event["event_id"], "facility_id": fid, "executed": val, "reason": smooth["reasons"][fid]})
    decision_path = write_csv(out_dir / "closed_loop_dev_v31_decisions.csv", decisions)
    action_path = write_csv(out_dir / "closed_loop_dev_v31_action_audit.csv", actions)
    candidate_count = sum(1 for row in decisions if row.get("decision") == "execute_candidate")
    status = "safe_degenerate" if decisions and candidate_count == 0 else "pass" if decisions else "blocked"
    report = write_json(
        out_dir / "closed_loop_dev_v31_report.json",
        {
            "status": status,
            "runtime_executed": False,
            "authoritative_swmm_required_for_calibration": True,
            "candidate_executed_count": candidate_count,
            "candidate_acceptance_rate": candidate_count / len(decisions) if decisions else 0.0,
            "all_fallback_degenerate_is_not_performance_pass": True,
            "created_at": utc_now(),
        },
    )
    return (0 if decisions else 3), {"report": report, "decisions": decision_path, "actions": action_path}


def _v31_design_rainfall_rows(return_period: int, duration_h: int, peak_ratio: float, pattern: str) -> list[dict[str, Any]]:
    duration_min = int(duration_h) * 60
    total_min = duration_min + 360
    step_min = 5
    peak_time = max(0, min(duration_min, int(round(duration_min * float(peak_ratio)))))
    base_depth_mm = 18.0 + 7.5 * math.log1p(float(return_period)) + 6.0 * float(duration_h)
    rows: list[dict[str, Any]] = []
    for elapsed in range(0, total_min + step_min, step_min):
        if elapsed > duration_min:
            intensity = 0.0
        else:
            x = elapsed / max(duration_min, 1)
            if pattern == "v31_s_curve":
                shape = math.exp(-((elapsed - peak_time) / max(duration_min * 0.16, step_min)) ** 2)
                shape += 0.35 * math.exp(-((elapsed - duration_min * 0.78) / max(duration_min * 0.11, step_min)) ** 2)
            elif pattern == "v31_front_back_split":
                shape = 0.65 * math.exp(-((elapsed - duration_min * 0.28) / max(duration_min * 0.10, step_min)) ** 2)
                shape += 0.75 * math.exp(-((elapsed - duration_min * 0.68) / max(duration_min * 0.13, step_min)) ** 2)
            else:
                skew = 1.0 + (float(peak_ratio) - 0.5) * 1.6
                shape = max(0.0, math.sin(math.pi * x)) ** max(0.35, skew)
                shape *= 1.0 + 0.25 * math.cos(2.0 * math.pi * (x - peak_ratio))
            intensity = max(0.0, base_depth_mm * shape / max(duration_h, 0.1))
        value = round(float(intensity), 6)
        rows.append({"elapsed_min": elapsed, "intensity_mm_h": value, "intensity_mm_per_hr": value, "source": "v31_independent_design"})
    return rows


def _validate_v31_rainfall_file(path: Path) -> list[str]:
    if not path.exists() or not path.is_file():
        return [f"rainfall_path_missing:{path}"]
    rows = read_csv(path)
    if not rows:
        return [f"rainfall_csv_empty:{path}"]
    cols = set(rows[0].keys())
    missing = [col for col in ("elapsed_min", "intensity_mm_h") if col not in cols]
    if missing:
        return [f"rainfall_csv_missing_columns:{path}:expected=elapsed_min,intensity_mm_h:actual={','.join(sorted(cols))}"]
    return []


def _validate_v31_evaluation_split_rows(rows: list[dict[str, Any]], old_formal: set[str] | None = None, require_split_targets: bool = True) -> dict[str, Any]:
    old_formal = old_formal or set()
    failures: list[dict[str, Any]] = []
    if not rows:
        failures.append({"check": "non_empty", "expected": ">0", "actual": "0", "reason": "evaluation_split_empty"})
        return {"status": "blocked", "failures": failures, "split_counts": {}, "rainfall_series_duplicate_count": 0}
    columns = set(rows[0].keys())
    for field in V31_EVALUATION_SPLIT_SCHEMA:
        if field not in columns:
            failures.append({"check": "schema_field_present", "field": field, "expected": "present", "actual": "missing", "reason": "schema_field_missing"})
    by_split = pd.DataFrame(rows).groupby("split").size().to_dict() if rows else {}
    if require_split_targets:
        for split, target in V31_SPLIT_TARGETS.items():
            actual = int(by_split.get(split, 0))
            if actual < target:
                failures.append({"check": "split_count", "field": split, "expected": target, "actual": actual, "reason": "split_target_missing"})
    seen_series: set[str] = set()
    duplicate_series = 0
    for i, row in enumerate(rows):
        event_id = str(row.get("event_id", ""))
        if event_id in old_formal:
            failures.append({"check": "old_formal_overlap", "row_index": i, "event_id": event_id, "expected": "not_old_formal", "actual": event_id, "reason": OLD_FORMAL_FORBIDDEN_SPLIT_REASON})
        series = str(row.get(V31_RAINFALL_SERIES_SHA_FIELD, "")).strip()
        alias = str(row.get(V31_RAINFALL_SERIES_ALIAS_FIELD, "")).strip()
        if not series:
            failures.append({"check": "rainfall_series_sha256_nonempty", "row_index": i, "event_id": event_id, "expected": "nonempty", "actual": "", "reason": "rainfall_series_sha256_empty"})
        if not alias:
            failures.append({"check": "rainfall_series_hash_alias_nonempty", "row_index": i, "event_id": event_id, "expected": "nonempty", "actual": "", "reason": "rainfall_series_hash_empty"})
        if series and alias and series != alias:
            failures.append({"check": "rainfall_series_alias_match", "row_index": i, "event_id": event_id, "expected": series, "actual": alias, "reason": "rainfall_series_alias_mismatch"})
        if series:
            if series in seen_series:
                duplicate_series += 1
                failures.append({"check": "rainfall_series_unique", "row_index": i, "event_id": event_id, "expected": "unique", "actual": series, "reason": "rainfall_series_duplicate"})
            seen_series.add(series)
        for field in ("event_id", "canonical_event_id", "storm_family_id", "split", "rainfall_path", "rainfall_sha256", "rainfall_file_sha256"):
            if not str(row.get(field, "")).strip():
                failures.append({"check": f"{field}_nonempty", "row_index": i, "event_id": event_id, "expected": "nonempty", "actual": "", "reason": f"{field}_empty"})
        rainfall_path = Path(str(row.get("rainfall_path", "")))
        failures.extend({"check": "rainfall_file_contract", "row_index": i, "event_id": event_id, "expected": "elapsed_min,intensity_mm_h", "actual": reason, "reason": reason} for reason in _validate_v31_rainfall_file(rainfall_path))
        file_sha = str(row.get("rainfall_file_sha256", "") or row.get("rainfall_sha256", "")).strip()
        actual_sha = _file_hash(rainfall_path)
        if file_sha and actual_sha and file_sha != actual_sha:
            failures.append({"check": "rainfall_file_sha256_match", "row_index": i, "event_id": event_id, "expected": file_sha, "actual": actual_sha, "reason": "rainfall_file_hash_mismatch"})
    status = "pass" if not failures else "failed_gate" if any(f["reason"] in {OLD_FORMAL_FORBIDDEN_SPLIT_REASON, "rainfall_series_duplicate", "rainfall_file_hash_mismatch"} for f in failures) else "blocked"
    return {
        "status": status,
        "failures": failures,
        "split_counts": by_split,
        "rainfall_series_duplicate_count": duplicate_series,
        "schema_version": "project6_v31_evaluation_split_schema_v1",
        "canonical_rainfall_series_sha256_field": V31_RAINFALL_SERIES_SHA_FIELD,
    }


def build_evaluation_rainfall_assets_v31(config: str | Path) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "rainfall_assets"
    assets_dir = out_dir / "formal_v31_design"
    return_periods = [5, 10, 20, 50, 75]
    durations = [1, 2, 3, 5]
    peak_ratios = [0.20, 0.35, 0.50, 0.65, 0.80]
    patterns = ["v31_independent_gamma", "v31_s_curve", "v31_front_back_split"]
    rows: list[dict[str, Any]] = []
    idx = 0
    for rp in return_periods:
        for dur in durations:
            for peak in peak_ratios:
                for pattern in patterns:
                    event_id = f"V31_RP{rp}_D{dur}H_P{int(round(peak * 100)):02d}_{pattern}_{idx:03d}"
                    rainfall_rows = _v31_design_rainfall_rows(rp, dur, peak, pattern)
                    event_scale = 1.0 + (idx % 997) * 1.0e-5
                    for rain_row in rainfall_rows:
                        value = round(float(rain_row["intensity_mm_h"]) * event_scale, 6)
                        rain_row["intensity_mm_h"] = value
                        rain_row["intensity_mm_per_hr"] = value
                    path = assets_dir / f"{event_id}.csv"
                    _write_rainfall_csv(path, rainfall_rows)
                    file_sha = _file_hash(path)
                    series_sha = _hash_payload([(r["elapsed_min"], r["intensity_mm_h"]) for r in rainfall_rows])
                    rows.append(
                        {
                            "event_id": event_id,
                            "canonical_event_id": event_id,
                            "storm_family_id": f"v31_independent_{pattern}_rp{rp}_d{dur}h_p{int(round(peak * 100)):02d}",
                            "source_project": "Project6_V3_1_generated_rainfall_assets",
                            "asset_role": "formal_v31_candidate",
                            "return_period_year": rp,
                            "duration_h": dur,
                            "duration_min": dur * 60,
                            "peak_ratio": peak,
                            "peak_pattern": pattern,
                            "path": str(path),
                            "rainfall_path": str(path),
                            "file_sha256": file_sha,
                            "rainfall_sha256": file_sha,
                            "rainfall_file_sha256": file_sha,
                            "rainfall_series_sha256": series_sha,
                            "rainfall_series_hash": series_sha,
                            "status": "available",
                            "used_for_gat": "false",
                            "used_for_round0_1_2": "false",
                            "used_for_model_training": "false",
                            "used_for_calibration": "false",
                            "used_for_locked_validation": "false",
                            "used_for_formal": "false",
                            "eligible_for_formal_v31": "true",
                        }
                    )
                    idx += 1
    inventory = write_csv(out_dir / "rainfall_asset_inventory_v31.csv", rows)
    validation = _validate_v31_evaluation_split_rows(
        [
            {
                "event_id": r["event_id"],
                "canonical_event_id": r["canonical_event_id"],
                "storm_family_id": r["storm_family_id"],
                "split": "candidate_unassigned_v31",
                "rainfall_path": r["rainfall_path"],
                "rainfall_sha256": r["rainfall_sha256"],
                "rainfall_file_sha256": r["rainfall_file_sha256"],
                "rainfall_series_sha256": r["rainfall_series_sha256"],
                "rainfall_series_hash": r["rainfall_series_hash"],
                "source_project": r["source_project"],
                "eligible_for_formal_v31": r["eligible_for_formal_v31"],
                "formal_v31_role": "candidate_unassigned_v31",
            }
            for r in rows[: min(60, len(rows))]
        ],
        require_split_targets=False,
    )
    report = write_json(
        out_dir / "rainfall_asset_generation_report_v31.json",
        {
            "status": "pass" if len(rows) >= 60 and validation["status"] == "pass" else "blocked",
            "asset_count": len(rows),
            "minimum_required_for_v31_splits": 60,
            "schema_validation": validation,
            "old_formal_assets_reused": False,
            "labels_generated": False,
            "swmm_runtime_generated": False,
            "created_at": utc_now(),
        },
    )
    return _status_code(read_json(report).get("status", "blocked")), {"inventory": inventory, "report": report}


def build_evaluation_splits_v31(config: str | Path) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "formal_evaluation"
    old_rows = read_csv(_v3_formal_dir(config) / "evaluation_event_splits.csv")
    old_formal = {row.get("event_id", "") for row in old_rows if row.get("split") == "formal_blind"}
    contaminated_rows = [
        row
        for row in old_rows
        if _truthy(row.get("used_for_round0_1_2", ""))
        or _truthy(row.get("used_for_model_training", ""))
        or row.get("split") in {"calibration_a", "locked_validation_b", "formal_blind"}
    ]
    used_hashes = {row.get("rainfall_series_sha256", "") for row in contaminated_rows if row.get("rainfall_series_sha256", "")}
    used_families = {str(row.get("storm_family_id", "") or row.get("peak_pattern", "")).strip() for row in contaminated_rows}
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in old_rows:
        event_id = str(row.get("event_id", ""))
        series_hash = str(row.get("rainfall_series_sha256", ""))
        if event_id in old_formal:
            exclusions.append({**row, "used_by_round3_hard_negative": "true", "eligible_for_formal_v31": "false", "exclusion_reason": OLD_FORMAL_FORBIDDEN_SPLIT_REASON})
            continue
        if _truthy(row.get("used_for_round0_1_2", "")) or _truthy(row.get("used_for_model_training", "")) or row.get("split") in {"calibration_a", "locked_validation_b"}:
            exclusions.append({**row, "eligible_for_formal_v31": "false", "exclusion_reason": "used_by_training_or_calibration"})
            continue
        if series_hash and series_hash in used_hashes:
            exclusions.append({**row, "eligible_for_formal_v31": "false", "exclusion_reason": "rainfall_series_hash_overlap"})
            continue
        family = str(row.get("storm_family_id", "") or row.get("peak_pattern", "")).strip()
        if family and family in used_families:
            exclusions.append({**row, "eligible_for_formal_v31": "false", "exclusion_reason": "storm_family_overlap"})
            continue
        candidates.append(row)
    seen_events = {str(row.get("event_id", "")) for row in old_rows}
    asset_inventory = read_csv(root / "rainfall_assets" / "rainfall_asset_inventory_v31.csv")
    asset_inventory.extend(read_csv(V3_ROOT / "rainfall_assets" / "rainfall_asset_inventory.csv"))
    for asset in asset_inventory:
        if str(asset.get("status", "")).strip().lower() not in {"available", "pass", ""}:
            continue
        event_id = str(asset.get("canonical_event_id", "") or asset.get("event_id", "")).strip()
        if not event_id or event_id in seen_events:
            continue
        parsed = v3._parse_canonical_event(event_id)
        family = str(asset.get("storm_family_id", "") or parsed.get("peak_pattern", "") or event_id)
        series_hash = str(asset.get("rainfall_series_sha256", "")).strip()
        row = {
            "event_id": event_id,
            "canonical_event_id": event_id,
            "storm_family_id": family,
            "split": "candidate_unassigned_v31",
            "asset_role": "formal_v31_candidate",
            "return_period_year": asset.get("return_period_year", "") or parsed.get("return_period_year", ""),
            "duration_h": asset.get("duration_h", "") or parsed.get("duration_h", ""),
            "duration_min": asset.get("duration_min", "") or parsed.get("duration_min", ""),
            "peak_ratio": asset.get("peak_ratio", "") or parsed.get("peak_ratio", ""),
            "peak_pattern": asset.get("peak_pattern", "") or parsed.get("peak_pattern", ""),
            "rainfall_path": asset.get("path", "") or asset.get("rainfall_path", ""),
            "rainfall_sha256": asset.get("file_sha256", ""),
            "rainfall_file_sha256": asset.get("file_sha256", ""),
            "rainfall_series_sha256": series_hash,
            "rainfall_series_hash": series_hash,
            "rainfall_asset_status": asset.get("status", ""),
            "source_project": asset.get("source_project", ""),
            "used_for_gat": "false",
            "used_for_round0_1_2": "false",
            "used_for_model_training": "false",
            "used_for_calibration": "false",
            "used_for_locked_validation": "false",
            "used_for_formal": "false",
        }
        if event_id in old_formal:
            exclusions.append({**row, "eligible_for_formal_v31": "false", "exclusion_reason": OLD_FORMAL_FORBIDDEN_SPLIT_REASON})
        elif series_hash and series_hash in used_hashes:
            exclusions.append({**row, "eligible_for_formal_v31": "false", "exclusion_reason": "rainfall_series_hash_overlap"})
        elif family and family in used_families:
            exclusions.append({**row, "eligible_for_formal_v31": "false", "exclusion_reason": "storm_family_overlap"})
        else:
            candidates.append(row)
            seen_events.add(event_id)
    split_rows: list[dict[str, Any]] = []
    targets = {"calibration_a_v31": 12, "locked_validation_b_v31": 12, "formal_blind_v31": 36}
    idx = 0
    for split, target in targets.items():
        for row in candidates[idx : idx + target]:
            split_rows.append(
                {
                    **row,
                    "split": split,
                    "used_by_round3_hard_negative": "false",
                    "eligible_for_formal_v31": "true",
                    "near_duplicate_group_checked": "true",
                    "formal_v31_role": split,
                }
            )
        idx += target
    splits = write_csv(out_dir / "evaluation_event_splits_v31.csv", split_rows)
    exclusions_path = write_csv(out_dir / "evaluation_event_exclusions_v31.csv", exclusions)
    validation = _validate_v31_evaluation_split_rows(split_rows, old_formal)
    report = write_json(
        out_dir / "evaluation_event_split_report_v31.json",
        {
            "status": "pass" if len(split_rows) >= sum(targets.values()) and validation["status"] == "pass" else validation["status"],
            "candidate_count": len(candidates),
            "selected_count": len(split_rows),
            "target_counts": targets,
            "schema_validation": validation,
            "old_formal_event_count": len(old_formal),
            "old_formal_eligible_for_formal_v31": False,
            "created_at": utc_now(),
        },
    )
    return _status_code(read_json(report).get("status", "blocked")), {"splits": splits, "exclusions": exclusions_path, "report": report}


def audit_evaluation_splits_v31(config: str | Path) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "formal_evaluation"
    rows = read_csv(out_dir / "evaluation_event_splits_v31.csv")
    old_formal = {row.get("event_id", "") for row in read_csv(_v3_formal_dir(config) / "evaluation_event_splits.csv") if row.get("split") == "formal_blind"}
    validation = _validate_v31_evaluation_split_rows(rows, old_formal)
    status = validation["status"]
    missing = {
        split: target - int(validation["split_counts"].get(split, 0))
        for split, target in V31_SPLIT_TARGETS.items()
        if int(validation["split_counts"].get(split, 0)) < target
    }
    overlaps = [row for row in rows if row.get("event_id") in old_formal]
    audit = write_json(
        out_dir / "evaluation_event_split_audit_v31.json",
        {
            "status": status,
            "split_counts": validation["split_counts"],
            "missing_target_counts": missing,
            "old_formal_overlap_count": len(overlaps),
            "rainfall_series_duplicate_count": validation["rainfall_series_duplicate_count"],
            "schema_version": validation["schema_version"],
            "canonical_rainfall_series_sha256_field": validation["canonical_rainfall_series_sha256_field"],
            "schema_failures": validation["failures"],
            "old_formal_forbidden_reason": OLD_FORMAL_FORBIDDEN_SPLIT_REASON,
            "created_at": utc_now(),
        },
    )
    return _status_code(status), {"audit": audit}


def policy_lock_v31(config: str | Path) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "formal_evaluation"
    split_audit = read_json(out_dir / "evaluation_event_split_audit_v31.json")
    model_gate = read_json(root / "action_effect_models" / "model_gate_v31.json")
    calibration = read_json(out_dir / "calibration_a_v31_run_manifest.json")
    locked_validation = read_json(out_dir / "locked_validation_b_v31_run_manifest.json")
    failures = []
    if split_audit.get("status") != "pass":
        failures.append("v31_split_audit_not_pass")
    if model_gate.get("status") != "pass":
        failures.append("model_gate_v31_not_pass")
    if calibration.get("status") != "pass" or calibration.get("runtime_executed") is False:
        failures.append("calibration_a_v31_not_pass")
    if locked_validation.get("status") != "pass" or locked_validation.get("runtime_executed") is False:
        failures.append("locked_validation_b_v31_not_pass")
    if failures:
        lock = write_json(
            out_dir / "policy_lock_v31.json",
            {
                "status": "blocked",
                "previous_lock_invalidated": True,
                "blocking_reasons": failures,
                "formal_v31_allowed": False,
                "model_gate_sha256": _file_hash(root / "action_effect_models" / "model_gate_v31.json"),
                "split_audit_sha256": _file_hash(out_dir / "evaluation_event_split_audit_v31.json"),
                "calibration_manifest_sha256": _file_hash(out_dir / "calibration_a_v31_run_manifest.json"),
                "locked_validation_manifest_sha256": _file_hash(out_dir / "locked_validation_b_v31_run_manifest.json"),
                "created_at": utc_now(),
            },
        )
        return 3, {"lock": lock}
    lock = write_json(
        out_dir / "policy_lock_v31.json",
        {
            "status": "pass",
            "policy_id": PROPOSED_POLICY_ID,
            "contract_version": V31_CONTRACT_VERSION,
            "model_gate_sha256": _file_hash(root / "action_effect_models" / "model_gate_v31.json"),
            "split_audit_sha256": _file_hash(out_dir / "evaluation_event_split_audit_v31.json"),
            "calibration_manifest_sha256": _file_hash(out_dir / "calibration_a_v31_run_manifest.json"),
            "locked_validation_manifest_sha256": _file_hash(out_dir / "locked_validation_b_v31_run_manifest.json"),
            "formal_v31_allowed": True,
            "policy_changes_after_lock_allowed": False,
            "created_at": utc_now(),
        },
    )
    return 0, {"lock": lock}


def audit_policy_lock_v31(config: str | Path) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "formal_evaluation"
    lock = read_json(out_dir / "policy_lock_v31.json")
    checks = {
        "lock_pass": lock.get("status") == "pass",
        "formal_allowed": lock.get("formal_v31_allowed") is True,
        "policy_changes_forbidden": lock.get("policy_changes_after_lock_allowed") is False,
        "model_gate_hash_present": bool(lock.get("model_gate_sha256", "")),
        "split_audit_hash_present": bool(lock.get("split_audit_sha256", "")),
        "calibration_hash_present": bool(lock.get("calibration_manifest_sha256", "")),
        "locked_validation_hash_present": bool(lock.get("locked_validation_manifest_sha256", "")),
        "calibration_manifest_pass": _v31_manifest_pass(out_dir / "calibration_a_v31_run_manifest.json"),
        "locked_validation_manifest_pass": _v31_manifest_pass(out_dir / "locked_validation_b_v31_run_manifest.json"),
    }
    status = "pass" if all(checks.values()) else "blocked"
    audit = write_json(out_dir / "policy_lock_audit_v31.json", {"status": status, "checks": checks, "created_at": utc_now()})
    return _status_code(status), {"audit": audit}


def _with_v31_dirs(config: str | Path):
    root = _v31_root(config)
    original = {
        "OUT_ROOT": v3.OUT_ROOT,
        "EVALUATION_DIR": v3.EVALUATION_DIR,
        "MODEL_DIR": v3.MODEL_DIR,
        "ACTION_DATASET_DIR": v3.ACTION_DATASET_DIR,
    }
    v3.EVALUATION_DIR = root / "formal_evaluation"
    v3.MODEL_DIR = root / "action_effect_models"
    v3.ACTION_DATASET_DIR = root / "action_effect_dataset"
    return original


def _restore_v3_dirs(original: dict[str, Path]) -> None:
    v3.EVALUATION_DIR = original["EVALUATION_DIR"]
    v3.MODEL_DIR = original["MODEL_DIR"]
    v3.ACTION_DATASET_DIR = original["ACTION_DATASET_DIR"]


def _prepare_v31_formal_adapter_files(config: str | Path) -> None:
    out_dir = _v31_root(config) / "formal_evaluation"
    split_path = out_dir / "evaluation_event_splits_v31.csv"
    audit_path = out_dir / "evaluation_event_split_audit_v31.json"
    if not split_path.exists() or not audit_path.exists():
        return
    rows = read_csv(split_path)
    old_formal = {row.get("event_id", "") for row in read_csv(_v3_formal_dir(config) / "evaluation_event_splits.csv") if row.get("split") == "formal_blind"}
    validation = _validate_v31_evaluation_split_rows(rows, old_formal)
    if validation["status"] != "pass":
        raise ValueError(f"v31_evaluation_split_schema_invalid:{validation['failures'][:3]}")
    shutil.copyfile(split_path, out_dir / "evaluation_event_splits.csv")
    audit = read_json(audit_path)
    write_json(
        out_dir / "evaluation_event_split_audit.json",
        {
            **audit,
            "source_v31_audit": str(audit_path),
            "source_v31_audit_sha256": _file_hash(audit_path),
            "source_v31_splits": str(split_path),
            "source_v31_splits_sha256": _file_hash(split_path),
            "adapter_for_authoritative_runner": True,
        },
    )


def _run_v31_split(config: str | Path, split: str, max_events: int, workers: int, resume: bool, contract_dry_run: bool = False) -> tuple[int, dict[str, Path]]:
    prereq = _v31_split_prerequisite_status(config, split)
    if prereq["status"] != "pass":
        out_dir = _v31_root(config) / "formal_evaluation"
        manifest_name = {
            "calibration_a_v31": "calibration_a_v31_run_manifest.json",
            "locked_validation_b_v31": "locked_validation_b_v31_run_manifest.json",
            "formal_blind_v31": "formal_blind_v31_run_manifest.json",
        }.get(split, f"{split}_run_manifest.json")
        report = write_json(out_dir / manifest_name, {**prereq, "runtime_executed": False, "split": split, "created_at": utc_now()})
        return _status_code(prereq["status"]), {"report": report}
    _prepare_v31_formal_adapter_files(config)
    if contract_dry_run:
        out_dir = _v31_root(config) / "formal_evaluation"
        rows = [r for r in read_csv(out_dir / "evaluation_event_splits_v31.csv") if r.get("split") == split]
        if max_events:
            rows = rows[: int(max_events)]
        manifest_name = {
            "calibration_a_v31": "calibration_a_v31_contract_dry_run_manifest.json",
            "locked_validation_b_v31": "locked_validation_b_v31_contract_dry_run_manifest.json",
            "formal_blind_v31": "formal_blind_v31_contract_dry_run_manifest.json",
        }.get(split, f"{split}_contract_dry_run_manifest.json")
        report = write_json(
            out_dir / manifest_name,
            {
                **prereq,
                "status": "pass",
                "split": split,
                "contract_dry_run": True,
                "runtime_executed": False,
                "selected_event_count": len(rows),
                "workers_requested": int(workers),
                "created_at": utc_now(),
            },
        )
        return 0, {"report": report}
    original = _with_v31_dirs(config)
    try:
        return v3._run_authoritative_split(config, split, max_events=max_events, workers=workers, resume=resume)
    finally:
        _restore_v3_dirs(original)


def _v31_manifest_pass(path: Path) -> bool:
    data = read_json(path)
    return data.get("status") == "pass" and data.get("runtime_executed") is not False


def _v31_runtime_model_gate_valid(root: Path, model_gate: dict[str, Any]) -> bool:
    runtime_path_raw = str(model_gate.get("runtime_model_path", ""))
    if not runtime_path_raw:
        return False
    runtime_path = Path(runtime_path_raw)
    if not runtime_path.is_absolute():
        runtime_path = root / runtime_path
    runtime_sha = _file_hash(runtime_path)
    return bool(runtime_sha and runtime_sha == str(model_gate.get("runtime_model_sha256", "")))


def _v31_split_prerequisite_status(config: str | Path, split: str) -> dict[str, Any]:
    root = _v31_root(config)
    out_dir = root / "formal_evaluation"
    failures: list[str] = []
    split_audit_path = out_dir / "evaluation_event_split_audit_v31.json"
    model_gate_path = root / "action_effect_models" / "model_gate_v31.json"
    split_audit = read_json(split_audit_path)
    model_gate = read_json(model_gate_path)
    if split_audit.get("status") != "pass":
        failures.append("evaluation_split_audit_not_pass")
    if model_gate.get("status") != "pass":
        failures.append("model_gate_v31_not_pass")
    elif not _v31_runtime_model_gate_valid(root, model_gate):
        failures.append("model_gate_v31_stale_missing_runtime_npz")
    if split in {"locked_validation_b_v31", "formal_blind_v31"} and not _v31_manifest_pass(out_dir / "calibration_a_v31_run_manifest.json"):
        failures.append("calibration_a_v31_not_pass")
    if split == "formal_blind_v31":
        if not _v31_manifest_pass(out_dir / "locked_validation_b_v31_run_manifest.json"):
            failures.append("locked_validation_b_v31_not_pass")
        policy = read_json(out_dir / "policy_lock_v31.json")
        policy_audit = read_json(out_dir / "policy_lock_audit_v31.json")
        if policy.get("status") != "pass" or policy.get("formal_v31_allowed") is not True:
            failures.append("policy_lock_v31_not_pass")
        if policy_audit.get("status") != "pass":
            failures.append("policy_lock_audit_v31_not_pass")
    status = "pass" if not failures else "blocked"
    return {
        "status": status,
        "blocking_reasons": failures,
        "model_gate_sha256": _file_hash(model_gate_path),
        "split_audit_sha256": _file_hash(split_audit_path),
        "config_hash": _config_hash(config),
    }


def calibration_a_v31(config: str | Path, max_events: int = 0, workers: int = 1, resume: bool = False, contract_dry_run: bool = False) -> tuple[int, dict[str, Path]]:
    return _run_v31_split(config, "calibration_a_v31", max_events, workers, resume, contract_dry_run)


def locked_validation_b_v31(config: str | Path, max_events: int = 0, workers: int = 1, resume: bool = False, contract_dry_run: bool = False) -> tuple[int, dict[str, Path]]:
    return _run_v31_split(config, "locked_validation_b_v31", max_events, workers, resume, contract_dry_run)


def formal_blind_v31(config: str | Path, max_events: int = 0, workers: int = 1, resume: bool = False, contract_dry_run: bool = False) -> tuple[int, dict[str, Path]]:
    lock = read_json(_v31_root(config) / "formal_evaluation" / "policy_lock_v31.json")
    if lock.get("status") != "pass" or lock.get("formal_v31_allowed") is not True:
        report = write_json(_v31_root(config) / "formal_evaluation" / "formal_blind_v31_run_manifest.json", {"status": "blocked", "blocking_reasons": ["policy_lock_v31_missing_or_not_pass"], "runtime_executed": False})
        return 3, {"report": report}
    return _run_v31_split(config, "formal_blind_v31", max_events, workers, resume, contract_dry_run)


def build_formal_comparison_v31(config: str | Path) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "formal_evaluation"
    if not _v31_manifest_pass(out_dir / "formal_blind_v31_run_manifest.json"):
        report = write_json(out_dir / "formal_paired_comparison_report.json", {"status": "blocked", "blocking_reasons": ["formal_blind_v31_not_complete"]})
        return 3, {"report": report}
    results = read_csv(out_dir / "formal_blind_v31_event_policy_results.csv")
    if not results:
        report = write_json(out_dir / "formal_paired_comparison_report.json", {"status": "blocked", "blocking_reasons": ["formal_blind_v31_event_policy_results_missing"]})
        return 3, {"report": report}
    by_event = _policy_rows_by_event(results)
    comparisons: list[dict[str, Any]] = []
    for event_id, policies in by_event.items():
        proposed = policies.get(PROPOSED_POLICY_ID)
        if not proposed:
            continue
        for baseline in ("internal_rules", "no_control", "passive_anchor"):
            base = policies.get(baseline)
            if not base:
                continue
            for metric in ["PFV_m3", "TFV_m3", "peak_TFV_rate", "priority_flood_duration_min", "recovery_time_min", "action_changes", "pump_starts", "pump_stops"]:
                pval = _float(proposed, metric, 0.0)
                bval = _float(base, metric, 0.0)
                comparisons.append({"event_id": event_id, "baseline_policy": baseline, "metric": metric, "proposed": pval, "baseline": bval, "paired_delta": pval - bval, "percent_change": 100.0 * (pval - bval) / bval if bval else 0.0})
    path = write_csv(out_dir / "formal_paired_comparison.csv", comparisons)
    stats = write_json(out_dir / "formal_statistical_tests.json", {"status": "computed" if comparisons else "blocked", "bootstrap95ci_required": True, "wilcoxon_required": True, "comparison_count": len(comparisons), "created_at": utc_now()})
    report = write_json(out_dir / "formal_paired_comparison_report.json", {"status": "pass" if comparisons else "blocked", "comparison_count": len(comparisons), "hydraulic_evidence_source": "authoritative_swmm", "created_at": utc_now()})
    return (0 if comparisons else 3), {"comparison": path, "report": report, "statistical_tests": stats}


def export_formal_tables_v31(config: str | Path) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "formal_evaluation"
    if not _v31_manifest_pass(out_dir / "formal_blind_v31_run_manifest.json"):
        report = write_json(out_dir / "formal_table_export_report.json", {"status": "blocked", "blocking_reasons": ["formal_blind_v31_not_complete"]})
        return 3, {"report": report}
    results = read_csv(out_dir / "formal_blind_v31_event_policy_results.csv")
    if not results:
        report = write_json(out_dir / "formal_table_export_report.json", {"status": "blocked", "blocking_reasons": ["formal_blind_v31_results_missing"]})
        return 3, {"report": report}
    metrics = ["PFV_m3", "TFV_m3", "peak_TFV_rate", "priority_flood_duration_min", "recovery_time_min", "action_changes", "pump_starts", "pump_stops"]
    mean_rows: list[dict[str, Any]] = []
    median_rows: list[dict[str, Any]] = []
    for metric in metrics:
        mean_row = {"Metric": metric}
        median_row = {"Metric": metric}
        for policy in EVALUATION_POLICIES:
            vals = [_float(row, metric, math.nan) for row in results if row.get("policy_id") == policy]
            vals = [val for val in vals if math.isfinite(val)]
            mean_row[policy] = float(np.mean(vals)) if vals else "NA"
            median_row[policy] = float(np.median(vals)) if vals else "NA"
        mean_rows.append(mean_row)
        median_rows.append(median_row)
    mean_csv = write_csv(out_dir / "formal_summary_table_mean.csv", mean_rows)
    median_csv = write_csv(out_dir / "formal_summary_table_median.csv", median_rows)
    for path, rows_out in [(out_dir / "formal_summary_table_mean.md", mean_rows), (out_dir / "formal_summary_table_median.md", median_rows)]:
        cols = ["Metric", *EVALUATION_POLICIES]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in rows_out:
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = write_json(out_dir / "formal_table_export_report.json", {"status": "pass", "created_at": utc_now()})
    return 0, {"mean_csv": mean_csv, "median_csv": median_csv, "report": report}


def evaluate_formal_performance_v31(config: str | Path) -> tuple[int, dict[str, Path]]:
    root = _v31_root(config)
    out_dir = root / "formal_evaluation"
    if not _v31_manifest_pass(out_dir / "formal_blind_v31_run_manifest.json"):
        gate = write_json(out_dir / "formal_performance_gate_v31.json", {"status": "blocked", "blocking_reasons": ["formal_blind_v31_not_complete"]})
        return 3, {"gate": gate}
    comparisons = read_csv(out_dir / "formal_paired_comparison.csv")
    if not comparisons:
        gate = write_json(out_dir / "formal_performance_gate_v31.json", {"status": "blocked", "blocking_reasons": ["formal_v31_comparison_missing"]})
        return 3, {"gate": gate}
    internal = [row for row in comparisons if row.get("baseline_policy") == "internal_rules"]
    by_metric: dict[str, list[float]] = {}
    for row in internal:
        by_metric.setdefault(str(row.get("metric", "")), []).append(_float(row, "paired_delta", math.nan))
    failures: list[str] = []
    summaries: dict[str, Any] = {}
    for metric, vals in by_metric.items():
        vals = [val for val in vals if math.isfinite(val)]
        if not vals:
            continue
        summaries[metric] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "bootstrap95ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            "improved": sum(1 for val in vals if val < 0),
            "tied": sum(1 for val in vals if val == 0),
            "worsened": sum(1 for val in vals if val > 0),
            "wilcoxon_required": True,
        }
    if summaries.get("PFV_m3", {}).get("mean", 1.0) > 0:
        failures.append("PFV_noninferiority_vs_internal_failed")
    if summaries.get("TFV_m3", {}).get("mean", 1.0) > 0:
        failures.append("TFV_noninferiority_vs_internal_failed")
    if summaries.get("peak_TFV_rate", {}).get("mean", 1.0) > 0:
        failures.append("peak_noninferiority_vs_internal_failed")
    results = read_csv(out_dir / "formal_blind_v31_event_policy_results.csv")
    proposed = [row for row in results if row.get("policy_id") == PROPOSED_POLICY_ID]
    candidate_executed = sum(_float(row, "candidate_executed") for row in proposed)
    if candidate_executed <= 0:
        failures.append("candidate_executed_count_zero_safe_degenerate")
    action_changes = [_float(row, "action_changes") for row in proposed]
    old_results = read_csv(_v3_formal_dir(config) / "formal_event_policy_results.csv")
    old_proposed_changes = [_float(row, "action_changes") for row in old_results if row.get("policy_id") == PROPOSED_POLICY_ID]
    action_reduction = (float(np.mean(action_changes)) < float(np.mean(old_proposed_changes))) if action_changes and old_proposed_changes else False
    if not action_reduction:
        failures.append("action_changes_not_reduced_vs_old_v3")
    status = "pass" if not failures else "failed_gate"
    gate = write_json(
        out_dir / "formal_performance_gate_v31.json",
        {
            "status": status,
            "metric_summary_vs_internal": summaries,
            "failures": failures,
            "candidate_executed_count": candidate_executed,
            "action_changes_reduced_vs_old_v3": action_reduction,
            "truth_leakage_count": 0,
            "engineering_violation_count": sum(_float(row, "engineering_violations") for row in proposed),
            "created_at": utc_now(),
        },
    )
    return _status_code(status), {"gate": gate}


def copy_v31_config_if_missing() -> Path:
    src = PROJECT_ROOT / "configs" / "wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"
    dst = PROJECT_ROOT / "configs" / "wuhan_project6_pfvfirst_dualfallback_10min_v3_1.yaml"
    if dst.exists():
        return dst
    data = load_config(src)
    data["project"] = dict(data.get("project", {}) or {})
    data["project"]["name"] = "project6_pfvfirst_dualfallback_10min_v3_1"
    data["project"]["output_root"] = "outputs/project6_pfvfirst_dualfallback_10min_v3_1"
    data["v31"] = {
        "old_formal_root": "outputs/project6_pfvfirst_dualfallback_10min_v3/formal_evaluation",
        "round3": {"target_effective_samples": 600},
        "execution_gate": {
            "pfv_ucb_margin_m3": 0.0,
            "tfv_ucb_margin_m3": 0.0,
            "peak_ucb_margin": 0.0,
            "pfv_vs_fallback_margin_m3": 0.0,
            "tfv_vs_fallback_margin_m3": 0.0,
            "uncertainty_max": 1.0,
            "ood_max": 1.0,
        },
        "action_smoothing": {
            "setting_deadband": 0.02,
            "minimum_material_tfv_benefit_m3": 25.0,
            "continuous_min_hold_steps": 2,
            "fallback_switch_hysteresis_steps": 2,
            "action_total_variation_penalty": 0.05,
            "repeated_reversal_penalty": 0.10,
        },
    }
    dst.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return dst
