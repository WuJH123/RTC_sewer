from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path_value: object, project_root: str | Path) -> Path:
    p = Path(str(path_value or ""))
    if not p.is_absolute():
        p = Path(project_root) / p
    return p


def _bool_gate(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(int(value))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "passed"}
    return False


def residual_report_passed(report: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not _bool_gate(report.get("validation_gate_passed", False)):
        failures.append("residual_model_gate_passed")
    smoke = report.get("rolling_horizon_smoke_eligibility", {})
    if isinstance(smoke, Mapping) and "passed" in smoke and not _bool_gate(smoke.get("passed")):
        failures.append("residual_smoke_eligibility")
    return not failures, failures


def core_residual_ids(actuators: pd.DataFrame, residual_actuator_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    ids = actuators["actuator_id"].astype(str).tolist()
    residual = [str(aid) for aid in residual_actuator_ids]
    missing = sorted(set(residual) - set(ids))
    if missing:
        raise ValueError(f"residual10 actuator ids missing from canonical action order: {missing}")
    core = [aid for aid in ids if aid not in set(residual)]
    return core, residual


def assert_residual_only_changes_residual_columns(
    core_action_seq: np.ndarray,
    combined_action_seq: np.ndarray,
    *,
    canonical_action_ids: Sequence[str],
    residual_actuator_ids: Sequence[str],
    atol: float = 1.0e-7,
) -> None:
    core = np.asarray(core_action_seq, dtype=float)
    combined = np.asarray(combined_action_seq, dtype=float)
    if core.shape != combined.shape:
        raise AssertionError(f"shape_mismatch:{core.shape}:{combined.shape}")
    ids = [str(aid) for aid in canonical_action_ids]
    residual = set(str(aid) for aid in residual_actuator_ids)
    core_columns = [idx for idx, aid in enumerate(ids) if aid not in residual]
    if core_columns and np.any(np.abs(core[:, core_columns] - combined[:, core_columns]) > float(atol)):
        changed = [
            ids[idx]
            for idx in core_columns
            if np.any(np.abs(core[:, idx] - combined[:, idx]) > float(atol))
        ]
        raise AssertionError(f"core26_modified:{changed}")


def build_strict_preflight(
    *,
    cfg: Mapping[str, Any],
    actuators: pd.DataFrame,
    project_root: str | Path,
) -> dict[str, Any]:
    controller = dict(cfg.get("controller", {}) or {})
    temporal = dict(controller.get("temporal_joint", {}) or {})
    hierarchical = dict(temporal.get("hierarchical", {}) or {})
    safety = dict(temporal.get("safety", {}) or {})
    residual_ids = [str(value) for value in hierarchical.get("residual_actuator_ids", [])]
    try:
        core_ids, residual_ids = core_residual_ids(actuators, residual_ids)
    except ValueError as exc:
        core_ids, residual_ids = [], residual_ids
        residual_id_error = str(exc)
    else:
        residual_id_error = ""

    path_keys = {
        "core26_policy_path": hierarchical.get("core26_policy_path") or temporal.get("candidate_search", {}).get("engineering_template_path"),
        "residual10_model_path": hierarchical.get("residual10_model_path") or temporal.get("model_path"),
        "residual10_model_report": hierarchical.get("residual10_model_report") or hierarchical.get("residual_validation_report"),
        "uncertainty_model_path": hierarchical.get("uncertainty_model_path"),
        "empirical_guard_path": hierarchical.get("empirical_guard_path"),
    }
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    failed: list[str] = []
    for key, value in path_keys.items():
        if not value:
            failed.append(f"{key}_nonempty")
            paths[key] = ""
            continue
        p = _resolve(value, project_root)
        paths[key] = str(p)
        if not p.exists():
            failed.append(f"{key}_exists")
        elif p.is_file():
            hashes[key] = sha256_file(p)

    report_passed = False
    report_failures: list[str] = []
    report_path = Path(paths.get("residual10_model_report", ""))
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report_failures = [f"residual_report_parse_error:{exc}"]
        else:
            report_passed, report_failures = residual_report_passed(report)
    else:
        report_failures = ["residual_report_missing"]
    failed.extend(report_failures)

    prediction_horizon_min = float(temporal.get("prediction_horizon_min", 0.0) or 0.0)
    move_horizon_min = float(temporal.get("move_horizon_min", 30.0) or 30.0)
    event_budget = bool(safety.get("event_pfv_budget_enabled", False))
    checks = {
        "controller_mode": str(controller.get("mode", "")) == "hierarchical_core26_residual10",
        "core26_policy_path": bool(paths.get("core26_policy_path")) and Path(paths["core26_policy_path"]).exists(),
        "residual10_model_path": bool(paths.get("residual10_model_path")) and Path(paths["residual10_model_path"]).exists(),
        "residual10_model_report": bool(paths.get("residual10_model_report")) and Path(paths["residual10_model_report"]).exists(),
        "residual_model_gate_passed": bool(report_passed),
        "uncertainty_model_path": bool(paths.get("uncertainty_model_path")) and Path(paths["uncertainty_model_path"]).exists(),
        "empirical_guard_path": bool(paths.get("empirical_guard_path")) and Path(paths["empirical_guard_path"]).exists(),
        "prediction_horizon_min": prediction_horizon_min >= 120.0,
        "move_horizon_min": abs(move_horizon_min - 30.0) <= 1.0e-6,
        "event_pfv_budget_enabled": event_budget,
        "core26_actuator_count": len(core_ids) == 26,
        "residual10_actuator_count": len(residual_ids) == 10,
    }
    if residual_id_error:
        failed.append("residual10_actuator_ids_present")
    failed.extend([name for name, ok in checks.items() if not ok and name not in failed])
    failed = sorted(set(failed))
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "paths": paths,
        "hashes": hashes,
        "prediction_horizon_min": prediction_horizon_min,
        "move_horizon_min": move_horizon_min,
        "event_pfv_budget_enabled": event_budget,
        "core26_actuator_count": len(core_ids),
        "residual10_actuator_count": len(residual_ids),
        "core26_actuator_ids": core_ids,
        "residual10_actuator_ids": residual_ids,
        "residual_id_error": residual_id_error,
    }
