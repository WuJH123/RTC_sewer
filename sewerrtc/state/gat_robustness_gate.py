from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gat_audit import sha256_file
from .gat_robustness import _gate_check, _write_gate
from .gat_selection import EXPECTED_CHECKPOINT_SHA256, SELECTED_REGISTRY_NAME


def _c(check_id: str, status: str, observed: Any, required: Any, evidence: Path, reason: str = "", remediation: str = "") -> dict[str, Any]:
    return _gate_check(
        check_id=check_id,
        status=status,
        observed_value=observed,
        required_value=required,
        evidence_path=evidence,
        blocking_reason=reason,
        remediation=remediation,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _find_primary_lock(gat_dir: Path) -> Path:
    for candidate_dir in [gat_dir, *gat_dir.parents]:
        candidate = candidate_dir / "gat_primary_selection_lock.json"
        if candidate.exists():
            return candidate
    return gat_dir / "gat_primary_selection_lock.json"


def _status_from_rows(rows: list[dict[str, str]], default: str = "incomplete", check_id: str | None = None) -> str:
    if not rows:
        return default
    if check_id is not None:
        # Some evidence files are audit tables rather than gate-check tables:
        # they have a row-level ``status`` but no ``check_id`` column. In that
        # case, summarize the table status instead of treating the missing
        # column as missing evidence.
        if all("check_id" not in row for row in rows):
            return _status_from_rows(rows, default=default, check_id=None)
        for row in rows:
            if row.get("check_id") == check_id:
                status = str(row.get("status", "")).lower()
                if status in {"pass", "fail", "incomplete", "not_applicable"}:
                    return status
                return default
        return default
    statuses = {str(row.get("status", "")).lower() for row in rows}
    if "fail" in statuses or "failed" in statuses:
        return "fail"
    if "incomplete" in statuses or "diagnostic_only" in statuses or "unknown" in statuses:
        return "incomplete"
    if statuses <= {"pass", "computed", "completed", ""}:
        return "pass"
    return default


def evaluate_gat_robustness_gate(gat_dir: Path) -> tuple[int, Path]:
    lock_path = _find_primary_lock(gat_dir)
    gate_path = gat_dir / "gat_sr0p15_robustness_gate.json"
    if not lock_path.exists():
        _write_gate(
            gate_path,
            [
                _gate_check(
                    check_id="sr0p15_selection_lock_valid",
                    status="incomplete",
                    observed_value="missing",
                    required_value="present",
                    evidence_path=lock_path,
                    blocking_reason="sr0p15 primary GAT lock is missing",
                    remediation="run SelectPrimaryGAT with sr0p15 and acknowledgement",
                )
            ],
        )
        return 3, gate_path
    lock = _read_json(lock_path)
    if lock.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        _write_gate(
            gate_path,
            [
                _gate_check(
                    check_id="sr0p15_selection_lock_valid",
                    status="fail",
                    observed_value=lock.get("checkpoint_sha256"),
                    required_value=EXPECTED_CHECKPOINT_SHA256,
                    evidence_path=lock_path,
                    blocking_reason="selection lock checkpoint hash mismatch",
                    remediation="rerun AuditGAT and SelectPrimaryGAT",
                )
            ],
        )
        return 6, gate_path

    provenance_path = gat_dir / "gat_sr0p15_validation_provenance_audit.csv"
    leakage_path = gat_dir / "gat_sr0p15_validation_leakage_audit.csv"
    node_metrics_path = gat_dir / "gat_sr0p15_node_group_metrics.csv"
    priority_path = gat_dir / "gat_sr0p15_priority_leaveout_audit.csv"
    sentinel_path = gat_dir / "gat_sr0p15_sentinel_leaveout_audit.csv"
    highwater_path = gat_dir / "gat_sr0p15_highwater_phase_audit.csv"
    sensor_matrix_path = gat_dir / "gat_sr0p15_sensor_failure_completion_matrix.csv"
    sensor_summary_path = gat_dir / "gat_sr0p15_sensor_failure_summary.csv"
    latency_path = gat_dir / "gat_sr0p15_latency_repeatability_audit.csv"
    latency_summary_path = gat_dir / "gat_sr0p15_latency_summary.json"
    strict_path = gat_dir / "gat_strict_load_audit.csv"

    provenance_rows = _read_csv(provenance_path)
    leakage_rows = _read_csv(leakage_path)
    priority_rows = _read_csv(priority_path)
    sentinel_rows = _read_csv(sentinel_path)
    highwater_rows = _read_csv(highwater_path)
    sensor_rows = _read_csv(sensor_matrix_path)
    latency_rows = _read_csv(latency_path)
    leakage_status = _status_from_rows(leakage_rows, check_id="no_training_event_leakage")
    provenance_status = _status_from_rows(provenance_rows, check_id="validation_provenance_complete")
    sensor_complete = bool(sensor_rows) and all(row.get("status") == "pass" for row in sensor_rows)
    latency_complete = bool(latency_rows) and all(row.get("status") == "computed" and row.get("p95_ms") not in {"", None} for row in latency_rows)
    checks = [
        _c("sr0p15_selection_lock_valid", "pass", "present", "present", lock_path),
        _c("strict_load_passed", "pass" if lock.get("strict_load_status") == "strict_loaded" else "fail", lock.get("strict_load_status"), "strict_loaded", strict_path),
        _c("compatible_strict", "pass" if lock.get("compatibility_status") == "compatible_strict" else "fail", lock.get("compatibility_status"), "compatible_strict", lock_path),
        _c("validation_provenance_complete", provenance_status, provenance_status, "pass", provenance_path, "" if provenance_status == "pass" else "validation provenance evidence is incomplete"),
        _c("no_training_event_leakage", leakage_status, leakage_status, "pass", leakage_path, "" if leakage_status == "pass" else "training-event leakage is not ruled out"),
        _c("unobserved_metrics_exist", "pass" if node_metrics_path.exists() else "incomplete", node_metrics_path.exists(), True, node_metrics_path),
        _c("priority_leaveout_complete", "pass" if len(priority_rows) >= 8 else "incomplete", len(priority_rows), ">=8 rows", priority_path),
        _c("sentinel_leaveout_complete", "pass" if len(sentinel_rows) >= 2 else "incomplete", len(sentinel_rows), ">=2 rows", sentinel_path),
        _c("highwater_phase_complete", "pass" if highwater_rows else "incomplete", bool(highwater_rows), True, highwater_path),
        _c("sensor_failure_execution_complete", "pass" if sensor_complete else "incomplete", sensor_complete, True, sensor_matrix_path, "" if sensor_complete else "sensor failure completion matrix is incomplete"),
        _c("sensor_failure_performance_gate", "not_applicable", "uncalibrated", "calibrated threshold", sensor_summary_path),
        _c("latency_measurement_complete", "pass" if latency_complete else "incomplete", latency_complete, True, latency_path, "" if latency_complete else "latency p95 or required measurements are missing"),
        _c("latency_budget_gate", "not_applicable", "uncalibrated", "frozen threshold", latency_summary_path),
    ]
    _write_gate(gate_path, checks)
    gate = _read_json(gate_path)
    if gate.get("failed_checks"):
        return 5, gate_path
    if gate.get("incomplete_checks"):
        return 3, gate_path
    return 0, gate_path
