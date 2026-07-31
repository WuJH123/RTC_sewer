"""Execution-status validation for Project6 PFV-first V3."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


VALID_STATUSES = {
    "not_started",
    "running",
    "scaffold_only",
    "completed",
    "disabled",
    "failed",
    "failed_gate",
    "blocked",
    "contract_mismatch",
}

NULL_MARKER_STATUSES = {
    "disabled",
    "failed",
    "failed_gate",
    "blocked",
    "contract_mismatch",
    "scaffold_only",
}


def validate_execution_status(payload: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    for field in (
        "stage",
        "status",
        "config_path",
        "config_sha256",
        "finished_at",
        "inputs",
        "outputs",
        "completion_marker",
        "failure_reason",
    ):
        if field not in payload:
            errors.append(f"missing:{field}")
    status = str(payload.get("status", ""))
    marker = payload.get("completion_marker")
    if status and status not in VALID_STATUSES:
        errors.append(f"invalid_status:{status}")
    if marker == "":
        errors.append("completion_marker_empty_string")
    if status == "completed":
        if marker is None:
            errors.append("completed_requires_completion_marker")
        elif not str(marker).strip():
            errors.append("completed_requires_non_empty_completion_marker")
    if status in NULL_MARKER_STATUSES and marker is not None:
        errors.append(f"{status}_requires_null_completion_marker")
    return errors


def validate_no_forbidden_completion_marker(stage: str, execution_root: str | Path) -> list[str]:
    root = Path(execution_root)
    marker = root / f"{stage}_COMPLETED.json"
    return [f"forbidden_completion_marker:{marker}"] if marker.exists() else []


def validate_completion_marker_path(path: object) -> list[str]:
    if path is None:
        return ["completion_marker_missing"]
    if str(path) == "":
        return ["completion_marker_empty_string"]
    return []
