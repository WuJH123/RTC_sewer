from __future__ import annotations

from pathlib import Path

from sewerrtc.execution.status_contract import (
    validate_completion_marker_path,
    validate_execution_status,
    validate_no_forbidden_completion_marker,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"


def test_disabled_status_requires_json_null_completion_marker() -> None:
    payload = {
        "stage": "BuildDataset",
        "status": "disabled",
        "config_path": "configs/x.yaml",
        "config_sha256": "abc",
        "finished_at": "2026-01-01T00:00:00Z",
        "inputs": {},
        "outputs": {},
        "completion_marker": None,
        "failure_reason": "not_implemented",
    }
    assert validate_execution_status(payload) == []


def test_scaffold_only_status_requires_json_null_completion_marker() -> None:
    payload = {
        "stage": "InitCoverageSchema",
        "status": "scaffold_only",
        "config_path": "configs/x.yaml",
        "config_sha256": "abc",
        "finished_at": "2026-01-01T00:00:00Z",
        "inputs": {},
        "outputs": {},
        "completion_marker": None,
        "failure_reason": "",
    }
    assert validate_execution_status(payload) == []


def test_completed_status_requires_non_empty_marker() -> None:
    payload = {
        "stage": "Audit",
        "status": "completed",
        "config_path": "configs/x.yaml",
        "config_sha256": "abc",
        "finished_at": "2026-01-01T00:00:00Z",
        "inputs": {},
        "outputs": {},
        "completion_marker": None,
        "failure_reason": "",
    }
    assert "completed_requires_completion_marker" in validate_execution_status(payload)
    payload["completion_marker"] = ""
    errors = validate_execution_status(payload)
    assert "completion_marker_empty_string" in errors
    assert "completed_requires_non_empty_completion_marker" in errors


def test_disabled_stage_must_not_have_completion_marker_file(tmp_path: Path) -> None:
    assert validate_no_forbidden_completion_marker("BuildDataset", tmp_path) == []
    (tmp_path / "BuildDataset_COMPLETED.json").write_text("{}", encoding="utf-8")
    assert validate_no_forbidden_completion_marker("BuildDataset", tmp_path)


def test_completion_marker_empty_string_invalid() -> None:
    assert "completion_marker_empty_string" in validate_completion_marker_path("")


def test_runner_declares_stable_exit_codes_and_disabled_stages() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "exit 0" in text
    assert "ExitCode 2" in text
    assert "ExitCode 3" in text
    assert "ExitCode 4" in text
    assert "ExitCode 6" in text
    assert "ExitCode 7" in text
    assert 'if ($BuildDataset) { Disable-Stage "BuildDataset" }' in text
    assert 'if ($TrainPilot) { Disable-Stage "TrainPilot" }' in text
    assert 'if ($MinimalGate) { Disable-Stage "MinimalGate" }' in text
    assert 'if ($RunSmoke) { Disable-Stage "RunSmoke" }' in text
