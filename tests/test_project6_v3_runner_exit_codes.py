from __future__ import annotations

import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
CONFIG = ROOT / "configs" / "wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"
COVERAGE_DIR = ROOT / "outputs" / "project6_pfvfirst_dualfallback_10min_v3" / "coverage"
EXECUTION_DIR = ROOT / "outputs" / "project6_pfvfirst_dualfallback_10min_v3" / "execution_status"


def run_stage(*stage_args: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(RUNNER),
        *stage_args,
        "-Python",
        str(PYTHON),
        "-Config",
        str(CONFIG),
    ]
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


@contextmanager
def restored_directory(path: Path):
    backup = path.with_name(path.name + ".pytest_backup")
    if backup.exists():
        shutil.rmtree(backup)
    if path.exists():
        shutil.copytree(path, backup)
        shutil.rmtree(path)
    try:
        yield
    finally:
        if path.exists():
            shutil.rmtree(path)
        if backup.exists():
            shutil.copytree(backup, path)
            shutil.rmtree(backup)


def test_build_dataset_disabled_exit_code_2() -> None:
    result = run_stage("-BuildDataset")
    assert result.returncode == 2
    status_path = EXECUTION_DIR / "BuildDataset.status.json"
    assert status_path.exists()
    assert "not_implemented" in status_path.read_text(encoding="utf-8")
    assert not (EXECUTION_DIR / "BuildDataset_COMPLETED.json").exists()


def test_multiple_stage_selection_exit_code_7() -> None:
    result = run_stage("-Status", "-Audit")
    assert result.returncode == 7


def test_status_success_exit_code_0() -> None:
    result = run_stage("-Status")
    assert result.returncode == 0


def test_audit_success_exit_code_0() -> None:
    result = run_stage("-Audit")
    assert result.returncode == 0


def test_init_coverage_schema_populated_file_exit_code_3() -> None:
    with restored_directory(COVERAGE_DIR):
        first = run_stage("-InitCoverageSchema")
        assert first.returncode == 0
        manifest = COVERAGE_DIR / "candidate_manifest_preview.csv"
        with manifest.open("a", encoding="utf-8") as fh:
            fh.write("case_001\n")
        blocked = run_stage("-InitCoverageSchema")
        assert blocked.returncode == 3
        status_path = EXECUTION_DIR / "InitCoverageSchema.status.json"
        assert status_path.exists()
        assert "populated_coverage_artifact_exists" in status_path.read_text(encoding="utf-8")
