from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "126_plan_information_coverage_cases.py"
spec = importlib.util.spec_from_file_location("coverage_script", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
coverage_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage_script)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_header(path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)


def test_first_init_creates_schema_files(tmp_path: Path) -> None:
    # This test patches the module-level safe path by calling the lower-level
    # writer on the real schema helpers via monkeypatch-like direct assignment.
    old_safe = coverage_script.SAFE_COVERAGE_DIR
    try:
        coverage_script.SAFE_COVERAGE_DIR = tmp_path
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("coverage_targets:\n  per_facility_effective_cases_min: 60\n", encoding="utf-8")
        report = coverage_script.plan_cases(cfg, tmp_path)
        assert report["status"] == "scaffold_only"
        assert (tmp_path / "coverage_gap_audit.csv").exists()
        assert (tmp_path / "coverage_cells_schema.csv").exists()
        assert (tmp_path / "candidate_manifest_preview.csv").exists()
    finally:
        coverage_script.SAFE_COVERAGE_DIR = old_safe


def test_reinit_empty_schema_preserves_hash(tmp_path: Path) -> None:
    old_safe = coverage_script.SAFE_COVERAGE_DIR
    try:
        coverage_script.SAFE_COVERAGE_DIR = tmp_path
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("coverage_targets:\n  per_facility_effective_cases_min: 60\n", encoding="utf-8")
        coverage_script.plan_cases(cfg, tmp_path)
        manifest = tmp_path / "candidate_manifest_preview.csv"
        before = file_hash(manifest)
        coverage_script.plan_cases(cfg, tmp_path)
        assert file_hash(manifest) == before
    finally:
        coverage_script.SAFE_COVERAGE_DIR = old_safe


def test_populated_manifest_is_blocked(tmp_path: Path) -> None:
    old_safe = coverage_script.SAFE_COVERAGE_DIR
    try:
        coverage_script.SAFE_COVERAGE_DIR = tmp_path
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("coverage_targets:\n  per_facility_effective_cases_min: 60\n", encoding="utf-8")
        coverage_script.plan_cases(cfg, tmp_path)
        manifest = tmp_path / "candidate_manifest_preview.csv"
        with manifest.open("a", encoding="utf-8") as fh:
            fh.write("case_001\n")
        with pytest.raises(coverage_script.CoverageBlocked, match="populated_coverage_artifact_exists"):
            coverage_script.plan_cases(cfg, tmp_path)
    finally:
        coverage_script.SAFE_COVERAGE_DIR = old_safe


def test_force_requires_acknowledgement(tmp_path: Path) -> None:
    old_safe = coverage_script.SAFE_COVERAGE_DIR
    try:
        coverage_script.SAFE_COVERAGE_DIR = tmp_path
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("coverage_targets:\n  per_facility_effective_cases_min: 60\n", encoding="utf-8")
        coverage_script.plan_cases(cfg, tmp_path)
        with pytest.raises(coverage_script.CoverageBlocked, match="force_requires_acknowledge_data_loss"):
            coverage_script.plan_cases(cfg, tmp_path, force=True, acknowledge=False)
    finally:
        coverage_script.SAFE_COVERAGE_DIR = old_safe


def test_non_v3_target_path_cannot_force(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("coverage_targets:\n  per_facility_effective_cases_min: 60\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe_coverage_target"):
        coverage_script.plan_cases(cfg, tmp_path / "outside", force=True, acknowledge=True)


def test_force_with_acknowledgement_creates_backup(tmp_path: Path) -> None:
    old_safe = coverage_script.SAFE_COVERAGE_DIR
    try:
        coverage_script.SAFE_COVERAGE_DIR = tmp_path
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("coverage_targets:\n  per_facility_effective_cases_min: 60\n", encoding="utf-8")
        coverage_script.plan_cases(cfg, tmp_path)
        manifest = tmp_path / "candidate_manifest_preview.csv"
        with manifest.open("a", encoding="utf-8") as fh:
            fh.write("case_001\n")
        report = coverage_script.plan_cases(cfg, tmp_path, force=True, acknowledge=True)
        assert report["backups"]
        assert any("candidate_manifest_preview.csv" in item for item in report["backups"])
    finally:
        coverage_script.SAFE_COVERAGE_DIR = old_safe
