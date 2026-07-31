"""Initialize Project6 V3 information-coverage schemas.

This stage is scaffold-only. It must not overwrite populated coverage files and
must not create a completion marker or unlock downstream stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.data.coverage_contract import CANDIDATE_MANIFEST_FIELDS, COVERAGE_CELL_FIELDS, classify_coverage


SAFE_COVERAGE_DIR = PROJECT_ROOT / "outputs" / "project6_pfvfirst_dualfallback_10min_v3" / "coverage"


class CoverageBlocked(RuntimeError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def row_count_csv(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    return max(len(rows) - 1, 0)


def csv_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        return next(reader, [])


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def csv_header_text(fields: Iterable[str]) -> str:
    return ",".join(fields) + "\n"


def ensure_schema_csv(
    path: Path,
    fields: Iterable[str],
    *,
    force: bool,
    acknowledge: bool,
    backups: list[str],
) -> str:
    expected = list(fields)
    if path.exists():
        rows = row_count_csv(path)
        if rows > 0 and not (force and acknowledge):
            raise CoverageBlocked(f"populated_coverage_artifact_exists:{path}")
        if rows == 0 and csv_header(path) == expected and not force:
            return "existing_empty_schema_verified"
        if not force:
            raise CoverageBlocked(f"coverage_schema_mismatch:{path}")
        backup = path.with_name(path.name + "." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".bak")
        shutil.copy2(path, backup)
        backups.append(str(backup))
    atomic_write_text(path, csv_header_text(expected))
    return "created_or_reinitialized_schema"


def assert_safe_target(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    safe = SAFE_COVERAGE_DIR.resolve()
    if resolved != safe:
        raise ValueError(f"unsafe_coverage_target:{resolved}; expected:{safe}")


def plan_cases(config_path: Path, out_dir: Path, *, force: bool = False, acknowledge: bool = False) -> dict[str, Any]:
    assert_safe_target(out_dir)
    if force and not acknowledge:
        raise CoverageBlocked("force_requires_acknowledge_data_loss")

    cfg = load_config(config_path)
    targets = cfg.get("coverage_targets", {})
    planning_targets = {
        k: int(v)
        for k, v in targets.items()
        if isinstance(v, int) and not k.endswith("_max") and k not in {"batch_effective_candidate_max"}
    }
    maximum_constraints = {
        k: int(v)
        for k, v in targets.items()
        if isinstance(v, int) and (k.endswith("_max") or k == "batch_effective_candidate_max")
    }
    current = {key: 0 for key in planning_targets}
    coverage_rows = classify_coverage(current, planning_targets)

    out_dir.mkdir(parents=True, exist_ok=True)
    backups: list[str] = []
    coverage_path = out_dir / "coverage_gap_audit.csv"
    cells_path = out_dir / "coverage_cells_schema.csv"
    manifest_path = out_dir / "candidate_manifest_preview.csv"
    metadata_path = out_dir / "coverage_schema_metadata.json"

    # The audit file is scaffold output and may be regenerated only while it
    # remains header-only or under explicit force+acknowledge.
    coverage_status = ensure_schema_csv(
        coverage_path,
        ["coverage_key", "current", "minimum", "status", "gap"],
        force=force,
        acknowledge=acknowledge,
        backups=backups,
    )
    cells_status = ensure_schema_csv(cells_path, COVERAGE_CELL_FIELDS, force=force, acknowledge=acknowledge, backups=backups)
    manifest_status = ensure_schema_csv(
        manifest_path,
        CANDIDATE_MANIFEST_FIELDS,
        force=force,
        acknowledge=acknowledge,
        backups=backups,
    )

    metadata = {
        "schema_version": "project6_v3_coverage_schema_metadata_v1",
        "status": "scaffold_only",
        "completion_marker": None,
        "unlocks_downstream": False,
        "coverage_gap_audit": str(coverage_path),
        "coverage_cells_schema": str(cells_path),
        "candidate_manifest_preview": str(manifest_path),
    }
    atomic_write_text(metadata_path, json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    report = {
        "config": str(config_path),
        "coverage_audit": str(coverage_path),
        "coverage_cells_schema": str(cells_path),
        "candidate_manifest_preview": str(manifest_path),
        "coverage_schema_metadata": str(metadata_path),
        "maximum_constraints_report_only": maximum_constraints,
        "coverage_targets_scaffold": coverage_rows,
        "artifact_status": {
            "coverage_gap_audit": coverage_status,
            "coverage_cells_schema": cells_status,
            "candidate_manifest_preview": manifest_status,
        },
        "backups": backups,
        "status": "scaffold_only",
        "completion_marker": None,
        "unlocks_downstream": False,
        "note": "Populate candidate rows only after EventCatalog, CheckpointCatalog, Internal PFV opportunity scan, and same-state clone checks are available.",
    }
    atomic_write_text(out_dir / "coverage_plan_report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force-reinitialize-empty-coverage", action="store_true")
    ap.add_argument("--acknowledge-data-loss", action="store_true")
    args = ap.parse_args()
    try:
        report = plan_cases(
            Path(args.config),
            Path(args.out_dir),
            force=args.force_reinitialize_empty_coverage,
            acknowledge=args.acknowledge_data_loss,
        )
    except CoverageBlocked as exc:
        print(json.dumps({"status": "blocked", "failure_reason": str(exc), "completion_marker": None}, indent=2))
        raise SystemExit(3)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
