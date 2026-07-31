from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from typing import Any

from sewerrtc._project_root import PROJECT_ROOT
from sewerrtc.contracts.prompt3a import OUT_ROOT, sha256_file, utc_now, write_csv, write_json
from sewerrtc.data.event_identity import canonical_event_id, storm_family_id
from sewerrtc.data.rainfall_series import normalized_series_hash, read_rainfall_series, timestamp_hash


_PROJECT4_ROOT = Path(os.environ.get("PROJECT4_ROOT", PROJECT_ROOT.parent / "Project4"))
_PROJECT5_ROOT = Path(os.environ.get("PROJECT5_ROOT", PROJECT_ROOT.parent / "Project5"))

ALLOWED_RAINFALL_DIRS = [
    _PROJECT4_ROOT / "outputs" / "rainfall_library",
    _PROJECT5_ROOT / "outputs" / "pystorms_beta" / "rainfall_library",
    PROJECT_ROOT / "data" / "rainfall_library",
    OUT_ROOT / "rainfall_assets",
]

INDEX_COLUMNS = [
    "asset_id",
    "canonical_event_id",
    "path",
    "source_project",
    "source_priority",
    "file_sha256",
    "rainfall_series_sha256",
    "timestamp_sha256",
    "row_count",
    "interval_min",
    "unit",
    "total_depth_mm",
    "peak_intensity_mm_h",
    "peak_time_min",
    "parser_id",
    "parser_version",
    "source_mode",
    "status",
    "provenance",
]


def _source_project(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for name in ["project6", "project5", "project4", "project3", "project2"]:
        if name in parts:
            return name.replace("project", "Project")
    return "unknown"


def _source_priority(path: Path) -> int:
    source = _source_project(path)
    return {"Project6": 1, "Project5": 2, "Project4": 3}.get(source, 9)


def _event_id_from_csv(path: Path) -> str:
    if path.name.lower() == "rainfall_event_table.csv":
        return ""
    return path.stem


def build_rainfall_asset_index(out_dir: str | Path = OUT_ROOT / "rainfall_assets") -> tuple[int, dict[str, Any], list[Path]]:
    out_dir = Path(out_dir)
    inventory: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for root in ALLOWED_RAINFALL_DIRS:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.csv")):
            event_id = _event_id_from_csv(path)
            if not event_id:
                excluded.append({"path": str(path), "reason": "rainfall_event_table_not_event_series", "file_sha256": sha256_file(path)})
                continue
            series = read_rainfall_series(path)
            if series["status"] != "parsed":
                excluded.append({"path": str(path), "reason": series["status"], "file_sha256": sha256_file(path)})
                continue
            source_project = _source_project(path)
            inventory.append(
                {
                    "asset_id": f"{source_project}_{event_id}_{sha256_file(path)[:10]}",
                    "canonical_event_id": canonical_event_id(event_id),
                    "path": str(path),
                    "source_project": source_project,
                    "source_priority": _source_priority(path),
                    "file_sha256": sha256_file(path),
                    "rainfall_series_sha256": normalized_series_hash(series["intensity_values"]),
                    "timestamp_sha256": timestamp_hash(series["timestamp_values"]),
                    "row_count": series["row_count"],
                    "interval_min": series["interval_min"],
                    "unit": "mm_per_hour",
                    "total_depth_mm": series["total_depth_mm"],
                    "peak_intensity_mm_h": series["peak_intensity_mm_h"],
                    "peak_time_min": series["peak_time_min"],
                    "parser_id": "project6_rainfall_csv_elapsed_intensity",
                    "parser_version": "v1",
                    "source_mode": "external_file",
                    "status": "available",
                    "provenance": "fixed_allowed_rainfall_directories",
                }
            )
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        by_event[str(row["canonical_event_id"])].append(row)
    duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    resolution: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for event_id, rows in by_event.items():
        rows_sorted = sorted(rows, key=lambda r: (int(r["source_priority"]), str(r["path"])))
        selected[event_id] = rows_sorted[0]
        hashes = {r["rainfall_series_sha256"] for r in rows_sorted}
        status = "resolved_unique"
        if len(rows_sorted) > 1 and len(hashes) == 1:
            status = "resolved_equivalent_duplicates"
            duplicates.extend(rows_sorted)
        elif len(hashes) > 1:
            status = "conflicting_assets"
            conflicts.extend(rows_sorted)
        resolution.append(
            {
                "canonical_event_id": event_id,
                "storm_family_id": storm_family_id(event_id),
                "selected_asset_id": selected[event_id]["asset_id"],
                "selected_path": selected[event_id]["path"],
                "asset_count": len(rows_sorted),
                "resolution_status": status,
                "series_hash_count": len(hashes),
            }
        )
    files = [
        write_csv(out_dir / "rainfall_asset_inventory.csv", inventory, INDEX_COLUMNS),
        write_csv(out_dir / "rainfall_asset_duplicate_audit.csv", duplicates),
        write_csv(out_dir / "rainfall_asset_conflict_audit.csv", conflicts),
        write_csv(out_dir / "rainfall_asset_resolution_audit.csv", resolution),
        write_json(out_dir / "rainfall_asset_index_report.json", {"status": "completed", "created_at": utc_now(), "asset_count": len(inventory), "excluded_file_count": len(excluded), "unique_event_count": len(by_event), "conflicting_event_count": len({r["canonical_event_id"] for r in conflicts})}),
        write_csv(out_dir / "rainfall_asset_excluded_files.csv", excluded),
    ]
    return 0, {"status": "completed", "asset_count": len(inventory), "unique_event_count": len(by_event), "outputs": [str(p) for p in files]}, files


def load_selected_rainfall_assets(inventory_path: str | Path) -> dict[str, dict[str, str]]:
    rows = []
    path = Path(inventory_path)
    if not path.exists():
        return {}
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    selected: dict[str, dict[str, str]] = {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("canonical_event_id", "")].append(row)
    for event_id, event_rows in grouped.items():
        if not event_id:
            continue
        selected[event_id] = sorted(event_rows, key=lambda r: (int(r.get("source_priority") or 9), r.get("path", "")))[0]
    return selected
