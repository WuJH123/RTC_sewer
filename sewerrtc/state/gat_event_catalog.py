from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from sewerrtc._project_root import PROJECT_ROOT


def _sibling_project_root(name: str) -> Path:
    env_var = f"{name.upper()}_ROOT"
    return Path(os.environ.get(env_var, PROJECT_ROOT.parent / name))


PROJECT_ROOTS = [_sibling_project_root(f"Project{i}") for i in range(2, 7)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash an existing file or a directory tree without treating directories as files."""
    if not path.exists():
        return ""
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        return ""
    h = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        rel = child.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(child).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def _sha256_existing_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return sha256_file(path)


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _event_parts(event_id: str) -> dict[str, str]:
    m = re.match(r"^(T[^_]+)_D(\d+)_(.+)$", event_id)
    if not m:
        return {"rain_id": "", "duration_min": "", "pattern": "", "storm_family_id": event_id}
    return {
        "rain_id": m.group(1),
        "duration_min": m.group(2),
        "pattern": m.group(3),
        "storm_family_id": f"{m.group(1)}_{m.group(3)}",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def event_id_from_source(source: str) -> str:
    head = _norm(source).split(":", 1)[0]
    name = Path(head).name
    if "__" in name:
        return name.split("__", 1)[0]
    if name.endswith(".csv"):
        return name[:-4]
    return ""


def scan_validation_assets() -> list[dict[str, Any]]:
    patterns = ("*rainfall_event_table.csv", "*selected_events.csv", "*risk_stratified_event_table.csv", "*event*.csv", "*manifest*.csv", "*trajectory*.csv")
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in PROJECT_ROOTS:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                name = path.name.lower()
                asset_type = "event_table"
                if "rainfall" in name:
                    asset_type = "rainfall_library"
                elif "selected_events" in name:
                    asset_type = "selected_event_table"
                elif "trajectory" in name or "manifest" in name:
                    asset_type = "trajectory_or_manifest"
                rows.append(
                    {
                        "source_project": root.name,
                        "path": str(path),
                        "SHA256": sha256_file(path),
                        "asset_type": asset_type,
                        "network_path": "",
                        "network_sha256": "",
                        "event_identity_availability": "event_id" if _has_column(path, "event_id") else "unknown",
                        "rainfall_identity_availability": "rainfall_csv" if _has_column(path, "rainfall_csv") else "unknown",
                        "trajectory_identity_availability": "path_or_manifest" if asset_type == "trajectory_or_manifest" else "unknown",
                        "full_node_truth_availability": "unknown",
                        "sr0p15_sensor_availability": "unknown",
                        "timestamps_availability": "unknown",
                        "eligibility_status": "inventory_only",
                    }
                )
    return rows


def _has_column(path: Path, column: str) -> bool:
    rows = _read_csv(path)
    return bool(rows and column in rows[0])


def candidate_events_from_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_event: dict[str, dict[str, Any]] = {}
    for asset in assets:
        path = Path(asset["path"])
        for row in _read_csv(path):
            event_id = _norm(row.get("event_id") or row.get("canonical_event_id") or row.get("rain_id"))
            if not event_id or event_id.startswith("T") is False:
                continue
            parts = _event_parts(event_id)
            rainfall_path = _norm(row.get("rainfall_csv") or row.get("rainfall_path") or row.get("rain_path"))
            rainfall_sha = _sha256_existing_file(Path(rainfall_path)) if rainfall_path else ""
            trajectory_path = _norm(row.get("trajectory_path") or row.get("detail_file") or row.get("internal_detail_file"))
            cache_path = _norm(row.get("cache_path"))
            cache_sha = _norm(row.get("cache_sha256"))
            if cache_path and not cache_sha:
                cache_sha = _sha256_existing_file(Path(cache_path))
            existing = by_event.setdefault(
                event_id,
                {
                    "event_id": event_id,
                    "canonical_event_id": event_id,
                    "storm_family_id": parts["storm_family_id"],
                    "source_project": asset["source_project"],
                    "rainfall_path": rainfall_path,
                    "rainfall_file_sha256": rainfall_sha,
                    "rainfall_series_sha256": rainfall_sha,
                    "trajectory_path": trajectory_path,
                    "trajectory_sha256": "",
                    "trajectory_path_type": "",
                    "network_path": _norm(row.get("network_path")),
                    "network_sha256": "",
                    "cache_path": cache_path,
                    "cache_sha256": cache_sha,
                    "full_node_truth_available": "false",
                    "sr0p15_sensor_available": "false",
                    "timestamps_available": "false",
                    "has_60min_history": "false",
                    "split": _norm(row.get("split") or row.get("intended_split") or "candidate_holdout"),
                    "provenance_path": asset["path"],
                    "provenance_sha256": asset["SHA256"],
                },
            )
            if rainfall_path and not existing.get("rainfall_path"):
                existing["rainfall_path"] = rainfall_path
                existing["rainfall_file_sha256"] = rainfall_sha
                existing["rainfall_series_sha256"] = rainfall_sha
            if cache_path:
                existing["cache_path"] = cache_path
                existing["cache_sha256"] = cache_sha
            if trajectory_path:
                existing["trajectory_path"] = trajectory_path
                trajectory_obj = Path(trajectory_path)
                if trajectory_obj.exists():
                    existing["trajectory_sha256"] = sha256_path(trajectory_obj)
                    existing["trajectory_path_type"] = "directory" if trajectory_obj.is_dir() else "file"
                    existing["timestamps_available"] = "true"
                    # Detailed SWMM CSVs are path evidence, but full 932-node truth
                    # must be proven by later state builders before eligible lock.
                    existing["full_node_truth_available"] = _norm(row.get("full_node_truth_available") or "false")
                    existing["sr0p15_sensor_available"] = _norm(row.get("sr0p15_sensor_available") or existing.get("sr0p15_sensor_available") or "false")
                    existing["has_60min_history"] = _norm(row.get("has_60min_history") or "false")
    return list(by_event.values())


def contaminated_from_sources(sources: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: dict[str, dict[str, Any]] = {}
    family_rows: dict[str, dict[str, Any]] = {}
    for source in sources:
        event_id = event_id_from_source(source)
        if not event_id:
            continue
        parts = _event_parts(event_id)
        event_rows[event_id] = {
            "event_id": event_id,
            "storm_family_id": parts["storm_family_id"],
            "contamination_source": "project4_gat_cache_sources",
            "source_ref": source,
        }
        family_rows[parts["storm_family_id"]] = {
            "storm_family_id": parts["storm_family_id"],
            "example_event_id": event_id,
            "contamination_source": "project4_gat_cache_sources",
        }
    return list(event_rows.values()), list(family_rows.values()), []


def load_cache_sources(cache_path: Path) -> list[str]:
    if not cache_path.exists():
        return []
    try:
        import numpy as np

        with np.load(cache_path, allow_pickle=True) as z:
            if "sources" not in z.files:
                return []
            return [str(value) for value in z["sources"].reshape(-1)]
    except Exception:
        return []


def write_event_catalog_outputs(out_dir: Path, assets: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    _write_csv(out_dir / "gat_validation_asset_inventory.csv", assets)
    _write_csv(out_dir / "gat_independent_validation_candidates.csv", candidates)
