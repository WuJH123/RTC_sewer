from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from sewerrtc.contracts.prompt3a import OUT_ROOT, read_csv, sha256_file, utc_now, write_csv, write_json
from sewerrtc.data.event_identity import canonical_event_id, storm_family_id
from sewerrtc.data.rainfall_asset_index import load_selected_rainfall_assets
from sewerrtc.data.split_contract import assign_split, audit_split_leakage
from sewerrtc.state.gat_event_catalog import scan_validation_assets


def build_event_catalog(out_dir: str | Path = OUT_ROOT / "event_catalog") -> tuple[int, dict[str, Any], list[Path]]:
    out_dir = Path(out_dir)
    assets = scan_validation_assets()
    holdout_rows = read_csv(OUT_ROOT / "gat" / "gat_independent_validation_manifest.csv")
    holdout_events = {row.get("event_id", "") for row in holdout_rows}
    rainfall_assets = load_selected_rainfall_assets(OUT_ROOT / "rainfall_assets" / "rainfall_asset_inventory.csv")
    rows: dict[str, dict[str, Any]] = {}
    for asset in assets:
        path = Path(asset["path"])
        for row in read_csv(path):
            event_id = row.get("event_id") or row.get("canonical_event_id") or row.get("rain_id") or ""
            if not event_id.startswith("T"):
                continue
            canonical = canonical_event_id(event_id)
            rainfall = row.get("rainfall_csv") or row.get("rainfall_path") or row.get("rain_path") or ""
            explicit_rainfall = Path(rainfall) if rainfall else None
            explicit_hash = sha256_file(explicit_rainfall) if explicit_rainfall and explicit_rainfall.is_file() else ""
            indexed = rainfall_assets.get(canonical, {})
            resolved_path = str(explicit_rainfall) if explicit_hash else indexed.get("path", "")
            resolved_file_hash = explicit_hash or indexed.get("file_sha256", "")
            resolved_series_hash = explicit_hash or indexed.get("rainfall_series_sha256", "")
            if explicit_hash:
                rainfall_resolution_status = "resolved_explicit_path"
            elif indexed:
                rainfall_resolution_status = "resolved_exact_basename"
            else:
                rainfall_resolution_status = "unresolved"
            record = rows.setdefault(
                event_id,
                {
                    "event_id": event_id,
                    "canonical_event_id": canonical,
                    "storm_family_id": storm_family_id(event_id),
                    "source_project": asset["source_project"],
                    "source_path": asset["path"],
                    "rainfall_path": resolved_path,
                    "rainfall_file_hash": resolved_file_hash,
                    "rainfall_file_sha256": resolved_file_hash,
                    "rainfall_series_hash": resolved_series_hash,
                    "rainfall_series_sha256": resolved_series_hash,
                    "rainfall_resolution_status": rainfall_resolution_status,
                    "start_time": "",
                    "end_time": "",
                    "duration_min": "",
                    "total_depth": "",
                    "peak_intensity": "",
                    "peak_time": "",
                    "peak_count": "",
                    "antecedent_condition": "",
                    "network_hash": "",
                    "trajectory_availability": "unknown",
                    "gat_training_seen": "unknown",
                    "gat_model_selection_seen": "unknown",
                    "gat_independent_holdout": str(event_id in holdout_events).lower(),
                    "action_effect_fit_eligible": str(event_id not in holdout_events and rainfall_resolution_status != "unresolved").lower(),
                    "round0_eligible": str(event_id not in holdout_events and rainfall_resolution_status != "unresolved").lower(),
                    "calibration_eligible": "false",
                    "formal_eligible": "false",
                    "near_duplicate_group": storm_family_id(event_id),
                    "provenance_status": "inventory",
                },
            )
            if not record.get("rainfall_path") and resolved_path:
                record["rainfall_path"] = resolved_path
                record["rainfall_file_hash"] = resolved_file_hash
                record["rainfall_file_sha256"] = resolved_file_hash
                record["rainfall_series_hash"] = resolved_series_hash
                record["rainfall_series_sha256"] = resolved_series_hash
                record["rainfall_resolution_status"] = rainfall_resolution_status
                if event_id not in holdout_events:
                    record["action_effect_fit_eligible"] = "true"
                    record["round0_eligible"] = "true"
            record["split"] = assign_split(record)
    catalog_rows = list(rows.values())
    unresolved = [row for row in catalog_rows if row.get("rainfall_resolution_status") == "unresolved"]
    leakage = audit_split_leakage(catalog_rows)
    files = [
        write_csv(out_dir / "event_catalog.csv", catalog_rows),
        write_json(out_dir / "event_provenance_audit.json", {"status": "completed", "created_at": utc_now(), "asset_count": len(assets), "event_count": len(catalog_rows)}),
        write_csv(out_dir / "event_near_duplicate_groups.csv", [{"near_duplicate_group": row["near_duplicate_group"], "event_id": row["event_id"], "storm_family_id": row["storm_family_id"]} for row in catalog_rows]),
        write_csv(out_dir / "event_split_manifest.csv", [{"event_id": row["event_id"], "storm_family_id": row["storm_family_id"], "split": row["split"], "round0_eligible": row["round0_eligible"]} for row in catalog_rows]),
        write_csv(out_dir / "event_split_leakage_audit.csv", leakage),
        write_csv(out_dir / "gat_seen_event_manifest.csv", [row for row in catalog_rows if row["gat_training_seen"] != "false"]),
        write_csv(out_dir / "gat_independent_holdout_event_manifest.csv", [row for row in catalog_rows if row["gat_independent_holdout"] == "true"]),
        write_csv(out_dir / "unresolved_rainfall_events.csv", unresolved),
    ]
    report = {"status": "completed" if not leakage else "failed_gate", "event_count": len(catalog_rows), "gat_holdout_event_count": len(holdout_events), "leakage_count": len(leakage), "unresolved_rainfall_event_count": len(unresolved), "round0_unlock_allowed": False}
    return (0 if not leakage else 5), report, files
