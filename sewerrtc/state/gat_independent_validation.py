from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sewerrtc._project_root import PROJECT_ROOT

from sewerrtc.state.gat_event_catalog import (
    candidate_events_from_assets,
    contaminated_from_sources,
    load_cache_sources,
    scan_validation_assets,
    sha256_file,
    write_event_catalog_outputs,
)
from sewerrtc.state.gat_holdout_eligibility import ELIGIBLE, REQUIRES_NEW_TRAJECTORY, classify_holdout_candidate


OUT_ROOT = PROJECT_ROOT / "outputs" / "project6_pfvfirst_dualfallback_10min_v3"
GAT_DIR = OUT_ROOT / "gat"
_PROJECT4_ROOT = Path(os.environ.get("PROJECT4_ROOT", PROJECT_ROOT.parent / "Project4"))
PROJECT4_CACHE = _PROJECT4_ROOT / "outputs" / "cache_paired_no_controls" / "transition_cache.npz"
EXPECTED_SR0P15_SHA256 = "11f40e6a36016202139e604f04c7d888b5ec3805511c46172ad968a7c20d0e20"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _hash_rows(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_contaminated_sets(gat_dir: Path = GAT_DIR) -> dict[str, set[str]]:
    sources = load_cache_sources(PROJECT4_CACHE)
    event_rows, family_rows, rain_rows = contaminated_from_sources(sources)
    # Preserve known leakage rows from the current diagnostic audit as explicit
    # contamination evidence.
    for row in _read_csv(gat_dir / "gat_sr0p15_validation_leakage_audit.csv"):
        event_id = row.get("validation_event") or row.get("matching_training_event")
        family = row.get("evidence") if row.get("match_type") == "storm_family_overlap" else ""
        if event_id:
            event_rows.append({"event_id": event_id, "storm_family_id": family, "contamination_source": "current_leakage_audit", "source_ref": row.get("match_type", "")})
        if family:
            family_rows.append({"storm_family_id": family, "example_event_id": event_id, "contamination_source": "current_leakage_audit"})
    return {
        "event_ids": {row["event_id"] for row in event_rows if row.get("event_id")},
        "storm_families": {row["storm_family_id"] for row in family_rows if row.get("storm_family_id")},
        "rainfall_hashes": {row.get("rainfall_series_sha256", "") for row in rain_rows if row.get("rainfall_series_sha256")},
        "trajectory_hashes": set(),
    }


def build_gat_independent_validation_catalog(gat_dir: Path = GAT_DIR) -> dict[str, Any]:
    gat_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest_rows = _read_csv(gat_dir / "gat_independent_validation_manifest.csv")
    assets = scan_validation_assets()
    candidates = candidate_events_from_assets(assets)

    sources = load_cache_sources(PROJECT4_CACHE)
    contaminated_events, contaminated_families, contaminated_rain_hashes = contaminated_from_sources(sources)
    contaminated = build_contaminated_sets(gat_dir)

    exclusion_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        decision = classify_holdout_candidate(candidate, contaminated)
        row = dict(candidate)
        row.update(
            {
                "eligibility_status": decision.status,
                "exclusion_reason": decision.exclusion_reason,
                "match_type": decision.match_type,
                "matching_contaminated_event": decision.matching_contaminated_event,
            }
        )
        candidate_rows.append(row)
        exclusion_rows.append(
            {
                "event_id": candidate.get("event_id", ""),
                "storm_family_id": candidate.get("storm_family_id", ""),
                "eligibility_status": decision.status,
                "exclusion_reason": decision.exclusion_reason,
                "match_type": decision.match_type,
                "matching_contaminated_event": decision.matching_contaminated_event,
                "rainfall_path": candidate.get("rainfall_path", ""),
                "rainfall_file_sha256": candidate.get("rainfall_file_sha256", ""),
            }
        )
        if decision.status == ELIGIBLE:
            manifest_rows.append(
                {
                    "holdout_id": f"gat_holdout_{len(manifest_rows):04d}",
                    "event_id": candidate.get("event_id", ""),
                    "storm_family_id": candidate.get("storm_family_id", ""),
                    "source_project": candidate.get("source_project", ""),
                    "split": "gat_independent_holdout",
                    "rainfall_path": candidate.get("rainfall_path", ""),
                    "rainfall_file_sha256": candidate.get("rainfall_file_sha256", ""),
                    "rainfall_series_sha256": candidate.get("rainfall_series_sha256", ""),
                    "trajectory_path": candidate.get("trajectory_path", ""),
                    "trajectory_sha256": candidate.get("trajectory_sha256", ""),
                    "network_path": candidate.get("network_path", ""),
                    "network_sha256": candidate.get("network_sha256", ""),
                    "timestamp_range": "",
                    "node_truth_path": candidate.get("trajectory_path", ""),
                    "sensor_input_path": candidate.get("trajectory_path", ""),
                    "sample_count": "",
                    "full_node_truth_available": candidate.get("full_node_truth_available", ""),
                    "sr0p15_sensor_available": candidate.get("sr0p15_sensor_available", ""),
                    "timestamps_available": candidate.get("timestamps_available", ""),
                    "has_60min_history": candidate.get("has_60min_history", ""),
                    "high_water_support": "",
                    "eligibility_evidence": candidate.get("provenance_path", ""),
                    "exclusion_audit_hash": "",
                }
            )
        elif decision.status == REQUIRES_NEW_TRAJECTORY:
            plan_rows.append(
                {
                    "planned_holdout_id": f"gat_planned_holdout_{len(plan_rows):04d}",
                    "rainfall_path": candidate.get("rainfall_path", ""),
                    "rainfall_file_sha256": candidate.get("rainfall_file_sha256", ""),
                    "rainfall_series_sha256": candidate.get("rainfall_series_sha256", ""),
                    "event_id": candidate.get("event_id", ""),
                    "storm_family_id": candidate.get("storm_family_id", ""),
                    "target_network": str(PROJECT_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"),
                    "required_outputs": "932-node depth truth, sr0p15 sensor input, strict timestamps, 60min history",
                    "required_timestep": "control-compatible <=10min source timestamps",
                    "required_node_truth": "all 932 nodes",
                    "required_sensor_input": "sr0p15 134 sensors",
                    "expected_duration": "",
                    "policy_mode": "no_control_or_internal_for_state_validation_only",
                    "intended_split": "gat_independent_holdout",
                    "provenance": candidate.get("provenance_path", ""),
                    "independence_evidence": decision.exclusion_reason,
                }
            )

    # A previous GenerateGATIndependentHoldoutTrajectories stage may have
    # created a real sr0p15 cache and manifest. Preserve those rows during a
    # later catalog refresh instead of overwriting them with the pre-generation
    # "requires_new_trajectory" scan.
    existing_manifest_keys = {(row.get("event_id", ""), row.get("cache_path", "")) for row in manifest_rows}
    for row in existing_manifest_rows:
        if not row.get("cache_path"):
            continue
        decision = classify_holdout_candidate(row, contaminated)
        if decision.status != ELIGIBLE:
            exclusion_rows.append(
                {
                    "event_id": row.get("event_id", ""),
                    "storm_family_id": row.get("storm_family_id", ""),
                    "eligibility_status": decision.status,
                    "exclusion_reason": decision.exclusion_reason,
                    "match_type": decision.match_type,
                    "matching_contaminated_event": decision.matching_contaminated_event,
                    "rainfall_path": row.get("rainfall_path", ""),
                    "rainfall_file_sha256": row.get("rainfall_file_sha256", ""),
                }
            )
            continue
        key = (row.get("event_id", ""), row.get("cache_path", ""))
        if key not in existing_manifest_keys:
            manifest_rows.append(dict(row))
            existing_manifest_keys.add(key)

    for row in manifest_rows:
        row["exclusion_audit_hash"] = _hash_rows(exclusion_rows)

    write_event_catalog_outputs(gat_dir, assets, candidate_rows)
    _write_csv(gat_dir / "gat_independent_validation_exclusion_audit.csv", exclusion_rows)
    _write_csv(gat_dir / "gat_contaminated_event_manifest.csv", contaminated_events)
    _write_csv(gat_dir / "gat_contaminated_storm_family_manifest.csv", contaminated_families)
    _write_csv(gat_dir / "gat_contaminated_rainfall_hashes.csv", contaminated_rain_hashes)
    _write_csv(gat_dir / "gat_model_selection_event_manifest.csv", contaminated_events)
    _write_csv(gat_dir / "gat_independent_validation_manifest.csv", manifest_rows)
    _write_csv(gat_dir / "gat_independent_trajectory_plan.csv", plan_rows)

    families = {row.get("storm_family_id") for row in manifest_rows if row.get("storm_family_id")}
    support_status = "sufficient_candidate_diversity" if len(manifest_rows) > 1 and len(families) > 1 else "insufficient_diversity"
    summary = {
        "status": "completed",
        "current_validation_status": "failed_due_to_exact_training_event_leakage",
        "robustness_status": "pending_independent_holdout_validation" if manifest_rows else "pending_new_independent_holdout",
        "eligible_event_count": len(manifest_rows),
        "eligible_storm_family_count": len(families),
        "requires_new_trajectory_count": len(plan_rows),
        "support_status": support_status,
        "support_threshold_status": "human_freeze_required",
        "round0_unlock_allowed": False,
        "allowed_to_enter_prompt3a": False,
        "created_at": _now(),
    }
    _write_json(gat_dir / "gat_independent_validation_catalog_report.json", summary)
    if not manifest_rows:
        _write_json(
            gat_dir / "gat_independent_validation_gap_report.json",
            {
                **summary,
                "missing_independent_events": "no existing event currently satisfies all holdout eligibility criteria",
                "contaminated_storm_family_count": len(contaminated.get("storm_families", set())),
                "required_next_step": "generate trajectories listed in gat_independent_trajectory_plan.csv, then rebuild catalog and lock manifest",
            },
        )
    return summary


def lock_gat_independent_validation_manifest(manifest_path: Path, gat_dir: Path = GAT_DIR, *, acknowledge: bool = False) -> tuple[int, Path]:
    lock_path = gat_dir / "gat_independent_validation_lock.json"
    if not acknowledge:
        return 7, lock_path
    rows = _read_csv(manifest_path)
    if not rows:
        return 3, lock_path
    contaminated = build_contaminated_sets(gat_dir)
    bad: list[str] = []
    for row in rows:
        cache_path = Path(str(row.get("cache_path", "")))
        if row.get("cache_path") and (not cache_path.exists() or not cache_path.is_file()):
            bad.append(f"{row.get('event_id', '')}:cache_path_missing")
            continue
        decision = classify_holdout_candidate(row, contaminated)
        if decision.status != ELIGIBLE:
            bad.append(f"{row.get('event_id', '')}:{decision.exclusion_reason}")
    if bad:
        _write_json(
            gat_dir / "gat_independent_validation_lock_rejected.json",
            {"status": "rejected", "reasons": bad, "created_at": _now()},
        )
        return 5, lock_path
    manifest_hash = sha256_file(manifest_path)
    families = {row.get("storm_family_id") for row in rows if row.get("storm_family_id")}
    lock = {
        "status": "locked",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "event_count": len(rows),
        "storm_family_count": len(families),
        "contaminated_set_hash": _hash_rows(_read_csv(gat_dir / "gat_contaminated_event_manifest.csv")),
        "exclusion_audit_hash": sha256_file(gat_dir / "gat_independent_validation_exclusion_audit.csv") if (gat_dir / "gat_independent_validation_exclusion_audit.csv").exists() else "",
        "network_hash": sha256_file(PROJECT_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"),
        "sr0p15_checkpoint_hash": EXPECTED_SR0P15_SHA256,
        "locked_at": _now(),
        "acknowledgement": True,
        "allowed_for_robustness_audit": True,
        "allowed_for_model_tuning": False,
    }
    _write_json(lock_path, lock)
    return 0, lock_path
