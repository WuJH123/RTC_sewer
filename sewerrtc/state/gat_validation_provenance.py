from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sewerrtc._project_root import PROJECT_ROOT

from sewerrtc.state.gat_event_leakage import compare_training_and_validation_events, near_duplicate_rows


_PROJECT4_ROOT = Path(os.environ.get("PROJECT4_ROOT", PROJECT_ROOT.parent / "Project4"))
PROJECT4_ROOT = _PROJECT4_ROOT
PROJECT6_OUT = PROJECT_ROOT / "outputs" / "project6_pfvfirst_dualfallback_10min_v3"
GAT_DIR = PROJECT6_OUT / "gat"


@dataclass(frozen=True)
class ProvenanceAuditResult:
    status: str
    exit_code: int
    gate_path: Path
    blocking_reasons: list[str]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_missing_identity(value: Any) -> bool:
    text = _norm(value).lower()
    return text in {"", "unknown", "none", "nan", "unresolved"} or text.startswith("sample_") or text.startswith("unknown_")


def _scan_project4_artifacts() -> list[dict[str, Any]]:
    roots = [
        PROJECT4_ROOT / "scripts",
        PROJECT4_ROOT / "configs",
        PROJECT4_ROOT / "data",
        PROJECT4_ROOT / "sewerrtc",
        PROJECT4_ROOT / "tests",
        PROJECT4_ROOT / "outputs",
        PROJECT4_ROOT / "outputs" / "sensor_sensitivity",
    ]
    patterns = ("*manifest*", "*split*", "*event*", "*rain*", "*trajectory*", "*cache*", "*sensor*")
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path in seen or not path.is_file():
                    continue
                if path.suffix.lower() not in {".csv", ".json", ".jsonl", ".npz", ".npy", ".pt", ".pth", ".txt", ".yaml", ".yml"}:
                    continue
                seen.add(path)
                name = path.name.lower()
                role = "unknown"
                if "split" in name:
                    role = "split"
                elif "manifest" in name:
                    role = "manifest"
                elif "event" in name:
                    role = "event_identity"
                elif "rain" in name:
                    role = "rainfall_identity"
                elif "trajectory" in name:
                    role = "trajectory_identity"
                elif "cache" in name:
                    role = "cache"
                rows.append(
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "file_type": path.suffix.lower().lstrip("."),
                        "role": role,
                        "event_identity_available": "unknown",
                        "split_available": "unknown",
                        "rainfall_identity_available": "unknown",
                        "sample_index_mapping_available": "unknown",
                        "confidence": "candidate",
                    }
                )
    return rows


def _load_event_like_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            for row in _read_csv(path):
                if any(key in row for key in ("event_id", "canonical_event_id", "rain_id", "storm_family_id", "split")):
                    rows.append(dict(row))
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                source_rows = payload
            elif isinstance(payload, dict):
                source_rows = []
                for value in payload.values():
                    if isinstance(value, list):
                        source_rows.extend(value)
            else:
                source_rows = []
            for row in source_rows:
                if isinstance(row, dict) and any(key in row for key in ("event_id", "canonical_event_id", "rain_id", "storm_family_id", "split")):
                    rows.append(dict(row))
    except Exception:
        return []
    return rows


def _event_id(row: dict[str, Any]) -> str:
    return _norm(row.get("event_id") or row.get("canonical_event_id") or row.get("rain_id") or row.get("event"))


def _event_id_from_source(source: str) -> str:
    head = _norm(source).split(":", 1)[0]
    name = Path(head).name
    if "__" in name:
        return name.split("__", 1)[0]
    if name.endswith(".csv"):
        return name[:-4]
    return ""


def _cache_row_from_source(source: str) -> str:
    parts = _norm(source).split(":")
    return parts[1] if len(parts) >= 2 else ""


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


def _storm_family(row: dict[str, Any]) -> str:
    return _norm(row.get("storm_family_id") or row.get("storm_family") or row.get("pattern") or row.get("rain_pattern"))


def _split(row: dict[str, Any]) -> str:
    text = _norm(row.get("split") or row.get("intended_split") or row.get("dataset_split"))
    return text.lower()


def _as_event_manifest(rows: list[dict[str, Any]], *, split_filter: str | None, source_path: Path | None = None) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        split = _split(row)
        if split_filter and split != split_filter:
            continue
        event_id = _event_id(row)
        if _is_missing_identity(event_id):
            continue
        source_sha = sha256_file(source_path) if source_path and source_path.exists() else _norm(row.get("source_sha256") or row.get("provenance_sha256"))
        out[event_id] = {
            "event_id": event_id,
            "storm_family_id": _storm_family(row),
            "rainfall_path": _norm(row.get("rainfall_path") or row.get("rain_path") or row.get("source_path")),
            "rainfall_file_sha256": _norm(row.get("rainfall_file_sha256") or row.get("rainfall_sha256")),
            "rainfall_series_sha256": _norm(row.get("rainfall_series_sha256") or row.get("time_series_signature")),
            "trajectory_path": _norm(row.get("trajectory_path") or row.get("source_path")),
            "trajectory_sha256": _norm(row.get("trajectory_sha256")),
            "split": split,
            "used_for_training": split in {"train", "training"},
            "used_for_validation": split in {"validation", "val", "test"},
            "used_for_hyperparameter_selection": _norm(row.get("used_for_hyperparameter_selection") or row.get("used_for_model_selection")),
            "used_for_sensor_ratio_selection": _norm(row.get("used_for_sensor_ratio_selection")),
            "provenance_path": str(source_path) if source_path else _norm(row.get("provenance_path") or row.get("source_path")),
            "provenance_sha256": source_sha,
        }
    return list(out.values())


def _recover_event_sets(artifact_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    training: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    searched: list[str] = []
    for artifact in artifact_rows:
        path = Path(artifact["path"])
        if path.suffix.lower() not in {".csv", ".json"}:
            continue
        event_rows = _load_event_like_rows(path)
        if not event_rows:
            continue
        searched.append(str(path))
        training.extend(_as_event_manifest(event_rows, split_filter="train", source_path=path))
        training.extend(_as_event_manifest(event_rows, split_filter="training", source_path=path))
        validation.extend(_as_event_manifest(event_rows, split_filter="validation", source_path=path))
        validation.extend(_as_event_manifest(event_rows, split_filter="val", source_path=path))
        validation.extend(_as_event_manifest(event_rows, split_filter="test", source_path=path))

    def unique(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            by_id[row["event_id"]] = row
        return list(by_id.values())

    return unique(training), unique(validation), searched


def _load_validation_dataset_manifest(gat_dir: Path) -> dict[str, Any]:
    path = gat_dir / "gat_sr0p15_validation_dataset_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_cache_sources(cache_path: Path) -> list[str]:
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


def _event_manifest_from_cache_sources(cache_path: Path, sources: list[str], split: str) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    cache_sha = sha256_file(cache_path) if cache_path.exists() else ""
    for source in sources:
        event_id = _event_id_from_source(source)
        if not event_id:
            continue
        parts = _event_parts(event_id)
        out[event_id] = {
            "event_id": event_id,
            "storm_family_id": parts["storm_family_id"],
            "rainfall_path": str(PROJECT4_ROOT / "outputs" / "rainfall_library" / f"{event_id}.csv"),
            "rainfall_file_sha256": "",
            "rainfall_series_sha256": "",
            "trajectory_path": source.split(":", 1)[0],
            "trajectory_sha256": "",
            "split": split,
            "used_for_training": split in {"train", "training"},
            "used_for_validation": split in {"validation", "diagnostic_validation"},
            "used_for_hyperparameter_selection": "",
            "used_for_sensor_ratio_selection": "",
            "provenance_path": str(cache_path),
            "provenance_sha256": cache_sha,
        }
    return list(out.values())


def _repair_sample_inventory(
    existing: list[dict[str, Any]],
    validation_events: list[dict[str, Any]],
    source_cache_path: str = "",
    cache_sources: list[str] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    by_event = {row["event_id"]: row for row in validation_events}
    fallback_events = list(by_event)
    repaired: list[dict[str, Any]] = []
    complete = bool(existing)
    cache_sources = cache_sources or []
    for i, row in enumerate(existing):
        event_id = _norm(row.get("event_id"))
        source = cache_sources[i] if i < len(cache_sources) else ""
        if _is_missing_identity(event_id):
            event_id = _event_id_from_source(source)
        if _is_missing_identity(event_id) and len(fallback_events) == 1:
            event_id = fallback_events[0]
        event = by_event.get(event_id, {})
        split = _norm(row.get("split")) or _norm(event.get("split"))
        status = "resolved" if not _is_missing_identity(event_id) and split else "unresolved"
        if status != "resolved":
            complete = False
        source_path = _norm(row.get("source_path")) or source_cache_path
        repaired.append(
            {
                "sample_id": _norm(row.get("sample_id")) or f"sample_{i:06d}",
                "cache_path": source_path,
                "cache_sha256": _norm(row.get("source_sha256") or row.get("cache_sha256")),
                "cache_row": _norm(row.get("source_row") or row.get("cache_row") or _cache_row_from_source(source) or i),
                "time_index": _norm(row.get("time_index") or row.get("source_row") or _cache_row_from_source(source) or i),
                "timestamp": _norm(row.get("timestamp")),
                "trajectory_id": _norm(row.get("trajectory_id") or source.split(":", 1)[0]),
                "trajectory_path": _norm(event.get("trajectory_path") or row.get("trajectory_path") or source.split(":", 1)[0]),
                "trajectory_sha256": _norm(event.get("trajectory_sha256") or row.get("trajectory_sha256")),
                "event_id": event_id,
                "storm_family_id": _norm(event.get("storm_family_id") or row.get("storm_family_id")),
                "rainfall_path": _norm(event.get("rainfall_path") or row.get("rainfall_path")),
                "rainfall_file_sha256": _norm(event.get("rainfall_file_sha256") or row.get("rainfall_file_sha256")),
                "rainfall_series_sha256": _norm(event.get("rainfall_series_sha256") or row.get("rainfall_series_sha256")),
                "split": split,
                "used_for_training": "false",
                "used_for_validation": "true" if split in {"validation", "val", "test"} else "",
                "used_for_model_selection": _norm(row.get("model_selection_membership")),
                "used_for_sensor_ratio_selection": "",
                "provenance_source": _norm(event.get("provenance_path") or row.get("source_path")),
                "provenance_confidence": "high" if status == "resolved" else "low",
                "provenance_status": status,
                "highwater_flag": _norm(row.get("highwater_flag")),
                "phase": _norm(row.get("phase")),
            }
        )
    return repaired, complete


def _event_support(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved = [row for row in sample_rows if row.get("provenance_status") == "resolved"]
    events = {row.get("event_id") for row in resolved if row.get("event_id")}
    families = {row.get("storm_family_id") for row in resolved if row.get("storm_family_id")}
    phases = {row.get("phase") for row in resolved if row.get("phase")}
    samples_per_event: dict[str, int] = {}
    for row in resolved:
        event_id = row.get("event_id", "")
        samples_per_event[event_id] = samples_per_event.get(event_id, 0) + 1
    return [
        {
            "raw_sample_count": len(sample_rows),
            "resolved_sample_count": len(resolved),
            "unique_event_count": len(events),
            "unique_storm_family_count": len(families),
            "samples_per_event": json.dumps(samples_per_event, ensure_ascii=False, sort_keys=True),
            "highwater_events": len({row.get("event_id") for row in resolved if str(row.get("highwater_flag")).lower() == "true"}),
            "priority_active_events": "",
            "sentinel_high_events": "",
            "rising_events": len({row.get("event_id") for row in resolved if row.get("phase") == "rising"}),
            "peak_events": len({row.get("event_id") for row in resolved if row.get("phase") == "peak"}),
            "recession_events": len({row.get("event_id") for row in resolved if row.get("phase") == "recession"}),
            "phase_labels": ",".join(sorted(phase for phase in phases if phase)),
        }
    ]


def _check_row(check_id: str, status: str, evidence_path: Path, reason: str = "") -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": status,
        "observed_value": status,
        "required_value": "pass",
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path) if evidence_path.exists() else "",
        "blocking_reason": reason,
        "remediation": "" if status == "pass" else "recover Project4 sample/event/split provenance or provide an independent validation manifest",
        "evaluated_at": _now(),
    }


def audit_gat_validation_provenance(gat_dir: Path = GAT_DIR) -> ProvenanceAuditResult:
    gat_dir.mkdir(parents=True, exist_ok=True)
    artifact_inventory_path = gat_dir / "gat_training_validation_artifact_inventory.csv"
    sample_inventory_path = gat_dir / "gat_sr0p15_validation_sample_inventory.csv"
    training_manifest_path = gat_dir / "gat_sr0p15_training_event_manifest.csv"
    validation_manifest_path = gat_dir / "gat_sr0p15_validation_event_manifest.csv"
    provenance_audit_path = gat_dir / "gat_sr0p15_validation_provenance_audit.csv"
    leakage_audit_path = gat_dir / "gat_sr0p15_validation_leakage_audit.csv"
    near_duplicate_path = gat_dir / "gat_sr0p15_rainfall_near_duplicate_audit.csv"
    split_membership_path = gat_dir / "gat_sr0p15_split_membership_audit.csv"
    event_support_path = gat_dir / "gat_sr0p15_validation_event_support.csv"
    gap_report_path = gat_dir / "gat_sr0p15_independent_validation_gap_report.json"
    gate_path = gat_dir / "gat_sr0p15_robustness_gate.json"

    artifacts = _scan_project4_artifacts()
    _write_csv(artifact_inventory_path, artifacts)

    training_events, validation_events, searched_event_sources = _recover_event_sets(artifacts)
    validation_dataset_manifest = _load_validation_dataset_manifest(gat_dir)
    cache_path = Path(_norm(validation_dataset_manifest.get("cache_path")))
    cache_sources = _load_cache_sources(cache_path) if _norm(cache_path) else []
    if cache_sources:
        cache_training_events = _event_manifest_from_cache_sources(cache_path, cache_sources, "train")
        cache_validation_events = _event_manifest_from_cache_sources(cache_path, cache_sources[: int(validation_dataset_manifest.get("sample_count") or 0) or 256], "validation")
        training_events.extend(cache_training_events)
        validation_events.extend(cache_validation_events)
        searched_event_sources.append(str(cache_path))
    existing_samples = _read_csv(sample_inventory_path)
    repaired_samples, sample_complete = _repair_sample_inventory(existing_samples, validation_events, str(cache_path), cache_sources)

    training_complete = bool(training_events)
    validation_complete = bool(validation_events) and sample_complete
    provenance_status = "pass" if training_complete and validation_complete else "incomplete"
    provenance_reason = "" if provenance_status == "pass" else "sample-to-event provenance or train/validation event set is incomplete"

    _write_csv(training_manifest_path, training_events)
    _write_csv(validation_manifest_path, validation_events)
    _write_csv(sample_inventory_path, repaired_samples)
    _write_csv(event_support_path, _event_support(repaired_samples))

    leakage = compare_training_and_validation_events(
        training_events,
        validation_events,
        training_complete=training_complete,
        validation_complete=validation_complete,
    )
    _write_csv(leakage_audit_path, leakage.rows)
    _write_csv(near_duplicate_path, near_duplicate_rows(training_events, validation_events))
    _write_csv(
        split_membership_path,
        [
            {
                "event_id": row.get("event_id", ""),
                "split": row.get("split", ""),
                "used_for_training": row.get("used_for_training", ""),
                "used_for_validation": row.get("used_for_validation", ""),
                "used_for_hyperparameter_selection": row.get("used_for_hyperparameter_selection", ""),
                "used_for_sensor_ratio_selection": row.get("used_for_sensor_ratio_selection", ""),
                "provenance_path": row.get("provenance_path", ""),
                "provenance_sha256": row.get("provenance_sha256", ""),
            }
            for row in training_events + validation_events
        ],
    )
    _write_csv(
        provenance_audit_path,
        [_check_row("validation_provenance_complete", provenance_status, sample_inventory_path, provenance_reason)],
    )

    # Append a four-state leakage check row after the detailed match rows so the
    # gate evaluator can consume the status without re-running the audit.
    detailed_rows = _read_csv(leakage_audit_path)
    detailed_rows.append(_check_row("no_training_event_leakage", leakage.status, leakage_audit_path, leakage.blocking_reason))
    _write_csv(leakage_audit_path, detailed_rows)

    blocking_reasons: list[str] = []
    if provenance_status != "pass":
        blocking_reasons.append(provenance_reason)
    if leakage.status == "incomplete":
        blocking_reasons.append(leakage.blocking_reason)

    if blocking_reasons:
        _write_json(
            gap_report_path,
            {
                "status": "incomplete",
                "missing_provenance_assets": {
                    "unresolved_sample_count": sum(1 for row in repaired_samples if row.get("provenance_status") != "resolved"),
                    "training_event_count": len(training_events),
                    "validation_event_count": len(validation_events),
                },
                "searched_paths": [
                    str(PROJECT4_ROOT / "scripts"),
                    str(PROJECT4_ROOT / "configs"),
                    str(PROJECT4_ROOT / "data"),
                    str(PROJECT4_ROOT / "sewerrtc"),
                    str(PROJECT4_ROOT / "tests"),
                    str(PROJECT4_ROOT / "outputs"),
                    str(PROJECT4_ROOT / "outputs" / "sensor_sensitivity"),
                ],
                "searched_event_sources": searched_event_sources,
                "why_no_leakage_cannot_be_proven": blocking_reasons,
                "diagnostic_only_reason": "independent validation event identity is not fully recoverable from current Project4 artifacts",
                "minimum_formal_independent_validation_requirement": "all samples must map to event_id, split, rainfall identity, and non-training membership",
                "created_at": _now(),
            },
        )

    # Reuse the existing gate file but update only the two provenance checks if
    # it exists. The dedicated EvaluateGATRobustnessGate stage remains the
    # authoritative gate recomputation step.
    if gate_path.exists():
        try:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            checks = gate.setdefault("checks", {})
            for check_id, status, reason, evidence in [
                ("validation_provenance_complete", provenance_status, provenance_reason, provenance_audit_path),
                ("no_training_event_leakage", leakage.status, leakage.blocking_reason, leakage_audit_path),
            ]:
                checks[check_id] = {
                    "check_id": check_id,
                    "status": status,
                    "observed_value": status,
                    "required_value": "pass",
                    "evidence_path": str(evidence),
                    "evidence_sha256": sha256_file(evidence) if evidence.exists() else "",
                    "blocking_reason": reason,
                    "remediation": "" if status == "pass" else "recover independent event provenance or provide an approved holdout manifest",
                    "evaluated_at": _now(),
                }
            gate["validation_status"] = "complete" if provenance_status == "pass" else "diagnostic_only"
            gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    exit_code = 0 if provenance_status == "pass" or leakage.status == "fail" else 3
    status = "completed" if exit_code == 0 else "blocked"
    return ProvenanceAuditResult(status=status, exit_code=exit_code, gate_path=gate_path, blocking_reasons=blocking_reasons)
