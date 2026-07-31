from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gat_audit import config_inp_path, sha256_file


SELECTED_REGISTRY_NAME = "sr0p15"
EXPECTED_SENSOR_COUNT = 134
EXPECTED_CHECKPOINT_SHA256 = "11f40e6a36016202139e604f04c7d888b5ec3805511c46172ad968a7c20d0e20"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _find_candidate(report: dict[str, Any], registry_name: str) -> dict[str, Any] | None:
    for candidate in report.get("candidates", []):
        if candidate.get("registry_name") == registry_name:
            return candidate
    return None


def _require(condition: bool, reason: str, failures: list[str]) -> None:
    if not condition:
        failures.append(reason)


def validate_primary_gat_selection(
    *,
    config_path: Path,
    decision_path: Path,
    gat_dir: Path,
    registry_name: str,
    acknowledgement: bool,
) -> tuple[bool, list[str], dict[str, Any]]:
    failures: list[str] = []
    decision = _read_json(decision_path) if decision_path.exists() else {}
    _require(acknowledgement, "missing_acknowledgement", failures)
    _require(registry_name == SELECTED_REGISTRY_NAME, "registry_name_must_be_sr0p15", failures)
    _require(decision.get("registry_name") == SELECTED_REGISTRY_NAME, "decision_contract_registry_not_sr0p15", failures)
    _require(decision.get("expected_sensor_count") == EXPECTED_SENSOR_COUNT, "decision_expected_sensor_count_not_134", failures)
    _require(
        str(decision.get("expected_checkpoint_sha256", "")).lower() == EXPECTED_CHECKPOINT_SHA256,
        "decision_checkpoint_hash_mismatch",
        failures,
    )

    report_path = gat_dir / "gat_compatibility_report.json"
    hashes_path = gat_dir / "gat_checkpoint_hashes.csv"
    strict_path = gat_dir / "gat_strict_load_audit.csv"
    node_path = gat_dir / "gat_node_mapping.csv"
    sensor_path = gat_dir / "gat_sensor_mapping.csv"
    graph_path = gat_dir / "gat_graph_signature_audit.csv"
    registry_path = gat_dir / "gat_external_registry.csv"
    required_paths = [report_path, hashes_path, strict_path, node_path, sensor_path, graph_path, registry_path]
    for path in required_paths:
        _require(path.exists(), f"missing_required_report:{path.name}", failures)

    candidate: dict[str, Any] | None = None
    checkpoint_hash: str | None = None
    strict_status = None
    node_rows: list[dict[str, str]] = []
    sensor_rows: list[dict[str, str]] = []
    graph_rows: list[dict[str, str]] = []
    registry_row: dict[str, str] = {}
    if report_path.exists():
        report = _read_json(report_path)
        candidate = _find_candidate(report, registry_name)
        _require(candidate is not None, "selected_candidate_missing_from_compatibility_report", failures)
        if candidate is not None:
            _require(candidate.get("compatibility_status") == "compatible_strict", "sr0p15_not_compatible_strict", failures)
            _require(candidate.get("strict_load_status") == "strict_loaded", "sr0p15_not_strict_loaded", failures)
            _require(int(candidate.get("sensor_count", -1)) == EXPECTED_SENSOR_COUNT, "sr0p15_sensor_count_not_134", failures)
            _require(candidate.get("node_count") == candidate.get("project6_node_count"), "sr0p15_node_count_mismatch", failures)
    if hashes_path.exists():
        for row in _read_csv(hashes_path):
            if row.get("registry_name") == registry_name:
                checkpoint_hash = str(row.get("sha256", "")).lower()
                break
        _require(checkpoint_hash == EXPECTED_CHECKPOINT_SHA256, "current_checkpoint_hash_mismatch", failures)
    if strict_path.exists():
        for row in _read_csv(strict_path):
            if row.get("registry_name") == registry_name:
                strict_status = row.get("strict_load_status")
                break
        _require(strict_status == "strict_loaded", "strict_load_status_not_pass", failures)
    if node_path.exists():
        node_rows = [row for row in _read_csv(node_path) if row.get("registry_name") == registry_name]
        _require(len(node_rows) > 0, "sr0p15_node_mapping_missing", failures)
        _require(all(row.get("mapping_status") == "mapped" for row in node_rows), "sr0p15_node_mapping_not_complete", failures)
    if sensor_path.exists():
        sensor_rows = [row for row in _read_csv(sensor_path) if row.get("registry_name") == registry_name]
        _require(len(sensor_rows) == EXPECTED_SENSOR_COUNT, "sr0p15_sensor_mapping_count_not_134", failures)
        _require(all(row.get("exists") == "True" for row in sensor_rows), "sr0p15_sensor_missing_in_project6", failures)
        _require(all(row.get("duplicate") == "False" for row in sensor_rows), "sr0p15_sensor_duplicates_present", failures)
    if graph_path.exists():
        graph_rows = [row for row in _read_csv(graph_path) if row.get("registry_name") == registry_name]
        p6_rows = [row for row in graph_rows if row.get("graph_name") == "project6_retrofit_inp_graph"]
        _require(
            any(row.get("comparison_status") == "matches_project4_edge_set" for row in p6_rows),
            "sr0p15_directed_edge_set_not_confirmed",
            failures,
        )
    if registry_path.exists():
        for row in _read_csv(registry_path):
            if row.get("registry_name") == registry_name:
                registry_row = row
                break
        _require(bool(registry_row), "sr0p15_registry_row_missing", failures)

    network_path = config_inp_path(config_path)
    evidence = {
        "decision": decision,
        "candidate": candidate or {},
        "checkpoint_hash": checkpoint_hash,
        "strict_load_status": strict_status,
        "node_mapping_rows": len(node_rows),
        "sensor_mapping_rows": len(sensor_rows),
        "registry_row": registry_row,
        "graph_rows": graph_rows,
        "report_hashes": {path.name: sha256_file(path) for path in required_paths if path.exists()},
        "network_path": str(network_path),
        "network_sha256": sha256_file(network_path),
        "config_sha256": sha256_file(config_path),
        "decision_sha256": sha256_file(decision_path),
    }
    return not failures, failures, evidence


def _hash_evidence(value: Any, source_file: Path, source_field: str) -> dict[str, Any]:
    return {
        "value": value,
        "source_file": str(source_file),
        "source_field": source_field,
        "hash_algorithm": "sha256",
        "canonicalization_version": "project6_v3_csv_json_canonical_v1",
    }


def build_primary_gat_lock(
    *,
    config_path: Path,
    decision_path: Path,
    gat_dir: Path,
    script_path: Path,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    candidate = evidence["candidate"]
    decision = evidence["decision"]
    registry_row = evidence.get("registry_row", {})
    strict_path = gat_dir / "gat_strict_load_audit.csv"
    graph_path = gat_dir / "gat_graph_signature_audit.csv"
    node_path = gat_dir / "gat_node_mapping.csv"
    sensor_path = gat_dir / "gat_sensor_mapping.csv"
    registry_path = gat_dir / "gat_external_registry.csv"
    project6_graph_rows = [
        row
        for row in evidence.get("graph_rows", [])
        if row.get("graph_name") == "project6_retrofit_inp_graph"
    ]
    edge_set_hash = (project6_graph_rows[0].get("directed_edge_list_hash") if project6_graph_rows else "") or candidate.get("graph_signature")
    return {
        "registry_name": SELECTED_REGISTRY_NAME,
        "sensor_ratio": decision.get("declared_sensor_ratio", 0.15),
        "sensor_count": candidate.get("sensor_count", EXPECTED_SENSOR_COUNT),
        "checkpoint_path": candidate.get("source_path") or decision.get("checkpoint_path"),
        "checkpoint_sha256": evidence.get("checkpoint_hash"),
        "model_class": "sewerrtc.models.gat_reconstructor.SparseGATReconstructor",
        "model_signature": {
            "node_count": candidate.get("node_count"),
            "input_dim": 10,
            "output_dim": candidate.get("node_count"),
            "hidden_dim": 256,
            "gat_heads": 4,
        },
        "state_dict_signature": _hash_evidence(registry_row.get("state_dict_key_signature"), registry_path, "state_dict_key_signature"),
        "node_ids_hash": _hash_evidence(candidate.get("node_order_hash"), node_path, "node_order_hash"),
        "node_order_hash": _hash_evidence(candidate.get("node_order_hash"), node_path, "ordered_training_node_sequence"),
        "edge_set_hash": _hash_evidence(edge_set_hash, graph_path, "directed_edge_list_hash"),
        "sensor_ids_hash": _hash_evidence(candidate.get("sensor_ids_hash"), sensor_path, "sensor_ids_hash"),
        "static_tensor_hash": _hash_evidence(registry_row.get("normalization_hash") or candidate.get("normalization_hash"), registry_path, "normalization_hash"),
        "strict_load_status": candidate.get("strict_load_status"),
        "compatibility_status": candidate.get("compatibility_status"),
        "human_selection_decision_path": str(decision_path),
        "human_selection_decision_sha256": evidence.get("decision_sha256"),
        "acknowledgement": True,
        "primary_gat_selected": True,
        "robustness_status": "pending",
        "round0_unlock_allowed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(script_path),
        "config_sha256": evidence.get("config_sha256"),
        "network_sha256": evidence.get("network_sha256"),
        "report_hashes": evidence.get("report_hashes", {}),
    }


def write_primary_gat_lock(
    *,
    config_path: Path,
    decision_path: Path,
    gat_dir: Path,
    script_path: Path,
    registry_name: str,
    acknowledgement: bool,
) -> tuple[int, dict[str, Any]]:
    ok, failures, evidence = validate_primary_gat_selection(
        config_path=config_path,
        decision_path=decision_path,
        gat_dir=gat_dir,
        registry_name=registry_name,
        acknowledgement=acknowledgement,
    )
    if not ok:
        return 6, {"status": "contract_mismatch", "failures": failures, "lock_written": False}
    lock = build_primary_gat_lock(
        config_path=config_path,
        decision_path=decision_path,
        gat_dir=gat_dir,
        script_path=script_path,
        evidence=evidence,
    )
    out = gat_dir / "gat_primary_selection_lock.json"
    out.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return 0, {"status": "locked", "lock_path": str(out), "lock": lock}
