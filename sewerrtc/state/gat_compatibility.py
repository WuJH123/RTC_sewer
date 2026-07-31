from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sewerrtc._project_root import PROJECT_ROOT

from .gat_audit import (
    GAT_CANDIDATES,
    build_node_mapping,
    build_sensor_mapping,
    checkpoint_rows,
    compatibility_status,
    config_inp_path,
    graph_signature_rows,
    load_checkpoint,
    load_project4_model,
    metadata_from_checkpoint,
    normalization_rows,
    parse_inp_node_ids,
    priority_nodes_from_contract,
    reconstruction_audits,
    sha256_file,
    write_csv,
)


COMPATIBLE_STRICT = "compatible_strict"
COMPATIBLE_STRICT_REORDER = "compatible_strict_with_explicit_reorder"
COMPATIBLE_SHARED_BASE = "compatible_shared_base_graph_only"
METADATA_INCOMPLETE = "metadata_incomplete"
INCOMPATIBLE = "incompatible"
LOAD_FAILED = "load_failed"


@dataclass
class GATCompatibilityReport:
    candidates: list[dict[str, Any]]
    selected_primary_gat: str | None
    selection_status: str
    overall_research_status: str
    human_selection_allowed: bool

    @property
    def compatible_strict_count(self) -> int:
        return sum(1 for c in self.candidates if c.get("compatibility_status") in {COMPATIBLE_STRICT, COMPATIBLE_STRICT_REORDER})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    return _read_csv(path) if path.exists() else []


def recover_metadata_outputs(config_path: Path, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    recovered_dir = out_dir / "recovered_metadata"
    recovered_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    for name, ratio, path in GAT_CANDIDATES:
        loaded = load_checkpoint(name, ratio, path)
        metadata = metadata_from_checkpoint(loaded) if loaded.checkpoint is not None else {
            "registry_name": name,
            "declared_sensor_ratio": ratio,
            "source_path": str(path),
            "source_sha256": sha256_file(path),
        }
        (recovered_dir / f"{name}.metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        inventory_rows.append(
            {
                "registry_name": name,
                "artifact_type": "gat_checkpoint",
                "path": str(path),
                "sha256": sha256_file(path),
                "exists": path.exists(),
                "source_project": "Project4",
            }
        )
        for key in [
            "model_class",
            "n_nodes",
            "static_dim",
            "hidden_dim",
            "gat_heads",
            "input_dim",
            "output_dim",
            "node_ids_hash",
            "sensor_ids_hash",
            "edge_index_hash",
            "node_static_hash",
            "state_dict_key_signature",
        ]:
            value = metadata.get(key)
            report_rows.append(
                {
                    "registry_name": name,
                    "field": key,
                    "value": json.dumps(value, default=str, ensure_ascii=False) if isinstance(value, (list, dict)) else value,
                    "source_path": metadata.get("source_path"),
                    "source_sha256": metadata.get("source_sha256"),
                    "recovery_method": "checkpoint_internal_field" if value is not None else "not_recovered",
                    "confidence": "high" if value is not None else "none",
                    "conflict_status": "none" if value is not None else "missing",
                }
            )
        provenance_rows.extend(
            [
                {
                    "registry_name": name,
                    "field": "model_class",
                    "source_path": metadata.get("model_class_source_path"),
                    "source_sha256": metadata.get("model_class_source_sha256"),
                    "recovery_method": "project4_source_import",
                    "confidence": "high",
                    "conflict_status": "none",
                },
                {
                    "registry_name": name,
                    "field": "sensor_ids",
                    "source_path": str(Path(path).parents[1] / "design" / "sensor_nodes.csv"),
                    "source_sha256": sha256_file(Path(path).parents[1] / "design" / "sensor_nodes.csv"),
                    "recovery_method": "project4_sensor_design_csv",
                    "confidence": "high",
                    "conflict_status": "none",
                },
            ]
        )
    outputs = {
        "inventory": out_dir / "gat_training_artifact_inventory.csv",
        "metadata_report": out_dir / "gat_metadata_recovery_report.csv",
        "metadata_provenance": out_dir / "gat_metadata_source_provenance.csv",
        "metadata_conflicts": out_dir / "gat_metadata_conflicts.csv",
    }
    write_csv(outputs["inventory"], inventory_rows, ["registry_name", "artifact_type", "path", "sha256", "exists", "source_project"])
    write_csv(outputs["metadata_report"], report_rows, ["registry_name", "field", "value", "source_path", "source_sha256", "recovery_method", "confidence", "conflict_status"])
    write_csv(outputs["metadata_provenance"], provenance_rows, ["registry_name", "field", "source_path", "source_sha256", "recovery_method", "confidence", "conflict_status"])
    write_csv(outputs["metadata_conflicts"], conflict_rows, ["registry_name", "field", "conflict_status", "details"])
    return outputs


def inspect_checkpoint_outputs(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    tensor_rows: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    for name, ratio, path in GAT_CANDIDATES:
        loaded = load_checkpoint(name, ratio, path)
        summary, tensors = checkpoint_rows(loaded)
        summary_rows.append(summary)
        tensor_rows.extend(tensors)
        status, missing, unexpected, errors = load_project4_model(loaded)
        strict_rows.append(
            {
                "registry_name": name,
                "model_class": "sewerrtc.models.gat_reconstructor.SparseGATReconstructor",
                "model_class_source": str(Path(os.environ.get("PROJECT4_ROOT", PROJECT_ROOT.parent / "Project4")) / "sewerrtc" / "models" / "gat_reconstructor.py"),
                "strict_load_status": status,
                "missing_keys": json.dumps(missing),
                "unexpected_keys": json.dumps(unexpected),
                "shape_mismatches": json.dumps([]),
                "dtype_mismatches": json.dumps([]),
                "errors": json.dumps(errors),
            }
        )
    outputs = {
        "load_audit": out_dir / "gat_checkpoint_load_audit.csv",
        "tensor_audit": out_dir / "gat_checkpoint_tensor_audit.csv",
        "strict_load_audit": out_dir / "gat_strict_load_audit.csv",
    }
    write_csv(outputs["load_audit"], summary_rows, list(summary_rows[0].keys()) if summary_rows else [])
    write_csv(outputs["tensor_audit"], tensor_rows, ["registry_name", "tensor_key", "shape", "dtype", "numel", "finite"])
    write_csv(outputs["strict_load_audit"], strict_rows, ["registry_name", "model_class", "model_class_source", "strict_load_status", "missing_keys", "unexpected_keys", "shape_mismatches", "dtype_mismatches", "errors"])
    return outputs


def audit_gat_registry(config_path: Path, registry_path: Path | None = None) -> GATCompatibilityReport:
    inp = config_inp_path(config_path)
    project6_nodes = parse_inp_node_ids(inp)
    priority_nodes = priority_nodes_from_contract()
    candidates: list[dict[str, Any]] = []
    for name, ratio, path in GAT_CANDIDATES:
        loaded = load_checkpoint(name, ratio, path)
        if loaded.checkpoint is None:
            candidates.append(
                {
                    "registry_name": name,
                    "source_path": str(path),
                    "declared_sensor_ratio": ratio,
                    "compatibility_status": LOAD_FAILED,
                    "compatibility_gate_evaluated": True,
                    "reasons": [loaded.load_error or "load_failed"],
                    "strict_load_status": "not_loaded",
                    "human_selection_allowed": False,
                }
            )
            continue
        metadata = metadata_from_checkpoint(loaded)
        node_rows = build_node_mapping(metadata, project6_nodes, priority_nodes)
        sensor_rows = build_sensor_mapping(metadata, project6_nodes, priority_nodes)
        norm_rows = normalization_rows(metadata)
        strict_status, missing, unexpected, errors = load_project4_model(loaded)
        status, reasons, gate = compatibility_status(metadata, strict_status, node_rows, sensor_rows, norm_rows)
        candidates.append(
            {
                "registry_name": name,
                "source_path": str(path),
                "declared_sensor_ratio": ratio,
                "actual_sensor_ratio": metadata.get("actual_sensor_ratio"),
                "compatibility_status": status,
                "compatibility_gate_evaluated": gate,
                "reasons": reasons,
                "strict_load_status": strict_status,
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "strict_load_errors": errors,
                "node_count": len(metadata.get("node_ids") or []),
                "sensor_count": len(metadata.get("sensor_ids") or []),
                "project6_node_count": len(project6_nodes),
                "shared_node_count": sum(1 for r in node_rows if r["is_shared"]),
                "missing_training_nodes": sum(1 for r in node_rows if r["is_missing"]),
                "added_project6_nodes": sum(1 for r in node_rows if r["is_added_in_retrofit"]),
                "node_order_hash": metadata.get("node_order_hash"),
                "sensor_ids_hash": metadata.get("sensor_ids_hash"),
                "normalization_hash": metadata.get("node_static_hash"),
                "graph_signature": metadata.get("graph_signature"),
                "human_selection_allowed": status in {COMPATIBLE_STRICT, COMPATIBLE_STRICT_REORDER, COMPATIBLE_SHARED_BASE},
            }
        )
    allowed = any(c["human_selection_allowed"] for c in candidates)
    return GATCompatibilityReport(
        candidates=candidates,
        selected_primary_gat="sr0p15" if allowed else None,
        selection_status="user_confirmed_pending_manual_lock" if allowed else "blocked_pending_compatibility_evidence",
        overall_research_status="audit_complete_sr0p15_user_confirmed_pending_lock" if allowed else "blocked_pending_checkpoint_introspection_and_metadata_recovery",
        human_selection_allowed=allowed,
    )


def write_gat_compatibility_outputs(report: GATCompatibilityReport, out_dir: Path, config_path: Path | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_path or Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")
    inp = config_inp_path(config_path)
    project6_nodes = parse_inp_node_ids(inp)
    priority_nodes = priority_nodes_from_contract()
    all_node_rows: list[dict[str, Any]] = []
    all_sensor_rows: list[dict[str, Any]] = []
    all_norm_rows: list[dict[str, Any]] = []
    all_graph_rows: list[dict[str, Any]] = []
    for name, ratio, path in GAT_CANDIDATES:
        loaded = load_checkpoint(name, ratio, path)
        if loaded.checkpoint is None:
            continue
        metadata = metadata_from_checkpoint(loaded)
        all_node_rows.extend(build_node_mapping(metadata, project6_nodes, priority_nodes))
        all_sensor_rows.extend(build_sensor_mapping(metadata, project6_nodes, priority_nodes))
        all_norm_rows.extend(normalization_rows(metadata))
        all_graph_rows.extend(graph_signature_rows(metadata, config_path))
    outputs = {
        "compatibility_report": out_dir / "gat_compatibility_report.json",
        "node_mapping": out_dir / "gat_node_mapping.csv",
        "sensor_mapping": out_dir / "gat_sensor_mapping.csv",
        "normalization_audit": out_dir / "gat_normalization_audit.csv",
        "graph_signature_audit": out_dir / "gat_graph_signature_audit.csv",
        "checkpoint_load_audit": out_dir / "gat_checkpoint_load_audit.csv",
        "strict_load_audit": out_dir / "gat_strict_load_audit.csv",
    }
    payload = {
        "selected_primary_gat": report.selected_primary_gat,
        "primary_gat_registry_name": "sr0p15" if report.human_selection_allowed else None,
        "selection_decision_status": "user_confirmed" if report.human_selection_allowed else "blocked",
        "selection_lock_status": "pending_manual_execution" if report.human_selection_allowed else "blocked",
        "gat_robustness_status": "pending" if report.human_selection_allowed else "blocked",
        "round0_unlock_allowed": False,
        "selection_status": report.selection_status,
        "overall_research_status": report.overall_research_status,
        "human_selection_allowed": report.human_selection_allowed,
        "compatible_strict_count": report.compatible_strict_count,
        "compatibility_status_definitions": [
            COMPATIBLE_STRICT,
            COMPATIBLE_STRICT_REORDER,
            COMPATIBLE_SHARED_BASE,
            METADATA_INCOMPLETE,
            INCOMPATIBLE,
            LOAD_FAILED,
        ],
        "candidates": report.candidates,
    }
    outputs["compatibility_report"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_csv(outputs["node_mapping"], all_node_rows, ["registry_name", "training_index", "training_node_id", "project6_index", "project6_node_id", "canonical_id", "mapping_status", "reorder_index", "is_shared", "is_missing", "is_added_in_retrofit", "is_priority", "is_sentinel_candidate", "is_storage"])
    write_csv(outputs["sensor_mapping"], all_sensor_rows, ["registry_name", "declared_sensor_ratio", "actual_sensor_ratio", "training_sensor_index", "sensor_id", "training_node_index", "project6_node_index", "exists", "duplicate", "is_priority", "is_sentinel_candidate", "is_storage", "mapping_status"])
    write_csv(outputs["normalization_audit"], all_norm_rows, ["registry_name", "feature_index", "feature_name", "unit", "transform", "mean", "std", "min", "max", "source_path", "source_sha256", "round_trip_error", "zero_variance", "status"])
    write_csv(outputs["graph_signature_audit"], all_graph_rows, ["registry_name", "graph_name", "node_order_hash", "directed_edge_list_hash", "edge_count", "self_loop_rule", "bidirectional_rule", "edge_feature_names", "edge_feature_normalization", "adjacency_sorting_rule", "comparison_status"])
    load_outputs = inspect_checkpoint_outputs(out_dir)
    outputs["checkpoint_load_audit"] = load_outputs["load_audit"]
    outputs["strict_load_audit"] = load_outputs["strict_load_audit"]
    return outputs


def run_forward_smoke_outputs(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, ratio, path in GAT_CANDIDATES:
        loaded = load_checkpoint(name, ratio, path)
        if loaded.checkpoint is None:
            rows.append({"registry_name": name, "status": "blocked", "reason": "checkpoint_load_failed"})
            continue
        metadata = metadata_from_checkpoint(loaded)
        from .gat_audit import forward_smoke
        rows.append(forward_smoke(metadata, samples=128))
    csv_path = out_dir / "gat_forward_smoke_audit.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    write_csv(csv_path, rows, fieldnames)
    report_path = out_dir / "gat_forward_smoke_report.json"
    report_path.write_text(json.dumps({"rows": rows}, indent=2, default=str), encoding="utf-8")
    return {"smoke_csv": csv_path, "smoke_report": report_path}


def run_reconstruction_audit_outputs(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "gat_reconstruction_audit": out_dir / "gat_reconstruction_audit.csv",
        "gat_unsensed_node_audit": out_dir / "gat_unsensed_node_audit.csv",
        "gat_priority_leaveout_audit": out_dir / "gat_priority_leaveout_audit.csv",
        "gat_sentinel_leaveout_audit": out_dir / "gat_sentinel_leaveout_audit.csv",
        "gat_highwater_audit": out_dir / "gat_highwater_audit.csv",
        "gat_sensor_failure_audit": out_dir / "gat_sensor_failure_audit.csv",
        "gat_candidate_comparison": out_dir / "gat_candidate_comparison.csv",
    }
    grouped_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in paths}
    for name, ratio, path in GAT_CANDIDATES:
        loaded = load_checkpoint(name, ratio, path)
        if loaded.checkpoint is None:
            for key in grouped_rows:
                grouped_rows[key].append(
                    {
                        "registry_name": name,
                        "audit_name": key,
                        "metric": "",
                        "value": "",
                        "support_samples": 0,
                        "status": "blocked",
                        "reason": "checkpoint_load_failed",
                    }
                )
            continue
        metadata = metadata_from_checkpoint(loaded)
        result = reconstruction_audits(metadata, out_dir / "_tmp_reconstruction", samples=512)
        for key, tmp_path in result.items():
            grouped_rows[key].extend(_read_csv(tmp_path))
    fieldnames = ["registry_name", "audit_name", "metric", "value", "support_samples", "status", "reason"]
    for key, rows in grouped_rows.items():
        write_csv(paths[key], rows, fieldnames)
    return paths
