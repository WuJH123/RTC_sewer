from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from sewerrtc._project_root import PROJECT_ROOT

from .gat_audit import checkpoint_rows, load_checkpoint, metadata_from_checkpoint

_PROJECT4_ROOT = Path(os.environ.get("PROJECT4_ROOT", PROJECT_ROOT.parent / "Project4"))

DEFAULT_PROJECT4_GAT_CANDIDATES = [
    ("sr0p05", 0.05, str(_PROJECT4_ROOT / "outputs" / "sensor_sensitivity" / "sr0p05" / "models" / "gat_sr0p10.pt")),
    ("sr0p10", 0.10, str(_PROJECT4_ROOT / "outputs" / "sensor_sensitivity" / "sr0p10" / "models" / "gat_sr0p10.pt")),
    ("sr0p15", 0.15, str(_PROJECT4_ROOT / "outputs" / "sensor_sensitivity" / "sr0p15" / "models" / "gat_sr0p10.pt")),
    ("sr0p20", 0.20, str(_PROJECT4_ROOT / "outputs" / "sensor_sensitivity" / "sr0p20" / "models" / "gat_sr0p10.pt")),
    ("sr0p30", 0.30, str(_PROJECT4_ROOT / "outputs" / "sensor_sensitivity" / "sr0p30" / "models" / "gat_sr0p10.pt")),
]


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json_sidecar(path: Path) -> dict:
    sidecars = [
        path.with_suffix(path.suffix + ".json"),
        path.with_suffix(".json"),
        path.parent / "metadata.json",
        path.parent / "gat_metadata.json",
    ]
    for sidecar in sidecars:
        if sidecar.exists():
            try:
                return json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - reported as metadata, not fatal.
                return {"metadata_read_error": str(exc), "metadata_path": str(sidecar)}
    return {}


@dataclass
class GATRegistryRecord:
    registry_name: str
    declared_sensor_ratio: float
    source_path: str
    source_project: str
    file_name: str
    exists: bool
    file_size_bytes: int | None
    sha256: str | None
    checkpoint_format: str
    checkpoint_metadata: str
    model_class: str | None
    state_dict_key_signature: str | None
    input_dim: int | None
    hidden_dim: int | None
    output_dim: int | None
    graph_node_count: int | None
    node_ids_source: str | None
    node_ids_hash: str | None
    node_order_hash: str | None
    sensor_count: int | None
    sensor_ids_source: str | None
    sensor_ids_hash: str | None
    normalization_source: str | None
    normalization_hash: str | None
    graph_edge_source: str | None
    graph_signature: str | None
    training_event_provenance: str | None
    compatibility_status: str
    incompatibility_reasons: str
    missing_metadata: str
    intended_use: str
    registered_at: str


def _metadata_value(metadata: dict, *keys: str):
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _json_compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_gat_registry(
    candidates: Iterable[tuple[str, float, str]],
    config_path: Path,
    intended_use: str,
) -> list[GATRegistryRecord]:
    now = datetime.now(timezone.utc).isoformat()
    records: list[GATRegistryRecord] = []
    for name, ratio, raw_path in candidates:
        path = Path(raw_path)
        exists = path.exists()
        loaded = load_checkpoint(name, ratio, path) if exists else None
        ckpt_metadata = metadata_from_checkpoint(loaded) if loaded and loaded.checkpoint is not None else {}
        sidecar_metadata = read_json_sidecar(path) if exists else {}
        metadata = {**sidecar_metadata, **ckpt_metadata}
        missing = []
        node_ids = _metadata_value(metadata, "node_ids", "nodes", "node_order")
        sensor_ids = _metadata_value(metadata, "sensor_ids", "sensor_nodes", "sensor_mask_ids")
        normalization = _metadata_value(metadata, "normalization", "norm", "normalizer")
        graph_signature = _metadata_value(metadata, "graph_signature", "edge_signature", "topology_signature")
        state_dict_keys = _metadata_value(metadata, "state_dict_keys", "keys")
        for field, value in {
            "node_ids": node_ids,
            "sensor_ids": sensor_ids,
            "normalization": normalization,
            "graph_signature": graph_signature,
            "state_dict_keys": state_dict_keys,
        }.items():
            if value in (None, "", []):
                missing.append(field)
        status = "present_unverified"
        if not exists:
            status = "missing_source"
        elif loaded and loaded.load_status == "load_failed":
            status = "load_failed"
        elif missing:
            status = "metadata_incomplete"
        records.append(
            GATRegistryRecord(
                registry_name=name,
                declared_sensor_ratio=ratio,
                source_path=str(path),
                source_project="Project4",
                file_name=path.name,
                exists=exists,
                file_size_bytes=os.path.getsize(path) if exists else None,
                sha256=sha256_file(path) if exists else None,
                checkpoint_format=path.suffix.lstrip(".") or "unknown",
                checkpoint_metadata=_json_compact(metadata),
                model_class=_metadata_value(metadata, "model_class", "class_name"),
                state_dict_key_signature=sha256_text(_json_compact(state_dict_keys)) if state_dict_keys else None,
                input_dim=_metadata_value(metadata, "input_dim", "in_channels"),
                hidden_dim=_metadata_value(metadata, "hidden_dim", "hidden_channels"),
                output_dim=_metadata_value(metadata, "output_dim", "out_channels"),
                graph_node_count=len(node_ids) if isinstance(node_ids, list) else _metadata_value(metadata, "graph_node_count", "node_count", "n_nodes"),
                node_ids_source="checkpoint_metadata" if node_ids else None,
                node_ids_hash=sha256_text(_json_compact(node_ids)) if node_ids else None,
                node_order_hash=sha256_text(_json_compact(node_ids)) if node_ids else None,
                sensor_count=len(sensor_ids) if isinstance(sensor_ids, list) else _metadata_value(metadata, "sensor_count"),
                sensor_ids_source="checkpoint_metadata" if sensor_ids else None,
                sensor_ids_hash=sha256_text(_json_compact(sensor_ids)) if sensor_ids else None,
                normalization_source="checkpoint_metadata" if normalization else None,
                normalization_hash=sha256_text(_json_compact(normalization)) if normalization else None,
                graph_edge_source="checkpoint_metadata" if graph_signature else None,
                graph_signature=graph_signature if isinstance(graph_signature, str) else (sha256_text(_json_compact(graph_signature)) if graph_signature else None),
                training_event_provenance=_json_compact(_metadata_value(metadata, "training_event_provenance", "training_events")),
                compatibility_status=status,
                incompatibility_reasons="" if exists else "source_checkpoint_missing",
                missing_metadata=";".join(missing),
                intended_use=intended_use,
                registered_at=now,
            )
        )
    return records


def write_gat_registry(records: list[GATRegistryRecord], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = out_dir / "gat_external_registry.csv"
    hashes_path = out_dir / "gat_checkpoint_hashes.csv"
    fieldnames = list(asdict(records[0]).keys()) if records else list(GATRegistryRecord.__dataclass_fields__.keys())
    with registry_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    with hashes_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["registry_name", "source_path", "sha256", "file_size_bytes", "exists"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "registry_name": record.registry_name,
                    "source_path": record.source_path,
                    "sha256": record.sha256,
                    "file_size_bytes": record.file_size_bytes,
                    "exists": record.exists,
                }
            )
    return {"registry": registry_path, "hashes": hashes_path}
