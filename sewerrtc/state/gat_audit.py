from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc._project_root import PROJECT_ROOT

PROJECT6_ROOT = PROJECT_ROOT
_PROJECT4_ROOT = Path(os.environ.get("PROJECT4_ROOT", PROJECT_ROOT.parent / "Project4"))
PROJECT4_ROOT = _PROJECT4_ROOT
PROJECT4_SENSOR_ROOT = PROJECT4_ROOT / "outputs" / "sensor_sensitivity"
PROJECT4_CACHE = PROJECT4_ROOT / "outputs" / "cache_paired_no_controls" / "transition_cache.npz"
PROJECT4_NODE_TABLE = PROJECT4_ROOT / "outputs" / "audit" / "node_table.csv"
PROJECT4_LINK_TABLE = PROJECT4_ROOT / "outputs" / "audit" / "link_table.csv"
PROJECT4_GAT_TRAIN_SCRIPT = PROJECT4_ROOT / "scripts" / "05_train_gat.py"
PROJECT4_GAT_MODEL = PROJECT4_ROOT / "sewerrtc" / "models" / "gat_reconstructor.py"

GAT_CANDIDATES = [
    ("sr0p05", 0.05, PROJECT4_SENSOR_ROOT / "sr0p05" / "models" / "gat_sr0p10.pt"),
    ("sr0p10", 0.10, PROJECT4_SENSOR_ROOT / "sr0p10" / "models" / "gat_sr0p10.pt"),
    ("sr0p15", 0.15, PROJECT4_SENSOR_ROOT / "sr0p15" / "models" / "gat_sr0p10.pt"),
    ("sr0p20", 0.20, PROJECT4_SENSOR_ROOT / "sr0p20" / "models" / "gat_sr0p10.pt"),
    ("sr0p30", 0.30, PROJECT4_SENSOR_ROOT / "sr0p30" / "models" / "gat_sr0p10.pt"),
]

STATIC_FEATURE_NAMES = ["invert", "max_depth", "ponded_area", "degree_in", "degree_out", "is_storage", "is_outfall"]
SENTINEL_CANDIDATES = {"MH0200770", "HS1355904"}


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_inp_node_ids(inp_path: Path) -> list[str]:
    nodes: list[str] = []
    current: str | None = None
    node_sections = {"JUNCTIONS", "OUTFALLS", "STORAGE", "DIVIDERS"}
    if not inp_path.exists():
        return nodes
    for raw in inp_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.strip("[]").upper()
            continue
        if current in node_sections:
            parts = line.split()
            if parts:
                nodes.append(parts[0])
    return nodes


def parse_inp_edge_ids(inp_path: Path, node_index: dict[str, int]) -> list[tuple[int, int, str]]:
    edges: list[tuple[int, int, str]] = []
    current: str | None = None
    link_sections = {"CONDUITS", "PUMPS", "ORIFICES", "WEIRS", "OUTLETS"}
    if not inp_path.exists():
        return edges
    for raw in inp_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.strip("[]").upper()
            continue
        if current in link_sections:
            parts = line.split()
            if len(parts) >= 3 and parts[1] in node_index and parts[2] in node_index:
                edges.append((node_index[parts[1]], node_index[parts[2]], parts[0]))
                edges.append((node_index[parts[2]], node_index[parts[1]], parts[0] + "__reverse"))
    return edges


def config_inp_path(config_path: Path) -> Path:
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    for raw in text.splitlines():
        if "wuhan_v8_storage_retrofit.inp" in raw:
            value = raw.split(":", 1)[-1].strip().strip("'\"")
            path = Path(value)
            return path if path.is_absolute() else PROJECT6_ROOT / path
    return PROJECT6_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"


def priority_nodes_from_contract() -> set[str]:
    contract = PROJECT6_ROOT / "docs" / "contracts" / "kpi_contract.json"
    if not contract.exists():
        return set()
    try:
        payload = json.loads(contract.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    pfv = payload.get("pfv", {}) if isinstance(payload.get("pfv"), dict) else {}
    path_text = pfv.get("priority_node_list_path")
    if not path_text:
        return set()
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT6_ROOT / path
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}


@dataclass
class LoadedCheckpoint:
    name: str
    path: Path
    declared_ratio: float
    checkpoint: dict[str, Any] | None
    state_dict: dict[str, Any]
    load_status: str
    weights_only_error: str | None
    trusted_full_pickle_used: bool
    load_error: str | None
    load_seconds: float


def load_checkpoint(name: str, ratio: float, path: Path) -> LoadedCheckpoint:
    start = time.perf_counter()
    weights_only_error = None
    trusted_full_pickle_used = False
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        weights_only_error = repr(exc)
        try:
            import torch
            resolved = path.resolve()
            trusted_root = PROJECT4_SENSOR_ROOT.resolve()
            if trusted_root not in resolved.parents:
                raise RuntimeError(f"full pickle refused for untrusted path: {resolved}")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            trusted_full_pickle_used = True
        except Exception as exc2:
            return LoadedCheckpoint(
                name=name,
                path=path,
                declared_ratio=ratio,
                checkpoint=None,
                state_dict={},
                load_status="load_failed",
                weights_only_error=weights_only_error,
                trusted_full_pickle_used=trusted_full_pickle_used,
                load_error=repr(exc2),
                load_seconds=time.perf_counter() - start,
            )
    state = extract_state_dict(ckpt)
    return LoadedCheckpoint(
        name=name,
        path=path,
        declared_ratio=ratio,
        checkpoint=ckpt if isinstance(ckpt, dict) else {"__object_type__": type(ckpt).__name__},
        state_dict=state,
        load_status="loaded_cpu_no_inference",
        weights_only_error=weights_only_error,
        trusted_full_pickle_used=trusted_full_pickle_used,
        load_error=None,
        load_seconds=time.perf_counter() - start,
    )


def extract_state_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = obj.get(key)
            if isinstance(value, dict):
                return {str(k).removeprefix("module."): v for k, v in value.items()}
        tensor_items = {str(k).removeprefix("module."): v for k, v in obj.items() if hasattr(v, "shape")}
        if tensor_items:
            return tensor_items
    return {}


def tensor_shape(value: Any) -> str:
    return "x".join(str(x) for x in tuple(value.shape)) if hasattr(value, "shape") else ""


def checkpoint_rows(loaded: LoadedCheckpoint) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ckpt = loaded.checkpoint or {}
    top_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else []
    tensor_rows = []
    total_params = 0
    dtype_counts: dict[str, int] = {}
    has_nan_inf = False
    for key, value in loaded.state_dict.items():
        dtype = str(getattr(value, "dtype", ""))
        numel = int(value.numel()) if hasattr(value, "numel") else 0
        total_params += numel
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        finite = None
        if hasattr(value, "isfinite"):
            try:
                finite = bool(value.isfinite().all().item())
                has_nan_inf = has_nan_inf or not finite
            except Exception:
                finite = False
                has_nan_inf = True
        tensor_rows.append(
            {
                "registry_name": loaded.name,
                "tensor_key": key,
                "shape": tensor_shape(value),
                "dtype": dtype,
                "numel": numel,
                "finite": finite,
            }
        )
    summary = {
        "registry_name": loaded.name,
        "source_path": str(loaded.path),
        "torch_load_status": loaded.load_status,
        "weights_only_error": loaded.weights_only_error,
        "trusted_full_pickle_used": loaded.trusted_full_pickle_used,
        "load_error": loaded.load_error,
        "load_seconds": loaded.load_seconds,
        "top_level_object_type": type(loaded.checkpoint).__name__ if loaded.checkpoint is not None else None,
        "top_level_keys": json.dumps(top_keys),
        "state_dict_location": "model" if "model" in ckpt else ("state_dict" if "state_dict" in ckpt else "model_state_dict" if "model_state_dict" in ckpt else "top_level_tensors"),
        "state_dict_key_count": len(loaded.state_dict),
        "parameter_count": total_params,
        "dtype_summary": json.dumps(dtype_counts, sort_keys=True),
        "nan_inf_status": "contains_nan_or_inf" if has_nan_inf else "finite_parameters",
        "n_nodes": ckpt.get("n_nodes"),
        "static_dim": ckpt.get("static_dim"),
        "hidden_dim": ckpt.get("hidden_dim"),
        "gat_heads": ckpt.get("gat_heads"),
    }
    return summary, tensor_rows


def metadata_from_checkpoint(loaded: LoadedCheckpoint) -> dict[str, Any]:
    ckpt = loaded.checkpoint or {}
    node_ids = [str(x) for x in ckpt.get("node_ids", [])]
    edge_index = ckpt.get("edge_index")
    node_static = ckpt.get("node_static")
    sensor_path = PROJECT4_SENSOR_ROOT / loaded.name / "design" / "sensor_nodes.csv"
    sensors = pd.read_csv(sensor_path) if sensor_path.exists() else pd.DataFrame()
    sensor_ids = sensors["node_id"].astype(str).tolist() if "node_id" in sensors else []
    return {
        "registry_name": loaded.name,
        "declared_sensor_ratio": loaded.declared_ratio,
        "source_path": str(loaded.path),
        "source_sha256": sha256_file(loaded.path),
        "checkpoint_format": "torch_pickle_pt",
        "model_class": "sewerrtc.models.gat_reconstructor.SparseGATReconstructor",
        "model_class_source_path": str(PROJECT4_GAT_MODEL),
        "model_class_source_sha256": sha256_file(PROJECT4_GAT_MODEL),
        "training_script_source_path": str(PROJECT4_GAT_TRAIN_SCRIPT),
        "training_script_source_sha256": sha256_file(PROJECT4_GAT_TRAIN_SCRIPT),
        "n_nodes": ckpt.get("n_nodes"),
        "static_dim": ckpt.get("static_dim"),
        "hidden_dim": ckpt.get("hidden_dim"),
        "gat_heads": ckpt.get("gat_heads"),
        "input_dim": int(loaded.state_dict["input.weight"].shape[1]) if "input.weight" in loaded.state_dict else None,
        "output_dim": int(ckpt.get("n_nodes")) if ckpt.get("n_nodes") is not None else None,
        "node_ids": node_ids,
        "node_ids_hash": sha256_json(node_ids) if node_ids else None,
        "node_order_hash": sha256_json(node_ids) if node_ids else None,
        "sensor_ids": sensor_ids,
        "sensor_ids_hash": sha256_json(sensor_ids) if sensor_ids else None,
        "sensor_count": len(sensor_ids),
        "actual_sensor_ratio": float(len(sensor_ids) / len(node_ids)) if node_ids else None,
        "node_static_shape": list(node_static.shape) if hasattr(node_static, "shape") else None,
        "node_static_hash": sha256_json(np.asarray(node_static).round(8).tolist()) if node_static is not None else None,
        "edge_index_shape": list(edge_index.shape) if hasattr(edge_index, "shape") else None,
        "edge_index_hash": sha256_json(np.asarray(edge_index).tolist()) if edge_index is not None else None,
        "state_dict_keys": list(loaded.state_dict.keys()),
        "state_dict_key_signature": sha256_json(list(loaded.state_dict.keys())),
        "normalization": {
            "state_depth": {"transform": "identity", "unit": "m", "source": "scripts/05_train_gat.py"},
            "rain": {"transform": "identity", "unit": "mm_per_hour", "source": "scripts/05_train_gat.py"},
            "node_static": {
                "transform": "zscore_precomputed_tensor_saved_in_checkpoint",
                "feature_names": STATIC_FEATURE_NAMES,
                "mean_std_source": "not_saved",
            },
        },
        "graph_signature": sha256_json(np.asarray(edge_index).tolist()) if edge_index is not None else None,
        "training_event_provenance": {
            "cache": str(PROJECT4_CACHE),
            "cache_sha256": sha256_file(PROJECT4_CACHE),
        },
    }


def load_project4_model(loaded: LoadedCheckpoint) -> tuple[str, list[str], list[str], list[str]]:
    if loaded.checkpoint is None:
        return "not_loaded", [], [], ["checkpoint_not_loaded"]
    try:
        SparseGATReconstructor = load_project4_gat_class()
    except Exception as exc:
        return "model_import_failed", [], [], [repr(exc)]
    ckpt = loaded.checkpoint
    try:
        model = SparseGATReconstructor(
            int(ckpt["n_nodes"]),
            int(ckpt["static_dim"]),
            int(ckpt["hidden_dim"]),
            int(ckpt.get("gat_heads", 4)),
        )
        result = model.load_state_dict(loaded.state_dict, strict=True)
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        return "strict_loaded", missing, unexpected, []
    except Exception as exc:
        return "strict_load_failed", [], [], [repr(exc)]


def build_node_mapping(metadata: dict[str, Any], project6_nodes: list[str], priority_nodes: set[str]) -> list[dict[str, Any]]:
    project6_index = {node: i for i, node in enumerate(project6_nodes)}
    training_nodes = metadata.get("node_ids", []) or []
    rows: list[dict[str, Any]] = []
    for i, node in enumerate(training_nodes):
        p6 = project6_index.get(node)
        rows.append(
            {
                "registry_name": metadata["registry_name"],
                "training_index": i,
                "training_node_id": node,
                "project6_index": p6,
                "project6_node_id": node if p6 is not None else "",
                "canonical_id": node,
                "mapping_status": "mapped" if p6 is not None else "missing_in_project6",
                "reorder_index": p6,
                "is_shared": p6 is not None,
                "is_missing": p6 is None,
                "is_added_in_retrofit": False,
                "is_priority": node in priority_nodes,
                "is_sentinel_candidate": node in SENTINEL_CANDIDATES,
                "is_storage": False,
            }
        )
    training_set = set(training_nodes)
    for node in project6_nodes:
        if node not in training_set:
            rows.append(
                {
                    "registry_name": metadata["registry_name"],
                    "training_index": "",
                    "training_node_id": "",
                    "project6_index": project6_index[node],
                    "project6_node_id": node,
                    "canonical_id": node,
                    "mapping_status": "added_in_retrofit_or_uncovered",
                    "reorder_index": "",
                    "is_shared": False,
                    "is_missing": False,
                    "is_added_in_retrofit": True,
                    "is_priority": node in priority_nodes,
                    "is_sentinel_candidate": node in SENTINEL_CANDIDATES,
                    "is_storage": False,
                }
            )
    return rows


def build_sensor_mapping(metadata: dict[str, Any], project6_nodes: list[str], priority_nodes: set[str]) -> list[dict[str, Any]]:
    training_index = {node: i for i, node in enumerate(metadata.get("node_ids", []) or [])}
    project6_index = {node: i for i, node in enumerate(project6_nodes)}
    seen: set[str] = set()
    rows = []
    for i, sensor in enumerate(metadata.get("sensor_ids", []) or []):
        duplicate = sensor in seen
        seen.add(sensor)
        rows.append(
            {
                "registry_name": metadata["registry_name"],
                "declared_sensor_ratio": metadata["declared_sensor_ratio"],
                "actual_sensor_ratio": metadata["actual_sensor_ratio"],
                "training_sensor_index": i,
                "sensor_id": sensor,
                "training_node_index": training_index.get(sensor),
                "project6_node_index": project6_index.get(sensor),
                "exists": sensor in project6_index,
                "duplicate": duplicate,
                "is_priority": sensor in priority_nodes,
                "is_sentinel_candidate": sensor in SENTINEL_CANDIDATES,
                "is_storage": False,
                "mapping_status": "mapped" if sensor in project6_index and not duplicate else ("duplicate" if duplicate else "missing_in_project6"),
            }
        )
    return rows


def normalization_rows(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for i, name in enumerate(["sparse_depth", "sensor_mask", "rain"] + STATIC_FEATURE_NAMES):
        if name == "sparse_depth":
            transform, source, status, unit = "identity", "scripts/05_train_gat.py", "recoverable", "m"
        elif name == "sensor_mask":
            transform, source, status, unit = "binary_mask", "sensor_nodes.csv", "recoverable", "unitless"
        elif name == "rain":
            transform, source, status, unit = "identity", "scripts/05_train_gat.py", "recoverable", "mm_per_hour"
        else:
            transform = "zscore_precomputed_checkpoint_tensor"
            source = "checkpoint.node_static"
            status = "recoverable_precomputed_tensor"
            unit = "mixed"
        rows.append(
            {
                "registry_name": metadata["registry_name"],
                "feature_index": i,
                "feature_name": name,
                "unit": unit,
                "transform": transform,
                "mean": "",
                "std": "",
                "min": "",
                "max": "",
                "source_path": metadata["source_path"] if source.startswith("checkpoint") else str(PROJECT4_GAT_TRAIN_SCRIPT),
                "source_sha256": metadata["source_sha256"] if source.startswith("checkpoint") else sha256_file(PROJECT4_GAT_TRAIN_SCRIPT),
                "round_trip_error": 0.0 if name in {"sparse_depth", "sensor_mask", "rain"} else "",
                "zero_variance": "",
                "status": status,
            }
        )
    return rows


def graph_signature_rows(metadata: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    inp = config_inp_path(config_path)
    p6_nodes = parse_inp_node_ids(inp)
    p6_edges = parse_inp_edge_ids(inp, {n: i for i, n in enumerate(p6_nodes)})
    ckpt_edge_index = None
    try:
        loaded = load_checkpoint(metadata["registry_name"], float(metadata["declared_sensor_ratio"]), Path(metadata["source_path"]))
        ckpt_edge_index = np.asarray((loaded.checkpoint or {}).get("edge_index"))
    except Exception:
        ckpt_edge_index = None
    p4_edges = set(map(tuple, ckpt_edge_index.T.tolist())) if ckpt_edge_index is not None and ckpt_edge_index.ndim == 2 else set()
    p6_edge_set = {(a, b) for a, b, _ in p6_edges}
    project4_hash = sha256_json(sorted(p4_edges)) if p4_edges else metadata.get("edge_index_hash")
    p6_hash = sha256_json(sorted(p6_edge_set)) if p6_edge_set else None
    if p4_edges and p6_edge_set and p4_edges == p6_edge_set:
        comparison_status = "matches_project4_edge_set"
    elif p6_hash == project4_hash:
        comparison_status = "matches_project4_hash"
    else:
        comparison_status = "differs_or_unverified"
    return [
        {
            "registry_name": metadata["registry_name"],
            "graph_name": "project4_training_graph",
            "node_order_hash": metadata.get("node_order_hash"),
            "directed_edge_list_hash": project4_hash,
            "edge_count": (metadata.get("edge_index_shape") or ["", ""])[1] if metadata.get("edge_index_shape") else "",
            "self_loop_rule": "GATConv_add_self_loops_true",
            "bidirectional_rule": "training_builder_added_forward_and_reverse",
            "edge_feature_names": "none",
            "edge_feature_normalization": "not_applicable",
            "adjacency_sorting_rule": "training_link_table_iteration_order",
            "comparison_status": "source_graph_signature_recovered",
        },
        {
            "registry_name": metadata["registry_name"],
            "graph_name": "project6_retrofit_inp_graph",
            "node_order_hash": sha256_json(p6_nodes) if p6_nodes else None,
            "directed_edge_list_hash": p6_hash,
            "edge_count": len(p6_edges),
            "self_loop_rule": "not_in_edge_hash",
            "bidirectional_rule": "parser_added_forward_and_reverse",
            "edge_feature_names": "none",
            "edge_feature_normalization": "not_applicable",
            "adjacency_sorting_rule": "inp_section_order",
            "comparison_status": comparison_status,
        },
    ]


def compatibility_status(metadata: dict[str, Any], strict_status: str, node_rows: list[dict[str, Any]], sensor_rows: list[dict[str, Any]], normalization: list[dict[str, Any]]) -> tuple[str, list[str], bool]:
    reasons: list[str] = []
    if strict_status != "strict_loaded":
        return "load_failed", [strict_status], False
    missing_nodes = [r for r in node_rows if r["mapping_status"] == "missing_in_project6"]
    added_nodes = [r for r in node_rows if r["mapping_status"] == "added_in_retrofit_or_uncovered"]
    missing_sensors = [r for r in sensor_rows if r["mapping_status"] == "missing_in_project6"]
    if missing_sensors:
        reasons.append("sensor_missing_in_project6")
    if missing_nodes:
        return "incompatible", reasons + ["training_nodes_missing_in_project6"], True
    if added_nodes:
        unresolved_norm = [r for r in normalization if r["status"] == "recoverable_precomputed_tensor"]
        if unresolved_norm:
            reasons.append("node_static_mean_std_not_saved_for_added_or_uncovered_nodes")
        return "compatible_shared_base_graph_only", reasons + ["project6_has_nodes_not_covered_by_project4_gat"], True
    if reasons:
        return "metadata_incomplete", reasons, True
    return "compatible_strict", [], True


def forward_smoke(metadata: dict[str, Any], samples: int = 128) -> dict[str, Any]:
    import torch

    loaded = load_checkpoint(metadata["registry_name"], float(metadata["declared_sensor_ratio"]), Path(metadata["source_path"]))
    if loaded.checkpoint is None:
        return {"registry_name": metadata["registry_name"], "status": "blocked", "reason": "checkpoint_load_failed"}
    SparseGATReconstructor = load_project4_gat_class()

    ckpt = loaded.checkpoint
    cache = np.load(PROJECT4_CACHE, allow_pickle=True)
    state = cache["state"][:samples].astype(np.float32)
    rain = cache["rain"][:samples].astype(np.float32)
    sensor_ids = set(metadata.get("sensor_ids") or [])
    node_ids = metadata.get("node_ids") or []
    sensor_idx = [i for i, n in enumerate(node_ids) if n in sensor_ids]
    if not sensor_idx:
        return {"registry_name": metadata["registry_name"], "status": "blocked", "reason": "no_sensor_ids"}
    mask = np.zeros(len(node_ids), dtype=np.float32)
    mask[sensor_idx] = 1.0
    model = SparseGATReconstructor(int(ckpt["n_nodes"]), int(ckpt["static_dim"]), int(ckpt["hidden_dim"]), int(ckpt.get("gat_heads", 4)))
    model.load_state_dict(loaded.state_dict, strict=True)
    model.eval()
    x = torch.tensor(state, dtype=torch.float32)
    r = torch.tensor(rain, dtype=torch.float32)
    m = torch.tensor(mask[None, :], dtype=torch.float32).expand(len(x), -1)
    ns = torch.tensor(ckpt["node_static"], dtype=torch.float32)
    ei = torch.tensor(ckpt["edge_index"], dtype=torch.long)
    start = time.perf_counter()
    with torch.no_grad():
        y1 = model(x, m, r, ns, ei)
        y2 = model(x, m, r, ns, ei)
    elapsed = time.perf_counter() - start
    pred = y1.detach().numpy()
    target = state
    err = pred - target
    denom = float(np.sum((target - target.mean()) ** 2))
    nse = float(1.0 - np.sum(err**2) / denom) if denom > 1e-9 else 0.0
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    repeat = float(torch.max(torch.abs(y1 - y2)).item())
    return {
        "registry_name": metadata["registry_name"],
        "status": "passed",
        "input_shape": list(x.shape),
        "output_shape": list(y1.shape),
        "finite_input": bool(np.isfinite(state).all()),
        "finite_output": bool(np.isfinite(pred).all()),
        "repeat_max_abs_diff": repeat,
        "single_sample_latency_ms": float(elapsed / max(1, len(x)) * 1000.0),
        "RMSE": rmse,
        "MAE": mae,
        "NSE": nse,
        "observed_sensor_count": len(sensor_idx),
        "unobserved_node_count": len(node_ids) - len(sensor_idx),
        "physical_min_output": float(np.min(pred)),
        "physical_max_output": float(np.max(pred)),
    }


def load_project4_gat_class():
    module_name = "_project4_sparse_gat_reconstructor"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, PROJECT4_GAT_MODEL)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot create import spec for {PROJECT4_GAT_MODEL}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.SparseGATReconstructor


def reconstruction_audits(metadata: dict[str, Any], out_dir: Path, samples: int = 512) -> dict[str, Path]:
    result = forward_smoke(metadata, samples=samples)
    paths = {
        "gat_reconstruction_audit": out_dir / "gat_reconstruction_audit.csv",
        "gat_unsensed_node_audit": out_dir / "gat_unsensed_node_audit.csv",
        "gat_priority_leaveout_audit": out_dir / "gat_priority_leaveout_audit.csv",
        "gat_sentinel_leaveout_audit": out_dir / "gat_sentinel_leaveout_audit.csv",
        "gat_highwater_audit": out_dir / "gat_highwater_audit.csv",
        "gat_sensor_failure_audit": out_dir / "gat_sensor_failure_audit.csv",
        "gat_candidate_comparison": out_dir / "gat_candidate_comparison.csv",
    }
    common = ["registry_name", "audit_name", "metric", "value", "support_samples", "status", "reason"]
    for name, path in paths.items():
        rows = []
        if result.get("status") == "passed":
            if name == "gat_reconstruction_audit":
                for metric in ["RMSE", "MAE", "NSE", "single_sample_latency_ms"]:
                    rows.append({"registry_name": metadata["registry_name"], "audit_name": name, "metric": metric, "value": result.get(metric), "support_samples": samples, "status": "computed", "reason": ""})
            elif name == "gat_candidate_comparison":
                rows.append({"registry_name": metadata["registry_name"], "audit_name": name, "metric": "NSE", "value": result.get("NSE"), "support_samples": samples, "status": "computed", "reason": ""})
            else:
                rows.append({"registry_name": metadata["registry_name"], "audit_name": name, "metric": "not_computed_in_minimal_smoke", "value": "", "support_samples": samples, "status": "blocked", "reason": "dedicated leaveout/highwater/sensor-failure perturbation not yet executed"})
        else:
            rows.append({"registry_name": metadata["registry_name"], "audit_name": name, "metric": "", "value": "", "support_samples": 0, "status": "blocked", "reason": result.get("reason")})
        write_csv(path, rows, common)
    return paths
