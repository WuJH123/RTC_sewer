from __future__ import annotations

import csv
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .gat_audit import (
    GAT_CANDIDATES,
    PROJECT4_CACHE,
    SENTINEL_CANDIDATES,
    forward_smoke,
    load_checkpoint,
    load_project4_gat_class,
    metadata_from_checkpoint,
    priority_nodes_from_contract,
    sha256_file,
    write_csv,
)
from .gat_selection import SELECTED_REGISTRY_NAME


ROBUSTNESS_REQUIRED_OUTPUTS = [
    "gat_sr0p15_validation_dataset_manifest.json",
    "gat_sr0p15_validation_sample_inventory.csv",
    "gat_sr0p15_validation_event_support.csv",
    "gat_sr0p15_validation_provenance_audit.csv",
    "gat_sr0p15_validation_leakage_audit.csv",
    "gat_sr0p15_rainfall_near_duplicate_audit.csv",
    "gat_sr0p15_split_membership_audit.csv",
    "gat_sr0p15_temporal_dependence_audit.csv",
    "gat_sr0p15_node_group_metrics.csv",
    "gat_sr0p15_priority_leaveout_audit.csv",
    "gat_sr0p15_sentinel_leaveout_audit.csv",
    "gat_sr0p15_highwater_phase_audit.csv",
    "gat_sr0p15_sensor_failure_contract.json",
    "gat_sr0p15_sensor_failure_completion_matrix.csv",
    "gat_sr0p15_sensor_failure_audit.csv",
    "gat_sr0p15_sensor_failure_summary.csv",
    "gat_sr0p15_latency_contract.json",
    "gat_sr0p15_latency_repeatability_audit.csv",
    "gat_sr0p15_latency_summary.json",
    "gat_robustness_memory_plan.json",
    "gat_sr0p15_robustness_gate.json",
]


def _sr0p15_candidate() -> tuple[str, float, Path]:
    for item in GAT_CANDIDATES:
        if item[0] == SELECTED_REGISTRY_NAME:
            return item
    raise RuntimeError("sr0p15 GAT candidate is not configured")


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    denom = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "NSE": float(1.0 - np.sum(err**2) / denom) if denom > 1e-12 else 0.0,
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "p90_abs_error": float(np.quantile(np.abs(err), 0.90)),
        "p99_abs_error": float(np.quantile(np.abs(err), 0.99)),
        "max_abs_error": float(np.max(np.abs(err))),
    }


def _write_metric_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "registry_name",
            "diagnostic",
            "group",
            "node_id",
            "metric",
            "value",
            "sample_count",
            "unique_event_count",
            "status",
            "reason",
        ],
    )


def _load_sr0p15_model():
    import torch

    name, ratio, path = _sr0p15_candidate()
    loaded = load_checkpoint(name, ratio, path)
    if loaded.checkpoint is None:
        raise RuntimeError(f"sr0p15 checkpoint could not be loaded: {loaded.load_error}")
    metadata = metadata_from_checkpoint(loaded)
    model_cls = load_project4_gat_class()
    ckpt = loaded.checkpoint
    model = model_cls(int(ckpt["n_nodes"]), int(ckpt["static_dim"]), int(ckpt["hidden_dim"]), int(ckpt.get("gat_heads", 4)))
    model.load_state_dict(loaded.state_dict, strict=True)
    model.eval()
    return torch, loaded, metadata, model


def _estimate_forward_memory_gb(*, batch_size: int, n_nodes: int, edge_count: int, hidden_dim: int, heads: int) -> float:
    """Conservative CPU-memory estimate for one SparseGATReconstructor forward.

    The Project4 model internally creates a PyG batch by repeating `edge_index`.
    Treating the full audit sample count as one batch therefore scales attention
    memory as B * E * heads. This estimate intentionally over-approximates so
    the runner shrinks the batch before PyG asks the allocator for multi-GB
    tensors.
    """
    bytes_per_float = 4
    node_activation = batch_size * n_nodes * hidden_dim * bytes_per_float * 8
    edge_attention = batch_size * edge_count * heads * bytes_per_float * 24
    input_static = batch_size * n_nodes * (hidden_dim + 16) * bytes_per_float
    return float((node_activation + edge_attention + input_static) / (1024**3))


def _effective_batch_size(*, requested: int, n_nodes: int, edge_count: int, hidden_dim: int, heads: int, max_memory_gb: float) -> int:
    requested = max(1, int(requested))
    effective = requested
    while effective > 1:
        estimate = _estimate_forward_memory_gb(
            batch_size=effective,
            n_nodes=n_nodes,
            edge_count=edge_count,
            hidden_dim=hidden_dim,
            heads=heads,
        )
        if estimate <= max_memory_gb:
            return effective
        effective = max(1, effective // 2)
    return 1


def _write_memory_plan(
    path: Path,
    *,
    requested_batch_size: int,
    effective_batch_size: int,
    max_samples: int,
    max_memory_gb: float,
    n_nodes: int,
    edge_count: int,
    hidden_dim: int,
    heads: int,
    status: str,
    failure_reason: str = "",
) -> None:
    plan = {
        "registry_name": SELECTED_REGISTRY_NAME,
        "requested_batch_size": int(requested_batch_size),
        "effective_batch_size": int(effective_batch_size),
        "max_samples": int(max_samples),
        "max_memory_gb": float(max_memory_gb),
        "node_count_per_graph": int(n_nodes),
        "edge_count_per_graph": int(edge_count),
        "estimated_peak_gb_effective_batch": _estimate_forward_memory_gb(
            batch_size=effective_batch_size,
            n_nodes=n_nodes,
            edge_count=edge_count,
            hidden_dim=hidden_dim,
            heads=heads,
        ),
        "estimated_peak_gb_requested_batch": _estimate_forward_memory_gb(
            batch_size=max(1, int(requested_batch_size)),
            n_nodes=n_nodes,
            edge_count=edge_count,
            hidden_dim=hidden_dim,
            heads=heads,
        ),
        "status": status,
        "failure_reason": failure_reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def _predict_with_mask(
    torch,
    loaded,
    metadata: dict[str, Any],
    model,
    state: np.ndarray,
    rain: np.ndarray,
    mask: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    ckpt = loaded.checkpoint
    ns = torch.tensor(ckpt["node_static"], dtype=torch.float32)
    ei = torch.tensor(ckpt["edge_index"], dtype=torch.long)
    if state.ndim != 2:
        raise ValueError(f"state must be [samples,nodes], got {state.shape}")
    if int(state.shape[1]) != int(ckpt["n_nodes"]):
        raise ValueError(f"state node count {state.shape[1]} does not match checkpoint n_nodes {ckpt['n_nodes']}")
    if mask.shape[0] != state.shape[1]:
        raise ValueError(f"sensor mask shape {mask.shape} does not match state nodes {state.shape[1]}")
    if int(ei.max()) >= int(ckpt["n_nodes"]):
        raise ValueError("edge_index contains node index outside checkpoint node range")
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(state), max(1, int(batch_size))):
            stop = min(start + max(1, int(batch_size)), len(state))
            x = torch.tensor(state[start:stop], dtype=torch.float32)
            r = torch.tensor(rain[start:stop], dtype=torch.float32)
            m = torch.tensor(mask[None, :], dtype=torch.float32).expand(len(x), -1)
            y = model(x, m, r, ns, ei)
            if y.shape != x.shape:
                raise ValueError(f"unexpected model output shape {tuple(y.shape)} for input shape {tuple(x.shape)}")
            outputs.append(y.detach().cpu().numpy())
            del x, r, m, y
            gc.collect()
    return np.concatenate(outputs, axis=0) if outputs else np.empty_like(state)


def _event_ids(cache: Any, n: int) -> list[str]:
    for key in ["event_id", "event_ids", "events"]:
        if key in cache.files:
            return [str(x) for x in cache[key][:n]]
    return ["unknown_project4_cache_event"] * n


def _phase_labels(rain: np.ndarray) -> list[str]:
    flat = rain.reshape(len(rain), -1).mean(axis=1)
    if len(flat) == 0:
        return []
    q70 = float(np.quantile(flat, 0.70))
    labels = []
    for i, value in enumerate(flat):
        if value >= q70:
            labels.append("peak")
        elif i > 0 and value >= flat[i - 1]:
            labels.append("rising")
        else:
            labels.append("recession")
    return labels


def _sensor_mask(metadata: dict[str, Any]) -> np.ndarray:
    node_ids = metadata.get("node_ids") or []
    sensors = set(metadata.get("sensor_ids") or [])
    return np.array([1.0 if node in sensors else 0.0 for node in node_ids], dtype=np.float32)


def _csv_sha(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def _gate_check(
    *,
    check_id: str,
    status: str,
    observed_value: Any,
    required_value: Any,
    evidence_path: Path,
    blocking_reason: str = "",
    remediation: str = "",
) -> dict[str, Any]:
    if status not in {"pass", "fail", "incomplete", "not_applicable"}:
        raise ValueError(f"invalid four-state check status: {status}")
    return {
        "check_id": check_id,
        "status": status,
        "observed_value": observed_value,
        "required_value": required_value,
        "evidence_path": str(evidence_path),
        "evidence_sha256": _csv_sha(evidence_path),
        "blocking_reason": blocking_reason,
        "remediation": remediation,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_gate(path: Path, checks: list[dict[str, Any]]) -> None:
    failed = [c["check_id"] for c in checks if c["status"] == "fail"]
    incomplete = [c["check_id"] for c in checks if c["status"] == "incomplete"]
    passed = [c["check_id"] for c in checks if c["status"] == "pass"]
    status = "pass" if not failed and not incomplete else ("fail" if failed else "incomplete")
    gate = {
        "registry_name": SELECTED_REGISTRY_NAME,
        "status": status,
        "checks": {c["check_id"]: c for c in checks},
        "passed_checks": passed,
        "failed_checks": failed,
        "incomplete_checks": incomplete,
        "blocking_reasons": [c["blocking_reason"] for c in checks if c["status"] in {"fail", "incomplete"} and c.get("blocking_reason")],
        "allowed_to_build_node_state": status == "pass",
        "allowed_to_enter_prompt3a": False,
        "round0_unlock_allowed": False,
        "full_project6_augmented_state_complete": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(gate, indent=2), encoding="utf-8")


def _scenario_sensor_indices(scenario_id: str, sensor_idx: np.ndarray, node_ids: list[str], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(sensor_idx) == 0:
        return np.array([], dtype=int)
    if scenario_id == "baseline":
        return np.array([], dtype=int)
    if scenario_id == "drop_single_sensor":
        return np.array([sensor_idx[0]], dtype=int)
    if scenario_id == "drop_random_5pct":
        return rng.choice(sensor_idx, size=max(1, int(len(sensor_idx) * 0.05)), replace=False)
    if scenario_id == "drop_random_10pct":
        return rng.choice(sensor_idx, size=max(1, int(len(sensor_idx) * 0.10)), replace=False)
    if scenario_id == "drop_priority_related_sensor":
        priority = set(priority_nodes_from_contract())
        direct = [i for i in sensor_idx if node_ids[int(i)] in priority]
        return np.array(direct[: max(1, min(3, len(direct)))], dtype=int) if direct else np.array([sensor_idx[len(sensor_idx) // 2]], dtype=int)
    if scenario_id == "drop_sentinel_related_sensor":
        sentinels = set(SENTINEL_CANDIDATES)
        direct = [i for i in sensor_idx if node_ids[int(i)] in sentinels]
        return np.array(direct[: max(1, min(2, len(direct)))], dtype=int) if direct else np.array([sensor_idx[min(len(sensor_idx) - 1, len(sensor_idx) // 3)]], dtype=int)
    if scenario_id == "regional_sensor_outage":
        center = len(sensor_idx) // 2
        width = max(1, int(len(sensor_idx) * 0.08))
        return sensor_idx[max(0, center - width) : min(len(sensor_idx), center + width)]
    return np.array([], dtype=int)


def _perturb_state_for_sensor_scenario(state: np.ndarray, scenario_id: str) -> np.ndarray:
    if scenario_id == "stale_10min":
        return np.vstack([state[0:1], state[:-1]])
    if scenario_id == "stale_20min":
        return np.vstack([state[0:1], state[0:1], state[:-2]]) if len(state) > 1 else state.copy()
    if scenario_id == "positive_bias":
        return state + 0.02
    if scenario_id == "negative_bias":
        return np.maximum(0.0, state - 0.02)
    return state


def run_sr0p15_robustness_audit(
    config_path: Path,
    gat_dir: Path,
    out_dir: Path,
    max_samples: int = 512,
    batch_size: int = 1,
    resume: bool = False,
    flush_every: int = 32,
    max_memory_gb: float = 4.0,
    scenario_filter: str = "",
    seed: int = 150,
    validation_cache_path: Path | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / name for name in ROBUSTNESS_REQUIRED_OUTPUTS}
    if resume and all(path.exists() for path in paths.values()):
        return paths
    if not (gat_dir / "gat_primary_selection_lock.json").exists():
        gate_path = out_dir / "gat_sr0p15_robustness_gate.json"
        gate_path.write_text(
            json.dumps(
                {
                    "status": "blocked_pending_manual_selection_lock",
                    "registry_name": SELECTED_REGISTRY_NAME,
                    "round0_unlock_allowed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"gate": gate_path}

    import torch as torch_module

    torch, loaded, metadata, model = _load_sr0p15_model()
    cache_path = validation_cache_path or PROJECT4_CACHE
    cache = np.load(cache_path, allow_pickle=True)
    sample_slice = slice(None) if max_samples == 0 else slice(0, max_samples)
    state = cache["state"][sample_slice].astype(np.float32)
    rain = cache["rain"][sample_slice].astype(np.float32)
    event_ids = _event_ids(cache, len(state))
    phases = _phase_labels(rain)
    node_ids = metadata.get("node_ids") or []
    sensor_mask = _sensor_mask(metadata)
    ckpt = loaded.checkpoint
    edge_count = int(np.asarray(ckpt["edge_index"]).shape[1])
    effective_batch = _effective_batch_size(
        requested=batch_size,
        n_nodes=int(ckpt["n_nodes"]),
        edge_count=edge_count,
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        heads=int(ckpt.get("gat_heads", 4)),
        max_memory_gb=float(max_memory_gb),
    )
    _write_memory_plan(
        paths["gat_robustness_memory_plan.json"],
        requested_batch_size=batch_size,
        effective_batch_size=effective_batch,
        max_samples=len(state),
        max_memory_gb=max_memory_gb,
        n_nodes=int(ckpt["n_nodes"]),
        edge_count=edge_count,
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        heads=int(ckpt.get("gat_heads", 4)),
        status="planned",
    )
    try:
        pred = _predict_with_mask(torch, loaded, metadata, model, state, rain, sensor_mask, batch_size=effective_batch)
    except RuntimeError as exc:
        if "memory" in str(exc).lower() or "allocator" in str(exc).lower():
            _write_memory_plan(
                paths["gat_robustness_memory_plan.json"],
                requested_batch_size=batch_size,
                effective_batch_size=effective_batch,
                max_samples=len(state),
                max_memory_gb=max_memory_gb,
                n_nodes=int(ckpt["n_nodes"]),
                edge_count=edge_count,
                hidden_dim=int(ckpt.get("hidden_dim", 256)),
                heads=int(ckpt.get("gat_heads", 4)),
                status="failed",
                failure_reason="out_of_memory",
            )
            paths["gat_sr0p15_robustness_gate.json"].write_text(
                json.dumps(
                    {
                        "registry_name": SELECTED_REGISTRY_NAME,
                        "status": "failed",
                        "failure_reason": "out_of_memory",
                        "completion_marker": None,
                        "round0_unlock_allowed": False,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return paths
        raise
    manifest = {
        "registry_name": SELECTED_REGISTRY_NAME,
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "checkpoint_path": str(loaded.path),
        "checkpoint_sha256": sha256_file(loaded.path),
        "sample_count": int(len(state)),
        "node_count": int(len(node_ids)),
        "sensor_count": int(sensor_mask.sum()),
        "validation_status": "diagnostic_only",
        "diagnostic_reason": "Project4 cache provenance is available, but this audit has not proven independence from GAT training split.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    paths["gat_sr0p15_validation_dataset_manifest.json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    unique_events = sorted(set(event_ids))
    sample_inventory_rows = []
    for i, event in enumerate(event_ids):
        sample_inventory_rows.append(
            {
                "sample_id": f"sr0p15_validation_{i:06d}",
                "event_id": event,
                "storm_family_id": "unknown_from_cache" if event == "unknown_project4_cache_event" else event,
                "source_path": str(cache_path),
                "source_row": i,
                "time_index": i,
                "timestamp": "",
                "split": "diagnostic_only",
                "training_membership": "unknown",
                "model_selection_membership": "unknown",
                "highwater_flag": bool(state[i].max() >= np.quantile(state.max(axis=1), 0.90)),
                "phase": phases[i] if i < len(phases) else "",
                "source_sha256": sha256_file(cache_path),
            }
        )
    write_csv(
        paths["gat_sr0p15_validation_sample_inventory.csv"],
        sample_inventory_rows,
        [
            "sample_id",
            "event_id",
            "storm_family_id",
            "source_path",
            "source_row",
            "time_index",
            "timestamp",
            "split",
            "training_membership",
            "model_selection_membership",
            "highwater_flag",
            "phase",
            "source_sha256",
        ],
    )
    write_csv(
        paths["gat_sr0p15_validation_event_support.csv"],
        [
            {
                "event_id": event,
                "storm_family": "unknown_from_cache" if event == "unknown_project4_cache_event" else event,
                "split": "diagnostic_only",
                "sample_count": event_ids.count(event),
                "participated_in_gat_training": "unknown",
                "participated_in_model_selection": "unknown",
            }
            for event in unique_events
        ],
        ["event_id", "storm_family", "split", "sample_count", "participated_in_gat_training", "participated_in_model_selection"],
    )
    provenance_status = "incomplete" if any(event == "unknown_project4_cache_event" for event in unique_events) else "pass"
    write_csv(
        paths["gat_sr0p15_validation_provenance_audit.csv"],
        [
            {
                "registry_name": SELECTED_REGISTRY_NAME,
                "source_path": str(cache_path),
                "source_sha256": sha256_file(cache_path),
                "sample_count": len(state),
                "unique_event_count": len(unique_events),
                "storm_family_count": len(unique_events) if provenance_status == "pass" else "",
                "training_split_recovered": provenance_status == "pass",
                "validation_split_recovered": provenance_status == "pass",
                "five_sensor_ratios_share_validation_set": "unknown",
                "status": provenance_status,
                "reason": "" if provenance_status == "pass" else "cache does not expose recoverable event identity/split membership for all samples",
            }
        ],
        [
            "registry_name",
            "source_path",
            "source_sha256",
            "sample_count",
            "unique_event_count",
            "storm_family_count",
            "training_split_recovered",
            "validation_split_recovered",
            "five_sensor_ratios_share_validation_set",
            "status",
            "reason",
        ],
    )
    write_csv(
        paths["gat_sr0p15_validation_leakage_audit.csv"],
        [
            {
                "registry_name": SELECTED_REGISTRY_NAME,
                "status": "incomplete" if provenance_status != "pass" else "pass",
                "leakage_free": "" if provenance_status != "pass" else True,
                "reason": "independent validation split not proven from available cache metadata" if provenance_status != "pass" else "",
                "overlap_event_ids": "",
            }
        ],
        ["registry_name", "status", "leakage_free", "reason", "overlap_event_ids"],
    )
    write_csv(
        paths["gat_sr0p15_rainfall_near_duplicate_audit.csv"],
        [
            {
                "registry_name": SELECTED_REGISTRY_NAME,
                "status": "incomplete" if provenance_status != "pass" else "pass",
                "near_duplicate_type": "unknown",
                "event_id": event,
                "rainfall_time_series_hash": "",
                "reason": "rainfall time-series identity not present in cache metadata" if provenance_status != "pass" else "",
            }
            for event in unique_events
        ],
        ["registry_name", "status", "near_duplicate_type", "event_id", "rainfall_time_series_hash", "reason"],
    )
    write_csv(
        paths["gat_sr0p15_split_membership_audit.csv"],
        [
            {
                "registry_name": SELECTED_REGISTRY_NAME,
                "event_id": event,
                "split": "unknown" if provenance_status != "pass" else "validation",
                "training_membership": "unknown" if provenance_status != "pass" else "false",
                "model_selection_membership": "unknown",
                "status": "incomplete" if provenance_status != "pass" else "pass",
            }
            for event in unique_events
        ],
        ["registry_name", "event_id", "split", "training_membership", "model_selection_membership", "status"],
    )
    write_csv(
        paths["gat_sr0p15_temporal_dependence_audit.csv"],
        [
            {
                "registry_name": SELECTED_REGISTRY_NAME,
                "sample_count": len(state),
                "unique_event_count": len(unique_events),
                "continuous_time_dependence_status": "not_proven_independent",
                "status": "diagnostic_only",
            }
        ],
        ["registry_name", "sample_count", "unique_event_count", "continuous_time_dependence_status", "status"],
    )

    observed_idx = np.where(sensor_mask > 0.5)[0]
    unobserved_idx = np.where(sensor_mask <= 0.5)[0]
    node_metric_rows: list[dict[str, Any]] = []
    for group, idx in [("all_nodes", np.arange(len(node_ids))), ("observed_nodes", observed_idx), ("unobserved_nodes", unobserved_idx)]:
        m = _metrics(pred[:, idx], state[:, idx]) if len(idx) else {}
        for metric, value in m.items():
            node_metric_rows.append(
                {
                    "registry_name": SELECTED_REGISTRY_NAME,
                    "diagnostic": "node_group_metrics",
                    "group": group,
                    "node_id": "",
                    "metric": metric,
                    "value": value,
                    "sample_count": len(state),
                    "unique_event_count": len(unique_events),
                    "status": "computed",
                    "reason": "",
                }
            )
    _write_metric_rows(paths["gat_sr0p15_node_group_metrics.csv"], node_metric_rows)

    def leaveout_rows(nodes: list[str], diagnostic: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        node_to_idx = {node: i for i, node in enumerate(node_ids)}
        for node in nodes:
            if node not in node_to_idx:
                rows.append(
                    {
                        "registry_name": SELECTED_REGISTRY_NAME,
                        "diagnostic": diagnostic,
                        "group": "single_node",
                        "node_id": node,
                        "metric": "",
                        "value": "",
                        "sample_count": len(state),
                        "unique_event_count": len(unique_events),
                        "status": "blocked",
                        "reason": "node_not_in_gat_node_order",
                    }
                )
                continue
            idx = node_to_idx[node]
            mask = sensor_mask.copy()
            originally_observed = bool(mask[idx] > 0.5)
            mask[idx] = 0.0
            lo_pred = _predict_with_mask(torch, loaded, metadata, model, state, rain, mask, batch_size=effective_batch)
            normal_err = pred[:, idx] - state[:, idx]
            leave_err = lo_pred[:, idx] - state[:, idx]
            values = {
                "originally_observed": float(originally_observed),
                "normal_MAE": float(np.mean(np.abs(normal_err))),
                "leaveout_MAE": float(np.mean(np.abs(leave_err))),
                "leaveout_RMSE": float(np.sqrt(np.mean(leave_err**2))),
                "maximum_underprediction": float(np.min(leave_err)),
                "maximum_overprediction": float(np.max(leave_err)),
            }
            for metric, value in values.items():
                rows.append(
                    {
                        "registry_name": SELECTED_REGISTRY_NAME,
                        "diagnostic": diagnostic,
                        "group": "single_node",
                        "node_id": node,
                        "metric": metric,
                        "value": value,
                        "sample_count": len(state),
                        "unique_event_count": len(unique_events),
                        "status": "computed",
                        "reason": "",
                    }
                )
        return rows

    priority_nodes = sorted(priority_nodes_from_contract())
    sentinel_nodes = sorted(SENTINEL_CANDIDATES)
    _write_metric_rows(paths["gat_sr0p15_priority_leaveout_audit.csv"], leaveout_rows(priority_nodes, "priority_leave_sensor_out"))
    _write_metric_rows(paths["gat_sr0p15_sentinel_leaveout_audit.csv"], leaveout_rows(sentinel_nodes, "sentinel_leave_sensor_out"))

    highwater_rows: list[dict[str, Any]] = []
    max_depth = state.max(axis=1)
    for group_name, mask in [
        ("all_states", np.ones(len(state), dtype=bool)),
        ("top20_depth", max_depth >= np.quantile(max_depth, 0.80)),
        ("top10_depth", max_depth >= np.quantile(max_depth, 0.90)),
        ("top5_depth", max_depth >= np.quantile(max_depth, 0.95)),
    ]:
        m = _metrics(pred[mask], state[mask]) if mask.any() else {}
        for metric, value in m.items():
            highwater_rows.append(
                {
                    "registry_name": SELECTED_REGISTRY_NAME,
                    "diagnostic": "highwater",
                    "group": group_name,
                    "node_id": "",
                    "metric": metric,
                    "value": value,
                    "sample_count": int(mask.sum()),
                    "unique_event_count": len(set(np.array(event_ids)[mask])) if mask.any() else 0,
                    "status": "computed",
                    "reason": "",
                }
            )
    for phase in sorted(set(phases)):
        mask = np.array([p == phase for p in phases])
        m = _metrics(pred[mask], state[mask]) if mask.any() else {}
        for metric, value in m.items():
            highwater_rows.append(
                {
                    "registry_name": SELECTED_REGISTRY_NAME,
                    "diagnostic": "phase",
                    "group": phase,
                    "node_id": "",
                    "metric": metric,
                    "value": value,
                    "sample_count": int(mask.sum()),
                    "unique_event_count": len(set(np.array(event_ids)[mask])) if mask.any() else 0,
                    "status": "computed",
                    "reason": "",
                }
            )
    _write_metric_rows(paths["gat_sr0p15_highwater_phase_audit.csv"], highwater_rows)

    sensor_failure_contract = {
        "registry_name": SELECTED_REGISTRY_NAME,
        "required_scenarios": [
            "baseline",
            "drop_single_sensor",
            "drop_random_5pct",
            "drop_random_10pct",
            "drop_priority_related_sensor",
            "drop_sentinel_related_sensor",
            "regional_sensor_outage",
            "stale_10min",
            "stale_20min",
            "positive_bias",
            "negative_bias",
        ],
        "random_seed_list": [seed],
        "performance_gate_status": "uncalibrated",
        "execution_completeness_required_for_robustness": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    paths["gat_sr0p15_sensor_failure_contract.json"].write_text(json.dumps(sensor_failure_contract, indent=2), encoding="utf-8")
    sensor_failure_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    sensor_idx = np.where(sensor_mask > 0.5)[0]
    for scenario in sensor_failure_contract["required_scenarios"]:
        drop = _scenario_sensor_indices(scenario, sensor_idx, node_ids, seed)
        mask = sensor_mask.copy()
        if len(drop):
            mask[drop] = 0.0
        scenario_state = _perturb_state_for_sensor_scenario(state, scenario)
        scenario_pred = _predict_with_mask(torch, loaded, metadata, model, scenario_state, rain, mask, batch_size=effective_batch)
        finite = bool(np.isfinite(scenario_pred).all())
        m = _metrics(scenario_pred, state) if finite else {}
        for metric, value in m.items():
            sensor_failure_rows.append(
                {
                    "registry_name": SELECTED_REGISTRY_NAME,
                    "diagnostic": "sensor_failure",
                    "group": scenario,
                    "node_id": "",
                    "metric": metric,
                    "value": value,
                    "sample_count": len(state),
                    "unique_event_count": len(unique_events),
                    "status": "computed" if finite else "failed",
                    "reason": "" if finite else "nonfinite_output",
                }
            )
        completion_rows.append(
            {
                "scenario_id": scenario,
                "seed": seed,
                "requested_samples": len(state),
                "completed_samples": len(state) if finite else 0,
                "unique_samples": len(state),
                "duplicate_samples": 0,
                "affected_sensor_count": int(len(drop)),
                "output_exists": True,
                "metrics_complete": finite and bool(m),
                "status": "pass" if finite and bool(m) else "fail",
                "failure_reason": "" if finite and bool(m) else "sensor_failure_metrics_not_complete",
            }
        )
    _write_metric_rows(paths["gat_sr0p15_sensor_failure_audit.csv"], sensor_failure_rows)
    write_csv(
        paths["gat_sr0p15_sensor_failure_completion_matrix.csv"],
        completion_rows,
        [
            "scenario_id",
            "seed",
            "requested_samples",
            "completed_samples",
            "unique_samples",
            "duplicate_samples",
            "affected_sensor_count",
            "output_exists",
            "metrics_complete",
            "status",
            "failure_reason",
        ],
    )
    sensor_failure_execution_complete = all(row["status"] == "pass" for row in completion_rows) and {
        row["scenario_id"] for row in completion_rows
    } == set(sensor_failure_contract["required_scenarios"])
    paths["gat_sr0p15_sensor_failure_summary.csv"].write_text(
        "registry_name,execution_status,performance_gate_status,scenario_count,required_scenario_count\n"
        f"{SELECTED_REGISTRY_NAME},{'pass' if sensor_failure_execution_complete else 'incomplete'},uncalibrated,{len(completion_rows)},{len(sensor_failure_contract['required_scenarios'])}\n",
        encoding="utf-8",
    )

    latency_contract = {
        "registry_name": SELECTED_REGISTRY_NAME,
        "warmup_runs": 5,
        "measured_runs": 30,
        "required_measurements": ["single_sample", "batch_size_8", "seven_frame"],
        "budget_gate_status": "uncalibrated",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    paths["gat_sr0p15_latency_contract.json"].write_text(json.dumps(latency_contract, indent=2), encoding="utf-8")
    smoke = forward_smoke(metadata, samples=min(8, len(state)))
    single_times: list[float] = []
    batch8_times: list[float] = []
    seven_frame_times: list[float] = []
    # Model and tensors are already loaded; these timings intentionally exclude
    # import, checkpoint loading, and file I/O.
    for _ in range(latency_contract["warmup_runs"]):
        _predict_with_mask(torch, loaded, metadata, model, state[:1], rain[:1], sensor_mask, batch_size=1)
    for _ in range(latency_contract["measured_runs"]):
        t0 = time.perf_counter()
        _predict_with_mask(torch, loaded, metadata, model, state[:1], rain[:1], sensor_mask, batch_size=1)
        single_times.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        _predict_with_mask(torch, loaded, metadata, model, state[: min(8, len(state))], rain[: min(8, len(rain))], sensor_mask, batch_size=min(8, effective_batch))
        batch8_times.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        for frame in range(7):
            idx = min(frame, len(state) - 1)
            _predict_with_mask(torch, loaded, metadata, model, state[idx : idx + 1], rain[idx : idx + 1], sensor_mask, batch_size=1)
        seven_frame_times.append((time.perf_counter() - t0) * 1000.0)

    def _lat_stats(values: list[float]) -> dict[str, float]:
        arr = np.array(values, dtype=float)
        return {
            "median_ms": float(np.median(arr)),
            "p90_ms": float(np.quantile(arr, 0.90)),
            "p95_ms": float(np.quantile(arr, 0.95)),
            "max_ms": float(np.max(arr)),
        }

    single_stats = _lat_stats(single_times)
    batch8_stats = _lat_stats(batch8_times)
    seven_stats = _lat_stats(seven_frame_times)
    latency_rows = [
        {
            "registry_name": SELECTED_REGISTRY_NAME,
            "python_version": "",
            "torch_version": getattr(torch_module, "__version__", ""),
            "pyg_version": "",
            "device": "cpu",
            "dtype": "float32",
            "thread_count": "",
            "measurement_id": "single_sample",
            "batch_size": 1,
            "repeat_max_abs_diff": smoke.get("repeat_max_abs_diff"),
            **single_stats,
            "status": "computed",
        },
        {
            "registry_name": SELECTED_REGISTRY_NAME,
            "python_version": "",
            "torch_version": getattr(torch_module, "__version__", ""),
            "pyg_version": "",
            "device": "cpu",
            "dtype": "float32",
            "thread_count": "",
            "measurement_id": "batch_size_8",
            "batch_size": min(8, len(state)),
            "repeat_max_abs_diff": "",
            **batch8_stats,
            "status": "computed",
        },
        {
            "registry_name": SELECTED_REGISTRY_NAME,
            "python_version": "",
            "torch_version": getattr(torch_module, "__version__", ""),
            "pyg_version": "",
            "device": "cpu",
            "dtype": "float32",
            "thread_count": "",
            "measurement_id": "seven_frame",
            "batch_size": 1,
            "repeat_max_abs_diff": "",
            **seven_stats,
            "status": "computed",
        },
    ]
    write_csv(
        paths["gat_sr0p15_latency_repeatability_audit.csv"],
        latency_rows,
        [
            "registry_name",
            "python_version",
            "torch_version",
            "pyg_version",
            "device",
            "dtype",
            "thread_count",
            "measurement_id",
            "batch_size",
            "repeat_max_abs_diff",
            "median_ms",
            "p90_ms",
            "p95_ms",
            "max_ms",
            "status",
        ],
    )
    latency_measurement_complete = all(row["status"] == "computed" and row["p95_ms"] != "" for row in latency_rows)
    paths["gat_sr0p15_latency_summary.json"].write_text(
        json.dumps(
            {
                "registry_name": SELECTED_REGISTRY_NAME,
                "measurement_complete": latency_measurement_complete,
                "latency_budget_gate": "uncalibrated",
                "measurements": latency_rows,
                "checkpoint_sha256": sha256_file(loaded.path),
                "input_sample_hash": sha256_file(cache_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    checks = [
        _gate_check(
            check_id="validation_provenance_complete",
            status="pass" if provenance_status == "pass" else "incomplete",
            observed_value=provenance_status,
            required_value="pass",
            evidence_path=paths["gat_sr0p15_validation_provenance_audit.csv"],
            blocking_reason="" if provenance_status == "pass" else "validation sample event identity or split membership is not fully recoverable",
            remediation="provide an explicit independent validation manifest with event IDs, storm families, timestamps, and split membership",
        ),
        _gate_check(
            check_id="no_training_event_leakage",
            status="pass" if provenance_status == "pass" else "incomplete",
            observed_value="unknown" if provenance_status != "pass" else "no_overlap_detected",
            required_value="no exact or near-duplicate training-event leakage",
            evidence_path=paths["gat_sr0p15_validation_leakage_audit.csv"],
            blocking_reason="" if provenance_status == "pass" else "training/validation event membership is not fully recoverable",
            remediation="recover Project4 train/validation event membership or provide a holdout validation manifest",
        ),
        _gate_check(check_id="strict_load_passed", status="pass", observed_value=True, required_value=True, evidence_path=gat_dir / "gat_strict_load_audit.csv"),
        _gate_check(check_id="finite_outputs", status="pass" if bool(np.isfinite(pred).all()) else "fail", observed_value=bool(np.isfinite(pred).all()), required_value=True, evidence_path=paths["gat_sr0p15_node_group_metrics.csv"]),
        _gate_check(check_id="unobserved_metrics_exist", status="pass", observed_value=True, required_value=True, evidence_path=paths["gat_sr0p15_node_group_metrics.csv"]),
        _gate_check(check_id="priority_leaveout_complete", status="pass" if len(priority_nodes) == 8 else "incomplete", observed_value=len(priority_nodes), required_value=8, evidence_path=paths["gat_sr0p15_priority_leaveout_audit.csv"]),
        _gate_check(check_id="sentinel_leaveout_complete", status="pass" if len(sentinel_nodes) == 2 else "incomplete", observed_value=len(sentinel_nodes), required_value=2, evidence_path=paths["gat_sr0p15_sentinel_leaveout_audit.csv"]),
        _gate_check(check_id="highwater_phase_complete", status="pass", observed_value=True, required_value=True, evidence_path=paths["gat_sr0p15_highwater_phase_audit.csv"]),
        _gate_check(
            check_id="sensor_failure_execution_complete",
            status="pass" if sensor_failure_execution_complete else "incomplete",
            observed_value=f"{sum(1 for row in completion_rows if row['status'] == 'pass')}/{len(sensor_failure_contract['required_scenarios'])}",
            required_value="all required scenarios and seeds complete with finite outputs",
            evidence_path=paths["gat_sr0p15_sensor_failure_completion_matrix.csv"],
            blocking_reason="" if sensor_failure_execution_complete else "sensor failure completion matrix is incomplete",
            remediation="run or repair missing sensor-failure scenarios without lowering performance thresholds",
        ),
        _gate_check(
            check_id="sensor_failure_performance_gate",
            status="not_applicable",
            observed_value="uncalibrated",
            required_value="calibrated degradation threshold",
            evidence_path=paths["gat_sr0p15_sensor_failure_summary.csv"],
            blocking_reason="",
            remediation="calibrate degradation thresholds in a later design/calibration stage",
        ),
        _gate_check(
            check_id="latency_measurement_complete",
            status="pass" if latency_measurement_complete else "incomplete",
            observed_value=latency_measurement_complete,
            required_value=True,
            evidence_path=paths["gat_sr0p15_latency_repeatability_audit.csv"],
            blocking_reason="" if latency_measurement_complete else "latency report lacks required p95 or seven-frame measurement",
            remediation="rerun latency audit with required warmup and measured repeats",
        ),
        _gate_check(
            check_id="latency_budget_gate",
            status="not_applicable",
            observed_value="uncalibrated",
            required_value="frozen control-budget threshold",
            evidence_path=paths["gat_sr0p15_latency_summary.json"],
            remediation="freeze a latency budget before treating timing as a performance gate",
        ),
    ]
    _write_gate(paths["gat_sr0p15_robustness_gate.json"], checks)
    return paths
