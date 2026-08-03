"""Materialise a development-only causal GAT history for the qualification pass.

Unlike the legacy bridge, the state signature excludes the checkpoint action,
which may already contain the newly applied Candidate transition.  Same-state
matching uses checkpoint depth, rainfall history through the checkpoint, and
actual actions only through t-5.  This mirrors the pre-action counterfactual
semantics used by Formal Raw Readmission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.models.temporal_sparse_gat_v42 import TemporalSparseGATReconstructorV42
from sewerrtc.v4.v42_fast_e2e import make_causal_rainfall_forecast
from sewerrtc.v4.v42_step1_dataset import (
    _build_usecols,
    _detail_extract_window,
    _sensor_layout,
    load_graph_assets,
)

QUALIFICATION_CONTRACT = "PROJECT6_V42_QUALIFICATION_FIRST_PASS_V1"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"input table is empty: {path}")
    return frame


def _detail(path: Path, required: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    available = set(map(str, header.columns))
    missing = [column for column in required if column not in available]
    if missing:
        raise KeyError(f"qualification GAT history detail missing columns: {missing[:10]}")
    return pd.read_csv(path, usecols=required, low_memory=False).loc[:, required]


def _cached_detail(
    cache: OrderedDict[str, pd.DataFrame],
    path: Path,
    required: list[str],
    max_items: int,
) -> pd.DataFrame:
    key = str(path.resolve())
    if key in cache:
        value = cache.pop(key)
        cache[key] = value
        return value
    value = _detail(path, required)
    cache[key] = value
    while len(cache) > max_items:
        cache.popitem(last=False)
    return value


def _bounds(path: Path) -> tuple[float, float]:
    values = pd.to_numeric(pd.read_csv(path, usecols=["elapsed_min"])["elapsed_min"], errors="coerce").dropna()
    if values.empty:
        raise ValueError("elapsed_min has no finite values")
    return float(values.min()), float(values.max())


def _pre_action_signature(detail: pd.DataFrame, checkpoint: float, graph: Any) -> str:
    extracted = _detail_extract_window(detail, checkpoint, graph.node_ids, graph.facility_ids)
    if extracted is None:
        raise ValueError("detail cannot reconstruct Step1 window at checkpoint")
    depth_at_checkpoint = np.ascontiguousarray(extracted["depth_history"][-1], dtype=np.float64)
    rainfall_history = np.ascontiguousarray(extracted["rainfall"], dtype=np.float64)
    # The t action may already be the Candidate transition.  Same-state identity
    # is therefore based on actual readback only through t-5.
    pre_action_history = np.ascontiguousarray(extracted["actions"][:-1], dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(depth_at_checkpoint.tobytes())
    digest.update(rainfall_history.tobytes())
    digest.update(pre_action_history.tobytes())
    return digest.hexdigest()


def _reconstruct(
    detail: pd.DataFrame,
    checkpoint: float,
    model: TemporalSparseGATReconstructorV42,
    graph: Any,
    mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchors = [checkpoint - 60.0 + 5.0 * index for index in range(13)]
    extracted = []
    for anchor in anchors:
        item = _detail_extract_window(detail, anchor, graph.node_ids, graph.facility_ids)
        if item is None:
            raise ValueError(f"missing exact causal Step1 window at anchor={anchor:.6f}")
        extracted.append(item)

    mask_history = np.broadcast_to(mask[None, :], (13, graph.n_nodes)).astype(np.float32, copy=True)
    sparse = np.stack([item["depth_history"] * mask_history for item in extracted]).astype(np.float32)
    masks = np.broadcast_to(mask_history[None, :, :], (13, 13, graph.n_nodes)).copy().astype(np.float32)
    rainfall = np.stack([item["rainfall"] for item in extracted]).astype(np.float32)
    actions = np.stack([item["actions"] for item in extracted]).astype(np.float32)

    with torch.no_grad():
        prediction = model(
            sparse_depth_history=torch.from_numpy(sparse).to(device),
            sensor_mask_history=torch.from_numpy(masks).to(device),
            rainfall_history=torch.from_numpy(rainfall).to(device),
            historical_actions=torch.from_numpy(actions).to(device),
            node_static=torch.from_numpy(graph.node_static).to(device),
            link_static=torch.from_numpy(graph.link_static).to(device),
            edge_index=torch.from_numpy(graph.edge_index).to(device),
            action_node_map=torch.from_numpy(graph.action_node_map).to(device),
        )
    history = prediction.depth_mean.detach().cpu().numpy().astype(np.float32)
    uncertainty = prediction.depth_std.detach().cpu().numpy().astype(np.float32)
    if history.shape != (13, graph.n_nodes):
        raise RuntimeError(f"unexpected reconstructed history shape: {history.shape}")
    return history, uncertainty, extracted[-1]["rainfall"].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--step1-window-manifest", type=Path, required=False)
    parser.add_argument("--history-source-manifest", type=Path, required=False)
    parser.add_argument("--step1-model-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--min-rainfall-groups", type=int, default=65)
    parser.add_argument("--sensor-ratio", type=float, default=0.10)
    parser.add_argument("--sensor-layout-seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--gat-layers", type=int, default=3)
    parser.add_argument("--detail-cache-items", type=int, default=8)
    args = parser.parse_args()

    if args.detail_cache_items < 2:
        raise ValueError("detail-cache-items must be >=2")
    frame = _read(args.input_manifest)
    history_source_path = args.history_source_manifest or (
        args.output_manifest.parent.parent / "QUALIFICATION_GAT_HISTORY_SOURCE_MANIFEST.parquet"
    )
    history_sources = _read(history_source_path)
    missing_history_columns = sorted({"state_key", "history_detail_path", "compatible"} - set(history_sources.columns))
    if missing_history_columns:
        raise KeyError(f"qualification history source manifest missing columns: {missing_history_columns}")
    if "qualification_only" not in frame.columns or not bool(frame["qualification_only"].astype(bool).all()):
        raise RuntimeError("qualification GAT bridge accepts qualification-only input")

    graph = load_graph_assets(args.project_root)
    mask, sensor_indices, sensor_sha = _sensor_layout(
        graph.n_nodes, args.sensor_ratio, args.sensor_layout_seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalSparseGATReconstructorV42(
        n_nodes=graph.n_nodes,
        n_facilities=graph.n_facilities,
        node_static_dim=graph.node_static.shape[1],
        link_static_dim=graph.link_static.shape[1],
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        gat_layers=args.gat_layers,
    ).to(device)
    model.load_state_dict(
        torch.load(args.step1_model_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    model.eval()

    required = _build_usecols(graph.node_ids, graph.facility_ids)
    cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
    bounds_cache: dict[str, tuple[float, float]] = {}
    output_rows: list[pd.Series] = []
    failures: list[dict[str, Any]] = []
    successful_groups: set[str] = set()
    successful_states: set[str] = set()

    for state_index, (state_key, state_rows) in enumerate(frame.groupby("state_key", sort=True), start=1):
        first = state_rows.iloc[0]
        rainfall_group = str(first["split_group_key"])
        event_id = str(first.get("event_id", ""))
        checkpoint = float(first["checkpoint_min"])
        candidate_path = Path(str(first["source_detail_path_candidate"]))
        failure_reason = "no_same_rainfall_history"
        try:
            candidate_detail = _cached_detail(cache, candidate_path, required, args.detail_cache_items)
            candidate_signature = _pre_action_signature(candidate_detail, checkpoint, graph)
            mapping = history_sources[
                history_sources["state_key"].astype(str).eq(str(state_key))
                & history_sources["compatible"].fillna(False).astype(bool)
            ]
            if mapping.empty:
                raise FileNotFoundError("no_history_source_mapping")
            history_path = Path(str(mapping.iloc[0]["history_detail_path"]))
            if not history_path.exists():
                failure_reason = "history_file_missing"
                raise FileNotFoundError(history_path)
            key = str(history_path.resolve())
            if key not in bounds_cache:
                bounds_cache[key] = _bounds(history_path)
            lower, upper = bounds_cache[key]
            if lower > checkpoint - 120.0 + 1.0e-6:
                failure_reason = "coverage_start_too_late"
                raise FileNotFoundError(failure_reason)
            if upper < checkpoint - 1.0e-6:
                failure_reason = "coverage_end_too_early"
                raise FileNotFoundError(failure_reason)
            history_detail = _cached_detail(cache, history_path, required, args.detail_cache_items)
            if _pre_action_signature(history_detail, checkpoint, graph) != candidate_signature:
                failure_reason = "pre_action_signature_mismatch"
                raise FileNotFoundError(failure_reason)
            for anchor in [checkpoint - 60.0 + 5.0 * index for index in range(13)]:
                if _detail_extract_window(history_detail, anchor, graph.node_ids, graph.facility_ids) is None:
                    failure_reason = "missing_gat_anchor"
                    raise FileNotFoundError(failure_reason)
            history_path = history_path.resolve()

            history, uncertainty, observed_rainfall = _reconstruct(
                history_detail, checkpoint, model, graph, mask, device
            )
            forecast = make_causal_rainfall_forecast(observed_rainfall)
            for _, source_row in state_rows.iterrows():
                record = source_row.copy()
                record["history_source_detail_path"] = str(history_path)
                record["history_depth"] = json.dumps(history.tolist(), allow_nan=False)
                record["gat_depth_std_history_mean"] = float(uncertainty.mean())
                record["gat_depth_std_current_mean"] = float(uncertainty[-1].mean())
                record["rainfall_forecast"] = json.dumps(forecast.tolist(), allow_nan=False)
                record["state_source"] = "gat_sparse_reconstruction"
                record["history_input_contract"] = "gat_compatible_causal_state"
                record["reconstructor_contract"] = "qualification_temporal_v42"
                record["reconstructed_history_contract"] = "PROJECT6_V42_CAUSAL_RECONSTRUCTED_HISTORY_V1"
                record["current_frame_repetition_used"] = False
                record["authoritative_swmm_history_used_as_online_input"] = False
                record["realized_future_rainfall_used_online"] = False
                record["future_SWMM_trajectories_supervision_only"] = True
                record["sensor_layout_sha256"] = sensor_sha
                record["qualification_contract_id"] = QUALIFICATION_CONTRACT
                record["qualification_only"] = True
                record["development_only"] = True
                record["formal_mainline_authorized"] = False
                output_rows.append(record)
            successful_groups.add(rainfall_group)
            successful_states.add(str(state_key))
        except Exception as exc:
            failures.append(
                {
                    "rainfall_group": rainfall_group,
                    "state_key": str(state_key),
                    "event_id": event_id,
                    "checkpoint_min": checkpoint,
                    "candidate_detail": str(candidate_path),
                    "required_start_min": checkpoint - 120.0,
                    "failure_reason": failure_reason,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if state_index % 10 == 0:
            print(
                json.dumps(
                    {
                        "stage": "qualification_causal_gat_history",
                        "processed_states": state_index,
                        "output_states": len(successful_states),
                        "output_groups": len(successful_groups),
                        "failed_states": len(failures),
                        "cache_items": len(cache),
                    },
                    allow_nan=False,
                ),
                flush=True,
            )

    result = pd.DataFrame(output_rows)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if not result.empty:
        result.to_parquet(args.output_manifest, index=False)
    status = "pass" if len(successful_groups) >= args.min_rainfall_groups else "fail"
    audit = {
        "contract_id": QUALIFICATION_CONTRACT,
        "stage": "qualification_causal_gat_history",
        "status": status,
        "qualification_only": True,
        "development_only": True,
        "formal_mainline_authorized": False,
        "input_rows": int(len(frame)),
        "history_source_manifest": str(history_source_path),
        "output_rows": int(len(result)),
        "input_states": int(frame["state_key"].astype(str).nunique()),
        "output_states": int(len(successful_states)),
        "input_rainfall_groups": int(frame["split_group_key"].astype(str).nunique()),
        "output_rainfall_groups": int(len(successful_groups)),
        "minimum_rainfall_groups": int(args.min_rainfall_groups),
        "lost_rainfall_groups": sorted(set(frame["split_group_key"].astype(str)) - successful_groups),
        "failed_states": int(len(failures)),
        "failure_reason_counts": pd.Series([item["failure_reason"] for item in failures]).value_counts().to_dict() if failures else {},
        "failure_examples": failures[:100],
        "sensor_ratio": float(args.sensor_ratio),
        "sensor_count": int(len(sensor_indices)),
        "sensor_layout_sha256": sensor_sha,
        "pre_action_signature_excludes_checkpoint_action": True,
        "current_frame_repetition_used": False,
        "authoritative_swmm_history_used_as_online_input": False,
        "realized_future_rainfall_used_online": False,
    }
    audit_path = args.output_manifest.parent / "QUALIFICATION_GAT_HISTORY_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if status == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
