"""Materialise the Formal F2 causal 13-frame GAT history."""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sewerrtc.models.temporal_sparse_gat_v42 import TemporalSparseGATReconstructorV42
from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID, read_table
from sewerrtc.v4.v42_fast_e2e import make_causal_rainfall_forecast
from sewerrtc.v4.v42_step1_dataset import (
    _build_usecols,
    _detail_extract_window,
    _sensor_layout,
    load_graph_assets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _detail(path: Path, required: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    missing = [c for c in required if c not in set(map(str, header.columns))]
    if missing:
        raise KeyError(f"formal GAT history detail missing required columns: {missing[:10]}")
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


def _signature(detail: pd.DataFrame, checkpoint: float, graph: Any) -> tuple[np.ndarray, ...]:
    extracted = _detail_extract_window(detail, checkpoint, graph.node_ids, graph.facility_ids)
    if extracted is None:
        raise ValueError("detail cannot reconstruct Step1 window at checkpoint")
    # The checkpoint action is the branch transition. History identity must
    # compare only the common pre-action prefix, not that branch action.
    return (
        extracted["depth_history"][-1].astype(np.float64),
        extracted["actions"][:-1].astype(np.float64),
        extracted["rainfall"].astype(np.float64),
    )


def _same(left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...]) -> bool:
    return all(
        a.shape == b.shape and np.allclose(a, b, atol=1.0e-6, rtol=0.0)
        for a, b in zip(left, right)
    )


def _validate_history(
    path: Path,
    *,
    checkpoint: float,
    candidate_signature: tuple[np.ndarray, ...],
    required: list[str],
    graph: Any,
    cache: OrderedDict[str, pd.DataFrame],
    cache_items: int,
) -> pd.DataFrame:
    lower, upper = _bounds(path)
    if lower > checkpoint - 120.0 + 1.0e-6:
        raise ValueError("history source coverage starts after checkpoint-120")
    if upper < checkpoint - 1.0e-6:
        raise ValueError("history source coverage ends before checkpoint")
    detail = _cached_detail(cache, path, required, cache_items)
    if not _same(candidate_signature, _signature(detail, checkpoint, graph)):
        raise ValueError("history source pre-action signature mismatch")
    for anchor in (checkpoint - 60.0 + 5.0 * i for i in range(13)):
        if _detail_extract_window(detail, anchor, graph.node_ids, graph.facility_ids) is None:
            raise ValueError(f"missing exact GAT anchor: {anchor:.6f}")
    return detail


def _reconstruct(
    detail: pd.DataFrame,
    checkpoint: float,
    model: TemporalSparseGATReconstructorV42,
    graph: Any,
    mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    examples = []
    for anchor in (checkpoint - 60.0 + 5.0 * i for i in range(13)):
        item = _detail_extract_window(detail, anchor, graph.node_ids, graph.facility_ids)
        if item is None:
            raise ValueError(f"missing exact causal Step1 window at anchor={anchor:.6f}")
        examples.append(item)
    masked = np.broadcast_to(mask[None, :], (13, graph.n_nodes)).astype(np.float32, copy=True)
    sparse = np.stack([item["depth_history"] * masked for item in examples]).astype(np.float32)
    masks = np.broadcast_to(masked[None, :, :], (13, 13, graph.n_nodes)).copy().astype(np.float32)
    rainfall = np.stack([item["rainfall"] for item in examples]).astype(np.float32)
    actions = np.stack([item["actions"] for item in examples]).astype(np.float32)
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
        raise RuntimeError(f"unexpected formal reconstructed history shape: {history.shape}")
    return history, uncertainty, examples[-1]["rainfall"].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--input-manifest", type=Path, required=True)
    ap.add_argument("--step1-window-manifest", type=Path, required=True)
    ap.add_argument("--history-source-manifest", type=Path, required=True)
    ap.add_argument("--step1-model-dir", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path, required=True)
    ap.add_argument("--min-rainfall-groups", type=int, default=69)
    ap.add_argument("--sensor-ratio", type=float, default=0.1)
    ap.add_argument("--sensor-layout-seed", type=int, default=42)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--gat-layers", type=int, default=3)
    ap.add_argument("--detail-cache-items", type=int, default=12)
    args = ap.parse_args()
    if args.detail_cache_items < 2:
        raise ValueError("detail-cache-items must be >=2")

    frame_all = read_table(args.input_manifest)
    frame = frame_all[pd.to_numeric(frame_all["checkpoint_min"], errors="coerce") >= 120.0].copy()
    if frame.empty:
        raise RuntimeError("Formal GAT materialisation has no checkpoint >=120 min states")
    required_source = {"state_key", "history_detail_path", "compatible", "formal_generation_id"}
    sources = read_table(args.history_source_manifest)
    missing = sorted(required_source - set(sources.columns))
    if missing:
        raise KeyError(f"Formal history source manifest missing columns: {missing}")
    if not bool((sources["formal_generation_id"].astype(str) == FORMAL_GENERATION_ID).all()):
        raise RuntimeError("history source manifest generation does not match Formal F2")
    sources = sources[sources["compatible"].fillna(False).astype(bool)].copy()
    frame = frame[frame["state_key"].astype(str).isin(sources["state_key"].astype(str))].copy()
    if frame.empty:
        raise RuntimeError("no raw states remain after Formal history-source admission")
    if not bool(frame["training_admission_authorized"].astype(bool).all()):
        raise RuntimeError("formal GAT materialiser requires raw-authorized Step2 rows")

    graph = load_graph_assets(args.project_root)
    mask, indices, sensor_sha = _sensor_layout(graph.n_nodes, args.sensor_ratio, args.sensor_layout_seed)
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
    model.load_state_dict(torch.load(args.step1_model_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    required = _build_usecols(graph.node_ids, graph.facility_ids)
    source_by_state = {str(row.state_key): row for row in sources.itertuples(index=False)}
    cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
    output: list[pd.Series] = []
    failures: list[dict[str, Any]] = []
    successful_groups: set[str] = set()
    groups = list(frame.groupby("state_key", sort=True))
    for index, (state, group) in enumerate(groups, start=1):
        first = group.iloc[0]
        rainfall = str(first["split_group_key"])
        checkpoint = float(first["checkpoint_min"])
        candidate_path = Path(str(first["source_detail_path_candidate"]))
        try:
            candidate = _cached_detail(cache, candidate_path, required, args.detail_cache_items)
            candidate_signature = _signature(candidate, checkpoint, graph)
            source = source_by_state.get(str(state))
            if source is None:
                raise FileNotFoundError("no Formal history source mapping for state")
            history_path = Path(str(source.history_detail_path))
            history = _validate_history(
                history_path,
                checkpoint=checkpoint,
                candidate_signature=candidate_signature,
                required=required,
                graph=graph,
                cache=cache,
                cache_items=args.detail_cache_items,
            )
            reconstructed, uncertainty, observed = _reconstruct(history, checkpoint, model, graph, mask, device)
            forecast = make_causal_rainfall_forecast(observed)
            for _, row in group.iterrows():
                record = row.copy()
                record["history_source_detail_path"] = str(history_path.resolve())
                record["history_depth"] = json.dumps(reconstructed.tolist(), allow_nan=False)
                record["gat_depth_std_history_mean"] = float(uncertainty.mean())
                record["gat_depth_std_current_mean"] = float(uncertainty[-1].mean())
                record["rainfall_forecast"] = json.dumps(forecast.tolist(), allow_nan=False)
                record["state_source"] = "gat_sparse_reconstruction"
                record["history_input_contract"] = "gat_compatible_causal_state"
                record["reconstructor_contract"] = "formal_temporal_v42"
                record["reconstructed_history_contract"] = "PROJECT6_V42_CAUSAL_RECONSTRUCTED_HISTORY_V1"
                record["current_frame_repetition_used"] = False
                record["authoritative_swmm_history_used_as_online_input"] = False
                record["realized_future_rainfall_used_online"] = False
                record["future_SWMM_trajectories_supervision_only"] = True
                record["sensor_layout_sha256"] = sensor_sha
                output.append(record)
            successful_groups.add(rainfall)
        except Exception as exc:
            failures.append({
                "rainfall_group": rainfall,
                "state_key": str(state),
                "checkpoint_min": checkpoint,
                "detail_path_candidate": str(candidate_path),
                "required_history_start_min": checkpoint - 120.0,
                "failed_anchor": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
        if index % 25 == 0:
            print(json.dumps({"stage": "formal_f2_step2_gat_history", "processed_states": index, "total_states": len(groups), "output_rows": len(output), "failed_states": len(failures), "detail_cache_items": len(cache)}, allow_nan=False), flush=True)

    result = pd.DataFrame(output)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if not result.empty:
        result.to_parquet(args.output_manifest, index=False)
    audit = {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "stage": "formal_f2_step2_gat_history",
        "status": "pass" if len(successful_groups) >= args.min_rainfall_groups else "fail",
        "development_only": False,
        "formal_mainline_authorized": False,
        "input_rows_before_checkpoint_gate": int(len(frame_all)),
        "input_rows": int(len(frame)),
        "output_rows": int(len(result)),
        "input_states": int(frame["state_key"].astype(str).nunique()),
        "output_states": int(result["state_key"].astype(str).nunique()) if not result.empty else 0,
        "input_rainfall_groups": int(frame["split_group_key"].astype(str).nunique()),
        "output_rainfall_groups": int(len(successful_groups)),
        "lost_rainfall_groups": sorted(set(frame["split_group_key"].astype(str)) - successful_groups),
        "minimum_rainfall_groups": args.min_rainfall_groups,
        "failed_states": len(failures),
        "failure_examples": failures[:200],
        "history_source_manifest": str(args.history_source_manifest),
        "state_source": "gat_sparse_reconstruction",
        "reconstructed_history_contract": "PROJECT6_V42_CAUSAL_RECONSTRUCTED_HISTORY_V1",
        "current_frame_repetition_used": False,
        "authoritative_swmm_history_used_as_online_input": False,
        "realized_future_rainfall_used_online": False,
        "sensor_ratio": args.sensor_ratio,
        "sensor_count": len(indices),
        "sensor_layout_sha256": sensor_sha,
        "detail_cache_limit": args.detail_cache_items,
    }
    audit_path = args.output_manifest.parent / "FORMAL_F2_STEP2_GAT_HISTORY_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if audit["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
