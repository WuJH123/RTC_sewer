"""Bounded, no-SWMM audit of Step2 action sensitivity within states."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_v42_step2_fast import BRANCHES, _forward, _graph_indices, _hash_model, _tensorise
from sewerrtc.v4.formal_f2 import read_table
from sewerrtc.v4.models_v42.hydraulic_multi_reference import MultiReferenceHydraulicSurrogate
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import (
    _load_graph_topology,
    build_surrogate_action_node_map,
)


def _batch_indices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield np.arange(start, min(start + batch_size, n), dtype=np.int64)


def _required_columns(manifest: Path) -> list[str]:
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(manifest).schema.names)
    columns = ["state_key", "history_depth", "history_actions_readback", "rainfall_forecast", "pfv_delta", "tfv_delta", "peak_delta"]
    for branch in BRANCHES:
        action = "action_dynamic_internal_input_readback" if branch == "dynamic_internal" and "action_dynamic_internal_input_readback" in names else f"action_{branch}_readback"
        columns.extend([action, f"trajectory_depth_{branch}", f"trajectory_flood_{branch}"])
    missing = sorted(set(columns) - names)
    if missing:
        raise KeyError(f"manifest missing chunk columns: {missing}")
    return list(dict.fromkeys(columns))


def _iter_manifest_batches(manifest: Path, state_keys: set[str], batch_size: int):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(manifest)
    columns = _required_columns(manifest)
    for batch in parquet.iter_batches(batch_size=int(batch_size), columns=columns, use_threads=True):
        frame = batch.to_pandas()
        frame["state_key"] = frame["state_key"].astype(str)
        selected = frame[frame["state_key"].isin(state_keys)].reset_index(drop=True)
        if not selected.empty:
            yield selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--states", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--model-root", type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")

    if args.manifest.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        state_frame = pd.read_parquet(args.manifest, columns=["state_key"])
        state_keys = set(sorted(state_frame["state_key"].astype(str).unique())[: max(1, args.states)])
        meta_parts = []
        for batch in _iter_manifest_batches(args.manifest, state_keys, args.batch_size):
            meta_parts.append(batch[["state_key", "pfv_delta", "tfv_delta"]].copy())
        selected = pd.concat(meta_parts, ignore_index=True)
        batch_factory = lambda: _iter_manifest_batches(args.manifest, state_keys, args.batch_size)
    else:
        frame = read_table(args.manifest)
        state_keys = set(sorted(frame["state_key"].astype(str).unique())[: max(1, args.states)])
        selected = frame[frame["state_key"].astype(str).isin(state_keys)].reset_index(drop=True)
        batch_factory = lambda: [selected]
    graph = _load_graph_topology(args.project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_map = build_surrogate_action_node_map(graph).astype(np.float32)
    graph_tensors = (
        torch.from_numpy(graph["edge_index"].astype(np.int64)).to(device),
        torch.from_numpy(graph["node_static"].astype(np.float32)).to(device),
        torch.from_numpy(action_map).to(device),
        _graph_indices(graph, "is_storage", device),
        _graph_indices(graph, "is_outfall", device),
    )
    priority = torch.as_tensor(
        get_pfv_core_node_indices(list(graph["node_ids"])),
        dtype=torch.long,
        device=device,
    )
    predictions = []
    model_root = args.model_root or (
        args.project_root
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/step2/models"
    )
    for seed in args.seeds:
        model_path = model_root / f"seed_{seed}/best_model.pt"
        model = MultiReferenceHydraulicSurrogate(
            n_nodes=int(graph["n_nodes"]),
            n_facilities=int(graph["n_facilities"]),
            state_feature_dim=1,
            static_feature_dim=int(graph["node_static"].shape[1]),
            hidden_dim=64,
            gat_heads=4,
            gat_layers=3,
            horizon=12,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        predicted_pfv = []
        predicted_tfv = []
        with torch.inference_mode():
            for frame_batch in batch_factory():
                tensor_data = _tensorise(frame_batch)
                output = _forward(model, tensor_data, graph_tensors, priority, device)
                predicted_pfv.append(output["pfv_delta"].detach().cpu().numpy())
                predicted_tfv.append(output["tfv_delta"].detach().cpu().numpy())
        del output, tensor_data
        if device.type == "cuda":
            torch.cuda.empty_cache()
        predictions.append(
            {
                "seed": seed,
                "model_sha256": _hash_model(model),
                "pfv_delta": np.concatenate(predicted_pfv),
                "tfv_delta": np.concatenate(predicted_tfv),
            }
        )
        del model, predicted_pfv, predicted_tfv

    pfv = np.stack([item["pfv_delta"] for item in predictions])
    tfv = np.stack([item["tfv_delta"] for item in predictions])
    selected["pred_pfv_delta_mean"] = pfv.mean(axis=0)
    selected["pred_tfv_delta_mean"] = tfv.mean(axis=0)
    rows = []
    for state_key, group in selected.groupby("state_key", sort=True):
        idx = group.index.to_numpy()
        rows.append(
            {
                "state_key": str(state_key),
                "rows": int(len(group)),
                "label_pfv_std_m3": float(group["pfv_delta"].astype(float).std()),
                "label_tfv_std_m3": float(group["tfv_delta"].astype(float).std()),
                "predicted_pfv_range_m3": float(
                    selected.loc[idx, "pred_pfv_delta_mean"].max()
                    - selected.loc[idx, "pred_pfv_delta_mean"].min()
                ),
                "predicted_tfv_range_m3": float(
                    selected.loc[idx, "pred_tfv_delta_mean"].max()
                    - selected.loc[idx, "pred_tfv_delta_mean"].min()
                ),
            }
        )
    label_pfv_median = float(np.median([x["label_pfv_std_m3"] for x in rows]))
    predicted_pfv_median = float(
        np.median([x["predicted_pfv_range_m3"] for x in rows])
    )
    ratio = predicted_pfv_median / max(label_pfv_median, 1.0e-12)
    if ratio < 0.01:
        interpretation = (
            "candidate actions reach the model, but PFV action sensitivity remains "
            "collapsed when predicted PFV variation is orders of magnitude below "
            "within-state labels"
        )
    else:
        interpretation = (
            "candidate actions produce measurable PFV variation; safety classification "
            "and within-state ranking require separate authoritative audit"
        )
    result = {
        "audit_id": "V42_STEP2_ACTION_SENSITIVITY_V1",
        "read_only": True,
        "new_swmm_started": False,
        "manifest": str(args.manifest),
        "model_seeds": [item["seed"] for item in predictions],
        "states_requested": int(args.states),
        "batch_size": int(args.batch_size),
        "action_map_source": "build_surrogate_action_node_map",
        "action_map_nonzero": int(np.count_nonzero(action_map)),
        "states_audited": len(rows),
        "rows_audited": int(len(selected)),
        "state_rows": rows,
        "median_label_pfv_std_m3": label_pfv_median,
        "median_label_tfv_std_m3": float(np.median([x["label_tfv_std_m3"] for x in rows])),
        "median_predicted_pfv_range_m3": predicted_pfv_median,
        "predicted_to_label_pfv_sensitivity_ratio": ratio,
        "median_predicted_tfv_range_m3": float(np.median([x["predicted_tfv_range_m3"] for x in rows])),
        "model_hashes": {str(item["seed"]): item["model_sha256"] for item in predictions},
        "interpretation": interpretation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
