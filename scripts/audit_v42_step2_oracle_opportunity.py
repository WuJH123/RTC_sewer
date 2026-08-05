"""Read-only physical-oracle opportunity audit for accepted Step2 trajectories.

The oracle is computed from already-recorded authoritative candidate outcomes;
this script never starts SWMM and never changes the PFV budget.
"""
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

from scripts.train_v42_step2_formal_f2 import _add_causal_dynamic_internal_input
from scripts.train_v42_step2_fast import _forward, _graph_indices, _hash_model, _tensorise
from sewerrtc.v4.formal_f2 import read_table
from sewerrtc.v4.models_v42.hydraulic_multi_reference import MultiReferenceHydraulicSurrogate
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import (
    _load_graph_topology,
    build_surrogate_action_node_map,
)


def _arr(value: object) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=np.float64)


def _model_predictions(frame: pd.DataFrame, root: Path, model_root: Path, seed: int, batch_size: int) -> tuple[np.ndarray, np.ndarray, str]:
    graph = _load_graph_topology(root)
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
        get_pfv_core_node_indices(list(graph["node_ids"])), dtype=torch.long, device=device
    )
    tensor_data = _tensorise(frame)
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
    path = model_root / f"seed_{seed}" / "best_model.pt"
    if not path.exists():
        path = model_root / "best_model.pt"
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    pfv, tfv = [], []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            idx = torch.arange(start, min(start + batch_size, len(frame)), dtype=torch.long)
            batch = {key: value.index_select(0, idx) for key, value in tensor_data.items()}
            out = _forward(model, batch, graph_tensors, priority, device)
            pfv.append(out["pfv_delta"].detach().cpu().numpy())
            tfv.append(out["tfv_delta"].detach().cpu().numpy())
    digest = _hash_model(model)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(pfv), np.concatenate(tfv), digest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-root", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    frame = read_table(args.manifest).reset_index(drop=True)
    frame = _add_causal_dynamic_internal_input(frame, args.project_root)
    graph = _load_graph_topology(args.project_root)
    priority = get_pfv_core_node_indices(list(graph["node_ids"]))
    nc_pfv = np.asarray([
        float(_arr(value)[:, priority].sum() * 600.0)
        for value in frame["trajectory_flood_no_control"]
    ])
    actual_pfv = frame["pfv_delta"].astype(float).to_numpy()
    actual_tfv = frame["tfv_delta"].astype(float).to_numpy()
    budget = 100.0 + 0.05 * nc_pfv
    candidate = [_arr(value) for value in frame["action_candidate_readback"]]
    hold = [_arr(value) for value in frame["action_hold_previous_readback"]]
    nonhold = np.asarray([
        bool(np.max(np.abs(c[:3] - h[:3])) > 1.0e-6)
        for c, h in zip(candidate, hold)
    ])
    actual_safe = np.isfinite(actual_pfv) & (actual_pfv <= budget)

    frame["actual_no_control_pfv_m3"] = nc_pfv
    frame["actual_pfv_budget_m3"] = budget
    frame["actual_safe"] = actual_safe
    frame["actual_nonhold"] = nonhold

    predicted_pfv = predicted_tfv = None
    model_hash = None
    if args.model_root:
        predicted_pfv, predicted_tfv, model_hash = _model_predictions(
            frame, args.project_root, args.model_root, args.seed, args.batch_size
        )
        frame["predicted_pfv_delta_m3"] = predicted_pfv
        frame["predicted_tfv_delta_m3"] = predicted_tfv
        frame["predicted_safe"] = np.isfinite(predicted_pfv) & (predicted_pfv <= budget)

    rows = []
    selected_rows = []
    for state_key, group in frame.groupby("state_key", sort=True):
        indices = group.index.to_numpy()
        safe = actual_safe[indices]
        safe_nonhold = safe & nonhold[indices]
        safe_improving = safe & (actual_tfv[indices] < 0.0)
        oracle_idx = indices[safe]
        oracle_tfv = float(np.min(actual_tfv[oracle_idx])) if len(oracle_idx) else None
        selected_idx = None
        if predicted_pfv is not None:
            pred_safe = np.isfinite(predicted_pfv[indices]) & (predicted_pfv[indices] <= budget[indices])
            if pred_safe.any():
                local = np.flatnonzero(pred_safe)
                selected_idx = int(indices[local[np.argmin(predicted_tfv[indices[local]])]])
        selected_actual_tfv = float(actual_tfv[selected_idx]) if selected_idx is not None else None
        selected_actual_pfv = float(actual_pfv[selected_idx]) if selected_idx is not None else None
        regret = selected_actual_tfv - oracle_tfv if selected_actual_tfv is not None and oracle_tfv is not None else None
        if selected_idx is not None:
            selected_rows.append({
                "state_key": str(state_key),
                "selected_row": selected_idx,
                "selected_actual_safe": bool(actual_safe[selected_idx]),
                "selected_actual_nonhold": bool(nonhold[selected_idx]),
                "selected_actual_pfv_delta_m3": selected_actual_pfv,
                "selected_actual_tfv_delta_m3": selected_actual_tfv,
                "oracle_best_tfv_delta_m3": oracle_tfv,
                "selection_regret_m3": regret,
            })
        rows.append({
            "state_key": str(state_key),
            "rainfall_sha256": str(group.iloc[0]["rainfall_sha256"]),
            "event_id": str(group.iloc[0]["event_id"]),
            "candidate_count": int(len(group)),
            "actual_safe_count": int(safe.sum()),
            "actual_safe_fraction": float(safe.mean()),
            "actual_safe_nonhold_count": int(safe_nonhold.sum()),
            "actual_safe_tfv_improving_count": int(safe_improving.sum()),
            "oracle_best_tfv_delta_m3": oracle_tfv,
            "oracle_tfv_gain_m3": -oracle_tfv if oracle_tfv is not None else None,
            "predicted_safe_count": int(np.sum(predicted_pfv[indices] <= budget[indices])) if predicted_pfv is not None else None,
            "predicted_selection_available": selected_idx is not None,
            "selected_actual_safe": bool(actual_safe[selected_idx]) if selected_idx is not None else None,
            "selected_actual_tfv_delta_m3": selected_actual_tfv,
            "selection_regret_m3": regret,
        })

    positive_oracle_gains = [
        float(item["oracle_tfv_gain_m3"])
        for item in rows
        if item["oracle_tfv_gain_m3"] is not None and float(item["oracle_tfv_gain_m3"]) > 0.0
    ]
    regrets = [
        float(item["selection_regret_m3"])
        for item in selected_rows
        if item["selection_regret_m3"] is not None
    ]
    summary = {
        "audit_id": "V42_STEP2_ORACLE_OPPORTUNITY_V1",
        "read_only": True,
        "new_swmm_started": False,
        "action_map_source": "build_surrogate_action_node_map",
        "action_map_nonzero": int(np.count_nonzero(build_surrogate_action_node_map(graph))),
        "manifest": str(args.manifest),
        "rows": int(len(frame)),
        "states": len(rows),
        "rainfall_groups": int(frame["rainfall_sha256"].nunique()),
        "actual_safe_rows": int(actual_safe.sum()),
        "actual_safe_row_fraction": float(actual_safe.mean()),
        "states_with_actual_safe_candidate": int(sum(item["actual_safe_count"] > 0 for item in rows)),
        "states_with_actual_safe_nonhold_candidate": int(sum(item["actual_safe_nonhold_count"] > 0 for item in rows)),
        "states_with_actual_safe_tfv_improvement": int(sum(item["actual_safe_tfv_improving_count"] > 0 for item in rows)),
        "positive_oracle_gain_states": len(positive_oracle_gains),
        "mean_positive_oracle_gain_m3": float(np.mean(positive_oracle_gains)) if positive_oracle_gains else None,
        "median_positive_oracle_gain_m3": float(np.median(positive_oracle_gains)) if positive_oracle_gains else None,
        "model_seed": args.seed if predicted_pfv is not None else None,
        "model_sha256": model_hash,
        "predicted_selection_states": int(sum(item["predicted_selection_available"] for item in rows)),
        "predicted_selection_actual_safe_states": int(sum(item["selected_actual_safe"] is True for item in rows)),
        "predicted_selection_false_safe_rate": (
            float(sum(item["selected_actual_safe"] is not True for item in selected_rows) / len(selected_rows))
            if selected_rows else None
        ),
        "predicted_selection_mean_regret_m3": float(np.mean([item["selection_regret_m3"] for item in selected_rows if item["selection_regret_m3"] is not None])) if selected_rows else None,
        "predicted_selection_median_regret_m3": float(np.median(regrets)) if regrets else None,
        "predicted_selection_p90_regret_m3": float(np.percentile(regrets, 90)) if regrets else None,
        "interpretation": "actual oracle is from recorded authoritative candidate trajectories; model selection fields are diagnostic only",
        "state_rows": rows,
        "selected_rows": selected_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"state_rows", "selected_rows"}}, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
