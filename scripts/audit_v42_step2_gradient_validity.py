"""Bounded no-SWMM autograd/finite-difference validity audit for Step2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_v42_step2_action_sensitivity import _iter_manifest_batches
from scripts.train_v42_step2_fast import _forward, _graph_indices, _tensorise
from sewerrtc.v4.models_v42.hydraulic_multi_reference import MultiReferenceHydraulicSurrogate
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology, build_surrogate_action_node_map


def _first_states(manifest: Path, state_count: int, batch_size: int):
    import pandas as pd

    wanted = set(
        sorted(pd.read_parquet(manifest, columns=["state_key"])["state_key"].astype(str).unique())
        [: max(1, state_count)]
    )
    rows: list[dict] = []
    for batch in _iter_manifest_batches(manifest, wanted, batch_size):
        for _, row in batch[batch["state_key"].isin(wanted)].iterrows():
            key = str(row["state_key"])
            if not any(str(item["state_key"]) == key for item in rows):
                rows.append(row.to_dict())
        if len(rows) >= len(wanted):
            break
    if not rows:
        raise RuntimeError("no selected states found")
    return pd.DataFrame(rows).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument("--finite-difference-samples", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.01)
    args = parser.parse_args()

    frame = _first_states(args.manifest, args.states, args.batch_size)
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
    priority = torch.as_tensor(get_pfv_core_node_indices(list(graph["node_ids"])), dtype=torch.long, device=device)
    cpu_data = _tensorise(frame)
    batch = {key: value.to(device) for key, value in cpu_data.items()}
    base_action = batch["action_candidate"].detach()
    results = []
    actuator_ids = [str(x) for x in graph.get("actuator_ids", [])]
    binary = {"ADD301.2", "ADD301.3"}
    candidate_indices = [i for i, aid in enumerate(actuator_ids) if aid not in binary] or list(range(base_action.shape[-1]))

    for seed in args.seeds:
        model_path = args.model_root / f"seed_{seed}/best_model.pt"
        model = MultiReferenceHydraulicSurrogate(
            n_nodes=int(graph["n_nodes"]), n_facilities=int(graph["n_facilities"]),
            state_feature_dim=1, static_feature_dim=int(graph["node_static"].shape[1]),
            hidden_dim=64, gat_heads=4, gat_layers=3, horizon=12,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        action = base_action.clone().requires_grad_(True)
        local_batch = dict(batch)
        local_batch["action_candidate"] = action
        prediction = _forward(model, local_batch, graph_tensors, priority, device)
        tfv = prediction["tfv_delta"]
        pfv = prediction["pfv_delta"]
        grad_tfv = torch.autograd.grad(tfv.sum(), action, retain_graph=True)[0]
        grad_pfv = torch.autograd.grad(pfv.sum(), action)[0]
        finite_grad = bool(torch.isfinite(grad_tfv).all() and torch.isfinite(grad_pfv).all())
        nonzero_tfv = int((grad_tfv[:, :3, candidate_indices].abs() > 1.0e-8).sum().item())
        nonzero_pfv = int((grad_pfv[:, :3, candidate_indices].abs() > 1.0e-8).sum().item())
        comparisons = []
        n = min(len(frame), max(1, int(args.finite_difference_samples)))
        for row_index in range(n):
            facility = candidate_indices[row_index % len(candidate_indices)]
            plus = base_action.clone()
            plus[row_index, :3, facility] = torch.clamp(plus[row_index, :3, facility] + float(args.epsilon), 0.0, 1.0)
            perturbed = dict(batch)
            perturbed["action_candidate"] = plus
            with torch.inference_mode():
                plus_prediction = _forward(model, perturbed, graph_tensors, priority, device)
            fd = float((plus_prediction["tfv_delta"][row_index] - tfv.detach()[row_index]).item()) / float(args.epsilon)
            analytic = float(grad_tfv[row_index, 0, facility].detach().item())
            if abs(fd) > 1.0e-6 and abs(analytic) > 1.0e-6:
                comparisons.append(bool(np.sign(fd) == np.sign(analytic)))
        results.append({
            "seed": int(seed), "model_path": str(model_path), "states": int(len(frame)),
            "finite_gradients": finite_grad, "nonzero_tfv_gradient_entries": nonzero_tfv,
            "nonzero_pfv_gradient_entries": nonzero_pfv,
            "finite_difference_comparisons": int(len(comparisons)),
            "finite_difference_sign_agreement": float(np.mean(comparisons)) if comparisons else None,
        })
        del model, prediction, grad_tfv, grad_pfv, action
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "audit_id": "V42_STEP2_GRADIENT_VALIDITY_V1", "read_only": True, "new_swmm_started": False,
        "manifest": str(args.manifest), "states_audited": int(len(frame)), "model_seeds": list(args.seeds),
        "action_map_source": "build_surrogate_action_node_map", "action_map_nonzero": int(np.count_nonzero(action_map)),
        "epsilon": float(args.epsilon), "results": results,
        "finite_gradient_all_seeds": bool(all(item["finite_gradients"] for item in results)),
        "finite_difference_sign_agreement_mean": float(np.mean([item["finite_difference_sign_agreement"] for item in results if item["finite_difference_sign_agreement"] is not None])) if any(item["finite_difference_sign_agreement"] is not None for item in results) else None,
        "interpretation": "PASS if all gradients are finite, nonzero action paths exist, and finite-difference signs are directionally consistent; this is a model-validity gate, not authoritative hydraulic validation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
