"""Formal F2 trainer for the four-reference hydraulic surrogate.

Input must already have raw four-reference admission, actual Engineering36
readback, causal 13-frame sparse-GAT history and an explicit Step2 target
contract materialised by ``materialize_v42_step2_target_contract.py``.

Default Formal contract: CONTROL_CORE
    depth + node flooding + storage volume + managed-facility flow.

Optional extension: FULL_HYDRAULIC
    CONTROL_CORE + explicit outfall discharge.

PFV/TFV/Peak remain deterministic functions of the predicted node-flooding
trajectory; there is no independent KPI shortcut head.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_v42_step2_fast import (
    _all_branch_columns,
    _batch_indices,
    _evaluate,
    _forward,
    _graph_indices,
    _hash_model,
    _slice,
    _targets,
    _tensorise,
)
from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID, read_table
from sewerrtc.v4.models_v42.hydraulic_multi_reference import MultiReferenceHydraulicSurrogate
from sewerrtc.v4.models_v42.hydraulic_trajectory_losses import HydraulicLossWeights, HydraulicTrajectoryLoss
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import (
    SURROGATE_ACTION_MAP_CONTRACT,
    _load_graph_topology,
    build_surrogate_action_node_map,
)


def _rank(groups: list[str], seed: int) -> list[str]:
    return sorted(
        groups,
        key=lambda g: (hashlib.sha256(f"formal-f2:{seed}:{g}".encode()).hexdigest(), g),
    )


def _split(frame, seed: int, min_train: int):
    groups = sorted(frame.split_group_key.astype(str).unique())
    if len(groups) < min_train + 4:
        raise RuntimeError(f"formal Step2 needs >={min_train + 4} groups; got {len(groups)}")
    ranked = _rank(groups, seed)
    n = len(ranked)
    nv = max(2, round(0.1 * n))
    nc = max(2, round(0.1 * n))
    while n - nv - nc < min_train and (nv > 2 or nc > 2):
        if nv >= nc and nv > 2:
            nv -= 1
        elif nc > 2:
            nc -= 1
    if n - nv - nc < min_train:
        raise RuntimeError("formal Step2 cannot maintain minimum train groups")
    vg = ranked[:nv]
    cg = ranked[nv : nv + nc]
    tg = ranked[nv + nc :]
    return (
        frame[frame.split_group_key.astype(str).isin(tg)].copy(),
        frame[frame.split_group_key.astype(str).isin(vg)].copy(),
        frame[frame.split_group_key.astype(str).isin(cg)].copy(),
        tg,
        vg,
        cg,
    )


def _selection_key(validation: dict, metric: str) -> tuple[float, ...]:
    """Return a deterministic checkpoint-selection key.

    ``loss`` preserves the historical Formal default.  ``control`` is an
    explicit repair mode: validation PFV/TFV direction and error are selected
    before the aggregate hydraulic loss, without changing training targets or
    the runtime PFV gate.
    """
    if metric == "loss":
        return (float(validation["loss"]),)
    if metric == "control":
        return (
            -float(validation["pfv_delta_sign_accuracy"]),
            -float(validation["tfv_delta_sign_accuracy"]),
            float(validation["pfv_delta_mae"]),
            float(validation["tfv_delta_mae"]),
            float(validation["loss"]),
        )
    raise ValueError(f"unsupported Step2 selection metric: {metric}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/step2/FORMAL_F2_STEP2_CONTROL_CORE_MANIFEST.parquet",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--gat-layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument(
        "--selection-metric",
        choices=("loss", "control"),
        default="loss",
        help="checkpoint selection; control is an explicit PFV/TFV repair mode",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--min-train-groups", type=int, default=65)
    ap.add_argument(
        "--target-contract",
        choices=("CONTROL_CORE", "FULL_HYDRAULIC"),
        default="CONTROL_CORE",
    )
    args = ap.parse_args()

    frame = read_table(args.manifest)
    if frame.empty:
        raise ValueError("formal Step2 target manifest is empty")
    for column in (
        "training_admission_authorized",
        "raw_independent_oracle_all_pass",
        "actual_readback_verified",
    ):
        if column not in frame or not bool(frame[column].astype(bool).all()):
            raise RuntimeError(f"formal Step2 requires all {column}=True")
    for column, expected in {
        "state_source": "gat_sparse_reconstruction",
        "history_input_contract": "gat_compatible_causal_state",
        "reconstructor_contract": "formal_temporal_v42",
        "reconstructed_history_contract": "PROJECT6_V42_CAUSAL_RECONSTRUCTED_HISTORY_V1",
    }.items():
        if column not in frame or not bool(frame[column].astype(str).eq(expected).all()):
            raise RuntimeError(f"formal Step2 {column} contract mismatch")
    for column in (
        "current_frame_repetition_used",
        "authoritative_swmm_history_used_as_online_input",
        "realized_future_rainfall_used_online",
    ):
        if column not in frame or bool(frame[column].astype(bool).any()):
            raise RuntimeError(f"formal Step2 leakage contract violated: {column}")

    if "step2_target_contract" not in frame:
        raise RuntimeError("formal Step2 manifest has not been target-contract materialized")
    observed_contracts = set(frame["step2_target_contract"].astype(str))
    if observed_contracts != {args.target_contract}:
        raise RuntimeError(
            f"formal Step2 target-contract mismatch: observed={sorted(observed_contracts)} expected={args.target_contract}"
        )
    has_storage = _all_branch_columns(frame, "storage_volume")
    has_facility = _all_branch_columns(frame, "facility_flow")
    has_outfall = _all_branch_columns(frame, "outfall_flow")
    if not (has_storage and has_facility):
        raise RuntimeError("Formal CONTROL_CORE requires storage and managed-facility-flow supervision")
    if args.target_contract == "FULL_HYDRAULIC" and not has_outfall:
        raise RuntimeError("Formal FULL_HYDRAULIC requires explicit outfall-flow supervision")
    if "no_control_all_open_verified" not in frame or not bool(frame["no_control_all_open_verified"].astype(bool).all()):
        raise RuntimeError("Formal Step2 requires verified all-open No-control actions")

    train_f, val_f, cal_f, train_groups, val_groups, cal_groups = _split(
        frame, args.split_seed, args.min_train_groups
    )
    train = _tensorise(train_f)
    val = _tensorise(val_f)
    cal = _tensorise(cal_f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = _load_graph_topology(args.project_root)
    edge_index = torch.from_numpy(graph["edge_index"].astype(np.int64)).to(device)
    node_static = torch.from_numpy(graph["node_static"].astype(np.float32)).to(device)
    action_map = torch.from_numpy(
        build_surrogate_action_node_map(graph).astype(np.float32)
    ).to(device)
    priority = torch.as_tensor(
        get_pfv_core_node_indices(list(graph["node_ids"])), dtype=torch.long, device=device
    )
    storage_idx = _graph_indices(graph, "is_storage", device)
    outfall_idx = (
        _graph_indices(graph, "is_outfall", device)
        if has_outfall
        else torch.empty(0, dtype=torch.long, device=device)
    )
    graph_tensors = (edge_index, node_static, action_map, storage_idx, outfall_idx)

    model = MultiReferenceHydraulicSurrogate(
        n_nodes=int(graph["n_nodes"]),
        n_facilities=int(graph["n_facilities"]),
        state_feature_dim=1,
        static_feature_dim=int(graph["node_static"].shape[1]),
        hidden_dim=args.hidden_dim,
        gat_heads=4,
        gat_layers=args.gat_layers,
        horizon=12,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = HydraulicTrajectoryLoss(
        HydraulicLossWeights(
            depth=0.5,
            node_flooding=2.0,
            storage=0.35,
            facility_flow=0.35,
            outfall_flow=0.35 if args.target_contract == "FULL_HYDRAULIC" else 0.0,
            kpi_consistency=0.75,
        ),
        require_storage_targets=True,
        require_facility_flow_targets=True,
        require_outfall_flow_targets=args.target_contract == "FULL_HYDRAULIC",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_model.pt"
    history = []
    best_key: tuple[float, ...] | None = None
    best_epoch = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for idx in _batch_indices(len(train_f), args.batch_size, shuffle=True, seed=args.seed + epoch):
            batch = _slice(train, idx)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, batch, graph_tensors, priority, device)
            target = _targets(batch, device)
            losses = loss_fn(prediction, target)
            loss = loss_fn.total(losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += float(loss.detach().item()) * len(idx)
            seen += len(idx)
        validation = _evaluate(
            model, val, graph_tensors, priority, device, args.batch_size, loss_fn
        )
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "validation": validation,
        }
        selection_key = _selection_key(validation, args.selection_metric)
        row["selection_metric"] = args.selection_metric
        row["selection_key"] = list(selection_key)
        history.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
        if best_key is None or selection_key < best_key:
            best_key = selection_key
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    train_report = _evaluate(
        model, train, graph_tensors, priority, device, args.batch_size, loss_fn
    )
    val_report = _evaluate(
        model, val, graph_tensors, priority, device, args.batch_size, loss_fn
    )
    cal_report = _evaluate(
        model, cal, graph_tensors, priority, device, args.batch_size, loss_fn
    )
    report = {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "stage": "formal_f2_step2_single_seed",
        "status": "pass",
        "development_only": False,
        "formal_mainline_authorized": False,
        "formal_model": "MultiReferenceHydraulicSurrogate",
        "four_reference_shared_model": True,
        "trajectory_first_kpi_derivation": True,
        "training_admission_authorized": True,
        "raw_independent_oracle_all_pass": True,
        "action_authority": "actual_readback_setting",
        "history_input_contract": "gat_compatible_causal_state",
        "rainfall_group_isolated_split": True,
        "formal_target_domain_only": True,
        "step2_target_contract": args.target_contract,
        "control_core_target_coverage_complete": True,
        "full_hydraulic_target_coverage_complete": bool(
            args.target_contract == "FULL_HYDRAULIC" and has_outfall
        ),
        "outfall_supervised": bool(args.target_contract == "FULL_HYDRAULIC"),
        "storage_supervised": True,
        "facility_flow_supervised": True,
        "no_control_all_open_verified": True,
        "surrogate_action_map_contract": SURROGATE_ACTION_MAP_CONTRACT,
        "surrogate_action_map_nonzero": int(torch.count_nonzero(action_map).item()),
        "model_selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "train_cases": len(train_f),
        "validation_cases": len(val_f),
        "calibration_cases": len(cal_f),
        "train_rainfall_groups": train_groups,
        "validation_rainfall_groups": val_groups,
        "calibration_rainfall_groups": cal_groups,
        "train_rainfall_group_count": len(train_groups),
        "validation_rainfall_group_count": len(val_groups),
        "calibration_rainfall_group_count": len(cal_groups),
        "surrogate_model_sha256": _hash_model(model),
        "train": train_report,
        "validation": val_report,
        "calibration": cal_report,
        "history": history,
    }
    (args.output_dir / "formal_step2_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    (args.output_dir / "split_groups.json").write_text(
        json.dumps(
            {"train": train_groups, "validation": val_groups, "calibration": cal_groups},
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
