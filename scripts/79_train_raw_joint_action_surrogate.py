from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.models.raw_joint_action_surrogate import RawJointActionSurrogate
from sewerrtc.models.raw_joint_assets import build_raw_joint_assets
from sewerrtc.models.raw_joint_training import (
    aggregate_effect_targets,
    direction_accuracy,
    finite_metric_at_least,
    load_dynamics_warm_start,
    noninferiority_metrics,
    resolve_event_group_indices,
    robust_scale,
)


def _assets(cfg: dict, node_ids: list[str], action_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = pd.read_csv(cfg_path(cfg, "outputs.audit") / "node_table.csv").set_index("node_id")
    links = pd.read_csv(cfg_path(cfg, "outputs.audit") / "link_table.csv").set_index("link_id")
    actuators = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv").set_index("actuator_id").loc[action_ids]
    static_cols = ["invert", "max_depth", "ponded_area", "degree_in", "degree_out", "is_storage", "is_outfall"]
    node_static = np.asarray([nodes.loc[n, static_cols].to_numpy(float) if n in nodes.index else np.zeros(len(static_cols)) for n in node_ids], dtype=np.float32)
    node_static = (node_static - node_static.mean(0, keepdims=True)) / np.maximum(node_static.std(0, keepdims=True), 1e-6)
    idx = {node: i for i, node in enumerate(node_ids)}
    edges = []
    for row in links.itertuples():
        if str(row.from_node) in idx and str(row.to_node) in idx:
            edges.extend([(idx[str(row.from_node)], idx[str(row.to_node)]), (idx[str(row.to_node)], idx[str(row.from_node)])])
    edge_index = np.asarray(edges or [(i, i) for i in range(len(node_ids))], dtype=np.int64).T
    amap = np.zeros((len(action_ids), len(node_ids)), dtype=np.float32)
    for j, aid in enumerate(action_ids):
        row = links.loc[aid] if aid in links.index else None
        if row is not None:
            for node, weight in ((str(row.from_node), 1.0), (str(row.to_node), 0.6)):
                if node in idx: amap[j, idx[node]] = weight
        if amap[j].sum() <= 0: amap[j] = 1.0 / len(node_ids)
        else: amap[j] /= amap[j].sum()
    link_type = actuators["link_type"].astype(str).str.lower()
    features = np.stack([
        link_type.eq("pump"), link_type.eq("orifice"), link_type.eq("weir"),
        actuators["near_storage"].astype(bool), actuators["storage_control_type"].astype(str).eq("storage_inlet"),
        actuators["storage_control_type"].astype(str).eq("storage_outlet"), actuators["is_existing_rtc"].astype(bool),
        actuators["is_physically_controllable"].astype(bool),
    ], axis=1).astype(np.float32)
    priority = [x.strip() for x in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text().splitlines() if x.strip()]
    storage = nodes[nodes["node_type"].astype(str).eq("storage")].index.astype(str).tolist()
    return node_static, edge_index, amap, features, np.asarray([idx[x] for x in priority if x in idx]), np.asarray([idx[x] for x in storage if x in idx])


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    return selected.mean() if selected.numel() else values.sum() * 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan_project6.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--hidden-dim", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="outputs/models_raw_joint_dev")
    ap.add_argument("--require-same-state", action="store_true")
    ap.add_argument("--model-name", default="raw_joint_action_surrogate.pt")
    ap.add_argument("--dynamics-warm-start", default="")
    args = ap.parse_args()
    cfg = load_config(args.config); root = cfg_path(cfg, "project_root")
    temporal_cfg = (((cfg.get("controller", {}) or {}).get("temporal_joint", {}) or {}))
    validation_cfg = temporal_cfg.get("training_validation", {}) or {}
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    data = np.load(Path(args.dataset), allow_pickle=True)
    label_semantics = str(data["label_semantics"].item()) if "label_semantics" in data else "paired_trajectory_difference"
    if args.require_same_state and label_semantics != "same_state_candidate_minus_no_control":
        raise ValueError(f"Formal raw-joint training requires same-state labels, got {label_semantics}")
    event_ids = data["event_ids"].astype(str)
    embedded_split = data["split"].astype(str) if "split" in data else None
    train_idx, val_idx, val_events = resolve_event_group_indices(event_ids, embedded_split)
    action_ids = data["action_ids"].astype(str).tolist() if "action_ids" in data else pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")["actuator_id"].astype(str).tolist()
    node_ids = data["node_ids"].astype(str).tolist() if "node_ids" in data else pd.read_csv(cfg_path(cfg, "outputs.audit") / "node_table.csv")["node_id"].astype(str).tolist()
    node_static, edge_index, amap, afeat, priority_idx, storage_idx = build_raw_joint_assets(cfg, node_ids, action_ids)
    tensors = {key: torch.as_tensor(data[key], dtype=torch.float32) for key in ["state", "candidate_action_seq", "reference_action_seq", "rain_seq", "reference_risk_rate_seq", "delta_risk_rate_seq", "priority_depth_seq", "storage_level_seq", "target_state_seq"]}
    mask = torch.ones(len(action_ids), dtype=torch.float32)
    architecture_version = "priority_aware_v2"
    model = RawJointActionSurrogate(n_nodes=len(node_ids), n_actions=len(action_ids), node_static_dim=node_static.shape[1], actuator_feature_dim=afeat.shape[1], horizon_steps=tensors["rain_seq"].shape[1], hidden_dim=int(args.hidden_dim), heads=4, architecture_version=architecture_version).to(device)
    dynamics_warm_start_path = None
    dynamics_warm_start_parameters: list[str] = []
    if args.dynamics_warm_start:
        dynamics_warm_start_path = root / args.dynamics_warm_start if not Path(args.dynamics_warm_start).is_absolute() else Path(args.dynamics_warm_start)
        if not dynamics_warm_start_path.exists():
            raise FileNotFoundError(f"Missing dynamics warm-start checkpoint: {dynamics_warm_start_path}")
        dynamics_checkpoint = torch.load(dynamics_warm_start_path, map_location="cpu", weights_only=False)
        dynamics_warm_start_parameters = load_dynamics_warm_start(
            model,
            dynamics_checkpoint,
            node_ids=node_ids,
            action_ids=action_ids,
        )
    fixed = {"node_static": torch.as_tensor(node_static, device=device), "edge_index": torch.as_tensor(edge_index, device=device), "action_node_map": torch.as_tensor(amap, device=device), "actuator_features": torch.as_tensor(afeat, device=device), "priority_indices": torch.as_tensor(priority_idx, dtype=torch.long, device=device), "storage_indices": torch.as_tensor(storage_idx, dtype=torch.long, device=device)}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_rows = torch.as_tensor(train_idx, dtype=torch.long)
    delta_rate_scale = robust_scale(
        tensors["delta_risk_rate_seq"].index_select(0, train_rows),
        dimensions=(0, 1),
        minimum=0.01,
    )
    reference_rate_scale = robust_scale(
        tensors["reference_risk_rate_seq"].index_select(0, train_rows),
        dimensions=(0, 1),
        minimum=1.0,
    )
    train_target_aggregate = aggregate_effect_targets(
        tensors["reference_risk_rate_seq"].index_select(0, train_rows),
        tensors["delta_risk_rate_seq"].index_select(0, train_rows),
    )
    aggregate_scale = robust_scale(train_target_aggregate, dimensions=(0,), minimum=0.1)
    train_state_delta = (
        tensors["target_state_seq"].index_select(0, train_rows)
        - tensors["state"].index_select(0, train_rows)[:, None, :]
    )
    state_delta_scale = robust_scale(train_state_delta, minimum=0.05)
    tolerances = torch.tensor([
        float(validation_cfg.get("pfv_direction_tolerance_m3", 1.0)),
        float(validation_cfg.get("tfv_direction_tolerance_m3", 100.0)),
        float(validation_cfg.get("peak_direction_tolerance", 0.1)),
    ])
    rng = np.random.default_rng(20260712); history=[]
    for epoch in range(1, int(args.epochs)+1):
        model.train(); losses=[]; component_sums: dict[str, list[float]] = {}
        train_action_delta = tensors["candidate_action_seq"][train_idx] - tensors["reference_action_seq"][train_idx]
        nonzero_train = train_idx[train_action_delta.abs().amax(dim=(1, 2)).numpy() > 1.0e-8]
        zero_train = train_idx[train_action_delta.abs().amax(dim=(1, 2)).numpy() <= 1.0e-8]
        zero_limit = min(len(zero_train), max(1, len(nonzero_train) // 3))
        sampled_zero = rng.choice(zero_train, size=zero_limit, replace=False) if zero_limit else np.asarray([], dtype=int)
        epoch_indices = rng.permutation(np.concatenate([nonzero_train, sampled_zero]))
        for start in range(0, len(epoch_indices), int(args.batch_size)):
            batch_idx = epoch_indices[start:start+int(args.batch_size)]
            batch = {key: value[batch_idx].to(device) for key, value in tensors.items()}
            out = model(state=batch["state"], candidate_action_seq=batch["candidate_action_seq"], reference_action_seq=batch["reference_action_seq"], rain_seq=batch["rain_seq"], actuator_mask=mask[None].expand(len(batch_idx), -1).to(device), **fixed)
            active_rows = (batch["candidate_action_seq"] - batch["reference_action_seq"]).abs().amax(dim=(1, 2)) > 1.0e-8
            rate_scale = delta_rate_scale.to(device)[None, None, :]
            ref_scale = reference_rate_scale.to(device)[None, None, :]
            normalized_delta_error = torch.nn.functional.smooth_l1_loss(
                out["delta_risk_rate_seq"] / rate_scale,
                batch["delta_risk_rate_seq"] / rate_scale,
                reduction="none",
            ).mean(dim=(1, 2))
            l_sequence = _masked_mean(normalized_delta_error, active_rows)
            l_reference = torch.nn.functional.smooth_l1_loss(
                out["reference_risk_rate_seq"] / ref_scale,
                batch["reference_risk_rate_seq"] / ref_scale,
            )
            predicted_aggregate = torch.stack(
                [out["delta_PFV_H"], out["delta_TFV_H"], out["delta_peak"]], dim=1
            )
            target_aggregate = aggregate_effect_targets(
                batch["reference_risk_rate_seq"], batch["delta_risk_rate_seq"]
            )
            normalized_aggregate_error = torch.nn.functional.smooth_l1_loss(
                predicted_aggregate / aggregate_scale.to(device)[None, :],
                target_aggregate / aggregate_scale.to(device)[None, :],
                reduction="none",
            ).mean(dim=1)
            l_aggregate = _masked_mean(normalized_aggregate_error, active_rows)
            direction_terms = []
            for channel in range(3):
                valid = active_rows & (target_aggregate[:, channel].abs() > tolerances[channel].to(device))
                if bool(valid.any()):
                    direction_terms.append(torch.nn.functional.binary_cross_entropy_with_logits(
                        -predicted_aggregate[valid, channel] / aggregate_scale[channel].to(device),
                        (target_aggregate[valid, channel] < 0).float(),
                    ))
            l_direction = torch.stack(direction_terms).mean() if direction_terms else predicted_aggregate.sum() * 0.0
            predicted_state_delta = out["node_state_seq"] - batch["state"][:, None, :]
            target_state_delta = batch["target_state_seq"] - batch["state"][:, None, :]
            l_state = torch.nn.functional.smooth_l1_loss(
                predicted_state_delta / state_delta_scale.to(device),
                target_state_delta / state_delta_scale.to(device),
            )
            normalized_sigma = torch.clamp(
                out["delta_risk_sigma_seq"] / rate_scale,
                min=0.05,
                max=20.0,
            )
            normalized_residual = (out["delta_risk_rate_seq"] - batch["delta_risk_rate_seq"]) / rate_scale
            uncertainty_rows = (
                0.5 * (normalized_residual.detach() / normalized_sigma).square()
                + torch.log(normalized_sigma)
            ).mean(dim=(1, 2))
            l_uncertainty = _masked_mean(uncertainty_rows, active_rows)
            zero_rows = ~active_rows
            l_zero = _masked_mean(out["delta_risk_rate_seq"].abs().mean(dim=(1, 2)), zero_rows)
            # The controller consumes horizon effects, so aggregate magnitude
            # and direction receive priority over auxiliary state decoding.
            loss = 0.25*l_sequence + l_aggregate + 0.10*l_reference + 0.50*l_direction + 0.01*l_uncertainty + 0.02*l_state + l_zero
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.detach().cpu()))
            for name, value in {
                "sequence": l_sequence, "aggregate": l_aggregate, "reference": l_reference,
                "direction": l_direction, "uncertainty": l_uncertainty, "state": l_state,
                "zero": l_zero,
            }.items():
                component_sums.setdefault(name, []).append(float(value.detach().cpu()))
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"loss_{name}": float(np.mean(values)) for name, values in component_sums.items()},
        })
    model.eval()
    with torch.no_grad():
        batch = {key: value[val_idx].to(device) for key, value in tensors.items()}
        out = model(state=batch["state"], candidate_action_seq=batch["candidate_action_seq"], reference_action_seq=batch["reference_action_seq"], rain_seq=batch["rain_seq"], actuator_mask=mask[None].expand(len(val_idx), -1).to(device), **fixed)
        zero = model(state=batch["state"], candidate_action_seq=batch["reference_action_seq"], reference_action_seq=batch["reference_action_seq"], rain_seq=batch["rain_seq"], actuator_mask=mask[None].expand(len(val_idx), -1).to(device), **fixed)
        pred_delta = torch.stack([out["delta_PFV_H"], out["delta_TFV_H"], out["delta_peak"]], dim=1)
        target_delta = aggregate_effect_targets(batch["reference_risk_rate_seq"], batch["delta_risk_rate_seq"])
        direction = {
            "PFV": direction_accuracy(pred_delta[:, 0], target_delta[:, 0], tolerance=float(tolerances[0])),
            "TFV": direction_accuracy(pred_delta[:, 1], target_delta[:, 1], tolerance=float(tolerances[1])),
            "peak": direction_accuracy(pred_delta[:, 2], target_delta[:, 2], tolerance=float(tolerances[2])),
        }
        reference_pfv_h = batch["reference_risk_rate_seq"][:, :, 0].sum(dim=1) * 300.0
        pfv_noninferiority = noninferiority_metrics(
            pred_delta[:, 0], target_delta[:, 0], reference_pfv_h,
            absolute_margin=float(validation_cfg.get("pfv_abs_margin_m3", 100.0)),
            relative_margin=float(validation_cfg.get("pfv_rel_margin", 0.005)),
        )
        metrics = {
            "zero_action_relative_error": float(zero["delta_risk_rate_seq"].abs().sum().cpu() / torch.clamp(batch["reference_risk_rate_seq"].abs().sum(), min=1e-6).cpu()),
            "PFV_direction_accuracy": direction["PFV"]["accuracy"],
            "PFV_direction_samples": direction["PFV"]["count"],
            "TFV_direction_accuracy": direction["TFV"]["accuracy"],
            "TFV_direction_samples": direction["TFV"]["count"],
            "peak_direction_accuracy": direction["peak"]["accuracy"],
            "peak_direction_samples": direction["peak"]["count"],
            "PFV_noninferiority": pfv_noninferiority,
            "PFV_effect_MAE_m3": float((pred_delta[:, 0] - target_delta[:, 0]).abs().mean().cpu()),
            "TFV_effect_MAE_m3": float((pred_delta[:, 1] - target_delta[:, 1]).abs().mean().cpu()),
            "peak_effect_MAE": float((pred_delta[:, 2] - target_delta[:, 2]).abs().mean().cpu()),
            "risk_sequence_mae": float(torch.abs(out["delta_risk_rate_seq"]-batch["delta_risk_rate_seq"]).mean().cpu()),
            "state_mae": float(torch.abs(out["node_state_seq"]-batch["target_state_seq"]).mean().cpu()),
        }
    min_samples = int(validation_cfg.get("min_direction_samples", 12))
    gate_checks = {
        "zero_action": metrics["zero_action_relative_error"] < float(validation_cfg.get("max_zero_action_relative_error", 0.005)),
        "PFV_direction_sample_count": metrics["PFV_direction_samples"] >= min_samples,
        "TFV_direction_sample_count": metrics["TFV_direction_samples"] >= min_samples,
        "peak_direction_sample_count": metrics["peak_direction_samples"] >= min_samples,
        "PFV_direction": finite_metric_at_least(metrics["PFV_direction_accuracy"], float(validation_cfg.get("min_pfv_direction_accuracy", 0.85))),
        "TFV_direction": finite_metric_at_least(metrics["TFV_direction_accuracy"], float(validation_cfg.get("min_tfv_direction_accuracy", 0.80))),
        "peak_direction": finite_metric_at_least(metrics["peak_direction_accuracy"], float(validation_cfg.get("min_peak_direction_accuracy", 0.85))),
        "PFV_noninferiority": metrics["PFV_noninferiority"]["classification_accuracy"] >= float(validation_cfg.get("min_pfv_noninferiority_accuracy", 0.90)),
    }
    gate = bool(all(gate_checks.values()))
    out_dir=ensure_dir(root/args.out_dir); model_path=out_dir/args.model_name
    checkpoint={"model":model.state_dict(),"node_ids":node_ids,"action_ids":action_ids,"node_static":node_static,"edge_index":edge_index,"action_node_map":amap,"actuator_features":afeat,"priority_indices":priority_idx,"storage_indices":storage_idx,"hidden_dim":int(args.hidden_dim),"heads":4,"horizon_steps":int(tensors['rain_seq'].shape[1]),"architecture_version":architecture_version,"training_mask_scope":"canonical_36","label_semantics":label_semantics,"training_scales":{"delta_rate":delta_rate_scale.numpy(),"reference_rate":reference_rate_scale.numpy(),"aggregate":aggregate_scale.numpy(),"state_delta":float(state_delta_scale)},"dynamics_warm_start":str(dynamics_warm_start_path) if dynamics_warm_start_path else None,"dynamics_warm_start_parameters":dynamics_warm_start_parameters,"provenance":{"dataset":str(Path(args.dataset).resolve()),"train_events":sorted(set(event_ids[train_idx])),"val_events":sorted(val_events),"event_group_split":True,"validation_gate_passed":gate}}
    torch.save(checkpoint,model_path)
    report={"model":str(model_path),"dataset":str(args.dataset),"device":str(device),"architecture_version":architecture_version,"label_semantics":label_semantics,"dynamics_warm_start":str(dynamics_warm_start_path) if dynamics_warm_start_path else None,"dynamics_warm_start_parameter_count":len(dynamics_warm_start_parameters),"train_events":sorted(set(event_ids[train_idx])),"val_events":sorted(val_events),"event_group_split":True,"train_rows":len(train_idx),"val_rows":len(val_idx),"training_scales":{"delta_rate":delta_rate_scale.tolist(),"reference_rate":reference_rate_scale.tolist(),"aggregate":aggregate_scale.tolist(),"state_delta":float(state_delta_scale)},"metrics":metrics,"validation_gate_checks":gate_checks,"validation_gate_failures":[name for name, passed in gate_checks.items() if not passed],"validation_gate_passed":gate,"losses":history,"acceptance":"eligible_for_smoke" if gate and label_semantics == "same_state_candidate_minus_no_control" else "not_eligible_for_closed_loop"}
    report_path = out_dir / f"{Path(args.model_name).stem}_train_report.json"
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8"); print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == "__main__": main()
