"""Calibrate the Formal PFV-only one-sided UCB.

The sole hydraulic admission quantity is the complete PFV budget metric

    g = PFV_candidate - 1.05 * PFV_no_control,

and Formal admission requires ``UCB(g) <= 100 m3``.  Calibrating ``g`` directly
is important: the 5% No-control reference term is itself model-predicted and
must not be treated as deterministic.

Priority-node depth and global Peak remain diagnostic channels only. Three
Step2 seeds provide epistemic uncertainty. The current-generation Calibration
holdout is used only for uncertainty/safety calibration, not model-weight
training.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_v42_step2_fast import _forward, _slice, _tensorise
from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID, read_table, sha256_file
from sewerrtc.v4.models_v42.hydraulic_multi_reference import MultiReferenceHydraulicSurrogate
from sewerrtc.v4.v42_node_safety import priority_depth_limits_m
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology

PFV_ABSOLUTE_ALLOWANCE_M3 = 100.0
PFV_RELATIVE_ALLOWANCE_FRACTION = 0.05
PRIORITY_DEPTH_MAX_FRACTION = 0.95
PRIORITY_DEPTH_MIN_FREEBOARD_M = 0.05
DT_SEC = 600.0


def _conformal_quantile(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("empty conformal residual array")
    level = min(
        1.0,
        math.ceil((values.size + 1) * (1.0 - alpha)) / values.size,
    )
    try:
        return float(np.quantile(values, level, method="higher"))
    except TypeError:
        return float(np.quantile(values, level, interpolation="higher"))


def _validate_calibration_lineage(
    frame: pd.DataFrame,
    ledger: pd.DataFrame,
    model_reports: list[dict],
) -> tuple[list[str], set[str]]:
    groups = set(frame["split_group_key"].astype(str))
    allowed = set(
        ledger.loc[
            ledger["formal_f2_role"].astype(str).eq("calibration"),
            "rainfall_group_key",
        ].astype(str)
    )
    if not groups or not groups.issubset(allowed):
        raise RuntimeError(
            "Step2 calibration contains rainfall outside the current Calibration holdout: "
            f"{sorted(groups - allowed)[:10]}"
        )
    development: set[str] = set()
    for report in model_reports:
        development.update(map(str, report.get("train_rainfall_groups", [])))
        development.update(map(str, report.get("validation_rainfall_groups", [])))
        development.update(map(str, report.get("calibration_rainfall_groups", [])))
    overlap = groups & development
    if overlap:
        raise RuntimeError(
            f"current Calibration overlaps Step2 model-development groups: {sorted(overlap)[:10]}"
        )
    return sorted(groups), development


def _json_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=np.float64)


def _actual_no_control_pfv(
    frame: pd.DataFrame, priority_idx: list[int]
) -> np.ndarray:
    values: list[float] = []
    for raw in frame["trajectory_flood_no_control"]:
        flood = _json_array(raw)
        values.append(float(flood[:, priority_idx].sum() * DT_SEC))
    return np.asarray(values, dtype=float)


def _admission_risk(
    frame: pd.DataFrame,
    predicted_safe: np.ndarray,
    actual_safe: np.ndarray,
) -> dict[str, float | int | None]:
    predicted_safe = np.asarray(predicted_safe, dtype=bool)
    actual_safe = np.asarray(actual_safe, dtype=bool)
    false_safe = predicted_safe & ~actual_safe
    admitted = int(predicted_safe.sum())
    false_count = int(false_safe.sum())
    conditional = float(false_count / admitted) if admitted else None
    marginal = float(false_safe.mean())

    group_rates: list[float] = []
    if "split_group_key" in frame.columns:
        groups = frame["split_group_key"].astype(str).to_numpy()
        for group in sorted(set(groups)):
            mask = groups == group
            group_admitted = int(predicted_safe[mask].sum())
            if group_admitted:
                group_rates.append(
                    float(false_safe[mask].sum() / group_admitted)
                )
    return {
        "predicted_safe_count": admitted,
        "false_safe_count": false_count,
        "false_safe_rate_marginal": marginal,
        "false_safe_rate_among_admitted": conditional,
        "event_balanced_false_safe_rate_among_admitted": (
            float(np.mean(group_rates)) if group_rates else None
        ),
        "admitted_rainfall_group_count": len(group_rates),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument(
        "--calibration-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/calibration/FORMAL_F2_CALIBRATION_GAT_MANIFEST.parquet",
    )
    ap.add_argument(
        "--ledger",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare/FORMAL_F2_EVENT_LEDGER.csv",
    )
    ap.add_argument(
        "--models-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/step2/models",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/calibration/PFV_ONLY_SAFETY_CALIBRATION.json",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-calibration-groups", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--development-only", action="store_true")
    args = ap.parse_args()

    if not (0.0 < args.alpha < 0.5):
        raise ValueError("alpha must be in (0, 0.5)")
    frame = read_table(args.calibration_manifest)
    ledger = read_table(args.ledger)
    if frame.empty:
        raise ValueError("Formal F2 Step2 calibration manifest is empty")

    contracts = {
        "training_admission_authorized": True,
        "raw_independent_oracle_all_pass": True,
        "actual_readback_verified": True,
        "current_frame_repetition_used": False,
        "authoritative_swmm_history_used_as_online_input": False,
        "realized_future_rainfall_used_online": False,
    }
    for key, expected in contracts.items():
        if key not in frame:
            raise KeyError(f"calibration manifest missing {key}")
        observed = frame[key].astype(bool)
        if expected and not bool(observed.all()):
            raise RuntimeError(f"calibration contract failed: {key}")
        if not expected and bool(observed.any()):
            raise RuntimeError(f"calibration leakage contract failed: {key}")
    if not bool(
        frame["state_source"].astype(str).eq("gat_sparse_reconstruction").all()
    ):
        raise RuntimeError("calibration states are not causal sparse-GAT reconstructions")

    reports: list[dict] = []
    for seed in args.seeds:
        path = args.models_root / f"seed_{seed}" / "formal_step2_report.json"
        if not path.exists():
            raise FileNotFoundError(path)
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    calibration_groups, development_groups = _validate_calibration_lineage(
        frame, ledger, reports
    )
    if len(calibration_groups) < args.min_calibration_groups:
        raise RuntimeError(
            f"only {len(calibration_groups)} current Calibration groups; require {args.min_calibration_groups}"
        )
    target_contracts = {
        str(report.get("step2_target_contract", "")) for report in reports
    }
    if len(target_contracts) != 1 or "" in target_contracts:
        raise RuntimeError(
            f"Formal Step2 seeds have inconsistent target contracts: {target_contracts}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = _load_graph_topology(args.project_root)
    edge_index = torch.from_numpy(graph["edge_index"].astype(np.int64)).to(device)
    node_static = torch.from_numpy(graph["node_static"].astype(np.float32)).to(device)
    action_map = torch.from_numpy(graph["action_node_map"].astype(np.float32)).to(device)
    priority_idx = get_pfv_core_node_indices(list(graph["node_ids"]))
    priority = torch.as_tensor(priority_idx, dtype=torch.long, device=device)
    depth_limits = priority_depth_limits_m(
        args.project_root,
        priority_idx,
        max_depth_fraction=PRIORITY_DEPTH_MAX_FRACTION,
        minimum_freeboard_m=PRIORITY_DEPTH_MIN_FREEBOARD_M,
    )
    data = _tensorise(frame)

    pred_by_seed = {
        key: []
        for key in ("pfv_delta", "tfv_delta", "peak_delta", "no_control_pfv")
    }
    priority_depth_by_seed: list[np.ndarray] = []
    model_hashes: dict[str, str] = {}
    for seed, report in zip(args.seeds, reports):
        cfg = report.get("config", {})
        model = MultiReferenceHydraulicSurrogate(
            n_nodes=int(graph["n_nodes"]),
            n_facilities=int(graph["n_facilities"]),
            state_feature_dim=1,
            static_feature_dim=int(graph["node_static"].shape[1]),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            gat_heads=4,
            gat_layers=int(cfg.get("gat_layers", 3)),
            horizon=12,
        ).to(device)
        checkpoint = args.models_root / f"seed_{seed}" / "best_model.pt"
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()
        model_hashes[str(seed)] = str(report.get("surrogate_model_sha256", ""))
        outputs = {
            key: np.zeros(len(frame), dtype=float)
            for key in pred_by_seed
        }
        depth_out = np.zeros((len(frame), 12, len(priority_idx)), dtype=float)
        with torch.no_grad():
            for start in range(0, len(frame), args.batch_size):
                idx = np.arange(start, min(len(frame), start + args.batch_size))
                out = _forward(
                    model,
                    _slice(data, idx),
                    (edge_index, node_static, action_map),
                    priority,
                    device,
                )
                for key in ("pfv_delta", "tfv_delta", "peak_delta"):
                    outputs[key][idx] = out[key].detach().cpu().numpy()
                outputs["no_control_pfv"][idx] = (
                    out["kpi_no_control"]["pfv_m3"].detach().cpu().numpy()
                )
                depth_out[idx] = (
                    out["branches"]["candidate"]["node_depth"][:, :, priority]
                    .detach()
                    .cpu()
                    .numpy()
                )
        for key in pred_by_seed:
            pred_by_seed[key].append(outputs[key])
        priority_depth_by_seed.append(depth_out)

    ensemble = {
        key: np.stack(values, axis=0) for key, values in pred_by_seed.items()
    }
    budget_metric_ensemble = (
        ensemble["pfv_delta"]
        - PFV_RELATIVE_ALLOWANCE_FRACTION
        * np.maximum(ensemble["no_control_pfv"], 0.0)
    )
    depth_ensemble = np.stack(priority_depth_by_seed, axis=0)
    actual_delta = {
        key: frame[key].to_numpy(dtype=float)
        for key in ("pfv_delta", "tfv_delta", "peak_delta")
    }
    actual_nc_pfv = _actual_no_control_pfv(frame, priority_idx)
    actual_budget_metric = (
        actual_delta["pfv_delta"]
        - PFV_RELATIVE_ALLOWANCE_FRACTION * np.maximum(actual_nc_pfv, 0.0)
    )
    actual_priority_depth = np.stack(
        [
            _json_array(value)[:, priority_idx]
            for value in frame["trajectory_depth_candidate"]
        ],
        axis=0,
    )

    mean = {key: value.mean(axis=0) for key, value in ensemble.items()}
    std = {key: value.std(axis=0, ddof=1) for key, value in ensemble.items()}
    budget_mean = budget_metric_ensemble.mean(axis=0)
    budget_std = budget_metric_ensemble.std(axis=0, ddof=1)
    depth_mean = depth_ensemble.mean(axis=0)
    depth_std = depth_ensemble.std(axis=0, ddof=1)
    eps = 1.0e-6

    budget_std_resid = (
        actual_budget_metric - budget_mean
    ) / np.maximum(budget_std, eps)
    depth_std_resid = (
        actual_priority_depth - depth_mean
    ) / np.maximum(depth_std, eps)
    z_pfv = max(0.0, _conformal_quantile(budget_std_resid, args.alpha))
    z_depth = max(
        0.0, _conformal_quantile(depth_std_resid.reshape(-1), args.alpha)
    )
    confidence_z = float(z_pfv)

    budget_metric_ucb = budget_mean + confidence_z * budget_std
    predicted_pfv_safe = budget_metric_ucb <= PFV_ABSOLUTE_ALLOWANCE_M3
    actual_pfv_safe = actual_budget_metric <= PFV_ABSOLUTE_ALLOWANCE_M3
    admission_risk = _admission_risk(frame, predicted_pfv_safe, actual_pfv_safe)
    pfv_false_safe = float(admission_risk["false_safe_rate_marginal"])
    conditional_false_safe = admission_risk["false_safe_rate_among_admitted"]
    event_balanced_false_safe = admission_risk[
        "event_balanced_false_safe_rate_among_admitted"
    ]

    # Depth is strictly diagnostic. Use its own diagnostic conformal factor so
    # PFV calibration is not contaminated by a removed safety objective.
    depth_ucb = depth_mean + z_depth * depth_std
    predicted_depth_safe = np.all(
        depth_ucb <= depth_limits[None, None, :], axis=(1, 2)
    )
    actual_depth_safe = np.all(
        actual_priority_depth <= depth_limits[None, None, :], axis=(1, 2)
    )
    priority_depth_false_safe = float(
        np.mean(predicted_depth_safe & ~actual_depth_safe)
    )
    joint_predicted_safe = predicted_pfv_safe & predicted_depth_safe
    joint_actual_safe = actual_pfv_safe & actual_depth_safe
    joint_false_safe = float(np.mean(joint_predicted_safe & ~joint_actual_safe))

    peak_mae = float(
        np.mean(np.abs(mean["peak_delta"] - actual_delta["peak_delta"]))
    )
    tfv_mae = float(
        np.mean(np.abs(mean["tfv_delta"] - actual_delta["tfv_delta"]))
    )
    metric_scale = max(float(np.std(actual_budget_metric)), eps)
    uncertainty_score = np.abs(budget_std / metric_scale)
    uncertainty_limit = float(np.quantile(uncertainty_score, 0.99))

    # Marginal false-safe frequency alone is insufficient: a selector that
    # admits very few candidates could pass while nearly every admitted action
    # is unsafe. Require non-empty admission and <= alpha risk among admitted
    # candidates, both row-weighted and event-balanced.
    status = "pass"
    if int(admission_risk["predicted_safe_count"]) <= 0:
        status = "fail"
    if conditional_false_safe is None or float(conditional_false_safe) > args.alpha + 1.0e-12:
        status = "fail"
    if event_balanced_false_safe is None or float(event_balanced_false_safe) > args.alpha + 1.0e-12:
        status = "fail"

    payload = {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "stage": "formal_f2_pfv_only_safety_calibration",
        "status": status,
        "development_only": bool(args.development_only),
        "formal_mainline_authorized": False,
        "calibration_authority": (
            "revealed_diagnostic_calibration"
            if args.development_only
            else "current_generation_calibration_holdout"
        ),
        "calibration_manifest": str(args.calibration_manifest),
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
        "calibration_rainfall_groups": calibration_groups,
        "calibration_rainfall_group_count": len(calibration_groups),
        "training_or_internal_validation_overlap_count": len(
            set(calibration_groups) & development_groups
        ),
        "step2_target_contract": next(iter(target_contracts)),
        "model_hashes": model_hashes,
        "alpha": float(args.alpha),
        "confidence_z": confidence_z,
        "pfv_standardized_conformal_z": z_pfv,
        "pfv_budget_metric_standardized_conformal_z": z_pfv,
        "pfv_safety_statistic": "candidate_minus_1p05_no_control",
        "pfv_safety_inequality": "UCB(PFV_candidate-1.05*PFV_no_control)<=100m3",
        "pfv_budget_metric_std_scale": metric_scale,
        "priority_depth_standardized_conformal_z": z_depth,
        "pfv_absolute_allowance_m3": PFV_ABSOLUTE_ALLOWANCE_M3,
        "pfv_relative_allowance_fraction": PFV_RELATIVE_ALLOWANCE_FRACTION,
        "priority_depth_limit_contract": {
            "physical_metadata_authority": "raw_frozen_INP",
            "max_depth_fraction": PRIORITY_DEPTH_MAX_FRACTION,
            "minimum_freeboard_m": PRIORITY_DEPTH_MIN_FREEBOARD_M,
            "priority_node_limits_m": depth_limits.tolist(),
        },
        "pfv_false_safe_rate": pfv_false_safe,
        "pfv_false_safe_rate_marginal": pfv_false_safe,
        "pfv_false_safe_rate_among_admitted": conditional_false_safe,
        "pfv_event_balanced_false_safe_rate_among_admitted": event_balanced_false_safe,
        "pfv_predicted_safe_count": admission_risk["predicted_safe_count"],
        "pfv_false_safe_count": admission_risk["false_safe_count"],
        "pfv_admitted_rainfall_group_count": admission_risk[
            "admitted_rainfall_group_count"
        ],
        "priority_depth_false_safe_rate": priority_depth_false_safe,
        "joint_false_safe_rate_diagnostic_only": joint_false_safe,
        "peak_is_hard_safety_constraint": False,
        "peak_delta_ensemble_mae_m3s": peak_mae,
        "tfv_delta_ensemble_mae_m3": tfv_mae,
        "uncertainty_score_contract": "normalized_step2_ensemble_std_complete_pfv_budget_metric_diagnostic",
        "uncertainty_limit_99": uncertainty_limit,
        "safety_calibrated": status == "pass",
        "control_objective_contract": "PROJECT6_V42_PFV_ONLY_TFV_MIN_MPC_V2",
        "pfv_budget_applied": True,
        "priority_depth_hard_gate": False,
        "global_peak_hard_gate": False,
        "global_peak_objective_term": False,
        "uncertainty_role": "PFV_budget_UCB_only",
        "OOD_role": "diagnostic_only",
        "independent_OOD_gate": False,
        "independent_uncertainty_gate": False,
        "future_truth_used_for_online_decision": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)
    return 0 if status == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
