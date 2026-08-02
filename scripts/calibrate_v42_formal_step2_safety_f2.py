"""Calibrate Formal F2 PFV/Peak hard-safety UCBs on new F2 Calibration rainfalls.

The calibration manifest must be generated only from ledger role=calibration and
must contain the same causal GAT/four-reference contracts as Step2 training.
Three independently trained surrogates form the epistemic ensemble. A single
one-sided standardized conformal multiplier z is frozen for PFV and Peak so the
formal candidate builder can use mean + z*std without ad-hoc caller flags.
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
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology


def _conformal_quantile(values: np.ndarray, alpha: float) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("empty conformal residual array")
    level = min(1.0, math.ceil((v.size + 1) * (1.0 - alpha)) / v.size)
    try:
        return float(np.quantile(v, level, method="higher"))
    except TypeError:
        return float(np.quantile(v, level, interpolation="higher"))


def _validate_calibration_lineage(frame: pd.DataFrame, ledger: pd.DataFrame, model_reports: list[dict]) -> tuple[list[str], set[str]]:
    groups = set(frame["split_group_key"].astype(str))
    allowed = set(ledger.loc[ledger["formal_f2_role"].astype(str).eq("calibration"), "rainfall_group_key"].astype(str))
    if not groups or not groups.issubset(allowed):
        raise RuntimeError(f"Step2 safety calibration contains non-F2-calibration rainfall groups: {sorted(groups - allowed)[:10]}")
    training = set()
    for report in model_reports:
        training.update(map(str, report.get("train_rainfall_groups", [])))
        training.update(map(str, report.get("validation_rainfall_groups", [])))
        training.update(map(str, report.get("calibration_rainfall_groups", [])))
    overlap = groups & training
    if overlap:
        raise RuntimeError(f"new F2 safety calibration overlaps surrogate development groups: {sorted(overlap)[:10]}")
    return sorted(groups), training


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument(
        "--calibration-manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/calibration/FORMAL_F2_CALIBRATION_GAT_MANIFEST.parquet",
    )
    ap.add_argument(
        "--ledger",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare/FORMAL_F2_EVENT_LEDGER.csv",
    )
    ap.add_argument(
        "--models-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/step2/models",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/calibration/STEP2_SAFETY_CALIBRATION.json",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-calibration-groups", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    if not (0.0 < args.alpha < 0.5):
        raise ValueError("alpha must be in (0, 0.5)")
    frame = read_table(args.calibration_manifest)
    ledger = read_table(args.ledger)
    if frame.empty:
        raise ValueError("Formal F2 Step2 calibration manifest is empty")
    required_contracts = {
        "training_admission_authorized": True,
        "raw_independent_oracle_all_pass": True,
        "actual_readback_verified": True,
        "current_frame_repetition_used": False,
        "authoritative_swmm_history_used_as_online_input": False,
        "realized_future_rainfall_used_online": False,
    }
    for key, expected in required_contracts.items():
        if key not in frame:
            raise KeyError(f"calibration manifest missing {key}")
        observed = frame[key].astype(bool)
        if expected and not bool(observed.all()):
            raise RuntimeError(f"calibration contract failed: {key}")
        if not expected and bool(observed.any()):
            raise RuntimeError(f"calibration leakage contract failed: {key}")
    if not bool(frame["state_source"].astype(str).eq("gat_sparse_reconstruction").all()):
        raise RuntimeError("calibration states are not causal sparse-GAT reconstructions")

    reports = []
    for seed in args.seeds:
        path = args.models_root / f"seed_{seed}" / "formal_step2_report.json"
        if not path.exists():
            raise FileNotFoundError(path)
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    calibration_groups, training_groups = _validate_calibration_lineage(frame, ledger, reports)
    if len(calibration_groups) < args.min_calibration_groups:
        raise RuntimeError(f"only {len(calibration_groups)} F2 calibration groups; require {args.min_calibration_groups}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = _load_graph_topology(args.project_root)
    edge_index = torch.from_numpy(graph["edge_index"].astype(np.int64)).to(device)
    node_static = torch.from_numpy(graph["node_static"].astype(np.float32)).to(device)
    action_map = torch.from_numpy(graph["action_node_map"].astype(np.float32)).to(device)
    priority = torch.as_tensor(get_pfv_core_node_indices(list(graph["node_ids"])), dtype=torch.long, device=device)
    data = _tensorise(frame)
    pred_by_seed = {k: [] for k in ("pfv_delta", "tfv_delta", "peak_delta")}
    model_hashes: dict[str, str] = {}

    for seed, report in zip(args.seeds, reports):
        model = MultiReferenceHydraulicSurrogate(
            n_nodes=int(graph["n_nodes"]), n_facilities=int(graph["n_facilities"]),
            state_feature_dim=1, static_feature_dim=int(graph["node_static"].shape[1]),
            hidden_dim=64, gat_heads=4, gat_layers=3, horizon=12,
        ).to(device)
        checkpoint = args.models_root / f"seed_{seed}" / "best_model.pt"
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        model_hashes[str(seed)] = str(report.get("surrogate_model_sha256", ""))
        outputs = {k: np.zeros(len(frame), dtype=float) for k in pred_by_seed}
        with torch.no_grad():
            for start in range(0, len(frame), args.batch_size):
                idx = np.arange(start, min(len(frame), start + args.batch_size))
                out = _forward(model, _slice(data, idx), (edge_index, node_static, action_map), priority, device)
                for key in outputs:
                    outputs[key][idx] = out[key].detach().cpu().numpy()
        for key in pred_by_seed:
            pred_by_seed[key].append(outputs[key])

    ensemble = {key: np.stack(values, axis=0) for key, values in pred_by_seed.items()}
    actual = {key: frame[key].to_numpy(dtype=float) for key in ensemble}
    mean = {key: value.mean(axis=0) for key, value in ensemble.items()}
    std = {key: value.std(axis=0, ddof=1) for key, value in ensemble.items()}
    eps = 1e-6
    pfv_std_resid = (actual["pfv_delta"] - mean["pfv_delta"]) / np.maximum(std["pfv_delta"], eps)
    peak_std_resid = (actual["peak_delta"] - mean["peak_delta"]) / np.maximum(std["peak_delta"], eps)
    z_pfv = max(0.0, _conformal_quantile(pfv_std_resid, args.alpha))
    z_peak = max(0.0, _conformal_quantile(peak_std_resid, args.alpha))
    confidence_z = float(max(z_pfv, z_peak))
    pfv_ucb = mean["pfv_delta"] + confidence_z * std["pfv_delta"]
    peak_ucb = mean["peak_delta"] + confidence_z * std["peak_delta"]
    pfv_false_safe = float(np.mean((pfv_ucb <= 0.0) & (actual["pfv_delta"] > 0.0)))
    peak_false_safe = float(np.mean((peak_ucb <= 0.0) & (actual["peak_delta"] > 0.0)))
    joint_false_safe = float(np.mean((pfv_ucb <= 0.0) & (peak_ucb <= 0.0) & ((actual["pfv_delta"] > 0.0) | (actual["peak_delta"] > 0.0))))
    uncertainty_score = np.sqrt(np.square(std["pfv_delta"] / (np.std(actual["pfv_delta"]) + eps)) + np.square(std["peak_delta"] / (np.std(actual["peak_delta"]) + eps)))
    uncertainty_limit = float(np.quantile(uncertainty_score, 0.99))
    status = "pass" if max(pfv_false_safe, peak_false_safe, joint_false_safe) <= args.alpha + 1e-12 else "fail"
    payload = {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "stage": "formal_f2_step2_safety_calibration",
        "status": status,
        "development_only": False,
        "formal_mainline_authorized": False,
        "calibration_authority": "new_f2_calibration_rainfall_groups",
        "calibration_manifest": str(args.calibration_manifest),
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
        "calibration_rainfall_groups": calibration_groups,
        "calibration_rainfall_group_count": len(calibration_groups),
        "training_or_internal_validation_overlap_count": len(set(calibration_groups) & training_groups),
        "model_hashes": model_hashes,
        "alpha": float(args.alpha),
        "confidence_z": confidence_z,
        "pfv_standardized_conformal_z": z_pfv,
        "peak_standardized_conformal_z": z_peak,
        "pfv_false_safe_rate": pfv_false_safe,
        "peak_false_safe_rate": peak_false_safe,
        "joint_false_safe_rate": joint_false_safe,
        "uncertainty_score_contract": "normalized_step2_ensemble_std_pfv_peak",
        "uncertainty_limit_99": uncertainty_limit,
        "safety_calibrated": status == "pass",
        "future_truth_used_for_online_decision": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)
    return 0 if status == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
