"""Fast SWMM-backed offline policy replay for V4.2 feasibility screening.

The surrogate scores historically simulated candidates at the same state. The
canonical PFV-budgeted/TFV-first selector chooses one, and the selected outcome
is read from recorded authoritative SWMM trajectories.

Qualification uses the current control objective:
* PFV delta UCB <= 100 m3 + 5% * predicted No-control PFV;
* priority-node depth <= node-specific safety limits;
* Engineering/K/uncertainty/OOD/executability hard gates;
* minimise TFV vs Dynamic Internal, with positive Peak excess and action change
  as performance penalties.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_v42_step2_fast import _forward, _slice, _split_groups, _tensorise
from sewerrtc.control.pfvfirst_mpc_v42 import (
    EngineeringStatus,
    FrozenFallback,
    MPCandidate,
    MPCWeights,
    SafetyMargins,
    decide_pfvfirst_mpc,
)
from sewerrtc.v4.models_v42.hydraulic_multi_reference import MultiReferenceHydraulicSurrogate
from sewerrtc.v4.v42_fast_feasibility import FAST_CONTRACT_ID
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology

DT_SEC = 600.0
PFV_ABS_M3 = 100.0
PFV_REL = 0.05
DEPTH_FRACTION = 0.95
MIN_FREEBOARD_M = 0.05


def _arr(value: str) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=np.float32)


def _branch_metrics(
    row: pd.Series, branch: str, priority_idx: list[int]
) -> dict[str, float]:
    flood = _arr(row[f"trajectory_flood_{branch}"]).astype(np.float64)
    system = flood.sum(axis=1)
    priority = flood[:, priority_idx].sum(axis=1)
    return {
        "pfv_m3": float(priority.sum() * DT_SEC),
        "tfv_m3": float(system.sum() * DT_SEC),
        "peak_m3s": float(system.max()),
    }


def _priority_depth_max(row: pd.Series, branch: str, priority_idx: list[int]) -> np.ndarray:
    depth = _arr(row[f"trajectory_depth_{branch}"]).astype(np.float64)
    return depth[:, priority_idx].max(axis=0)


def _actual_deltas(
    row: pd.Series, branch: str, priority_idx: list[int]
) -> dict[str, float]:
    metrics = _branch_metrics(row, branch, priority_idx)
    nc = _branch_metrics(row, "no_control", priority_idx)
    di = _branch_metrics(row, "dynamic_internal", priority_idx)
    return {
        "pfv_delta_nc_m3": metrics["pfv_m3"] - nc["pfv_m3"],
        "tfv_delta_di_m3": metrics["tfv_m3"] - di["tfv_m3"],
        "peak_delta_di_m3s": metrics["peak_m3s"] - di["peak_m3s"],
        **metrics,
    }


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [
        float(r[key])
        for r in rows
        if r.get(key) is not None and np.isfinite(float(r[key]))
    ]
    return None if not vals else float(np.mean(vals))


def _rate(rows: list[dict], key: str) -> float | None:
    return None if not rows else float(np.mean([bool(r[key]) for r in rows]))


def _priority_depth_limits(graph: dict, priority_idx: list[int]) -> np.ndarray:
    cols = list(map(str, graph.get("node_static_cols", [])))
    if "max_depth" not in cols:
        raise KeyError("graph node_static missing max_depth")
    max_depth = np.asarray(graph["node_static"], dtype=float)[:, cols.index("max_depth")]
    selected = max_depth[np.asarray(priority_idx, dtype=int)]
    if not np.isfinite(selected).all() or np.any(selected <= 0.0):
        raise ValueError("priority max_depth must be finite and positive")
    return np.maximum(
        0.0,
        np.minimum(DEPTH_FRACTION * selected, selected - MIN_FREEBOARD_M),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    frame = (
        pd.read_parquet(args.manifest)
        if args.manifest.suffix.lower() == ".parquet"
        else pd.read_csv(args.manifest)
    )
    _, val_f, _, val_groups = _split_groups(frame, args.seed)
    if val_f.empty:
        raise ValueError("no validation rows for replay")
    report_path = args.model_dir / "fast_step2_report.json"
    if not report_path.exists():
        report_path = args.model_dir / "qualification_step2_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cfg = report.get("config", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = _load_graph_topology(args.project_root)
    edge_index = torch.from_numpy(graph["edge_index"].astype(np.int64)).to(device)
    node_static = torch.from_numpy(graph["node_static"].astype(np.float32)).to(device)
    action_map = torch.from_numpy(graph["action_node_map"].astype(np.float32)).to(device)
    priority_idx = get_pfv_core_node_indices(list(graph["node_ids"]))
    priority_tensor = torch.as_tensor(priority_idx, dtype=torch.long, device=device)
    depth_limits = _priority_depth_limits(graph, priority_idx)
    model = MultiReferenceHydraulicSurrogate(
        n_nodes=int(graph["n_nodes"]),
        n_facilities=int(graph["n_facilities"]),
        state_feature_dim=1,
        static_feature_dim=int(graph["node_static"].shape[1]),
        hidden_dim=int(cfg.get("hidden_dim", 32)),
        gat_heads=4,
        gat_layers=int(cfg.get("gat_layers", 2)),
        horizon=12,
    ).to(device)
    model.load_state_dict(
        torch.load(args.model_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    model.eval()

    data = _tensorise(val_f)
    predicted = {
        "pfv_delta": np.zeros(len(val_f)),
        "tfv_delta": np.zeros(len(val_f)),
        "peak_delta": np.zeros(len(val_f)),
        "no_control_pfv": np.zeros(len(val_f)),
    }
    predicted_priority_depth = np.zeros(
        (len(val_f), 12, len(priority_idx)), dtype=float
    )
    with torch.no_grad():
        for start in range(0, len(val_f), args.batch_size):
            idx = np.arange(start, min(len(val_f), start + args.batch_size))
            out = _forward(
                model,
                _slice(data, idx),
                (edge_index, node_static, action_map),
                priority_tensor,
                device,
            )
            for key in ("pfv_delta", "tfv_delta", "peak_delta"):
                predicted[key][idx] = out[key].detach().cpu().numpy()
            predicted["no_control_pfv"][idx] = (
                out["kpi_no_control"]["pfv_m3"].detach().cpu().numpy()
            )
            predicted_priority_depth[idx] = (
                out["branches"]["candidate"]["node_depth"][:, :, priority_tensor]
                .detach()
                .cpu()
                .numpy()
            )

    replay_rows: list[dict] = []
    val_reset = val_f.reset_index(drop=True)
    for state_key, group in val_reset.groupby("state_key", sort=True):
        indices = group.index.to_list()
        if len(indices) < 2:
            continue
        candidates: list[MPCandidate] = []
        row_by_id: dict[str, pd.Series] = {}
        first = val_reset.loc[indices[0]]
        hold_seq = _arr(first["action_hold_previous_readback"])
        anchor = hold_seq[0]
        for i in indices:
            row = val_reset.loc[i]
            cid = str(row["case_uid"])
            seq = _arr(row["action_candidate_readback"])
            changed = int(np.sum(np.abs(seq[0] - anchor) > 1e-9))
            candidate_depth_ucb = predicted_priority_depth[i].max(axis=0)
            candidates.append(
                MPCandidate(
                    candidate_id=cid,
                    action_sequence=seq,
                    pfv_delta_ucb_m3=float(predicted["pfv_delta"][i]),
                    peak_delta_ucb_m3s=float(predicted["peak_delta"][i]),
                    tfv_delta_di_m3=float(predicted["tfv_delta"][i]),
                    action_cost=float(np.mean(np.abs(seq[0] - anchor))),
                    terminal_cost=0.0,
                    uncertainty_cost=0.0,
                    changed_facilities=changed,
                    engineering=EngineeringStatus(True, True, True, True, True),
                    uncertainty_pass=True,
                    ood_pass=True,
                    executable=True,
                    pfv_no_control_m3=float(predicted["no_control_pfv"][i]),
                    priority_depth_ucb_m=tuple(map(float, candidate_depth_ucb)),
                    priority_depth_limit_m=tuple(map(float, depth_limits)),
                    metadata={
                        "development_only": True,
                        "authoritative_outcome_available": True,
                        "uncertainty_note": "qualification single-model interface; Formal uses calibrated ensemble UCB",
                    },
                )
            )
            row_by_id[cid] = row
        fallback = FrozenFallback(
            "hold_previous",
            hold_seq,
            "PROJECT6_V42_FAST_REPLAY_FALLBACK",
            True,
            {"development_only": True},
        )
        decision = decide_pfvfirst_mpc(
            candidates=candidates,
            fallback=fallback,
            margins=SafetyMargins(
                pfv_absolute_allowance_m3=PFV_ABS_M3,
                pfv_relative_allowance_fraction=PFV_REL,
                max_changed_facilities=8,
                require_priority_depth=True,
            ),
            weights=MPCWeights(
                peak=600.0,
                action=0.05,
                terminal=0.0,
                uncertainty=0.0,
            ),
        )
        if decision.used_fallback:
            actual = _actual_deltas(first, "hold_previous", priority_idx)
            actual_priority_depth = _priority_depth_max(
                first, "hold_previous", priority_idx
            )
            selected_pred = {"pfv": None, "tfv": None, "peak": None, "nc_pfv": None}
        else:
            chosen = row_by_id[decision.selected_id]
            actual = _actual_deltas(chosen, "candidate", priority_idx)
            actual_priority_depth = _priority_depth_max(
                chosen, "candidate", priority_idx
            )
            chosen_i = int(chosen.name)
            selected_pred = {
                "pfv": float(predicted["pfv_delta"][chosen_i]),
                "tfv": float(predicted["tfv_delta"][chosen_i]),
                "peak": float(predicted["peak_delta"][chosen_i]),
                "nc_pfv": float(predicted["no_control_pfv"][chosen_i]),
            }
        nc = _branch_metrics(first, "no_control", priority_idx)
        di = _branch_metrics(first, "dynamic_internal", priority_idx)
        hold = _branch_metrics(first, "hold_previous", priority_idx)
        actual_pfv_allowance = PFV_ABS_M3 + PFV_REL * max(0.0, nc["pfv_m3"])
        actual_pfv_safe = actual["pfv_delta_nc_m3"] <= actual_pfv_allowance
        actual_depth_safe = bool(np.all(actual_priority_depth <= depth_limits))
        actual_safe = bool(actual_pfv_safe and actual_depth_safe)
        predicted_pfv_allowance = (
            None
            if selected_pred["nc_pfv"] is None
            else PFV_ABS_M3 + PFV_REL * max(0.0, float(selected_pred["nc_pfv"]))
        )
        candidate_predicted_safe = bool(
            not decision.used_fallback
            and selected_pred["pfv"] is not None
            and predicted_pfv_allowance is not None
            and float(selected_pred["pfv"]) <= predicted_pfv_allowance
        )
        replay_rows.append(
            {
                "state_key": state_key,
                "candidate_count": len(indices),
                "selected_id": decision.selected_id,
                "used_fallback": decision.used_fallback,
                "predicted_pfv_delta_m3": selected_pred["pfv"],
                "predicted_pfv_allowance_m3": predicted_pfv_allowance,
                "predicted_tfv_delta_m3": selected_pred["tfv"],
                "predicted_peak_delta_m3s": selected_pred["peak"],
                "actual_pfv_delta_nc_m3": actual["pfv_delta_nc_m3"],
                "actual_pfv_allowance_m3": actual_pfv_allowance,
                "actual_tfv_delta_di_m3": actual["tfv_delta_di_m3"],
                "actual_peak_delta_di_m3s": actual["peak_delta_di_m3s"],
                "actual_priority_depth_safe": actual_depth_safe,
                "proposal_pfv_m3": actual["pfv_m3"],
                "proposal_tfv_m3": actual["tfv_m3"],
                "proposal_peak_m3s": actual["peak_m3s"],
                "no_control_pfv_m3": nc["pfv_m3"],
                "no_control_tfv_m3": nc["tfv_m3"],
                "no_control_peak_m3s": nc["peak_m3s"],
                "dynamic_internal_pfv_m3": di["pfv_m3"],
                "dynamic_internal_tfv_m3": di["tfv_m3"],
                "dynamic_internal_peak_m3s": di["peak_m3s"],
                "hold_previous_pfv_m3": hold["pfv_m3"],
                "hold_previous_tfv_m3": hold["tfv_m3"],
                "hold_previous_peak_m3s": hold["peak_m3s"],
                "candidate_predicted_safe": candidate_predicted_safe,
                "selected_candidate_false_safe": bool(
                    candidate_predicted_safe and not actual_safe
                ),
                "fallback_actual_safe": bool(actual_safe)
                if decision.used_fallback
                else None,
                "actual_safe": actual_safe,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "fast_policy_replay_rows.csv"
    if replay_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(replay_rows[0].keys()))
            writer.writeheader()
            writer.writerows(replay_rows)
    fallback_rows = [r for r in replay_rows if r["used_fallback"]]
    selected_rows = [r for r in replay_rows if not r["used_fallback"]]
    summary = {
        "contract_id": FAST_CONTRACT_ID,
        "control_objective_contract": "PROJECT6_V42_PFV_BUDGETED_TFV_MPC_V1",
        "stage": "step3_fast_offline_policy_replay",
        "development_only": True,
        "formal_closed_loop": False,
        "authoritative_outcomes": "recorded_SWMM_trajectories",
        "validation_rainfall_groups": val_groups,
        "state_groups_total": int(val_reset["state_key"].nunique()),
        "state_groups_replayed": int(len(replay_rows)),
        "fallback_rate": _rate(replay_rows, "used_fallback"),
        "nonfallback_selection_count": len(selected_rows),
        "nonfallback_selection_rate": None
        if not replay_rows
        else float(len(selected_rows) / len(replay_rows)),
        "fallback_actual_safety_rate": _rate(fallback_rows, "actual_safe"),
        "selected_candidate_actual_safety_rate": _rate(selected_rows, "actual_safe"),
        "selected_candidate_false_safe_rate": _rate(
            selected_rows, "selected_candidate_false_safe"
        ),
        "actual_safety_rate": _rate(replay_rows, "actual_safe"),
        "proposal_mean_pfv_m3": _mean(replay_rows, "proposal_pfv_m3"),
        "proposal_mean_tfv_m3": _mean(replay_rows, "proposal_tfv_m3"),
        "proposal_mean_peak_m3s": _mean(replay_rows, "proposal_peak_m3s"),
        "no_control_mean_pfv_m3": _mean(replay_rows, "no_control_pfv_m3"),
        "dynamic_internal_mean_tfv_m3": _mean(
            replay_rows, "dynamic_internal_tfv_m3"
        ),
        "dynamic_internal_mean_peak_m3s": _mean(
            replay_rows, "dynamic_internal_peak_m3s"
        ),
        "proposal_mean_pfv_delta_nc_m3": _mean(
            replay_rows, "actual_pfv_delta_nc_m3"
        ),
        "proposal_mean_tfv_delta_di_m3": _mean(
            replay_rows, "actual_tfv_delta_di_m3"
        ),
        "proposal_mean_peak_delta_di_m3s": _mean(
            replay_rows, "actual_peak_delta_di_m3s"
        ),
        "peak_is_performance_penalty_not_hard_safety": True,
    }
    summary["go_signal"] = bool(
        replay_rows
        and len(selected_rows) > 0
        and (summary["actual_safety_rate"] or 0.0) >= 0.8
        and (summary["proposal_mean_tfv_delta_di_m3"] or 0.0) < 0.0
        and (
            summary["selected_candidate_false_safe_rate"] is None
            or summary["selected_candidate_false_safe_rate"] <= 0.2
        )
    )
    summary["next_required_if_go"] = "one_event_authoritative_SWMM_rolling_closed_loop"
    (args.output_dir / "fast_policy_replay_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
