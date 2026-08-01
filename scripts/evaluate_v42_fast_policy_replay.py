"""Fast SWMM-backed *offline* policy replay for V4.2 feasibility screening.

The surrogate scores multiple historically simulated candidates at the same
state.  The canonical PFV-first selector chooses one using predicted deltas;
then the chosen action is scored with the already-recorded authoritative SWMM
trajectory.  This is much faster than a new rolling SWMM closed loop and is a
useful go/no-go screen, but it is not a substitute for formal closed-loop/blind
validation.
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


def _arr(value: str) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=np.float32)


def _branch_metrics(row: pd.Series, branch: str, priority_idx: list[int]) -> dict[str, float]:
    flood = _arr(row[f"trajectory_flood_{branch}"]).astype(np.float64)
    system = flood.sum(axis=1)
    priority = flood[:, priority_idx].sum(axis=1)
    return {
        "pfv_m3": float(priority.sum() * DT_SEC),
        "tfv_m3": float(system.sum() * DT_SEC),
        "peak_m3s": float(system.max()),
    }


def _actual_deltas(row: pd.Series, branch: str, priority_idx: list[int]) -> dict[str, float]:
    m = _branch_metrics(row, branch, priority_idx)
    nc = _branch_metrics(row, "no_control", priority_idx)
    di = _branch_metrics(row, "dynamic_internal", priority_idx)
    return {
        "pfv_delta_nc_m3": m["pfv_m3"] - nc["pfv_m3"],
        "tfv_delta_di_m3": m["tfv_m3"] - di["tfv_m3"],
        "peak_delta_di_m3s": m["peak_m3s"] - di["peak_m3s"],
        **m,
    }


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
    return None if not vals else float(np.mean(vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    frame = pd.read_parquet(args.manifest) if args.manifest.suffix.lower() == ".parquet" else pd.read_csv(args.manifest)
    _, val_f, _, val_groups = _split_groups(frame, args.seed)
    if val_f.empty:
        raise ValueError("no validation rows for replay")

    report = json.loads((args.model_dir / "fast_step2_report.json").read_text(encoding="utf-8"))
    cfg = report.get("config", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = _load_graph_topology(args.project_root)
    edge_index = torch.from_numpy(graph["edge_index"].astype(np.int64)).to(device)
    node_static = torch.from_numpy(graph["node_static"].astype(np.float32)).to(device)
    action_map = torch.from_numpy(graph["action_node_map"].astype(np.float32)).to(device)
    priority_idx = get_pfv_core_node_indices(list(graph["node_ids"]))
    priority_tensor = torch.as_tensor(priority_idx, dtype=torch.long, device=device)

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
    model.load_state_dict(torch.load(args.model_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()

    data = _tensorise(val_f)
    predicted = {"pfv_delta": np.zeros(len(val_f)), "tfv_delta": np.zeros(len(val_f)), "peak_delta": np.zeros(len(val_f))}
    with torch.no_grad():
        for start in range(0, len(val_f), args.batch_size):
            idx = np.arange(start, min(len(val_f), start + args.batch_size))
            batch = _slice(data, idx)
            out = _forward(model, batch, (edge_index, node_static, action_map), priority_tensor, device)
            for key in predicted:
                predicted[key][idx] = out[key].detach().cpu().numpy()

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
            action_cost = float(np.mean(np.abs(seq[0] - anchor)))
            candidates.append(
                MPCandidate(
                    candidate_id=cid,
                    action_sequence=seq,
                    pfv_delta_ucb_m3=float(predicted["pfv_delta"][i]),
                    peak_delta_ucb_m3s=float(predicted["peak_delta"][i]),
                    tfv_delta_di_m3=float(predicted["tfv_delta"][i]),
                    action_cost=action_cost,
                    terminal_cost=0.0,
                    uncertainty_cost=0.0,
                    changed_facilities=changed,
                    engineering=EngineeringStatus(True, True, True, True, True),
                    uncertainty_pass=True,
                    ood_pass=True,
                    executable=True,
                    metadata={"development_only": True, "authoritative_outcome_available": True},
                )
            )
            row_by_id[cid] = row
        fallback = FrozenFallback(
            fallback_id="hold_previous",
            action_sequence=hold_seq,
            contract_hash="PROJECT6_V42_FAST_REPLAY_FALLBACK",
            legal=True,
            metadata={"development_only": True},
        )
        decision = decide_pfvfirst_mpc(
            candidates=candidates,
            fallback=fallback,
            margins=SafetyMargins(max_changed_facilities=8),
            weights=MPCWeights(action=0.05, terminal=0.0, uncertainty=0.0),
        )
        if decision.used_fallback:
            actual = _actual_deltas(first, "hold_previous", priority_idx)
            selected_pred = {"pfv": None, "tfv": None, "peak": None}
        else:
            chosen = row_by_id[decision.selected_id]
            actual = _actual_deltas(chosen, "candidate", priority_idx)
            chosen_i = int(chosen.name)
            selected_pred = {
                "pfv": float(predicted["pfv_delta"][chosen_i]),
                "tfv": float(predicted["tfv_delta"][chosen_i]),
                "peak": float(predicted["peak_delta"][chosen_i]),
            }
        nc = _branch_metrics(first, "no_control", priority_idx)
        di = _branch_metrics(first, "dynamic_internal", priority_idx)
        hold = _branch_metrics(first, "hold_previous", priority_idx)
        actual_safe = bool(actual["pfv_delta_nc_m3"] <= 0.0 and actual["peak_delta_di_m3s"] <= 0.0)
        predicted_safe = bool(
            decision.used_fallback
            or ((selected_pred["pfv"] or 0.0) <= 0.0 and (selected_pred["peak"] or 0.0) <= 0.0)
        )
        replay_rows.append(
            {
                "state_key": state_key,
                "candidate_count": len(indices),
                "selected_id": decision.selected_id,
                "used_fallback": decision.used_fallback,
                "predicted_pfv_delta_m3": selected_pred["pfv"],
                "predicted_tfv_delta_m3": selected_pred["tfv"],
                "predicted_peak_delta_m3s": selected_pred["peak"],
                "actual_pfv_delta_nc_m3": actual["pfv_delta_nc_m3"],
                "actual_tfv_delta_di_m3": actual["tfv_delta_di_m3"],
                "actual_peak_delta_di_m3s": actual["peak_delta_di_m3s"],
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
                "predicted_safe": predicted_safe,
                "actual_safe": actual_safe,
                "false_safe": bool(predicted_safe and not actual_safe),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "fast_policy_replay_rows.csv"
    if replay_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(replay_rows[0].keys()))
            writer.writeheader()
            writer.writerows(replay_rows)
    summary = {
        "contract_id": FAST_CONTRACT_ID,
        "stage": "step3_fast_offline_policy_replay",
        "development_only": True,
        "formal_closed_loop": False,
        "authoritative_outcomes": "recorded_SWMM_trajectories",
        "validation_rainfall_groups": val_groups,
        "state_groups_total": int(val_reset["state_key"].nunique()),
        "state_groups_replayed": int(len(replay_rows)),
        "fallback_rate": None if not replay_rows else float(np.mean([r["used_fallback"] for r in replay_rows])),
        "actual_safety_rate": None if not replay_rows else float(np.mean([r["actual_safe"] for r in replay_rows])),
        "false_safe_rate": None if not replay_rows else float(np.mean([r["false_safe"] for r in replay_rows])),
        "proposal_mean_pfv_m3": _mean(replay_rows, "proposal_pfv_m3"),
        "proposal_mean_tfv_m3": _mean(replay_rows, "proposal_tfv_m3"),
        "proposal_mean_peak_m3s": _mean(replay_rows, "proposal_peak_m3s"),
        "no_control_mean_pfv_m3": _mean(replay_rows, "no_control_pfv_m3"),
        "dynamic_internal_mean_tfv_m3": _mean(replay_rows, "dynamic_internal_tfv_m3"),
        "dynamic_internal_mean_peak_m3s": _mean(replay_rows, "dynamic_internal_peak_m3s"),
        "proposal_mean_pfv_delta_nc_m3": _mean(replay_rows, "actual_pfv_delta_nc_m3"),
        "proposal_mean_tfv_delta_di_m3": _mean(replay_rows, "actual_tfv_delta_di_m3"),
        "proposal_mean_peak_delta_di_m3s": _mean(replay_rows, "actual_peak_delta_di_m3s"),
        "go_signal": bool(
            replay_rows
            and float(np.mean([r["actual_safe"] for r in replay_rows])) >= 0.8
            and (_mean(replay_rows, "actual_tfv_delta_di_m3") or 0.0) < 0.0
        ),
        "next_required_if_go": "one_event_authoritative_SWMM_rolling_closed_loop",
    }
    (args.output_dir / "fast_policy_replay_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
