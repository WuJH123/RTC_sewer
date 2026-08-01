"""Compare the fast V4.2 Proposed controller with drainage-control baselines.

No-control, Dynamic-Internal and Hold-Previous use the already recorded
authoritative SWMM branches.  Proposed uses the authoritative SWMM outcome of the
candidate selected by the fast PFV-first replay.  EFD, Auto-RBC and All-Close are
constructed from the same decision-time state; when a sufficiently close
historically simulated candidate exists its SWMM trajectory is used, otherwise
the fast Step2 surrogate supplies a clearly labelled screening estimate.

This mixed-authority table is intentionally development-only.  It answers the
"does this control chain have potential?" question without pretending that EFD
or Auto-RBC already have formal closed-loop SWMM evidence.
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

from scripts.evaluate_v42_fast_policy_replay import _branch_metrics
from scripts.train_v42_step2_fast import _forward, _split_groups, _tensorise
from sewerrtc.v4.models_v42.hydraulic_multi_reference import MultiReferenceHydraulicSurrogate
from sewerrtc.v4.v42_fast_e2e import (
    FAST_E2E_CONTRACT_ID,
    build_development_baseline_actions,
    nearest_recorded_action_proxy,
)
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_step1_dataset import load_graph_assets
from sewerrtc.v4.v42_trajectory_builder import NODE_STATIC_COLS, _load_graph_topology


BINARY_PUMP_IDS = {"add301.2", "add301.3"}


def _arr(value: object) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=np.float32)


def _surrogate_metrics(
    *,
    model,
    row: pd.Series,
    custom_action: np.ndarray,
    graph_tensors,
    priority_tensor: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    one = pd.DataFrame([row])
    data = _tensorise(one)
    data["action_candidate"] = torch.from_numpy(custom_action[None, ...].astype(np.float32))
    with torch.no_grad():
        out = _forward(model, data, graph_tensors, priority_tensor, device)
    kpi = out["kpi_candidate"]
    return {
        "pfv_m3": float(kpi["pfv_m3"].detach().cpu().numpy()[0]),
        "tfv_m3": float(kpi["tfv_m3"].detach().cpu().numpy()[0]),
        "peak_m3s": float(kpi["peak_m3s"].detach().cpu().numpy()[0]),
    }


def _pct_reduction(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None or not np.isfinite(reference) or abs(reference) <= 1.0e-12:
        return None
    return float(100.0 * (reference - value) / reference)


def _aggregate(rows: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for strategy, grp in rows.groupby("strategy", sort=True):
        result[str(strategy)] = {
            "states": int(len(grp)),
            "mean_pfv_m3": float(grp["pfv_m3"].mean()),
            "median_pfv_m3": float(grp["pfv_m3"].median()),
            "mean_tfv_m3": float(grp["tfv_m3"].mean()),
            "median_tfv_m3": float(grp["tfv_m3"].median()),
            "mean_peak_m3s": float(grp["peak_m3s"].mean()),
            "median_peak_m3s": float(grp["peak_m3s"].median()),
            "swmm_backed_fraction": float(grp["authority"].astype(str).str.startswith("authoritative_SWMM").mean()),
            "mean_proxy_distance": (
                None
                if grp["proxy_distance"].dropna().empty
                else float(grp["proxy_distance"].dropna().mean())
            ),
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--policy-replay-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--gat-layers", type=int, default=2)
    ap.add_argument("--proxy-tolerance", type=float, default=0.05)
    args = ap.parse_args()

    frame = pd.read_parquet(args.manifest) if args.manifest.suffix.lower() == ".parquet" else pd.read_csv(args.manifest)
    if frame.empty:
        raise ValueError("fast E2E baseline manifest is empty")
    _, val_f, _, val_groups = _split_groups(frame, args.seed)
    if val_f.empty:
        raise ValueError("no validation rows for fast E2E baseline comparison")

    replay_path = args.policy_replay_dir / "fast_policy_replay_rows.csv"
    replay = pd.read_csv(replay_path)
    if replay.empty:
        raise RuntimeError("policy replay has no SWMM-backed Proposed states")
    proposed_by_state = {str(r.state_key): r for r in replay.itertuples(index=False)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = _load_graph_topology(args.project_root)
    step1_graph = load_graph_assets(args.project_root)
    if [str(x) for x in graph["node_ids"]] != [str(x) for x in step1_graph.node_ids]:
        raise RuntimeError("baseline graph node order differs between Step1 and Step2")
    max_depth_idx = NODE_STATIC_COLS.index("max_depth")
    max_depth = step1_graph.node_static_raw[:, max_depth_idx].astype(np.float32)
    binary_indices = [
        i for i, fid in enumerate(graph["facility_ids"])
        if str(fid).casefold() in BINARY_PUMP_IDS
    ]

    edge_index = torch.from_numpy(graph["edge_index"].astype(np.int64)).to(device)
    node_static = torch.from_numpy(graph["node_static"].astype(np.float32)).to(device)
    action_map = torch.from_numpy(graph["action_node_map"].astype(np.float32)).to(device)
    priority_idx = get_pfv_core_node_indices(list(graph["node_ids"]))
    priority_tensor = torch.as_tensor(priority_idx, dtype=torch.long, device=device)
    graph_tensors = (edge_index, node_static, action_map)

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
    model.load_state_dict(torch.load(args.model_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()

    long_rows: list[dict[str, object]] = []
    for state_key, grp in val_f.groupby("state_key", sort=True):
        state_key = str(state_key)
        if state_key not in proposed_by_state:
            continue
        first = grp.iloc[0]
        proposed = proposed_by_state[state_key]
        nc = _branch_metrics(first, "no_control", priority_idx)
        di = _branch_metrics(first, "dynamic_internal", priority_idx)
        hold = _branch_metrics(first, "hold_previous", priority_idx)
        for strategy, metrics in (
            ("no_control", nc),
            ("internal_rule", di),
            ("hold_previous", hold),
            (
                "proposed_gat_surrogate_pfvfirst",
                {
                    "pfv_m3": float(proposed.proposal_pfv_m3),
                    "tfv_m3": float(proposed.proposal_tfv_m3),
                    "peak_m3s": float(proposed.proposal_peak_m3s),
                },
            ),
        ):
            long_rows.append(
                {
                    "state_key": state_key,
                    "split_group_key": str(first["split_group_key"]),
                    "strategy": strategy,
                    **metrics,
                    "authority": "authoritative_SWMM_recorded_branch_or_selected_candidate",
                    "proxy_distance": np.nan,
                }
            )

        history = _arr(first["history_depth"])
        hold_seq = _arr(first["action_hold_previous_readback"])
        anchor = hold_seq[0]
        schedules = build_development_baseline_actions(
            current_depth=history[-1],
            max_depth=max_depth,
            action_node_map=np.asarray(graph["action_node_map"], dtype=np.float32),
            anchor_action=anchor,
            binary_indices=binary_indices,
            max_changed_facilities=8,
        )
        candidate_rows = [row for _, row in grp.iterrows()]
        candidate_sequences = [_arr(row["action_candidate_readback"]) for row in candidate_rows]
        for strategy in ("efd", "auto_rbc", "all_close"):
            target = schedules[strategy]
            proxy_idx, distance = nearest_recorded_action_proxy(target, candidate_sequences)
            if distance <= float(args.proxy_tolerance):
                proxy_row = candidate_rows[proxy_idx]
                metrics = _branch_metrics(proxy_row, "candidate", priority_idx)
                authority = "authoritative_SWMM_nearest_recorded_action_proxy"
            else:
                metrics = _surrogate_metrics(
                    model=model,
                    row=first,
                    custom_action=target,
                    graph_tensors=graph_tensors,
                    priority_tensor=priority_tensor,
                    device=device,
                )
                authority = "surrogate_screen_no_close_recorded_action"
            long_rows.append(
                {
                    "state_key": state_key,
                    "split_group_key": str(first["split_group_key"]),
                    "strategy": strategy,
                    **metrics,
                    "authority": authority,
                    "proxy_distance": float(distance),
                }
            )

    rows = pd.DataFrame(long_rows)
    if rows.empty:
        raise RuntimeError("no common validation states between policy replay and baseline comparison")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "FAST_E2E_BASELINE_ROWS.csv", index=False)

    summary = _aggregate(rows)
    nc = summary.get("no_control", {})
    internal = summary.get("internal_rule", {})
    for strategy, metrics in summary.items():
        metrics["pfv_reduction_vs_no_control_pct"] = _pct_reduction(
            nc.get("mean_pfv_m3"), metrics.get("mean_pfv_m3")
        )
        metrics["tfv_reduction_vs_internal_pct"] = _pct_reduction(
            internal.get("mean_tfv_m3"), metrics.get("mean_tfv_m3")
        )
        metrics["peak_reduction_vs_internal_pct"] = _pct_reduction(
            internal.get("mean_peak_m3s"), metrics.get("mean_peak_m3s")
        )

    proposed_summary = summary.get("proposed_gat_surrogate_pfvfirst", {})
    replay_summary = json.loads(
        (args.policy_replay_dir / "fast_policy_replay_summary.json").read_text(encoding="utf-8")
    )
    potential_go = bool(
        replay_summary.get("go_signal")
        and proposed_summary
        and float(proposed_summary.get("mean_tfv_m3", np.inf))
        < float(internal.get("mean_tfv_m3", np.inf))
    )
    payload = {
        "contract_id": FAST_E2E_CONTRACT_ID,
        "stage": "fast_e2e_baseline_comparison",
        "development_only": True,
        "formal_mainline_authorized": False,
        "validation_rainfall_groups": val_groups,
        "common_state_count": int(rows["state_key"].nunique()),
        "required_strategies": [
            "proposed_gat_surrogate_pfvfirst",
            "efd",
            "auto_rbc",
            "all_close",
            "no_control",
            "internal_rule",
            "hold_previous",
        ],
        "all_required_strategies_present": all(
            x in summary
            for x in (
                "proposed_gat_surrogate_pfvfirst",
                "efd",
                "auto_rbc",
                "all_close",
                "no_control",
                "internal_rule",
                "hold_previous",
            )
        ),
        "authority_note": (
            "No-control/Internal/Hold/Proposed are recorded authoritative SWMM. "
            "EFD/Auto-RBC/All-close use a recorded SWMM candidate when the H3 action "
            "distance is within tolerance; otherwise they are surrogate-only screening values."
        ),
        "strategy_summary": summary,
        "policy_replay": replay_summary,
        "potential_go": potential_go,
        "next_if_go": "authoritative_SWMM_micro_closed_loop_all_required_baselines",
    }
    (args.output_dir / "FAST_E2E_BASELINE_COMPARISON.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    table = pd.DataFrame.from_dict(summary, orient="index").reset_index(names="strategy")
    table.to_csv(args.output_dir / "FAST_E2E_BASELINE_COMPARISON.csv", index=False)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
