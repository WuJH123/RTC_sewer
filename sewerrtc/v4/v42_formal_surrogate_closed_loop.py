"""Surrogate-only closed-loop diagnostic for Project6 V4.2 Formal.

This is stage 22 of the attribution chain.  It does not fabricate a passing
``surrogate_closed_loop`` evidence file from model metadata.  Starting from the
same authoritative prefix used by stage 21, it repeatedly:

1. reveals rainfall only through the current decision time;
2. runs the canonical PFV-budgeted selector with the Formal Step2 ensemble;
3. executes only the first action in the surrogate plant;
4. advances the surrogate hydraulic state by one 10-min step;
5. rolls the 13-frame state/action history and replans.

The current Dynamic-Internal action is taken from the authoritative Internal
trajectory at the same time and persisted causally over H120.  Future Internal
actions, future SWMM hydraulic states and future rainfall observations are never
passed to the selector.  The full authoritative CSV may be loaded offline, but
all online slices are explicitly bounded by ``elapsed_min <= current_time``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sewerrtc.simulation.kpi_metrics import compute_kpis
from sewerrtc.v4.v42_fast_e2e import make_causal_rainfall_forecast
from sewerrtc.v4.v42_formal_runtime import (
    CONTROL_INTERVAL_MIN,
    HORIZON_STEPS,
    FormalEventInput,
    _json_safe,
    load_actuators,
    load_model_bundle,
    predict_and_decide,
    sha256_file,
)


def _columns(ids: list[str], prefix: str) -> list[str]:
    return [f"{prefix}{item}" for item in ids]


def _row_at(frame: pd.DataFrame, elapsed: float) -> pd.Series:
    values = pd.to_numeric(frame["elapsed_min"], errors="coerce").to_numpy(float)
    idx = np.flatnonzero(np.isclose(values, float(elapsed), atol=1.0e-6, rtol=0.0))
    if len(idx) != 1:
        raise RuntimeError(f"expected one authoritative row at elapsed_min={elapsed}; got {len(idx)}")
    return frame.iloc[int(idx[0])]


def _predict_next_step(
    *,
    bundle,
    state_history: np.ndarray,
    historical_actions: np.ndarray,
    rainfall_forecast: np.ndarray,
    current_action: np.ndarray,
    command: np.ndarray,
    internal_current_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ensemble-mean Candidate depth/flood at t+10 only."""
    n_actions = bundle.graph.n_facilities
    candidate = np.repeat(np.asarray(current_action, np.float32)[None, :], HORIZON_STEPS, axis=0)
    candidate[0] = np.asarray(command, np.float32)
    no_control = np.ones((HORIZON_STEPS, n_actions), np.float32)
    internal = np.repeat(
        np.asarray(internal_current_action, np.float32)[None, :], HORIZON_STEPS, axis=0
    )
    hold = np.repeat(
        np.asarray(current_action, np.float32)[None, :], HORIZON_STEPS, axis=0
    )
    priority = torch.as_tensor(
        bundle.priority_indices, dtype=torch.long, device=bundle.device
    )
    depth: list[np.ndarray] = []
    flood: list[np.ndarray] = []
    with torch.inference_mode():
        for model in bundle.step2_models:
            out = model(
                state_history=torch.as_tensor(
                    np.asarray(state_history, np.float32)[None, :, :],
                    device=bundle.device,
                ),
                historical_actions=torch.as_tensor(
                    np.asarray(historical_actions, np.float32)[None, :, :],
                    device=bundle.device,
                ),
                rainfall_forecast=torch.as_tensor(
                    np.asarray(rainfall_forecast, np.float32)[None, :HORIZON_STEPS],
                    device=bundle.device,
                ),
                action_candidate=torch.as_tensor(candidate[None, :, :], device=bundle.device),
                action_no_control=torch.as_tensor(no_control[None, :, :], device=bundle.device),
                action_dynamic_internal=torch.as_tensor(internal[None, :, :], device=bundle.device),
                action_hold_previous=torch.as_tensor(hold[None, :, :], device=bundle.device),
                edge_index=bundle.edge_index,
                node_static=bundle.node_static,
                action_node_map=bundle.action_node_map,
                priority_node_indices=priority,
            )
            depth.append(
                out["branches"]["candidate"]["node_depth"][0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            flood.append(
                out["branches"]["candidate"]["node_flooding_rate"][0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return np.mean(depth, axis=0).astype(np.float32), np.mean(flood, axis=0).astype(np.float32)


def run_surrogate_closed_loop_event(
    event: FormalEventInput,
    *,
    project_root: str | Path,
    exact_proposed_detail: str | Path,
    exact_internal_detail: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    max_candidate_sequences: int = 64,
) -> dict[str, Any]:
    root = Path(project_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    proposed = pd.read_csv(exact_proposed_detail, low_memory=False)
    internal = pd.read_csv(exact_internal_detail, low_memory=False)
    if proposed.empty or internal.empty:
        raise RuntimeError("surrogate closed loop requires non-empty stage21 authoritative trajectories")
    bundle = load_model_bundle(root, device)
    actuators = load_actuators(root)
    node_ids = list(map(str, bundle.graph.node_ids))
    action_ids = list(map(str, bundle.graph.facility_ids))
    if action_ids != actuators["actuator_id"].astype(str).tolist():
        raise RuntimeError("surrogate closed loop Engineering36 order mismatch")
    depth_cols = _columns(node_ids, "h:")
    action_cols = _columns(action_ids, "setting:")
    flood_cols = _columns(node_ids, "flood:")
    for name, frame, columns in (
        ("Proposed", proposed, ["elapsed_min", "rainfall_mm_h", *depth_cols, *action_cols]),
        ("Internal", internal, ["elapsed_min", *action_cols]),
    ):
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise KeyError(f"stage21 {name} detail missing columns: {missing[:10]}")

    elapsed_all = pd.to_numeric(proposed["elapsed_min"], errors="raise").to_numpy(float)
    start_time = 120.0
    if not np.any(np.isclose(elapsed_all, start_time, atol=1.0e-6)):
        raise RuntimeError("stage21 trajectory lacks the frozen 120-min control start")
    prefix_times = [start_time - 60.0 + 5.0 * i for i in range(13)]
    state_history = np.stack(
        [
            _row_at(proposed, t)[depth_cols].to_numpy(np.float32)
            for t in prefix_times
        ],
        axis=0,
    )
    historical_actions = np.stack(
        [
            _row_at(proposed, t)[action_cols].to_numpy(np.float32)
            for t in prefix_times
        ],
        axis=0,
    )
    current_action = historical_actions[-1].copy()
    max_time = float(np.nanmax(elapsed_all))
    decision_times = np.arange(start_time, max_time - 1.0e-6, CONTROL_INTERVAL_MIN)
    priority_nodes = [node_ids[i] for i in bundle.priority_indices]
    records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    started = time.time()

    for current_time in decision_times:
        if current_time + CONTROL_INTERVAL_MIN > max_time + 1.0e-6:
            break
        visible = proposed.loc[
            pd.to_numeric(proposed["elapsed_min"], errors="coerce") <= current_time + 1.0e-6
        ]
        observed_rain = pd.to_numeric(
            visible["rainfall_mm_h"], errors="coerce"
        ).fillna(0.0).to_numpy(float).tolist()
        forecast = make_causal_rainfall_forecast(
            observed_rain, horizon_steps=HORIZON_STEPS
        )
        internal_action = _row_at(internal, current_time)[action_cols].to_numpy(np.float32)
        command, info = predict_and_decide(
            bundle=bundle,
            actuators=actuators,
            state_history=state_history,
            historical_actions=historical_actions,
            rainfall_forecast=forecast,
            current_action=current_action,
            internal_current_action=internal_action,
            # Stage22 isolates Step2/model-feedback from sparse-GAT error. OOD is
            # therefore not an applicable failure mode here; stage23 restores it.
            gat_ood_score=0.0,
            max_candidate_sequences=max_candidate_sequences,
        )
        next_depth, next_flood = _predict_next_step(
            bundle=bundle,
            state_history=state_history,
            historical_actions=historical_actions,
            rainfall_forecast=forecast,
            current_action=current_action,
            command=command,
            internal_current_action=internal_action,
        )
        midpoint_depth = 0.5 * (state_history[-1] + next_depth)
        next5 = current_time + 5.0
        next10 = current_time + 10.0
        rain5 = float(_row_at(proposed, next5)["rainfall_mm_h"])
        rain10 = float(_row_at(proposed, next10)["rainfall_mm_h"])
        state_history = np.concatenate(
            [state_history[2:], midpoint_depth[None, :], next_depth[None, :]], axis=0
        )
        historical_actions = np.concatenate(
            [
                historical_actions[2:],
                np.asarray(command, np.float32)[None, :],
                np.asarray(command, np.float32)[None, :],
            ],
            axis=0,
        )
        current_action = np.asarray(command, np.float32)
        row: dict[str, Any] = {
            "event_id": event.event_id,
            "elapsed_min": float(next10),
            "rainfall_mm_h": rain10,
            "rainfall_midpoint_mm_h": rain5,
        }
        for i, node in enumerate(node_ids):
            row[f"h:{node}"] = float(next_depth[i])
            row[f"flood:{node}"] = float(next_flood[i])
        for i, aid in enumerate(action_ids):
            row[f"setting:{aid}"] = float(command[i])
        records.append(row)
        decisions.append(
            {
                **info,
                "elapsed_min": float(current_time),
                "surrogate_state_feedback": True,
                "gat_ood_gate_not_applicable_in_stage22": True,
                "future_hydraulic_truth_used_online": False,
                "realized_future_rainfall_used_online": False,
                "rainfall_visibility_max_min": float(current_time),
            }
        )

    if not records or not decisions:
        raise RuntimeError("surrogate closed loop produced no decision/transition records")
    detail = pd.DataFrame(records)
    detail_path = output / "detail.csv"
    detail.to_csv(detail_path, index=False)
    decision_path = output / "decisions.jsonl"
    with decision_path.open("w", encoding="utf-8") as handle:
        for item in decisions:
            handle.write(
                json.dumps(_json_safe(item), ensure_ascii=False, allow_nan=False) + "\n"
            )
    kpis = compute_kpis(detail, priority_nodes, dt_sec=600)
    result = {
        "status": "pass",
        "event_id": event.event_id,
        "rainfall_sha256": event.rainfall_sha256,
        "surrogate_role": "hydraulic_surrogate_not_policy",
        "state_source": "surrogate_feedback_from_authoritative_prefix",
        "pfvfirst_mpc_v42": True,
        "authority": "formal_surrogate_diagnostic",
        "authoritative_prefix_end_min": start_time,
        "authoritative_hydraulic_truth_used_after_prefix": False,
        "realized_future_rainfall_used_online": False,
        "dynamic_internal_future_action_used_online": False,
        "decision_count": len(decisions),
        "detail_path": str(detail_path),
        "detail_sha256": sha256_file(detail_path),
        "decision_path": str(decision_path),
        "surrogate_model_sha256": bundle.surrogate_model_sha256,
        "gat_model_sha256": bundle.gat_model_sha256,
        "kpis": kpis,
        "runtime_sec": time.time() - started,
    }
    (output / "run_result.json").write_text(
        json.dumps(_json_safe(result), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return result
