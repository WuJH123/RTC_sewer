"""Safety wrapper around the V4.2 Formal authoritative runtime.

For Proposed and every non-Internal baseline, Engineering36 must be controlled by
the evaluated policy, not simultaneously by the INP's native [CONTROLS] rules.
This module creates a short-lived rule-free runtime INP while preserving the
physical network/rainfall definition. ``Internal`` alone runs the original INP
with its native rules. Proposed uses *both*: rule-free plant + native-rule causal
shadow advanced only to the current decision time.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.simulation.kpi_metrics import compute_kpis
from sewerrtc.simulation.pyswmm_runner import (
    _get_existing_links,
    _ids_from_container,
    _observed_action_from_links,
    physical_network_sha256,
)
from sewerrtc.v4.v42_formal_runtime import (
    FORMAL_OBJECTIVE_CONTRACT,
    HORIZON_STEPS,
    STATE_STEP_SEC,
    FormalEventInput,
    _frame,
    _is_decision_time,
    _json_safe,
    _record_row,
    load_actuators,
    load_model_bundle,
    predict_and_decide,
    reconstruct_history,
    run_baseline_event as _run_baseline_event,
    sha256_file,
)
from sewerrtc.v4.v42_fast_e2e import make_causal_rainfall_forecast


def build_rule_free_runtime_inp(source: str | Path, target: str | Path) -> Path:
    source = Path(source)
    target = Path(target)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    output: list[str] = []
    in_controls = False
    controls_seen = False
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            name = stripped[1:-1].strip().upper()
            if name == "CONTROLS":
                in_controls = True
                controls_seen = True
                output.append(raw)
                output.append(
                    "; Formal runtime: native control rules disabled; evaluated policy owns Engineering36."
                )
                continue
            if in_controls:
                in_controls = False
            output.append(raw)
            continue
        if in_controls:
            if stripped.startswith(";"):
                output.append(raw)
            continue
        output.append(raw)
    if not controls_seen:
        output.extend(
            [
                "",
                "[CONTROLS]",
                "; Formal runtime: no native control rules in source INP.",
            ]
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(output) + "\n", encoding="utf-8")
    if physical_network_sha256(source) != physical_network_sha256(target):
        raise RuntimeError(
            "removing [CONTROLS] changed the physical-network SHA; refusing Formal runtime clone"
        )
    return target.resolve()


def _runtime_event(event: FormalEventInput, output_dir: Path) -> FormalEventInput:
    digest = hashlib.sha256(
        f"{event.event_id}|{event.rainfall_sha256}|{event.input_sha256}".encode("utf-8")
    ).hexdigest()[:16]
    runtime_inp = build_rule_free_runtime_inp(
        event.inp_path,
        output_dir / "runtime_inp" / f"{digest}__no_native_controls.inp",
    )
    return replace(event, inp_path=runtime_inp)


def run_baseline_event(
    event: FormalEventInput,
    *,
    strategy: str,
    project_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir)
    if strategy == "Internal":
        result = _run_baseline_event(
            event,
            strategy=strategy,
            project_root=project_root,
            output_dir=out,
        )
        result["native_controls_preserved"] = True
        return result
    runtime_event = _runtime_event(event, out)
    result = _run_baseline_event(
        runtime_event,
        strategy=strategy,
        project_root=project_root,
        output_dir=out,
    )
    result["source_input_sha256"] = event.input_sha256
    result["runtime_rule_free_inp_sha256"] = runtime_event.input_sha256
    result["native_controls_disabled"] = True
    # The runtime copy is a policy-execution derivative, not a new physical
    # scenario. The frozen physical network SHA has already been verified equal.
    result["physical_network_sha256"] = physical_network_sha256(event.inp_path)
    return result


def run_proposed_event(
    event: FormalEventInput,
    *,
    project_root: str | Path,
    output_dir: str | Path,
    state_source: str = "gat_sparse_reconstruction",
    device: str = "auto",
    max_candidate_sequences: int = 64,
) -> dict[str, Any]:
    """Run rule-free Proposed plant plus causal native-rule Internal shadow."""
    from pyswmm import Links, Nodes, RainGages, Simulation

    root = Path(project_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_event = _runtime_event(event, out_dir)
    actuators = load_actuators(root)
    bundle = load_model_bundle(root, device)
    ids = actuators["actuator_id"].astype(str).tolist()
    if ids != [str(x) for x in bundle.graph.facility_ids]:
        raise RuntimeError("Formal Proposed actuator order differs from trained graph")
    priority_nodes = [str(bundle.graph.node_ids[i]) for i in bundle.priority_indices]
    detail_path = out_dir / "detail.csv"
    decision_path = out_dir / "decisions.jsonl"
    records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    started = time.time()

    # Main plant is rule-free so the evaluated policy owns Engineering36.
    # Internal shadow keeps the original native controls. It is advanced one
    # 5-min record step alongside the plant, and only its *current* readback is
    # exposed. Future shadow actions/states are never queried.
    with Simulation(str(runtime_event.inp_path)) as sim, Simulation(str(event.inp_path)) as internal_sim:
        sim.step_advance(STATE_STEP_SEC)
        internal_sim.step_advance(STATE_STEP_SEC)
        sim.start()
        internal_sim.start()
        nodes = Nodes(sim)
        links = Links(sim)
        gages = RainGages(sim)
        internal_links = Links(internal_sim)
        node_ids = _ids_from_container(nodes, "nodeid")
        node_objs = {nid: nodes[nid] for nid in node_ids}
        actuator_ids, link_objs = _get_existing_links(links, ids)
        internal_ids, internal_link_objs = _get_existing_links(internal_links, ids)
        if actuator_ids != ids or internal_ids != ids:
            raise RuntimeError("Formal Proposed plant/shadow do not expose Engineering36")
        rain_ids = _ids_from_container(gages, "raingageid")
        rain_obj = gages[rain_ids[0]] if rain_ids else None
        current_action = _observed_action_from_links(link_objs, ids, actuators)
        command = current_action.copy()
        shadow_iter = iter(internal_sim)
        for _ in sim:
            try:
                next(shadow_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "Dynamic-Internal causal shadow ended before Proposed plant"
                ) from exc
            pre = _frame(sim, node_objs, rain_obj, link_objs, ids, actuators)
            frames.append(pre)
            elapsed = float(pre["elapsed_min"])
            if _is_decision_time(elapsed):
                history, uncertainty, ood_score = reconstruct_history(
                    frames, bundle, state_source=state_source
                )
                historical_actions = np.stack(
                    [np.asarray(x["action"], np.float32) for x in frames[-13:]]
                )
                rainfall_forecast = make_causal_rainfall_forecast(
                    [float(x["rain"]) for x in frames], horizon_steps=HORIZON_STEPS
                )
                internal_current = _observed_action_from_links(
                    internal_link_objs, ids, actuators
                )
                command, info = predict_and_decide(
                    bundle=bundle,
                    actuators=actuators,
                    state_history=history,
                    historical_actions=historical_actions,
                    rainfall_forecast=rainfall_forecast,
                    current_action=current_action,
                    internal_current_action=internal_current,
                    gat_ood_score=ood_score,
                    max_candidate_sequences=max_candidate_sequences,
                )
                info.update(
                    {
                        "event_id": event.event_id,
                        "rainfall_sha256": event.rainfall_sha256,
                        "elapsed_min": elapsed,
                        "state_source": state_source,
                        "gat_uncertainty_mean": float(np.mean(uncertainty)),
                        "reconstructed_history_ready_before_mpc": True,
                        "reconstructed_history_contract": (
                            "PROJECT6_V42_CAUSAL_RECONSTRUCTED_HISTORY_V1"
                            if state_source == "gat_sparse_reconstruction"
                            else "true_state_diagnostic"
                        ),
                        "current_frame_repetition_used": False,
                        "authoritative_swmm_history_used_as_online_input": state_source
                        == "true_state",
                        "gat_uncertainty_used": state_source
                        == "gat_sparse_reconstruction",
                        "ood_gate_used": state_source
                        == "gat_sparse_reconstruction",
                        "sensor_layout_sha256": bundle.sensor_layout_sha256,
                        "plant_native_controls_disabled": True,
                        "internal_shadow_native_controls_preserved": True,
                        "internal_shadow_future_state_used_online": False,
                    }
                )
                decisions.append(info)
                for i, aid in enumerate(ids):
                    link_objs[aid].target_setting = float(command[i])
            readback = _observed_action_from_links(link_objs, ids, actuators)
            current_action = readback.copy()
            records.append(
                _record_row(
                    frame=pre,
                    event_id=event.event_id,
                    strategy="Proposed",
                    command=np.asarray(command),
                    readback=readback,
                    node_objs=node_objs,
                    link_objs=link_objs,
                    actuator_ids=ids,
                )
            )

    detail = pd.DataFrame(records)
    detail.to_csv(detail_path, index=False)
    with decision_path.open("w", encoding="utf-8") as handle:
        for item in decisions:
            handle.write(
                json.dumps(_json_safe(item), ensure_ascii=False, allow_nan=False)
                + "\n"
            )
    kpis = compute_kpis(detail, priority_nodes, dt_sec=STATE_STEP_SEC)
    result = {
        "status": "pass",
        "event_id": event.event_id,
        "rainfall_sha256": event.rainfall_sha256,
        "strategy": "Proposed",
        "authority": "authoritative_swmm",
        "state_source": state_source,
        "detail_path": str(detail_path),
        "detail_sha256": sha256_file(detail_path),
        "decision_path": str(decision_path),
        "source_input_sha256": event.input_sha256,
        "runtime_rule_free_inp_sha256": runtime_event.input_sha256,
        "physical_network_sha256": physical_network_sha256(event.inp_path),
        "kpis": kpis,
        "decision_count": len(decisions),
        "fallback_rate": (
            float(np.mean([bool(x.get("used_fallback", True)) for x in decisions]))
            if decisions
            else 1.0
        ),
        "canonical_pfvfirst_mpc_v42": True,
        "control_objective_contract": FORMAL_OBJECTIVE_CONTRACT,
        "gat_model_sha256": bundle.gat_model_sha256,
        "surrogate_model_sha256": bundle.surrogate_model_sha256,
        "fallback_contract_sha256": bundle.fallback_contract_sha256,
        "online_future_hydraulic_truth_used": False,
        "realized_future_rainfall_used_online": False,
        "dynamic_internal_online_forecast": "causal_current_native_rule_setting_persistence",
        "plant_native_controls_disabled": True,
        "internal_shadow_native_controls_preserved": True,
        "internal_shadow_future_state_used_online": False,
        "runtime_sec": time.time() - started,
    }
    (out_dir / "run_result.json").write_text(
        json.dumps(_json_safe(result), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return result
